"""Scrape a single fencer's history into the DB.

Bridges parsers + db. BFS orchestration lives in bfs.py.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

from . import db, parsers
from .http import HttpClient

log = logging.getLogger(__name__)

HISTORY_URL = "https://fencingtracker.com/p/{fencer_id}/{slug}/history"


@dataclass
class ScrapeStats:
    fencer_id: int
    url: str
    events_seen: int = 0
    bouts_inserted: int = 0
    bouts_seen: int = 0
    bouts_skipped: int = 0          # rows with no identifiable opponent
    fencers_discovered: int = 0
    legacy_opponents: int = 0       # opponents added with has_profile=0


def history_url(fencer_id: int, slug: Optional[str]) -> str:
    return HISTORY_URL.format(fencer_id=fencer_id, slug=slug or "x")


def scrape_fencer_history(
    conn: sqlite3.Connection,
    client: HttpClient,
    fencer_id: int,
    slug: Optional[str],
    *,
    enqueue_new_opponents: bool,
    opponent_depth: Optional[int],
    opponent_hops: Optional[int] = None,
    use_cache: bool = True,
) -> ScrapeStats:
    """Fetch + parse + persist one fencer's history. The site renders a fencer's
    entire history on a single page, so one fetch is complete.

    Opponents are always inserted so bouts keep referential integrity. Two optional
    discovery policies control whether an opponent becomes scrapeable:
      - `enqueue_new_opponents` + `opponent_depth`: BFS-style depth tagging (bootstrap).
      - `opponent_hops`: frontier-expansion budget — when >= 1, a new/undone opponent
        is queued for expansion with that many hops remaining (continual updates).

    `use_cache=False` forces a fresh fetch (used when refreshing for new events).
    """
    url = history_url(fencer_id, slug)
    stats = ScrapeStats(fencer_id=fencer_id, url=url)
    started = time.monotonic()
    error_message: Optional[str] = None
    try:
        html = client.get(url, use_cache=use_cache)
        birth_year = parsers.parse_birth_year(html)
        if birth_year is not None:
            db.set_fencer_birth_year(conn, fencer_id, birth_year)
        events = parsers.parse_history(html)
        stats.events_seen = len(events)
        for ev in events:
            _persist_event(
                conn, fencer_id, ev, opponent_depth, enqueue_new_opponents,
                opponent_hops, stats,
            )
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        log.exception("scrape_fencer_history failed for %s", fencer_id)
        raise
    finally:
        duration_ms = int((time.monotonic() - started) * 1000)
        db.log_scrape(
            conn,
            fencer_id=fencer_id,
            url=url,
            bouts_added=stats.bouts_inserted,
            fencers_added=stats.fencers_discovered,
            duration_ms=duration_ms,
            error_message=error_message,
        )
        conn.commit()
    return stats


def backfill_birth_years_from_cache(
    conn: sqlite3.Connection,
    client: HttpClient,
    *,
    only_missing: bool = True,
) -> dict:
    """Populate fencers.birth_year by re-parsing cached history HTML — no network.

    The history page (already fetched for every scraped fencer) carries the birth year
    in its hero header, so we can backfill the whole dataset from `.cache/` for free.
    """
    where = "WHERE has_profile = 1 AND scrape_status = 'done'"
    if only_missing:
        where += " AND birth_year IS NULL"
    rows = conn.execute(f"SELECT id, slug FROM fencers {where}").fetchall()
    stats = {"total": len(rows), "found": 0, "no_cache": 0, "no_year": 0}
    # Regex fast-path: the cached history pages can be multi-MB, so avoid a full
    # BeautifulSoup parse just to read the hero birth-year.
    fast = re.compile(r'person-hero__birth-year[^>]*>\s*((?:19|20)\d{2})')
    for r in rows:
        html = client._read_cache(history_url(r["id"], r["slug"]))
        if html is None:
            stats["no_cache"] += 1
            continue
        m = fast.search(html)
        by = int(m.group(1)) if m else parsers.parse_birth_year(html)
        if by is None:
            stats["no_year"] += 1
            continue
        db.set_fencer_birth_year(conn, r["id"], by)
        stats["found"] += 1
    conn.commit()
    return stats


def _persist_event(
    conn: sqlite3.Connection,
    focal_id: int,
    ev: parsers.ParsedEvent,
    opponent_depth: Optional[int],
    enqueue_new_opponents: bool,
    opponent_hops: Optional[int],
    stats: ScrapeStats,
) -> None:
    db.upsert_event(
        conn,
        event_id=ev.event_id,
        name=ev.tournament_name,
        classification=ev.classification,
        weapon=ev.weapon,
        gender=ev.gender,
        age_group=ev.age_group,
        rating_level=ev.rating_level,
        event_date=ev.event_date,
        raw_date=ev.raw_date,
    )
    stats.bouts_skipped += ev.skipped_bouts
    db.upsert_fencer_event_result(
        conn,
        fencer_id=focal_id,
        event_id=ev.event_id,
        seed=ev.focal_seed,
        placement=ev.focal_placement,
        field_size=ev.focal_field_size,
        rating_earned=ev.focal_rating,
    )

    # Y-8 (and similar) double round-robin pools repeat the same pair within one event.
    # Number repeats by row order so the PK can disambiguate.
    seq_counter: dict[tuple[int, str], int] = {}
    for bout in ev.bouts:
        stats.bouts_seen += 1
        # fencingtracker occasionally lists a fencer against themselves due to data-
        # entry errors on their end (e.g., merged duplicate profiles). Skip silently.
        if bout.opponent_id == focal_id:
            stats.bouts_skipped += 1
            log.debug("Skipping self-bout: fencer %s vs themselves in event %s",
                      focal_id, ev.event_id)
            continue
        is_new = db.ensure_fencer(
            conn,
            fencer_id=bout.opponent_id,
            name=bout.opponent_name,
            slug=bout.opponent_slug,
            club=bout.opponent_club,
            bfs_depth=opponent_depth if (enqueue_new_opponents and bout.opponent_has_profile) else None,
            has_profile=bout.opponent_has_profile,
            scrape_hops=opponent_hops if bout.opponent_has_profile else None,
            gender=ev.gender,   # the event's gender tags the opponent (single-gender only)
        )
        if is_new:
            stats.fencers_discovered += 1
            if not bout.opponent_has_profile:
                stats.legacy_opponents += 1

        # Canonical ordering
        if focal_id < bout.opponent_id:
            a_id, b_id = focal_id, bout.opponent_id
            a_score, b_score = bout.focal_score, bout.opp_score
        else:
            a_id, b_id = bout.opponent_id, focal_id
            a_score, b_score = bout.opp_score, bout.focal_score

        winner_id = focal_id if bout.focal_won else bout.opponent_id

        key = (bout.opponent_id, bout.bout_type)
        seq_counter[key] = seq_counter.get(key, 0) + 1
        bout_seq = seq_counter[key]

        inserted = db.insert_bout(
            conn,
            event_id=ev.event_id,
            fencer_a_id=a_id,
            fencer_b_id=b_id,
            fencer_a_score=a_score,
            fencer_b_score=b_score,
            winner_id=winner_id,
            bout_type=bout.bout_type,
            bout_seq=bout_seq,
            source_fencer_id=focal_id,
        )
        if inserted:
            stats.bouts_inserted += 1
