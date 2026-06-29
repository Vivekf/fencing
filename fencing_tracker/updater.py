"""The continual, idempotent update: one operation, run repeatedly.

All work is derived from DB state, so re-running just converges — safe to run on a
schedule. Three stages:

  Stage 0  Refresh upcoming events + fields (registrations and rosters change over
           time) — the field members become anchors for the scrape-core.
  Stage 1  Catch events that have happened: re-fetch the histories of already-scraped
           fencers whose data is stale (cache-bypassed, since the cache is permanent),
           upserting any new bouts.
  Stage 2  Grow the scrape-core by connectivity-gated, anchor-rooted expansion
           (k distinct core opponents, within `radius` hops of focal + field), capped.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from . import db, frontier, scraper, upcoming
from .http import HttpClient

log = logging.getLogger(__name__)


@dataclass
class UpdateStats:
    upcoming: upcoming.UpcomingStats = field(default_factory=upcoming.UpcomingStats)
    refreshed: int = 0              # stale histories re-pulled
    refresh_bouts_added: int = 0
    refresh_errors: int = 0
    frontier: frontier.FrontierStats = field(default_factory=frontier.FrontierStats)


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def run_update(
    conn: sqlite3.Connection,
    client: HttpClient,
    *,
    focal_id: int,
    focal_slug: Optional[str],
    refresh_after_days: int = 14,
    max_new: int = 2000,
    core_k: int = 3,
    radius: int = 2,
    youth_frac_min: float = 0.5,
    today_iso: Optional[str] = None,
    progress: bool = True,
) -> UpdateStats:
    today_iso = today_iso or _today_iso()
    stats = UpdateStats()

    # --- Stage 0: upcoming events + fields -----------------------------------
    log.info("Stage 0: refreshing upcoming events")
    upcoming.refresh_upcoming(
        conn, client, focal_id, focal_slug,
        today_iso=today_iso, stats=stats.upcoming,
    )

    # --- Stage 1: refresh stale histories (events that happened) --------------
    cutoff = (date.fromisoformat(today_iso) - timedelta(days=refresh_after_days)).isoformat()
    targets = db.fencers_to_refresh(conn, cutoff, focal_id)
    if youth_frac_min > 0:
        # Only maintain the relevant young cohort (+ focal always); don't re-pull the
        # teen/adult histories that the broad expansion happened to scrape.
        youth = frontier._youth_fraction(conn)
        targets = [r for r in targets if r["id"] == focal_id
                   or (youth.get(r["id"]) is None or youth.get(r["id"]) >= youth_frac_min)]
    log.info("Stage 1: %d fencer histories to refresh (stale before %s)", len(targets), cutoff)
    bar = None
    if progress and targets:
        from tqdm import tqdm
        bar = tqdm(total=len(targets), desc="refresh", unit="fencer")
    try:
        for row in targets:
            try:
                sub = scraper.scrape_fencer_history(
                    conn, client,
                    fencer_id=row["id"], slug=row["slug"],
                    enqueue_new_opponents=False,
                    opponent_depth=None,
                    opponent_hops=None,          # expansion is decided later by connectivity, not seeding
                    use_cache=False,             # must bypass the permanent cache to see new events
                )
                db.set_fencer_status(conn, row["id"], "done", history_pages=1)
                stats.refreshed += 1
                stats.refresh_bouts_added += sub.bouts_inserted
            except Exception:
                stats.refresh_errors += 1
                log.exception("Error refreshing fencer %s", row["id"])
            finally:
                conn.commit()
                if bar is not None:
                    bar.update(1)
    finally:
        if bar is not None:
            bar.close()

    # --- Stage 2: grow the scrape-core (connectivity-gated) -------------------
    log.info("Stage 2: growing scrape-core (k=%d, radius=%d, cap %s)", core_k, radius, max_new)
    stats.frontier = frontier.expand_frontier(
        conn, client, focal_id=focal_id, k=core_k, radius=radius,
        max_new=max_new, youth_frac_min=youth_frac_min, progress=progress,
    )

    return stats
