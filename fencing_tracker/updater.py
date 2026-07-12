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

from . import db, frontier, parsers, scraper, upcoming
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


# ---------------------------------------------------------------------------
# Event-driven update (Option B): scan the core's registrations for the watch-list,
# then ingest each completed event's whole field from its /results page. Far cheaper
# than re-scraping every stale fencer — see PLAN-update.md.
# ---------------------------------------------------------------------------

@dataclass
class EventDrivenStats:
    upcoming: upcoming.UpcomingStats = field(default_factory=upcoming.UpcomingStats)
    core_scanned: int = 0
    scan_errors: int = 0
    events_awaiting: int = 0
    events_ingested: int = 0
    events_unresolved: int = 0
    bouts_added: int = 0
    ingest_errors: int = 0
    frontier: frontier.FrontierStats = field(default_factory=frontier.FrontierStats)


def _bar(progress: bool, total: int, desc: str, unit: str):
    if not (progress and total):
        return None
    from tqdm import tqdm
    return tqdm(total=total, desc=desc, unit=unit)


def _core_scan_list(conn, core, focal_id, focal_slug, max_core_scan):
    """Fencers to scan for registrations: focal first, then the rest of the core (with a
    slug), capped at `max_core_scan` (None = all)."""
    slugs = dict(conn.execute("SELECT id, slug FROM fencers"))
    ordered = [(focal_id, focal_slug or slugs.get(focal_id))]
    ordered += [(fid, slugs.get(fid)) for fid in sorted(core)
                if fid != focal_id and slugs.get(fid)]
    return ordered[:max_core_scan] if max_core_scan else ordered


def _meta_match(pe, weapon, gender, age) -> bool:
    def eq(a, b):
        return (a or None) == (b or None)
    return eq(pe.weapon, weapon) and eq(pe.gender, gender) and eq(pe.age_group, age)


def _resolve_results_event_id(conn, client, ev, focal_id, focal_slug, *, max_tries=3):
    """Find the historical results event id for a past-dated prereg event by parsing a
    participant's history (focal first, then profiled registrants) and matching on
    (date, weapon, gender, age). No history is persisted — the whole field is ingested
    from the results page. Returns None if unresolved (results not posted yet)."""
    regs = db.upcoming_registrants(conn, ev["event_id"])
    candidates: list[tuple[int, str]] = []
    focal_reg = next((r for r in regs if r["fencer_id"] == focal_id), None)
    if focal_reg is not None:
        candidates.append((focal_id, focal_slug or focal_reg["slug"]))
    candidates += [(r["fencer_id"], r["slug"]) for r in regs
                   if r["fencer_id"] != focal_id and r["has_profile"] and r["slug"]]

    for fid, slug in candidates[:max_tries]:
        try:
            html = client.get(scraper.history_url(fid, slug), use_cache=False)
        except Exception:
            log.exception("resolve: history fetch failed for fencer %s", fid)
            continue
        for pe in parsers.parse_history(html):
            if pe.event_date == ev["event_date"] and _meta_match(
                pe, ev["weapon"], ev["gender"], ev["age_group"]
            ):
                return pe.event_id
    return None


def run_event_driven_update(
    conn: sqlite3.Connection,
    client: HttpClient,
    *,
    focal_id: int,
    focal_slug: Optional[str],
    core_k: int = 3,
    radius: int = 2,
    youth_frac_min: float = 0.5,
    max_core_scan: Optional[int] = None,
    max_new: int = 2000,
    today_iso: Optional[str] = None,
    progress: bool = True,
) -> EventDrivenStats:
    today_iso = today_iso or _today_iso()
    stats = EventDrivenStats()

    # --- Stage A: core-registration scan -> watch-list of upcoming events ------
    core, _, _ = frontier.compute_core(conn, focal_id, core_k, radius,
                                       youth_frac_min=youth_frac_min)
    scan_list = _core_scan_list(conn, core, focal_id, focal_slug, max_core_scan)
    log.info("Stage A: scanning registrations for %d/%d core fencer(s)",
             len(scan_list), len(core))
    seen_events: set[int] = set()
    bar = _bar(progress, len(scan_list), "reg-scan", "fencer")
    for fid, slug in scan_list:
        try:
            html = client.get(upcoming.summary_url(fid, slug), use_cache=False)
            for reg in parsers.parse_registrations(html):
                stats.upcoming.registrations_found += 1
                if reg.event_id in seen_events:          # dedupe rosters across the core
                    continue
                seen_events.add(reg.event_id)
                upcoming.ingest_event_roster(conn, client, reg,
                                             today_iso=today_iso, stats=stats.upcoming)
            stats.core_scanned += 1
        except Exception:
            stats.scan_errors += 1
            log.exception("registration scan failed for fencer %s", fid)
        finally:
            if bar:
                bar.update(1)
    if bar:
        bar.close()

    # --- Stage B: resolve + ingest completed events ---------------------------
    awaiting = db.upcoming_events_awaiting_results(conn, today_iso)
    stats.events_awaiting = len(awaiting)
    log.info("Stage B: %d past-dated event(s) awaiting results", len(awaiting))
    bar = _bar(progress, len(awaiting), "ingest", "event")
    for ev in awaiting:
        rid = _resolve_results_event_id(conn, client, ev, focal_id, focal_slug)
        if rid is None:
            stats.events_unresolved += 1
        else:
            try:
                sub = scraper.scrape_event_results(conn, client, rid, opponent_hops=radius)
                db.set_upcoming_results_event(conn, ev["event_id"], rid)
                conn.commit()
                stats.events_ingested += 1
                stats.bouts_added += sub.bouts_inserted
            except Exception:
                stats.ingest_errors += 1
                log.exception("results ingest failed for prereg %s -> %s",
                              ev["event_id"], rid)
        if bar:
            bar.update(1)
    if bar:
        bar.close()

    # --- Stage C: frontier expansion for newly-discovered fencers -------------
    log.info("Stage C: frontier expansion (max_new=%s)", max_new)
    stats.frontier = frontier.expand_frontier(
        conn, client, focal_id=focal_id, k=core_k, radius=radius,
        max_new=max_new, youth_frac_min=youth_frac_min, progress=progress,
    )
    return stats
