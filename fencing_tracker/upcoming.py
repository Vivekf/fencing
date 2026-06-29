"""Pull a fencer's upcoming (preregistered) events and their fields into the DB.

A fencer's summary page (`/p/{id}/{slug}`) has a *Registrations* section listing the
events they're entered in. Each links to `/event/{id}` (note: NO `/results` suffix —
that's how upcoming events are told apart from past ones), a preregistration roster
of every fencer in the field.

This module only records the registrations and fields. New field members are *seeded*
for frontier expansion (see `frontier.py`) rather than scraped inline, so the per-run
cap applies uniformly. Once an event's date has passed its results arrive via the
normal history refresh, so its roster is no longer re-pulled.

The roster pages also carry fencingtracker's strength / conservative-estimate numbers;
per project policy those are deliberately NOT captured.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Optional

from . import db, parsers
from .http import HttpClient

log = logging.getLogger(__name__)

SUMMARY_URL = "https://fencingtracker.com/p/{fencer_id}/{slug}"
EVENT_URL = "https://fencingtracker.com/event/{event_id}"


@dataclass
class UpcomingStats:
    registrations_found: int = 0
    future_events_scraped: int = 0
    past_events_skipped: int = 0
    registrants_total: int = 0
    new_fencers: int = 0          # field members not previously in `fencers`
    errors: int = 0


def summary_url(fencer_id: int, slug: Optional[str]) -> str:
    return SUMMARY_URL.format(fencer_id=fencer_id, slug=slug or "x")


def refresh_upcoming(
    conn: sqlite3.Connection,
    client: HttpClient,
    fencer_id: int,
    slug: Optional[str],
    *,
    today_iso: Optional[str] = None,
    stats: Optional[UpcomingStats] = None,
) -> UpcomingStats:
    """Refresh a fencer's upcoming events + fields. Field members become anchors for the
    scrape-core (frontier.compute_core); expansion itself is decided by connectivity, not
    seeded here. Past-dated events are left for the history refresh."""
    stats = stats or UpcomingStats()

    # Registrations and rosters change over time, so always re-fetch (skip the cache).
    summary_html = client.get(summary_url(fencer_id, slug), use_cache=False)
    regs = parsers.parse_registrations(summary_html)
    stats.registrations_found += len(regs)
    log.info("Found %d registration(s) for fencer %s", len(regs), fencer_id)

    for reg in regs:
        if today_iso and reg.event_date and reg.event_date < today_iso:
            stats.past_events_skipped += 1
            log.info("Event %s on %s has passed; results come via history refresh",
                     reg.event_id, reg.event_date)
            continue
        try:
            html = client.get(EVENT_URL.format(event_id=reg.event_id), use_cache=False)
            roster = parsers.parse_event_roster(html, event_id=reg.event_id)
        except Exception:
            stats.errors += 1
            log.exception("Failed to fetch/parse roster for event %s", reg.event_id)
            continue

        # fencingtracker occasionally lists a fencer twice; keep distinct entrants.
        unique_entries = list({e.fencer_id: e for e in roster.entries}.values())

        db.upsert_upcoming_event(
            conn,
            event_id=reg.event_id,
            tournament_name=roster.tournament_name or reg.tournament_name,
            event_name=roster.event_name or reg.event_name,
            classification=roster.classification,
            weapon=roster.weapon,
            gender=roster.gender,
            age_group=roster.age_group,
            venue=roster.venue,
            location=roster.location,
            start_datetime=roster.start_datetime,
            event_date=roster.event_date or reg.event_date,
            field_size=len(unique_entries),
        )

        registrants: list[tuple[int, Optional[str], Optional[str]]] = []
        for entry in unique_entries:
            is_new = db.ensure_fencer(
                conn,
                fencer_id=entry.fencer_id,
                name=entry.name,
                slug=entry.slug,
                club=entry.club,
                has_profile=True,
            )
            if is_new:
                stats.new_fencers += 1
            registrants.append((entry.fencer_id, entry.name, entry.club))

        db.replace_upcoming_registrants(conn, reg.event_id, registrants)
        stats.future_events_scraped += 1
        stats.registrants_total += len(registrants)
        conn.commit()
        log.info("Event %s: %d registrants", reg.event_id, len(registrants))

    return stats
