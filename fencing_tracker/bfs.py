"""BFS controller: scrape outward from the focal fencer up to a cap.

Cap semantics: stop after `cap` fencers have been scraped (status='done').
The discovered pool may be much larger than `cap`; those opponents stay in
`fencers` with `scrape_status='discovered'` so bouts retain referential
integrity but their own histories are not fetched.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field

from . import db, scraper
from .http import HttpClient

log = logging.getLogger(__name__)


@dataclass
class BfsStats:
    fencers_scraped: int = 0
    bouts_inserted: int = 0
    bouts_seen: int = 0
    bouts_skipped: int = 0
    events_seen: int = 0
    legacy_opponents: int = 0
    errors: int = 0
    by_depth: dict[int, int] = field(default_factory=dict)


def run_bfs(
    conn: sqlite3.Connection,
    client: HttpClient,
    focal_id: int,
    focal_name: str,
    focal_slug: str | None,
    cap: int,
    *,
    progress: bool = True,
) -> BfsStats:
    """Run BFS from focal fencer until `cap` fencers scraped or queue exhausted."""
    db.ensure_fencer(conn, focal_id, focal_name, focal_slug, bfs_depth=0)
    conn.commit()

    reset = db.reset_in_progress(conn)
    if reset:
        log.info("Reset %d 'in_progress' fencer(s) to 'discovered'", reset)
        conn.commit()

    stats = BfsStats()
    already_done = conn.execute(
        "SELECT COUNT(*) FROM fencers WHERE scrape_status = 'done'"
    ).fetchone()[0]

    bar = None
    if progress:
        from tqdm import tqdm
        bar = tqdm(total=cap, desc="BFS", unit="fencer", initial=already_done)

    try:
        while True:
            scraped_so_far = already_done + stats.fencers_scraped
            if scraped_so_far >= cap:
                log.info("Cap %d reached; stopping BFS.", cap)
                break

            fencer_row = db.next_to_scrape(conn)
            if fencer_row is None:
                log.info("Queue exhausted after %d scrapes.", scraped_so_far)
                break

            fencer_id = fencer_row["id"]
            slug = fencer_row["slug"]
            depth = fencer_row["bfs_depth"] or 0

            db.set_fencer_status(conn, fencer_id, "in_progress")
            conn.commit()

            try:
                sub = scraper.scrape_fencer_history(
                    conn,
                    client,
                    fencer_id=fencer_id,
                    slug=slug,
                    enqueue_new_opponents=True,
                    opponent_depth=depth + 1,
                )
                db.set_fencer_status(conn, fencer_id, "done", history_pages=1)
                stats.fencers_scraped += 1
                stats.bouts_inserted += sub.bouts_inserted
                stats.bouts_seen += sub.bouts_seen
                stats.bouts_skipped += sub.bouts_skipped
                stats.events_seen += sub.events_seen
                stats.legacy_opponents += sub.legacy_opponents
                stats.by_depth[depth] = stats.by_depth.get(depth, 0) + 1
                if bar is not None:
                    bar.update(1)
            except Exception as exc:
                stats.errors += 1
                db.set_fencer_status(
                    conn,
                    fencer_id,
                    "error",
                    error_message=f"{type(exc).__name__}: {exc}",
                )
                log.exception("Error scraping fencer %s", fencer_id)
            finally:
                conn.commit()
    finally:
        if bar is not None:
            bar.close()

    return stats
