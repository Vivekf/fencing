"""One-shot: scrape just Francesca (the focal fencer) to validate Phase A.

Usage:
    PYTHONPATH=. python scripts/scrape_focal.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from fencing_tracker import db, scraper
from fencing_tracker.http import HttpClient

FRANCESCA_ID = 100835605
FRANCESCA_SLUG = "Francesca-Farias"
FRANCESCA_NAME = "Francesca Farias"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    db_path = Path("fencing.db")
    cache_dir = Path(".cache")

    conn = db.connect(db_path)
    db.init_schema(conn)

    db.ensure_fencer(
        conn,
        fencer_id=FRANCESCA_ID,
        name=FRANCESCA_NAME,
        slug=FRANCESCA_SLUG,
        bfs_depth=0,
    )
    db.set_fencer_status(conn, FRANCESCA_ID, "in_progress")
    conn.commit()

    client = HttpClient(cache_dir=cache_dir, delay_seconds=1.5)

    stats = scraper.scrape_fencer_history(
        conn,
        client,
        fencer_id=FRANCESCA_ID,
        slug=FRANCESCA_SLUG,
        enqueue_new_opponents=True,
        opponent_depth=1,
    )
    db.set_fencer_status(conn, FRANCESCA_ID, "done")
    conn.commit()

    print()
    print("Scrape complete.")
    print(f"  events seen      : {stats.events_seen}")
    print(f"  bouts seen       : {stats.bouts_seen}")
    print(f"  bouts inserted   : {stats.bouts_inserted}")
    print(f"  new opponents    : {stats.fencers_discovered}")
    print()
    print("DB counts:")
    print(f"  fencers          : {conn.execute('SELECT COUNT(*) FROM fencers').fetchone()[0]}")
    print(f"  events           : {conn.execute('SELECT COUNT(*) FROM events').fetchone()[0]}")
    print(f"  bouts            : {conn.execute('SELECT COUNT(*) FROM bouts').fetchone()[0]}")
    print(f"  fencer_event_results : {conn.execute('SELECT COUNT(*) FROM fencer_event_results').fetchone()[0]}")

    print()
    print("Spot-check: Francesca vs Alexa Cahalane bout in event 41045:")
    row = conn.execute(
        """
        SELECT bout_type, fencer_a_id, fencer_a_score, fencer_b_id, fencer_b_score, winner_id
        FROM bouts
        WHERE event_id = 41045
          AND ((fencer_a_id = ? AND fencer_b_id = ?) OR (fencer_a_id = ? AND fencer_b_id = ?))
        """,
        (FRANCESCA_ID, 100357694, 100357694, FRANCESCA_ID),
    ).fetchone()
    print(f"  {dict(row) if row else 'NOT FOUND'}")

    conn.close()


if __name__ == "__main__":
    main()
