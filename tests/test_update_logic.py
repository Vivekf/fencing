"""Unit tests for the continual-update mechanics (hops queue + refresh targets).

These exercise the DB-level rules that drive frontier expansion and staleness
refresh, deterministically and without any network access.

Run: python -m pytest tests/test_update_logic.py -v
Or:  python tests/test_update_logic.py
"""

from __future__ import annotations

import sqlite3

from fencing_tracker import db


def _fresh_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_schema(conn)
    return conn


def _status(conn, fid):
    return conn.execute(
        "SELECT scrape_status, scrape_hops FROM fencers WHERE id=?", (fid,)
    ).fetchone()


def test_new_fencer_seeded_with_hops():
    conn = _fresh_db()
    assert db.ensure_fencer(conn, 1, "A", scrape_hops=2) is True
    row = _status(conn, 1)
    assert row["scrape_status"] == "discovered"
    assert row["scrape_hops"] == 2


def test_hops_raise_never_lowers_and_skips_done():
    conn = _fresh_db()
    db.ensure_fencer(conn, 1, "A", scrape_hops=1)
    # Raising to 2 should take effect.
    db.ensure_fencer(conn, 1, "A", scrape_hops=2)
    assert _status(conn, 1)["scrape_hops"] == 2
    # Lowering back to 1 must not reduce it.
    db.ensure_fencer(conn, 1, "A", scrape_hops=1)
    assert _status(conn, 1)["scrape_hops"] == 2
    # Once done, it is never re-queued.
    db.set_fencer_status(conn, 1, "done")
    db.set_scrape_hops(conn, 1, 0)
    db.ensure_fencer(conn, 1, "A", scrape_hops=2)
    assert _status(conn, 1)["scrape_hops"] == 0


def test_seed_only_affects_undone_profiled():
    conn = _fresh_db()
    db.ensure_fencer(conn, 1, "A")                       # discovered, hops 0
    db.ensure_fencer(conn, 2, "Legacy", has_profile=False)  # skipped, no profile
    db.ensure_fencer(conn, 3, "Done")
    db.set_fencer_status(conn, 3, "done")
    db.seed_fencer_for_expansion(conn, 1, 2)
    db.seed_fencer_for_expansion(conn, 2, 2)
    db.seed_fencer_for_expansion(conn, 3, 2)
    assert _status(conn, 1)["scrape_hops"] == 2
    assert _status(conn, 2)["scrape_hops"] == 0   # no profile -> not queued
    assert _status(conn, 3)["scrape_hops"] == 0   # done -> not queued


def test_next_to_expand_orders_by_hops_then_id():
    conn = _fresh_db()
    db.ensure_fencer(conn, 10, "low", scrape_hops=1)
    db.ensure_fencer(conn, 5, "high", scrape_hops=2)
    db.ensure_fencer(conn, 7, "high2", scrape_hops=2)
    db.ensure_fencer(conn, 9, "zero", scrape_hops=0)   # not queued
    assert db.count_pending_expansion(conn) == 3
    first = db.next_to_expand(conn)
    assert first["id"] == 5 and first["scrape_hops"] == 2   # highest hops, lowest id


def test_two_hop_decrement_chain():
    """Simulate the depth-2 rule the way the scraper drives it.

    new fencer (hops 2) -> opponent gets hops 1 (scraped) -> their opponent gets
    hops 0 (recorded, not scraped).
    """
    conn = _fresh_db()
    # Seed a brand-new fencer with budget 2.
    db.ensure_fencer(conn, 1, "New", scrape_hops=2)
    # Expand fencer 1: scraper seeds opponents with hops-1 = 1, then marks 1 done.
    db.ensure_fencer(conn, 2, "Opp", scrape_hops=1)
    db.set_fencer_status(conn, 1, "done"); db.set_scrape_hops(conn, 1, 0)
    assert _status(conn, 2)["scrape_hops"] == 1
    assert db.count_pending_expansion(conn) == 1   # only fencer 2 queued now
    # Expand fencer 2 (hops 1): opponents seeded with hops-1 = 0 -> not queued.
    db.ensure_fencer(conn, 3, "Opp2", scrape_hops=0)
    db.set_fencer_status(conn, 2, "done"); db.set_scrape_hops(conn, 2, 0)
    assert _status(conn, 3)["scrape_status"] == "discovered"
    assert _status(conn, 3)["scrape_hops"] == 0
    assert db.count_pending_expansion(conn) == 0   # frontier exhausted at depth 2


def test_fencers_to_refresh_stale_plus_focal():
    conn = _fresh_db()
    # Three done fencers with different last_scraped_at; one is the focal.
    for fid, name in [(1, "focal"), (2, "stale"), (3, "fresh")]:
        db.ensure_fencer(conn, fid, name)
        db.set_fencer_status(conn, fid, "done")
    conn.execute("UPDATE fencers SET last_scraped_at='2026-01-01T00:00:00+00:00' WHERE id=2")
    conn.execute("UPDATE fencers SET last_scraped_at='2026-06-27T00:00:00+00:00' WHERE id=3")
    conn.execute("UPDATE fencers SET last_scraped_at='2026-06-27T00:00:00+00:00' WHERE id=1")
    ids = {r["id"] for r in db.fencers_to_refresh(conn, "2026-06-14", focal_id=1)}
    assert ids == {1, 2}   # focal always (1) + stale (2); fresh (3) excluded


def _add_bout(conn, event_id, gender, a, b):
    conn.execute(
        "INSERT OR IGNORE INTO events (id, gender, first_seen_at) VALUES (?, ?, '2026-01-01')",
        (event_id, gender),
    )
    lo, hi = (a, b) if a < b else (b, a)
    conn.execute(
        """INSERT OR IGNORE INTO bouts
           (event_id, fencer_a_id, fencer_b_id, bout_type, bout_seq,
            fencer_a_score, fencer_b_score, winner_id, source_fencer_id)
           VALUES (?, ?, ?, 'Pool', 1, 5, 3, ?, ?)""",
        (event_id, lo, hi, lo, lo),
    )


def test_backfill_gender():
    conn = _fresh_db()
    for fid in (1, 2, 3, 4):
        db.ensure_fencer(conn, fid, f"f{fid}")
    _add_bout(conn, 100, "M", 1, 2)   # 1 & 2 only in a men's event -> 'M'
    _add_bout(conn, 200, "W", 3, 4)   # 3 & 4 only in a women's event -> 'W'
    _add_bout(conn, 300, "X", 1, 3)   # mixed: doesn't override single-gender signal
    n = db.backfill_gender(conn)
    g = {r["id"]: r["gender"] for r in conn.execute("SELECT id, gender FROM fencers")}
    assert g[1] == "M" and g[2] == "M"
    assert g[3] == "W" and g[4] == "W"
    assert n == 4


def test_expand_skips_male_without_focal_bout():
    conn = _fresh_db()
    FOCAL = 999
    db.ensure_fencer(conn, FOCAL, "focal", gender="W")
    db.set_fencer_status(conn, FOCAL, "done")
    db.ensure_fencer(conn, 1, "man", gender="M", scrape_hops=2)       # male, no focal bout
    db.ensure_fencer(conn, 2, "woman", gender="W", scrape_hops=2)     # female
    db.ensure_fencer(conn, 3, "unknown", scrape_hops=2)              # NULL gender
    db.ensure_fencer(conn, 4, "man-vs-focal", gender="M", scrape_hops=2)
    _add_bout(conn, 50, "X", 4, FOCAL)   # male who fenced focal directly
    # Without a focal, the gate is off: all 4 are pending.
    assert db.count_pending_expansion(conn) == 4
    # With the focal, the lone male without a focal bout (id 1) is excluded.
    assert db.count_pending_expansion(conn, FOCAL) == 3
    picked = set()
    while True:
        row = db.next_to_expand(conn, FOCAL)
        if row is None:
            break
        picked.add(row["id"])
        db.set_fencer_status(conn, row["id"], "done")
        db.set_scrape_hops(conn, row["id"], 0)
    assert picked == {2, 3, 4}   # never id 1


def test_refresh_excludes_stale_male_but_keeps_focal():
    conn = _fresh_db()
    FOCAL = 999
    for fid, g in [(FOCAL, "W"), (1, "M"), (2, "W"), (3, "M")]:
        db.ensure_fencer(conn, fid, f"f{fid}", gender=g)
        db.set_fencer_status(conn, fid, "done")
    conn.execute("UPDATE fencers SET last_scraped_at='2026-01-01T00:00:00+00:00'")  # all stale
    _add_bout(conn, 60, "X", 3, FOCAL)   # male 3 fenced focal -> kept
    ids = {r["id"] for r in db.fencers_to_refresh(conn, "2026-06-14", focal_id=FOCAL)}
    assert ids == {FOCAL, 2, 3}   # focal + stale female + male-who-fenced-focal; male 1 dropped


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
    print("\nAll update-logic tests passed.")
