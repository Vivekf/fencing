"""Tests for connectivity-gated scrape-core selection (frontier.compute_core)."""

from __future__ import annotations

import sqlite3

from fencing_tracker import db, frontier


def _bout(conn, eid, a, b):
    db.upsert_event(conn, eid, None, None, "epee", "W", "Y10", "U", "2026-01-01", "Jan 1, 2026")
    lo, hi = (a, b) if a < b else (b, a)
    db.insert_bout(conn, event_id=eid, fencer_a_id=lo, fencer_b_id=hi,
                   fencer_a_score=5, fencer_b_score=3, winner_id=lo,
                   bout_type="Pool", bout_seq=1, source_fencer_id=lo)


def _setup():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_schema(conn)
    FOCAL = 1
    # anchors = focal(1) + field(2,3); all women, scraped
    for fid, g in [(1, "W"), (2, "W"), (3, "W")]:
        db.ensure_fencer(conn, fid, f"f{fid}", slug=f"s{fid}", gender=g)
        db.set_fencer_status(conn, fid, "done")
    db.replace_upcoming_registrants(conn, 900, [(1, "f1", None), (2, "f2", None), (3, "f3", None)])
    db.upsert_upcoming_event(conn, 900, "T", "E", None, "epee", "W", "Y10", None, None, None, "2026-07-01", 3)
    # candidates (discovered)
    for fid, g in [(10, "W"), (11, "W"), (13, "M"), (14, "M"), (15, "W"), (20, "W")]:
        db.ensure_fencer(conn, fid, f"f{fid}", slug=f"s{fid}", gender=g)
    eid = 1000
    def bouts(a, partners):
        nonlocal eid
        for p in partners:
            _bout(conn, eid, a, p); eid += 1
    bouts(10, [1, 2, 3])      # 3 anchors -> admit (dist1)
    bouts(11, [1])            # 1 anchor -> reject (k=3)
    bouts(13, [1, 2, 3])      # male but fenced focal(1) -> admit
    bouts(14, [2, 3, 10])     # male, 3 core, never focal -> prune
    bouts(15, [2, 3, 10])     # 3 core -> admit (dist1 via 2,3)
    bouts(20, [10, 13, 15])   # 3 core, all dist1 -> dist2 -> admit only if radius>=2
    conn.commit()
    return conn, FOCAL


def test_core_k3_radius2():
    conn, FOCAL = _setup()
    core, targets, _ = frontier.compute_core(conn, FOCAL, k=3, radius=2)
    assert core == {1, 2, 3, 10, 13, 15, 20}, core
    assert set(targets) == {10, 13, 15, 20}, targets   # discovered members of core
    assert 11 not in core   # too few core connections
    assert 14 not in core   # male, never fenced focal


def test_radius_1_excludes_second_ring():
    conn, FOCAL = _setup()
    core, targets, _ = frontier.compute_core(conn, FOCAL, k=3, radius=1)
    assert 20 not in core   # distance 2 from anchors
    assert core == {1, 2, 3, 10, 13, 15}, core


def test_k2_is_looser():
    conn, FOCAL = _setup()
    core, _, _ = frontier.compute_core(conn, FOCAL, k=2, radius=2)
    assert 11 not in core   # 11 still only has 1 connection
    assert {10, 13, 15, 20}.issubset(core)


if __name__ == "__main__":
    test_core_k3_radius2(); print("OK: k=3 radius=2 core membership")
    test_radius_1_excludes_second_ring(); print("OK: radius=1 excludes 2nd ring")
    test_k2_is_looser(); print("OK: k=2 looser")
    print("\nAll frontier tests passed.")
