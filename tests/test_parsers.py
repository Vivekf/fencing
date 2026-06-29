"""Tests against the saved Francesca fixture.

Run: python -m pytest tests/ -v
Or:  python tests/test_parsers.py (uses plain asserts)
"""

from __future__ import annotations

from pathlib import Path

from fencing_tracker.parsers import parse_history

FIXTURE = Path(__file__).parent / "fixtures" / "francesca_history_p1.html"
FRANCESCA_ID = 100835605


def test_parses_all_events():
    html = FIXTURE.read_text(encoding="utf-8")
    events = parse_history(html)
    assert len(events) == 29, f"expected 29 events, got {len(events)}"


def test_total_bouts():
    html = FIXTURE.read_text(encoding="utf-8")
    events = parse_history(html)
    total = sum(len(e.bouts) for e in events)
    assert total == 220, f"expected 220 bouts, got {total}"


def test_first_event_metadata():
    html = FIXTURE.read_text(encoding="utf-8")
    events = parse_history(html)
    ev = events[0]
    assert ev.event_id == 41045
    assert ev.tournament_name == "Olympia D'Artagnan's Challenge 5B (Y10/Y14 Epee)"
    assert ev.classification == "Unrated Y-14 Women's Épée"
    assert ev.weapon == "epee"
    assert ev.gender == "W"
    assert ev.age_group == "Y14"
    assert ev.rating_level == "U"
    assert ev.event_date == "2026-05-17"
    assert ev.raw_date == "May 17, 2026"
    assert ev.focal_placement == 6
    assert ev.focal_field_size == 9
    assert ev.focal_seed == 7


def test_known_bout_vs_cahalane():
    html = FIXTURE.read_text(encoding="utf-8")
    events = parse_history(html)
    ev = events[0]  # Olympia event
    cahalane = next((b for b in ev.bouts if b.opponent_id == 100357694), None)
    assert cahalane is not None, "Expected to find Cahalane bout"
    assert cahalane.bout_type == "Pool"
    assert cahalane.focal_score == 2
    assert cahalane.opp_score == 5
    assert cahalane.focal_won is False
    assert cahalane.opponent_name == "CAHALANE Alexa"


def test_de_bout_present():
    """Confirm we capture DE bouts (T-rounds), not just pool."""
    html = FIXTURE.read_text(encoding="utf-8")
    events = parse_history(html)
    all_types = {b.bout_type for e in events for b in e.bouts}
    # We saw at least Pool, T8, T4, T2, T16, T32 in the fixture
    assert "Pool" in all_types
    assert any(t.startswith("T") for t in all_types), f"no DE bouts in {all_types}"


def test_all_opponent_ids_extracted():
    html = FIXTURE.read_text(encoding="utf-8")
    events = parse_history(html)
    for ev in events:
        for b in ev.bouts:
            assert b.opponent_id > 0
            assert b.opponent_name


def test_known_bout_has_profile_flag():
    """Modern opponents (long IDs) should have opponent_has_profile=True."""
    html = FIXTURE.read_text(encoding="utf-8")
    events = parse_history(html)
    ev = events[0]
    cahalane = next(b for b in ev.bouts if b.opponent_id == 100357694)
    assert cahalane.opponent_has_profile is True


def test_y8_double_round_robin():
    """Event 29559 (Olympia Y8 Epee) repeats three pool pairings."""
    html = FIXTURE.read_text(encoding="utf-8")
    events = parse_history(html)
    ev = next((e for e in events if e.event_id == 29559), None)
    assert ev is not None
    from collections import Counter
    pool_pairs = Counter(
        (b.opponent_id, b.bout_type) for b in ev.bouts if b.bout_type == "Pool"
    )
    repeats = {k: v for k, v in pool_pairs.items() if v > 1}
    assert len(repeats) == 3, f"expected 3 repeated pool pairings, got {repeats}"


if __name__ == "__main__":
    # Plain assertions so we don't require pytest just to validate
    test_parses_all_events()
    print("OK: 29 events")
    test_total_bouts()
    print("OK: 220 bouts")
    test_first_event_metadata()
    print("OK: first event metadata")
    test_known_bout_vs_cahalane()
    print("OK: Cahalane bout 2-5 loss in Pool")
    test_de_bout_present()
    print("OK: DE bouts present")
    test_all_opponent_ids_extracted()
    print("OK: every bout has opponent_id + name")
    test_known_bout_has_profile_flag()
    print("OK: opponent_has_profile flag")
    test_y8_double_round_robin()
    print("OK: Y-8 double round-robin detected (3 repeated pairs)")
    print("\nAll fixture tests passed.")
