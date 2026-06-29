"""Tests for the upcoming-events parsers against saved fixtures.

Run: python -m pytest tests/ -v
Or:  python tests/test_upcoming.py (uses plain asserts)
"""

from __future__ import annotations

from pathlib import Path

from fencing_tracker.parsers import parse_registrations, parse_event_roster

FIXTURES = Path(__file__).parent / "fixtures"
SUMMARY = FIXTURES / "francesca_summary.html"
ROSTER = FIXTURES / "event_18164_roster.html"
FRANCESCA_ID = 100835605


def test_parse_registrations():
    regs = parse_registrations(SUMMARY.read_text(encoding="utf-8"))
    assert len(regs) == 1, f"expected 1 upcoming registration, got {len(regs)}"
    r = regs[0]
    assert r.event_id == 18164
    assert r.tournament_name == "Summer Nationals and July Challenge"
    assert "Youth 10" in (r.event_name or "")
    assert r.event_date == "2026-07-05"


def test_registrations_ignore_past_events():
    """Past events link to /event/{id}/results and must not be returned."""
    regs = parse_registrations(SUMMARY.read_text(encoding="utf-8"))
    assert all(r.event_id == 18164 for r in regs)
    # The summary page contains ~30 past /results links; none should leak in.
    assert len(regs) == 1


def test_parse_event_roster_meta():
    roster = parse_event_roster(ROSTER.read_text(encoding="utf-8"), event_id=18164)
    assert roster.event_id == 18164
    assert roster.tournament_name == "Summer Nationals and July Challenge"
    assert "Youth 10" in (roster.event_name or "")
    assert roster.weapon == "epee"
    assert roster.gender == "W"
    assert roster.age_group == "Y10"
    assert roster.venue == "Oregon Convention Center"
    assert roster.location == "Portland, OR"
    assert roster.event_date == "2026-07-05"


def test_parse_event_roster_field():
    roster = parse_event_roster(ROSTER.read_text(encoding="utf-8"), event_id=18164)
    # 107 table rows, one fencer duplicated -> 106 distinct.
    assert len(roster.entries) == 107
    assert len({e.fencer_id for e in roster.entries}) == 106
    # The focal fencer is in her own event's field.
    assert FRANCESCA_ID in {e.fencer_id for e in roster.entries}


def test_roster_names_normalized():
    """Roster shows 'Last, First'; we store 'First Last' to match the rest of the DB."""
    roster = parse_event_roster(ROSTER.read_text(encoding="utf-8"), event_id=18164)
    chaney = next(e for e in roster.entries if e.fencer_id == 100757859)
    assert chaney.name == "Evelyn Chaney"
    assert chaney.slug == "Evelyn-Chaney"
    assert chaney.club == "Elite Fencing Academy"
    assert "," not in chaney.name


def test_roster_excludes_strength_columns():
    """RosterEntry must carry only factual identity, never strength numbers."""
    roster = parse_event_roster(ROSTER.read_text(encoding="utf-8"), event_id=18164)
    fields = roster.entries[0].__dict__.keys()
    for banned in ("strength", "conservative", "rank", "seed", "rating"):
        assert not any(banned in f.lower() for f in fields), f"unexpected field {banned!r}"


if __name__ == "__main__":
    test_parse_registrations()
    print("OK: 1 upcoming registration parsed")
    test_registrations_ignore_past_events()
    print("OK: past /results events ignored")
    test_parse_event_roster_meta()
    print("OK: roster event metadata")
    test_parse_event_roster_field()
    print("OK: 107 entries / 106 distinct, focal present")
    test_roster_names_normalized()
    print("OK: names normalized to 'First Last'")
    test_roster_excludes_strength_columns()
    print("OK: no strength columns captured")
    print("\nAll upcoming fixture tests passed.")
