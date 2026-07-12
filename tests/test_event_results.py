"""Tests for parse_event_results against a saved /event/{id}/results fixture.

Run directly (python tests/test_event_results.py) or via pytest.
"""

from pathlib import Path

from fencing_tracker.parsers import parse_event_results

FIXTURE = Path(__file__).parent / "fixtures" / "event_41875_results.html"


def _parsed():
    return parse_event_results(FIXTURE.read_text(encoding="utf-8"), 41875)


def test_field_and_meta():
    r = _parsed()
    assert r.event_name == "Y-10 Women's Épée"
    assert (r.weapon, r.gender, r.age_group) == ("epee", "W", "Y10")
    assert len(r.participants) == 104            # whole field from one request
    assert r.skipped_bouts == 0                  # every opponent name resolved


def test_bouts_are_symmetric_and_consistent():
    r = _parsed()
    assert len(r.bouts) == 411
    assert sum(b.bout_type == "Pool" for b in r.bouts) == 309
    assert sum(b.bout_type == "DE" for b in r.bouts) == 102
    for b in r.bouts:
        assert b.fencer_a_id < b.fencer_b_id                     # canonical ordering
        assert b.winner_id in (b.fencer_a_id, b.fencer_b_id)     # winner is a participant
        assert b.bout_type in ("Pool", "DE")
        # winner's score is the higher one
        hi = b.a_score if b.winner_id == b.fencer_a_id else b.b_score
        lo = b.b_score if b.winner_id == b.fencer_a_id else b.a_score
        assert hi >= lo


def test_known_bout():
    r = _parsed()
    wong = next(p for p in r.participants if p.raw_name == "WONG Isabelle")
    louvot = next(p for p in r.participants if p.raw_name == "LOUVOT Chloe")
    assert wong.placement == 1
    b = next(b for b in r.bouts
             if {b.fencer_a_id, b.fencer_b_id} == {wong.fencer_id, louvot.fencer_id}
             and b.bout_type == "Pool")
    assert b.winner_id == wong.fencer_id
    assert {b.a_score, b.b_score} == {5, 1}


class _FakeClient:
    """Serves the saved fixture instead of hitting the network."""
    def __init__(self, html):
        self.html = html

    def get(self, url, use_cache=True):
        return self.html


def test_ingest_and_idempotency(tmp_path=None):
    import tempfile
    from fencing_tracker import db, scraper
    path = (tmp_path / "t.db") if tmp_path is not None else tempfile.mktemp(suffix=".db")
    conn = db.connect(path)
    db.init_schema(conn)
    client = _FakeClient(FIXTURE.read_text(encoding="utf-8"))

    s = scraper.scrape_event_results(conn, client, 41875)
    assert s.participants == 104
    assert s.bouts_inserted == 411
    assert s.fencers_discovered == 104
    assert conn.execute("SELECT COUNT(*) FROM bouts").fetchone()[0] == 411
    assert db.event_results_ingested(conn, 41875) is True
    assert conn.execute(
        "SELECT COUNT(*) FROM fencer_event_results WHERE event_id=41875").fetchone()[0] == 104

    # Re-ingesting the same event must add nothing.
    s2 = scraper.scrape_event_results(conn, client, 41875)
    assert s2.bouts_inserted == 0
    assert s2.fencers_discovered == 0
    assert conn.execute("SELECT COUNT(*) FROM bouts").fetchone()[0] == 411
    conn.close()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
    print("\nAll event-results parser tests passed.")
