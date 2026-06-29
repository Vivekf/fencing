"""Tests for the Streamlit explorer.

Run: PYTHONPATH=. python tests/test_explorer.py
The data-layer checks run plainly; the app check uses Streamlit's AppTest.
"""

from __future__ import annotations

from explorer import data

FRANCESCA_ID = 100835605
CAHALANE_ID = 100357694


def test_perspective_counts():
    p = data.perspective_bouts()
    francesca = p[p["fencer_id"] == FRANCESCA_ID]
    assert len(francesca) == 220, f"expected 220 Francesca bouts, got {len(francesca)}"


def test_head_to_head_cahalane():
    fb = data.focal_bouts(FRANCESCA_ID)
    h2h = fb[fb["opponent_id"] == CAHALANE_ID]
    assert len(h2h) == 14, f"expected 14 bouts vs Cahalane, got {len(h2h)}"
    # The known pool loss 2-5 in event 41045.
    pool = h2h[(h2h["event_id"] == 41045) & (h2h["bout_type"] == "Pool")]
    assert len(pool) == 1
    row = pool.iloc[0]
    assert row["score_for"] == 2 and row["score_against"] == 5
    assert not bool(row["won"])


def test_pretty_name():
    assert data.pretty_name("CAHALANE Alexa") == "Alexa Cahalane"
    assert data.pretty_name("Francesca Farias") == "Francesca Farias"
    assert data.pretty_name("DE LA TORRE Maria") == "Maria De La Torre"


def test_summarize():
    fb = data.focal_bouts(FRANCESCA_ID)
    s = data.summarize(fb)
    assert s["bouts"] == 220
    assert s["wins"] + s["losses"] == 220
    assert 0 <= s["win_pct"] <= 100


def test_scraped_fencers():
    sf = data.scraped_fencers()
    # At least the original BFS cohort of 1002; grows as upcoming-event fields are
    # filled in (see `fencing_tracker upcoming`), so assert a floor, not an exact count.
    assert len(sf) >= 1002, f"expected >=1002 scraped fencers, got {len(sf)}"
    assert FRANCESCA_ID in set(sf["id"])


def test_common_opponents_runs():
    # Two fencers who almost certainly never met directly — just confirm it runs.
    co = data.common_opponents(FRANCESCA_ID, CAHALANE_ID)
    # Francesca and Cahalane HAVE met, but the function should still return a frame.
    assert co is not None


def test_app_runs_without_exception():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("explorer/app.py", default_timeout=120)
    at.run()
    assert not at.exception, f"app raised: {at.exception}"
    # Default focal is Francesca; Overview tab should show her bout KPI.
    metric_values = [m.value for m in at.metric]
    assert "220" in [str(v) for v in metric_values], \
        f"expected a 220-bout metric, got {metric_values}"


if __name__ == "__main__":
    test_perspective_counts()
    print("OK: perspective bout counts")
    test_head_to_head_cahalane()
    print("OK: head-to-head vs Cahalane (14 bouts, known 2-5 loss)")
    test_pretty_name()
    print("OK: name prettifier")
    test_summarize()
    print("OK: summarize")
    test_scraped_fencers()
    print("OK: 1002 scraped fencers")
    test_common_opponents_runs()
    print("OK: common opponents")
    test_app_runs_without_exception()
    print("OK: app.py runs in AppTest with no exception")
    print("\nAll explorer tests passed.")
