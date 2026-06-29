"""Bridge between the Streamlit explorer and the Phase-E analytics (model + simulation).

Fits the rating model once per session (cache_resource) and exposes:
  - recent_skill_map(): fencer_id -> estimated peer-relative skill
  - event_placement_df(event_id): per-registrant projected finish distribution
All read-only; the fit uses the same tuned config as the forecast.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import streamlit as st

from analytics import model as M, forecast as FC, simulate as S
from analytics.features import load_dataset
from explorer import data as D
from explorer.data import DB_PATH, display_name

SINCE, BIRTH_MIN, BIRTH_MAX = "2022-09", 2013, 2018
YOUTH_DE10 = {"Y8", "Y10"}
N_SIMS = 10000   # event Monte-Carlo runs — high, for smooth finish histograms
LEVEL_ORDER = {"Y8": 0, "Y10": 1, "Y12": 2, "Y14": 3, "Cadet": 4, "Junior": 5, "Senior": 6, "Vet": 7}


@st.cache_resource(show_spinner="Fitting rating model…")
def fit():
    ds = load_dataset(str(DB_PATH), core_only=True, since=SINCE,
                      birth_min=BIRTH_MIN, birth_max=BIRTH_MAX)
    fm = M.fit(ds, M.ModelConfig(rank=0, lam_s=0.05, lam_time=400))
    return ds, fm


@st.cache_data(show_spinner=False)
def recent_skill_map() -> dict[int, float]:
    """Club-adjusted ability per modelled fencer: recent skill s_i + club main-effect
    c_club_i (age-agnostic). s_i alone is only skill *relative to the club baseline*; adding
    c_club restores cross-club comparability. The placement simulation adds the age term on
    top (effective strength g = s + c + age)."""
    ds, fm = fit()
    s = FC.recent_skill(fm)
    ca, _ = _strength_inputs()
    C = len(fm.c)
    return {f: sv + (float(fm.c[ci]) if 0 <= (ci := ca.get(f, -1)) < C else 0.0)
            for f, sv in s.items()}


@st.cache_data(show_spinner=False)
def _strength_inputs():
    ds, _ = fit()
    ca = dict(zip(ds.bouts.fencer_a_id, ds.bouts.club_a))
    ca.update(zip(ds.bouts.fencer_b_id, ds.bouts.club_b))
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    by = {r[0]: r[1] for r in conn.execute("SELECT id, birth_year FROM fencers")}
    return ca, by


@st.cache_data(show_spinner=False)
def matchup_winprob(focal_id: int, opp_id: int) -> dict | None:
    """Model-estimated P(focal beats opp) in a single bout, for pool (to 5) and DE."""
    from analytics.model import _norm_cdf
    ds, fm = fit()
    ca, by = _strength_inputs()
    g = FC._effective_strength(fm, [focal_id, opp_id], ca, by)
    mu = float(g[0] - g[1])
    rated = set(fm.fencer_ids)
    return {
        "p_pool": float(_norm_cdf(mu / fm.sigma_pool)),
        "p_de": float(_norm_cdf(mu / fm.sigma_de)),
        "focal_rated": focal_id in rated, "opp_rated": opp_id in rated,
    }


def skill_percentile(skill: float | None, skills: np.ndarray) -> float | None:
    if skill is None or len(skills) == 0:
        return None
    return float((skills < skill).mean() * 100.0)


def _de_target(event_id: int) -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    row = conn.execute("SELECT age_group FROM upcoming_events WHERE event_id=?", (event_id,)).fetchone()
    return 10 if (row and row[0] in YOUTH_DE10) else 15


@st.cache_data(show_spinner="Simulating the event…")
def event_placement_df(event_id: int, n_sims: int = N_SIMS) -> pd.DataFrame:
    """One row per registrant: expected finish + the 10/25/50/75/90 finish percentiles,
    sorted by expected finish."""
    ds, fm = fit()
    field, g = FC.event_field_strengths(str(DB_PATH), fm, ds, event_id)
    n = len(field)
    if n < 2:
        return pd.DataFrame()
    places = S.simulate_all_placements(
        np.asarray(g, float), sigma_pool=fm.sigma_pool, sigma_de=fm.sigma_de,
        pool_target=5, de_target=_de_target(event_id), n_sims=n_sims)
    sk = recent_skill_map()
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    by = {r[0]: r[1] for r in conn.execute("SELECT id, birth_year FROM fencers")}
    rows = []
    for i, fid in enumerate(field):
        col = places[:, i]
        p10, p25, p50, p75, p90 = (int(np.percentile(col, q)) for q in (10, 25, 50, 75, 90))
        rows.append({
            "fencer_id": fid, "fencer": display_name(fid), "born": by.get(fid),
            "exp_finish": float(col.mean()), "skill": sk.get(fid),
            "p10": p10, "p25": p25, "p50": p50, "p75": p75, "p90": p90,
        })
    df = pd.DataFrame(rows).sort_values("exp_finish").reset_index(drop=True)
    df.insert(0, "proj_rank", np.arange(1, len(df) + 1))
    return df


@st.cache_data(show_spinner=False)
def skill_trajectory(focal_id: int) -> pd.DataFrame:
    """Monthly club-adjusted ability over time (s_im + club effect, age-agnostic)."""
    ds, fm = fit()
    fidx = {f: i for i, f in enumerate(fm.fencer_ids)}
    if focal_id not in fidx:
        return pd.DataFrame()
    ca, _ = _strength_inputs()
    C = len(fm.c); ci = ca.get(focal_id, -1)
    cterm = float(fm.c[ci]) if 0 <= ci < C else 0.0
    _, traj = fm._skill_lookup()
    rows = [{"month": pd.to_datetime(fm.months[mi] + "-01"), "skill": s + cterm}
            for mi, s in traj.get(fidx[focal_id], [])]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def birth_year_rank(focal_id: int) -> dict | None:
    """Focal's club-adjusted-ability rank among all rated fencers born in her exact birth
    year (her true same-age peers — no age confound)."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    by = {r[0]: r[1] for r in conn.execute("SELECT id, birth_year FROM fencers")}
    yr = by.get(focal_id)
    sk = recent_skill_map()
    if not yr or focal_id not in sk:
        return None
    cohort = {f: v for f, v in sk.items() if by.get(f) == yr}
    fs = cohort[focal_id]
    vals = np.fromiter(cohort.values(), float)
    return {"year": int(yr), "n": len(cohort), "rank": int((vals > fs).sum()) + 1,
            "pct": float((vals < fs).mean() * 100.0), "skill": fs}


@st.cache_data(show_spinner=False)
def eligibility_cohort_rank(focal_id: int) -> dict | None:
    """Focal's skill rank among everyone eligible for the LOWEST level she's entered
    (e.g. Y10) — i.e. rated fencers born on/after that level's eligibility floor."""
    ev = D.focal_upcoming_events(focal_id)
    if ev.empty or "age_group" not in ev:
        return None
    ages = [a for a in ev["age_group"] if a in LEVEL_ORDER]
    if not ages:
        return None
    low = min(ages, key=lambda a: LEVEL_ORDER[a])
    eids = ev.loc[ev["age_group"] == low, "event_id"]
    reg = D.upcoming_tables()["registrants"]
    members = reg.loc[reg["event_id"].isin(eids), "fencer_id"].unique()
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    by = {r[0]: r[1] for r in conn.execute("SELECT id, birth_year FROM fencers")}
    floors = [by[m] for m in members if by.get(m)]
    if not floors:
        return None
    floor = min(floors)
    sk = recent_skill_map()
    cohort = {f: s for f, s in sk.items() if (by.get(f) or 0) >= floor}
    if focal_id not in cohort:
        return None
    fs = cohort[focal_id]
    vals = np.fromiter(cohort.values(), float)
    return {"level": low, "floor": int(floor), "n": len(cohort),
            "rank": int((vals > fs).sum()) + 1, "pct": float((vals < fs).mean() * 100.0),
            "skill": fs}


@st.cache_data(show_spinner=False)
def placement_samples(event_id: int, fencer_id: int, n_sims: int = N_SIMS) -> np.ndarray:
    """Raw placement samples for one fencer in an event (for the distribution chart)."""
    ds, fm = fit()
    field, g = FC.event_field_strengths(str(DB_PATH), fm, ds, event_id)
    if fencer_id not in field:
        return np.array([])
    places = S.simulate_all_placements(
        np.asarray(g, float), sigma_pool=fm.sigma_pool, sigma_de=fm.sigma_de,
        pool_target=5, de_target=_de_target(event_id), n_sims=n_sims)
    return places[:, field.index(fencer_id)]
