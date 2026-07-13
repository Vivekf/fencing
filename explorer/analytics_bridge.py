"""Bridge between the Streamlit explorer and the Phase-E analytics (model + simulation).

Fits the rating model once per session (cache_resource) and exposes:
  - recent_skill_map(): fencer_id -> estimated peer-relative skill
  - event_placement_df(event_id): per-registrant projected finish distribution
All read-only; the fit uses the same tuned config as the forecast.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter

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
MIN_COHORT_BOUTS = 20   # below this a fencer is too sparsely rated to rank in a cohort
MIN_RYC = 10            # need this many serious (RYC+) bouts to score vs the experience curve
SCATTER_MIN_RYC = 25    # only plot fencers with a reasonable serious-bout sample


@st.cache_resource(show_spinner="Fitting rating model…")
def fit():
    ds = load_dataset(str(DB_PATH), core_only=True, since=SINCE,
                      birth_min=BIRTH_MIN, birth_max=BIRTH_MAX)
    fm = M.fit(ds, M.ModelConfig(rank=0, hier_club=True, lam_s=3.0, lam_cm=3.0, lam_time=400))
    return ds, fm


@st.cache_data(show_spinner=False)
def recent_skill_map() -> dict[int, float]:
    """Current (last-month, pointwise) ABSOLUTE ability s_i per modelled fencer.

    Under the hierarchical club model s_i already embeds the club prior — a fencer is shrunk
    toward their club mean when their record is thin and escapes it as they accumulate bouts —
    so s_i is directly cross-club comparable and there is NO separate club effect to add. The
    placement simulation adds only the age term on top (effective strength g = s + age)."""
    ds, fm = fit()
    return {f: float(sv) for f, sv in FC.recent_skill(fm).items()}


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


_SERIOUS_RE = re.compile(
    r"\bNAC\b|North American Cup|Summer Nationals|\bSYC\b|Super Youth|"
    r"\bROC\b|\bRYC\b|\bRJCC?\b|Regional", re.I)


@st.cache_data(show_spinner=False)
def _serious_events() -> set:
    """Event ids that are RYC+ (regional-and-up / 'serious')."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    return {r[0] for r in conn.execute("SELECT id, name FROM events")
            if r[1] and _SERIOUS_RE.search(r[1])}


@st.cache_data(show_spinner=False)
def _ryc_counts() -> dict[int, int]:
    """Per-fencer count of 'serious' (RYC+ / regional-and-up) epee bouts."""
    ds, _ = fit()
    serious = _serious_events()
    cnt = Counter()
    for f, e in zip(ds.bouts.fencer_a_id.to_numpy(), ds.bouts.event_id.to_numpy()):
        if int(e) in serious:
            cnt[int(f)] += 1
    for f, e in zip(ds.bouts.fencer_b_id.to_numpy(), ds.bouts.event_id.to_numpy()):
        if int(e) in serious:
            cnt[int(f)] += 1
    return dict(cnt)


@st.cache_data(show_spinner=False)
def experience_band() -> dict:
    """Per gender, fit skill = b0 + b1*ln(1+RYC+) on fencers with real serious experience
    (>= MIN_RYC RYC+ bouts and >= MIN_COHORT_BOUTS total). Deliberately NOT age-adjusted:
    being skilled young is itself talent, so we don't subtract it off (a negative,
    selection-driven age term would perversely penalize precocious fencers)."""
    sk, ryc = recent_skill_map(), _ryc_counts()
    cnt, gender = _fencer_meta()
    out = {}
    for g in ("W", "M"):
        fs = [f for f in sk if gender.get(f) == g and cnt.get(f, 0) >= MIN_COHORT_BOUTS
              and ryc.get(f, 0) >= MIN_RYC]
        if len(fs) < 30:
            continue
        x = np.log1p(np.array([ryc[f] for f in fs], float))
        y = np.array([sk[f] for f in fs])
        Xa = np.column_stack([np.ones(len(y)), x])
        b = np.linalg.lstsq(Xa, y, rcond=None)[0]
        resid = y - Xa @ b
        r2 = 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
        out[g] = {"b0": float(b[0]), "b1": float(b[1]),
                  "sigma": float(resid.std()), "r2": float(r2), "n": len(fs)}
    return out


@st.cache_data(show_spinner=False)
def experience_context(fencer_id: int) -> dict | None:
    """Skill vs. what serious experience predicts. None below MIN_RYC (too little serious
    experience to contextualize). z > 0 = over-performs their serious-competition volume."""
    sk = recent_skill_map()
    ryc = _ryc_counts().get(fencer_id, 0)
    if fencer_id not in sk or ryc < MIN_RYC:
        return None
    _, gender = _fencer_meta()
    band = experience_band().get(gender.get(fencer_id))
    if band is None:
        return None
    pred = band["b0"] + band["b1"] * np.log1p(ryc)
    act = sk[fencer_id]
    return {"ryc": ryc, "expected": pred, "actual": act, "sigma": band["sigma"],
            "z": (act - pred) / band["sigma"], "r2": band["r2"]}


@st.cache_data(show_spinner=False)
def experience_scatter(focal_id: int) -> dict | None:
    """Chart data: each cohort fencer (focal's gender, >= MIN_RYC) as (skill, residual-σ),
    where residual = (skill − experience-expected) / σ. Plus the focal point."""
    sk = recent_skill_map()
    if focal_id not in sk:
        return None
    cnt, gender = _fencer_meta()
    fg = gender.get(focal_id)
    band = experience_band().get(fg)
    ryc = _ryc_counts()
    if band is None or ryc.get(focal_id, 0) < MIN_RYC:
        return None

    def z(f):
        return (sk[f] - (band["b0"] + band["b1"] * np.log1p(ryc[f]))) / band["sigma"]

    fs = [f for f in sk if gender.get(f) == fg and cnt.get(f, 0) >= MIN_COHORT_BOUTS
          and ryc.get(f, 0) >= SCATTER_MIN_RYC]
    pts = pd.DataFrame({"skill": [sk[f] for f in fs], "z": [z(f) for f in fs],
                        "ryc": [ryc[f] for f in fs]})
    return {"points": pts,
            "focal": {"skill": sk[focal_id], "z": z(focal_id), "ryc": ryc.get(focal_id, 0)}}


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
    """Monthly absolute ability over time (s_im; age-agnostic). Under the hierarchical club
    model s_im already embeds the club prior, so no club effect is added."""
    ds, fm = fit()
    fidx = {f: i for i, f in enumerate(fm.fencer_ids)}
    if focal_id not in fidx:
        return pd.DataFrame()
    _, traj = fm._skill_lookup()
    rows = [{"month": pd.to_datetime(fm.months[mi] + "-01"), "skill": s}
            for mi, s in traj.get(fidx[focal_id], [])]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _fencer_meta():
    """(epee-bout-count, gender) per modelled fencer — for cohort gender/min-bouts filters."""
    ds, _ = fit()
    from collections import Counter
    cnt = Counter()
    for f in ds.bouts.fencer_a_id.to_numpy():
        cnt[int(f)] += 1
    for f in ds.bouts.fencer_b_id.to_numpy():
        cnt[int(f)] += 1
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    gender = {r[0]: r[1] for r in conn.execute("SELECT id, gender FROM fencers")}
    return dict(cnt), gender


@st.cache_data(show_spinner=False)
def birth_year_rank(focal_id: int) -> dict | None:
    """Focal's club-adjusted-ability rank among her TRUE same-age peers: rated fencers born
    in her exact birth year, of her gender, with enough bouts to be reliably rated."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    by = {r[0]: r[1] for r in conn.execute("SELECT id, birth_year FROM fencers")}
    yr = by.get(focal_id)
    sk = recent_skill_map()
    if not yr or focal_id not in sk:
        return None
    cnt, gender = _fencer_meta()
    fg = gender.get(focal_id)
    cohort = {f: v for f, v in sk.items()
              if by.get(f) == yr and cnt.get(f, 0) >= MIN_COHORT_BOUTS
              and (fg is None or gender.get(f) == fg)}
    if focal_id not in cohort:
        return None
    fs = cohort[focal_id]
    vals = np.fromiter(cohort.values(), float)
    return {"year": int(yr), "gender": fg, "n": len(cohort), "rank": int((vals > fs).sum()) + 1,
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
    cnt, gender = _fencer_meta()
    fg = gender.get(focal_id)
    cohort = {f: s for f, s in sk.items()
              if (by.get(f) or 0) >= floor and cnt.get(f, 0) >= MIN_COHORT_BOUTS
              and (fg is None or gender.get(f) == fg)}
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
