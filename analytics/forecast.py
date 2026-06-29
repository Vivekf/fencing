"""Forecast a focal fencer's results against an upcoming event's field.

Fits the tuned model (rank0, lam_time~200) and Elo on the birth-banded peer population,
blends their win probabilities (log-pool, the calibration-better combination), and
applies them to every registered opponent in a given upcoming event.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd

from .features import load_dataset
from . import model as M
from . import simulate as S


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _elo_ratings(ds, K=32.0, scale=400.0, base=1500.0):
    order = ds.bouts.sort_values(["month_idx", "de", "event_id"])
    R = defaultdict(lambda: base)
    for a, b, w in zip(order["fencer_a_id"].to_numpy(), order["fencer_b_id"].to_numpy(),
                       order["winner_id"].to_numpy()):
        ra, rb = R[a], R[b]
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / scale))
        sa = 1.0 if w == a else 0.0
        R[a] = ra + K * (sa - ea); R[b] = rb + K * ((1 - sa) - (1 - ea))
    return R, scale


def forecast_event(db_path, focal_id, event_id, *, since="2022-09",
                   birth_min=2013, birth_max=2018, blend_w=0.6, K=32.0):
    ds = load_dataset(db_path, core_only=True, since=since, birth_min=birth_min, birth_max=birth_max)
    fm = M.fit(ds, M.ModelConfig(rank=0, lam_s=0.05, lam_time=200))
    R, scale = _elo_ratings(ds, K=K)
    fidx = {f: i for i, f in enumerate(fm.fencer_ids)}

    # per-fencer static attributes from the fitted dataset
    club_a = dict(zip(ds.bouts.fencer_a_id, ds.bouts.club_a)); club_a.update(zip(ds.bouts.fencer_b_id, ds.bouts.club_b))
    cpop = dict(zip(ds.bouts.fencer_a_id, ds.bouts.club_pop_a)); cpop.update(zip(ds.bouts.fencer_b_id, ds.bouts.club_pop_b))

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    by = {r[0]: r[1] for r in conn.execute("SELECT id, birth_year FROM fencers")}
    name = {r[0]: r[1] for r in conn.execute("SELECT id, name FROM fencers")}
    field = [r[0] for r in conn.execute(
        "SELECT fencer_id FROM upcoming_event_registrants WHERE event_id=?", (event_id,))]
    f_birth = by.get(focal_id)

    rows, missing = [], 0
    for x in field:
        if x == focal_id or x not in fidx:
            missing += x != focal_id and x not in fidx
            continue
        # hypothetical pool bout: focal (a) vs x (b)
        rec = dict(
            a_idx=fidx[focal_id], b_idx=fidx[x],
            club_a=club_a.get(focal_id, -1), club_b=club_a.get(x, -1),
            club_pop_a=cpop.get(focal_id, -1), club_pop_b=cpop.get(x, -1),
            age_diff=(by.get(x, f_birth) - f_birth) if f_birth else 0.0,  # a_F - a_X = X_birth - F_birth
            de=0,  # pool
        )
        df = pd.DataFrame([rec])
        _, p_model = fm.predict(df)
        rf, rx = R.get(focal_id, 1500.0), R.get(x, 1500.0)
        p_elo = 1.0 / (1.0 + 10 ** ((rx - rf) / scale))
        p = 1.0 / (1.0 + np.exp(-(blend_w * _logit(p_model[0]) + (1 - blend_w) * _logit(p_elo))))
        rows.append((name.get(x, str(x)), float(p), float(p_model[0]), float(p_elo), by.get(x)))

    res = pd.DataFrame(rows, columns=["opponent", "p_win", "p_model", "p_elo", "born"])
    res = res.sort_values("p_win").reset_index(drop=True)
    return fm, res, missing


def recent_skill(fm, k=6):
    """Smoothed recent skill (mean of last up-to-k active months) per modelled fencer."""
    fidx = {f: i for i, f in enumerate(fm.fencer_ids)}
    _, traj = fm._skill_lookup()
    out = {}
    for f in fm.fencer_ids:
        t = traj.get(fidx[f], [])
        if t:
            out[f] = float(np.mean([s for _, s in t[-k:]]))
    return out


def _effective_strength(fm, fids, club_a, by, event_year=2026.5, cold_prior=0.0):
    """g_i = s_i + beta_age*age_i + club_main_effect. Uses the SAME recent-mean skill the
    explorer displays (so ability and outcome are consistent). Additive antisymmetric
    terms collapse to one number per fencer; unrated (cold) fencers get `cold_prior`
    (0 = population average; negative = the weaker-than-average prior for local unrateds)."""
    rs = recent_skill(fm)
    C = len(fm.c)
    g = np.zeros(len(fids))
    for k, x in enumerate(fids):
        s = rs.get(x, cold_prior)
        ci = club_a.get(x, -1); cterm = fm.c[ci] if 0 <= ci < C else 0.0
        age = (event_year - by[x]) if by.get(x) else 0.0
        g[k] = s + fm.beta_age * age + cterm
    return g


def event_field_strengths(db_path, fm, ds, event_id, *, extra_ids=()):
    """(field_ids, g) for an event's registrants (+ any extra_ids), using the fitted model."""
    club_a = dict(zip(ds.bouts.fencer_a_id, ds.bouts.club_a)); club_a.update(zip(ds.bouts.fencer_b_id, ds.bouts.club_b))
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    by = {r[0]: r[1] for r in conn.execute("SELECT id, birth_year FROM fencers")}
    field = list(dict.fromkeys(r[0] for r in conn.execute(
        "SELECT fencer_id FROM upcoming_event_registrants WHERE event_id=?", (event_id,))))
    for x in extra_ids:
        if x not in field:
            field.append(x)
    return field, _effective_strength(fm, field, club_a, by)


def forecast_placement(db_path, focal_id, event_id, *, since="2022-09",
                       birth_min=2013, birth_max=2018, n_sims=4000, de_target=10):
    """Monte-Carlo the event from the fitted model; return (placement array, entrant ids)."""
    ds = load_dataset(db_path, core_only=True, since=since, birth_min=birth_min, birth_max=birth_max)
    fm = M.fit(ds, M.ModelConfig(rank=0, lam_s=0.05, lam_time=200))
    field, g = event_field_strengths(db_path, fm, ds, event_id, extra_ids=(focal_id,))
    places = S.simulate_placements(
        g, field.index(focal_id), sigma_pool=fm.sigma_pool, sigma_de=fm.sigma_de,
        pool_target=5, de_target=de_target, n_sims=n_sims)
    return places, field


if __name__ == "__main__":
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else "fencing.db"
    F, EV = 100835605, 18164
    fm, res, missing = forecast_event(db_path, F, EV)
    p = res["p_win"].to_numpy()
    print(f"Forecast: Francesca vs Summer Nationals Y10WE field — {len(res)} opponents in model "
          f"({missing} not in band/model)")
    print(f"  expected wins (pool, neutral): {p.sum():.1f} / {len(p)}  ({100*p.mean():.0f}% avg)")
    print(f"  favored (>50%) against: {(p>0.5).sum()}/{len(p)}")
    print("\nToughest 8 (lowest win prob):")
    for _, r in res.head(8).iterrows():
        print(f"  {r.p_win:.2f}  (model {r.p_model:.2f} / elo {r.p_elo:.2f})  {r.opponent[:26]:26s} b{int(r.born) if r.born else '?'}")
    print("Easiest 8 (highest win prob):")
    for _, r in res.tail(8).iloc[::-1].iterrows():
        print(f"  {r.p_win:.2f}  (model {r.p_model:.2f} / elo {r.p_elo:.2f})  {r.opponent[:26]:26s} b{int(r.born) if r.born else '?'}")

    places, field = forecast_placement(db_path, F, EV)
    n = len(field)
    print(f"\n=== Placement distribution (Monte-Carlo, {len(places)} sims, field {n}) ===")
    print(f"  P(win)      : {(places==1).mean()*100:.1f}%")
    print(f"  P(final, top2): {(places<=2).mean()*100:.1f}%")
    print(f"  P(top 4)    : {(places<=4).mean()*100:.1f}%")
    print(f"  P(top 8)    : {(places<=8).mean()*100:.1f}%")
    print(f"  P(top 16)   : {(places<=16).mean()*100:.1f}%")
    print(f"  median place: {int(np.median(places))}   mean: {places.mean():.1f}")
    print(f"  10th-90th pct: {int(np.percentile(places,10))}-{int(np.percentile(places,90))}")
