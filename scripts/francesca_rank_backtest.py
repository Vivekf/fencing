"""Rank-correlation back-test on all of Francesca's events (leakage-free).

For each event she fenced: fit on bouts BEFORE that event's month, predict every field
member's effective strength g (skill + club/prior + age), and Spearman-correlate g with
actual finishing placement. Higher |rho| = the model orders the field better. Compares the
current fixed-effect club model vs. the hierarchical (credibility) club model.
"""
from __future__ import annotations
import sqlite3
import numpy as np
from analytics.features import load_dataset
from analytics import model as M, forecast as FC
from analytics.validate import _subset

F = 100835605
DB = "fencing_full.db"


def rankdata(a):
    a = np.asarray(a, float); n = len(a)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(n, int); inv[sorter] = np.arange(n)
    a_s = a[sorter]
    obs = np.r_[True, a_s[1:] != a_s[:-1]]
    dense = obs.cumsum()[inv]
    cnt = np.r_[np.nonzero(obs)[0], n]
    return 0.5 * (cnt[dense] + cnt[dense - 1] + 1)


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / d) if d > 0 else np.nan


def strengths(fm, ids, club_a, by, event_year):
    rs = FC.recent_skill(fm); hier = fm.config.hier_club
    C = len(fm.cm) if hier else len(fm.c)
    out = {}
    for x in ids:
        ci = club_a.get(x, -1)
        if hier:
            base = rs[x] if x in rs else (fm.cm[ci] if 0 <= ci < C else 0.0)
        else:
            base = rs.get(x, 0.0) + (fm.c[ci] if 0 <= ci < C else 0.0)
        age = (event_year - by[x]) if by.get(x) else 0.0
        out[x] = base + fm.beta_age * age
    return out


def run():
    ds = load_dataset(DB, core_only=True, since="2022-09", birth_min=2013, birth_max=2018)
    month_of = {mon: i for i, mon in enumerate(ds.months)}
    club_a = dict(zip(ds.bouts.fencer_a_id, ds.bouts.club_a))
    club_a.update(zip(ds.bouts.fencer_b_id, ds.bouts.club_b))
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    by = {r[0]: r[1] for r in conn.execute("SELECT id, birth_year FROM fencers")}
    events = conn.execute(
        """SELECT fer.event_id, e.event_date FROM fencer_event_results fer
           JOIN events e ON e.id = fer.event_id
           WHERE fer.fencer_id=? AND e.event_date IS NOT NULL AND fer.placement IS NOT NULL
           ORDER BY e.event_date""", (F,)).fetchall()

    configs = {"current": M.ModelConfig(rank=0, lam_s=0.05, lam_time=400),
               "hier": M.ModelConfig(rank=0, hier_club=True, lam_s=3.0, lam_cm=3.0, lam_time=400)}
    fits: dict = {}
    res = {k: [] for k in configs}
    used = 0
    for eid, edate in events:
        m = month_of.get(edate[:7])
        if not m:
            continue
        field = conn.execute(
            "SELECT fencer_id, placement FROM fencer_event_results WHERE event_id=? AND placement IS NOT NULL",
            (eid,)).fetchall()
        ids = [r[0] for r in field]; place = np.array([r[1] for r in field], float)
        if len(set(place)) < 5:            # need a rankable field
            continue
        used += 1
        for name, cfg in configs.items():
            key = (name, m)
            if key not in fits:
                fits[key] = M.fit(_subset(ds, ds.bouts["month_idx"].to_numpy() < m), cfg)
            g = strengths(fits[key], ids, club_a, by, int(edate[:4]) + 0.5)
            rho = spearman([g[i] for i in ids], place)     # strong -> low place => negative
            res[name].append((-rho, len(ids)))             # -rho so higher = better ordering
    print(f"Francesca events back-tested: {used}\n")
    print(f"{'model':10s} {'mean rank-corr':>15} {'size-weighted':>15} {'median':>9}")
    for name in configs:
        arr = np.array([r for r, _ in res[name]]); wts = np.array([w for _, w in res[name]])
        print(f"{name:10s} {np.nanmean(arr):>15.3f} {np.nansum(arr*wts)/wts.sum():>15.3f} {np.nanmedian(arr):>9.3f}")


if __name__ == "__main__":
    run()
