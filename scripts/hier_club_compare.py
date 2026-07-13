"""Compare the current fixed-effect club model vs. the hierarchical (credibility) club
model, in the same forward-chaining CV, sliced by the less-established fencer's bout count.

hier_club: club is a prior mean each fencer's absolute skill shrinks toward, with lam_s
controlling the credibility K (club "wears off" as N grows); unseen fencers fall back to
their club mean. Swept over lam_s.

    PYTHONPATH=. python scripts/hier_club_compare.py [db] [n_months]
"""
from __future__ import annotations

import sys
from collections import Counter

import numpy as np

from analytics.features import load_dataset
from analytics import model as M
from analytics.validate import _subset, default_test_months


def _logloss(won, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(won * np.log(p) + (1 - won) * np.log(1 - p)))


def run(db="fencing_full.db", n_months=8):
    ds = load_dataset(db, core_only=True, since="2022-09", birth_min=2013, birth_max=2018)
    tms = default_test_months(ds, n_months)
    configs = [
        ("current (fixed)", M.ModelConfig(rank=0, lam_s=0.05, lam_time=400)),
        ("hier lam_s=0.05", M.ModelConfig(rank=0, hier_club=True, lam_s=0.05, lam_cm=1.0, lam_time=400)),
        ("hier lam_s=0.3", M.ModelConfig(rank=0, hier_club=True, lam_s=0.3, lam_cm=1.0, lam_time=400)),
        ("hier lam_s=1.0", M.ModelConfig(rank=0, hier_club=True, lam_s=1.0, lam_cm=1.0, lam_time=400)),
        ("hier lam_s=3.0", M.ModelConfig(rank=0, hier_club=True, lam_s=3.0, lam_cm=1.0, lam_time=400)),
    ]
    WON, NMIN = [], []
    P = {name: [] for name, _ in configs}
    for m in tms:
        bmask = ds.bouts["month_idx"].to_numpy()
        train = _subset(ds, bmask < m)
        test = ds.bouts[bmask == m]
        if len(train.bouts) < 500 or len(test) == 0:
            continue
        cnt = Counter(train.bouts["a_idx"].tolist()) + Counter(train.bouts["b_idx"].tolist())
        a = test["a_idx"].to_numpy(); b = test["b_idx"].to_numpy()
        na = np.array([cnt.get(int(x), 0) for x in a]); nb = np.array([cnt.get(int(x), 0) for x in b])
        WON.append((test["winner_id"].to_numpy() == test["fencer_a_id"].to_numpy()).astype(float))
        NMIN.append(np.minimum(na, nb))
        for name, cfg in configs:
            fm = M.fit(train, cfg)
            _, p = fm.predict(test)
            P[name].append(p)
    WON = np.concatenate(WON); NMIN = np.concatenate(NMIN)
    P = {k: np.concatenate(v) for k, v in P.items()}

    buckets = [
        ("all", np.ones(len(WON), bool)),
        ("min N = 0", NMIN == 0),
        ("min N 1-5", (NMIN >= 1) & (NMIN <= 5)),
        ("min N 6-15", (NMIN >= 6) & (NMIN <= 15)),
        ("min N 16-40", (NMIN >= 16) & (NMIN <= 40)),
        ("min N 41+", NMIN >= 41),
    ]
    names = [n for n, _ in configs]
    print(f"db={db}  test months={[ds.months[i] for i in tms]}  bouts={len(WON)}")
    print(f"\nLOG-LOSS (lower better; * = best in row)")
    print(f"{'bucket':13s} {'n':>6}  " + "".join(f"{n:<17}" for n in names))
    for bname, mask in buckets:
        if mask.sum() == 0:
            continue
        vals = [_logloss(WON[mask], P[n][mask]) for n in names]
        best = int(np.argmin(vals))
        cells = "".join(f"{v:.4f}{'*' if i == best else ' '}       " for i, v in enumerate(vals))
        print(f"{bname:13s} {int(mask.sum()):>6}  {cells}")


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "fencing_full.db"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    run(db, n)
