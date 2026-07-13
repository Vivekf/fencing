"""Finer (lam_s, lam_cm) sweep for the hierarchical club model, forward-chaining CV.
Config per row; log-loss overall + by min(N_a,N_b) bucket."""
from __future__ import annotations
import sys
from collections import Counter
import numpy as np
from analytics.features import load_dataset
from analytics import model as M
from analytics.validate import _subset, default_test_months


def _ll(won, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(won * np.log(p) + (1 - won) * np.log(1 - p)))


def run(db="fencing_full.db", n_months=8):
    ds = load_dataset(db, core_only=True, since="2022-09", birth_min=2013, birth_max=2018)
    tms = default_test_months(ds, n_months)
    configs = [("current", M.ModelConfig(rank=0, lam_s=0.05, lam_time=400))]
    for ls in (1, 2, 3, 4, 6, 8):
        configs.append((f"hier ls={ls} lcm=1", M.ModelConfig(rank=0, hier_club=True, lam_s=ls, lam_cm=1.0, lam_time=400)))
    for lcm in (0.3, 3.0):
        configs.append((f"hier ls=3 lcm={lcm}", M.ModelConfig(rank=0, hier_club=True, lam_s=3.0, lam_cm=lcm, lam_time=400)))

    WON, NMIN = [], []
    P = {n: [] for n, _ in configs}
    for m in tms:
        bmask = ds.bouts["month_idx"].to_numpy()
        train = _subset(ds, bmask < m); test = ds.bouts[bmask == m]
        if len(train.bouts) < 500 or len(test) == 0:
            continue
        cnt = Counter(train.bouts["a_idx"].tolist()) + Counter(train.bouts["b_idx"].tolist())
        a = test["a_idx"].to_numpy(); b = test["b_idx"].to_numpy()
        NMIN.append(np.minimum([cnt.get(int(x), 0) for x in a], [cnt.get(int(x), 0) for x in b]))
        WON.append((test["winner_id"].to_numpy() == test["fencer_a_id"].to_numpy()).astype(float))
        for n, cfg in configs:
            fm = M.fit(train, cfg); _, p = fm.predict(test); P[n].append(p)
    WON = np.concatenate(WON); NMIN = np.concatenate(NMIN)
    P = {k: np.concatenate(v) for k, v in P.items()}
    B = [("all", np.ones(len(WON), bool)), ("N=0", NMIN == 0), ("1-5", (NMIN >= 1) & (NMIN <= 5)),
         ("6-15", (NMIN >= 6) & (NMIN <= 15)), ("16-40", (NMIN >= 16) & (NMIN <= 40)), ("41+", NMIN >= 41)]
    print(f"db={db}  bouts={len(WON)}  (log-loss; lower better)\n")
    print(f"{'config':20s} " + "".join(f"{bn:>8}" for bn, _ in B))
    base = {bn: _ll(WON[mk], P['current'][mk]) for bn, mk in B}
    for n, _ in configs:
        cells = "".join(f"{_ll(WON[mk], P[n][mk]):>8.4f}" for _, mk in B)
        print(f"{n:20s} {cells}")
    print("\n(Δ vs current, all bouts):")
    for n, _ in configs[1:]:
        print(f"  {n:20s} {_ll(WON, P[n]) - base['all']:+.4f}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "fencing_full.db",
        int(sys.argv[2]) if len(sys.argv) > 2 else 8)
