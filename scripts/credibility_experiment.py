"""Phase-1: does N-adaptive credibility shrinkage of skill beat the current model?

Post-hoc test (no re-fit): in each forward-chaining fold, scale each fencer's
carry-forward skill deviation by the Buhlmann factor w(N) = N/(N+K), where N is the
fencer's bout count in the training fold. K=0 recovers the current model (w=1). Skill is
already club-relative, so w<1 pulls a low-N fencer toward their club prior.

Scored by the same rolling-origin CV as analytics/validate.py, on log-loss, sliced by
min(N_a, N_b) — the less-established fencer in each bout, where shrinkage should help.

    PYTHONPATH=. python scripts/credibility_experiment.py [db] [n_months]
"""
from __future__ import annotations

import sys
from collections import Counter

import numpy as np

from analytics.features import load_dataset
from analytics import model as M
from analytics.validate import _subset, default_test_months


def _logloss(won, p):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(won * np.log(p) + (1 - won) * np.log(1 - p)))


def run(db="fencing_full.db", n_months=8, Ks=(0, 1, 2, 4, 8, 16, 32, 64, 128)):
    ds = load_dataset(db, core_only=True, since="2022-09", birth_min=2013, birth_max=2018)
    cfg = M.ModelConfig(rank=0, lam_s=0.05, lam_time=400)   # the deployed config
    tms = default_test_months(ds, n_months)

    WON, SIG, MU0, SA, SB, NA, NB = [], [], [], [], [], [], []
    for m in tms:
        bmask = ds.bouts["month_idx"].to_numpy()
        train = _subset(ds, bmask < m)
        test = ds.bouts[bmask == m]
        if len(train.bouts) < 500 or len(test) == 0:
            continue
        fm = M.fit(train, cfg)
        mu0, _ = fm.predict(test)
        cnt = Counter(train.bouts["a_idx"].tolist()) + Counter(train.bouts["b_idx"].tolist())
        a = test["a_idx"].to_numpy(); b = test["b_idx"].to_numpy()
        ls = fm.last_skill
        de = test["de"].to_numpy().astype(float)
        WON.append((test["winner_id"].to_numpy() == test["fencer_a_id"].to_numpy()).astype(float))
        SIG.append(np.where(de > 0.5, fm.sigma_de, fm.sigma_pool))
        MU0.append(mu0); SA.append(ls[a]); SB.append(ls[b])
        NA.append(np.array([cnt.get(int(x), 0) for x in a]))
        NB.append(np.array([cnt.get(int(x), 0) for x in b]))

    WON = np.concatenate(WON); SIG = np.concatenate(SIG); MU0 = np.concatenate(MU0)
    SA = np.concatenate(SA); SB = np.concatenate(SB)
    NA = np.concatenate(NA); NB = np.concatenate(NB)

    def wt(N, K):
        if K == 0:
            return (N > 0).astype(float)
        return N / (N + K)

    def logloss_K(K, mask):
        mu = MU0 + (wt(NA, K) - 1) * SA - (wt(NB, K) - 1) * SB
        return _logloss(WON[mask], M._norm_cdf(mu / SIG)[mask])

    nmin = np.minimum(NA, NB)
    buckets = [
        ("all", np.ones(len(WON), bool)),
        ("min N = 0", nmin == 0),
        ("min N 1-5", (nmin >= 1) & (nmin <= 5)),
        ("min N 6-15", (nmin >= 6) & (nmin <= 15)),
        ("min N 16-40", (nmin >= 16) & (nmin <= 40)),
        ("min N 41+", nmin >= 41),
    ]
    print(f"db={db}  test months={[ds.months[i] for i in tms]}  bouts scored={len(WON)}")
    print(f"\nLOG-LOSS by bucket x K   (K=0 = current model; * = best; lower better)")
    print(f"{'bucket':13s} {'n':>6}  " + "".join(f"K={k:<6}" for k in Ks))
    for name, mask in buckets:
        if mask.sum() == 0:
            continue
        vals = [logloss_K(K, mask) for K in Ks]
        best = int(np.argmin(vals))
        cells = "".join(f"{v:.4f}{'*' if i == best else ' '} " for i, v in enumerate(vals))
        delta = vals[best] - vals[0]
        print(f"{name:13s} {int(mask.sum()):>6}  {cells}  bestK={Ks[best]} ({delta:+.4f})")


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "fencing_full.db"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    run(db, n)
