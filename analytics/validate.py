"""Out-of-sample validation + hyperparameter selection for the rating model.

Uses forward-chaining (rolling-origin) cross-validation, held out by month: train on
all bouts before month m, predict the bouts in month m. Because every test bout is in
the future relative to training, whole events are held out automatically (no pool/DE
leakage within an event), and prediction uses each fencer's carry-forward skill — the
same thing we'll do to forecast a real upcoming event.

Metrics are pooled across all test months. Win prob = P(a beats b) = Phi(mu/sigma_type).
Reported both over all test bouts and over the "both fencers seen in training" subset
(the rest are cold-start, predictable only from age/club).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .features import Dataset, load_dataset
from . import model as M


def _subset(ds: Dataset, mask) -> Dataset:
    return Dataset(bouts=ds.bouts[mask].reset_index(drop=True),
                   months=ds.months, fencer_ids=ds.fencer_ids, club_names=ds.club_names,
                   popular_clubs=ds.popular_clubs)


def _metrics(z, won, mu, p, seen) -> dict:
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    def block(m):
        if m.sum() == 0:
            return {}
        zz, ww, mm, pp = z[m], won[m], mu[m], p[m]
        return {
            "n": int(m.sum()),
            "rmse": float(np.sqrt(np.mean((zz - mm) ** 2))),
            "logloss": float(-np.mean(ww * np.log(pp) + (1 - ww) * np.log(1 - pp))),
            "brier": float(np.mean((pp - ww) ** 2)),
            "acc": float(np.mean((pp > 0.5) == (ww > 0.5))),
        }
    out = {"all": block(np.ones(len(z), bool)), "seen": block(seen)}
    out["coverage"] = float(seen.mean())
    return out


def forward_chaining(ds: Dataset, cfg: M.ModelConfig, test_months: list[int]) -> dict:
    """Fit/predict for each test month; pool predictions and score once."""
    Z, WON, MU, P, SEEN = [], [], [], [], []
    for m in test_months:
        bmask = ds.bouts["month_idx"].to_numpy()
        train = _subset(ds, bmask < m)
        test = ds.bouts[bmask == m]
        if len(train.bouts) < 500 or len(test) == 0:
            continue
        fm = M.fit(train, cfg)
        mu, p = fm.predict(test)
        won = (test["winner_id"].to_numpy() == test["fencer_a_id"].to_numpy()).astype(float)
        Z.append(test["z"].to_numpy()); WON.append(won); MU.append(mu); P.append(p)
        SEEN.append(fm.both_seen(test))
    if not Z:
        return {}
    return _metrics(np.concatenate(Z), np.concatenate(WON), np.concatenate(MU),
                    np.concatenate(P), np.concatenate(SEEN))


def baseline_global_mean(ds: Dataset, test_months: list[int]) -> dict:
    Z, WON, MU, P, SEEN = [], [], [], [], []
    bmask = ds.bouts["month_idx"].to_numpy()
    for m in test_months:
        train = ds.bouts[bmask < m]; test = ds.bouts[bmask == m]
        if len(train) < 500 or len(test) == 0:
            continue
        mz = float(train["z"].mean())
        sig = float(train["z"].std()) or 1.0
        mu = np.full(len(test), mz)
        p = M._norm_cdf(mu / sig)
        won = (test["winner_id"].to_numpy() == test["fencer_a_id"].to_numpy()).astype(float)
        Z.append(test["z"].to_numpy()); WON.append(won); MU.append(mu); P.append(p)
        SEEN.append(np.zeros(len(test), bool))
    return _metrics(np.concatenate(Z), np.concatenate(WON), np.concatenate(MU),
                    np.concatenate(P), np.concatenate(SEEN)) if Z else {}


def elo_eval(ds: Dataset, test_months: list[int], K: float = 32.0,
             scale: float = 400.0, base: float = 1500.0) -> dict:
    """Standard win/loss Elo as a baseline, scored in the same forward-chaining CV.

    For each test month, ratings are trained on all earlier bouts (processed in
    approximate chronological order: month, then pools before DEs), frozen, and used
    to predict that month's win probabilities. Elo has no score-margin model, so RMSE(Z)
    is left NaN; the fair comparison is log-loss / Brier / accuracy on win/loss."""
    from collections import defaultdict
    b = ds.bouts
    order = b.sort_values(["month_idx", "de", "event_id"])
    mi_all = order["month_idx"].to_numpy()
    Z, WON, MU, P, SEEN = [], [], [], [], []
    for m in test_months:
        tr = order[mi_all < m]
        test = b[b["month_idx"].to_numpy() == m]
        if len(tr) < 500 or len(test) == 0:
            continue
        R = defaultdict(lambda: base); seen = set()
        for a, bb, w in zip(tr["fencer_a_id"].to_numpy(), tr["fencer_b_id"].to_numpy(),
                            tr["winner_id"].to_numpy()):
            ra, rb = R[a], R[bb]
            ea = 1.0 / (1.0 + 10 ** ((rb - ra) / scale))
            sa = 1.0 if w == a else 0.0
            R[a] = ra + K * (sa - ea); R[bb] = rb + K * ((1 - sa) - (1 - ea))
            seen.add(a); seen.add(bb)
        aa = test["fencer_a_id"].to_numpy(); bbb = test["fencer_b_id"].to_numpy()
        ra = np.array([R.get(x, base) for x in aa])
        rb = np.array([R.get(x, base) for x in bbb])
        p = 1.0 / (1.0 + 10 ** ((rb - ra) / scale))
        won = (test["winner_id"].to_numpy() == aa).astype(float)
        Z.append(test["z"].to_numpy()); WON.append(won)
        MU.append(np.full(len(test), np.nan)); P.append(p)
        SEEN.append(np.array([(x in seen and y in seen) for x, y in zip(aa, bbb)]))
    if not Z:
        return {}
    return _metrics(np.concatenate(Z), np.concatenate(WON), np.concatenate(MU),
                    np.concatenate(P), np.concatenate(SEEN))


def default_test_months(ds: Dataset, n: int = 6) -> list[int]:
    """The last `n` month indices that actually contain bouts."""
    present = sorted(ds.bouts["month_idx"].unique().tolist())
    return present[-n:]


def collect_model_elo(ds: Dataset, cfg: M.ModelConfig, test_months: list[int], K: float = 32.0):
    """Aligned per-bout predictions from the model and Elo on the same test bouts, for
    studying an ensemble. Returns dict of arrays: won, p_model, p_elo, seen, month."""
    from collections import defaultdict
    b = ds.bouts
    order = b.sort_values(["month_idx", "de", "event_id"]); mi = order["month_idx"].to_numpy()
    bmi = b["month_idx"].to_numpy()
    out = {k: [] for k in ("won", "p_model", "p_elo", "seen", "month")}
    for m in test_months:
        train = _subset(ds, bmi < m); test = b[bmi == m]
        if len(train.bouts) < 500 or len(test) == 0:
            continue
        fm = M.fit(train, cfg)
        _, pm = fm.predict(test)
        tr = order[mi < m]; R = defaultdict(lambda: 1500.0); seenset = set()
        for a, bb, w in zip(tr["fencer_a_id"].to_numpy(), tr["fencer_b_id"].to_numpy(),
                            tr["winner_id"].to_numpy()):
            ra, rb = R[a], R[bb]; ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
            sa = 1.0 if w == a else 0.0
            R[a] = ra + K * (sa - ea); R[bb] = rb + K * ((1 - sa) - (1 - ea))
            seenset.add(a); seenset.add(bb)
        aa = test["fencer_a_id"].to_numpy(); bbb = test["fencer_b_id"].to_numpy()
        ra = np.array([R.get(x, 1500.0) for x in aa]); rb = np.array([R.get(x, 1500.0) for x in bbb])
        pe = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        out["won"].append((test["winner_id"].to_numpy() == aa).astype(float))
        out["p_model"].append(pm); out["p_elo"].append(pe)
        out["seen"].append(fm.both_seen(test)); out["month"].append(np.full(len(test), m))
    return {k: np.concatenate(v) for k, v in out.items()}


def cross_validate(ds: Dataset, configs: list[tuple[str, M.ModelConfig]],
                   test_months: Optional[list[int]] = None) -> pd.DataFrame:
    test_months = test_months or default_test_months(ds)
    rows = []
    base = baseline_global_mean(ds, test_months)
    if base:
        rows.append(dict(name="baseline:global-mean", **base["all"], coverage=base["coverage"]))
    elo = elo_eval(ds, test_months, K=32)
    if elo:
        rows.append(dict(name="baseline:elo K=32", **elo["all"], coverage=elo["coverage"],
                         seen_logloss=elo["seen"].get("logloss"), seen_acc=elo["seen"].get("acc")))
    for name, cfg in configs:
        r = forward_chaining(ds, cfg, test_months)
        if r:
            rows.append(dict(name=name, **r["all"], coverage=r["coverage"],
                             seen_logloss=r["seen"].get("logloss"), seen_acc=r["seen"].get("acc")))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "fencing.db"
    ds = load_dataset(path, core_only=True, since="2022-09")
    tm = default_test_months(ds, 4)
    print(f"dataset {len(ds.bouts)} bouts; test months: {[ds.months[i] for i in tm]}", flush=True)
    configs = []
    for rank in (0, 1, 2, 4):
        for lt in (5, 30, 100):
            for luv in ([1] if rank == 0 else [5, 20]):
                configs.append((
                    f"r{rank}_lt{lt}_uv{luv}",
                    M.ModelConfig(rank=rank, lam_s=0.05, lam_time=lt,
                                  lam_u=luv, lam_v=luv, max_iter=2500),
                ))
    df = cross_validate(ds, configs, tm).sort_values("logloss")
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
