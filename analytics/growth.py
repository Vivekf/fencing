"""Stage-2 longitudinal growth model.

Stage 1 (analytics.model) produces a monthly skill panel s_{i,t}. This module explains /
projects that panel as a smooth population curve in AGE and EXPERIENCE plus a per-fencer
level (partial-pooled random intercept):

    s_{i,t} = b0 + g_age(age) + g_exp(experience) + alpha_i + eps

It is deliberately a *separate* stage so we keep the validated rating model intact and can
backtest the projector (growth curve) against a carry-forward / random-walk baseline.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd

ALL_FEATURES = ["age", "age2", "log_bouts", "tenure"]


def _month_decimal(ms: str) -> float:
    y, m = ms.split("-")
    return int(y) + (int(m) - 0.5) / 12.0


def build_panel(db_path: str, fm, ds) -> pd.DataFrame:
    """One row per fitted (fencer, month) skill cell, with age + experience covariates.

    age      = month_decimal - birth_year (birth year only is known)
    cum_bouts= bouts fenced up to and including that month  (experience volume)
    tenure   = months since the fencer's first observed bout (time in the sport)
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    by = {r[0]: r[1] for r in conn.execute("SELECT id, birth_year FROM fencers")}
    bm: dict[int, list[int]] = defaultdict(list)
    for f, m in zip(ds.bouts.fencer_a_id.to_numpy(), ds.bouts.month_idx.to_numpy()):
        bm[int(f)].append(int(m))
    for f, m in zip(ds.bouts.fencer_b_id.to_numpy(), ds.bouts.month_idx.to_numpy()):
        bm[int(f)].append(int(m))
    bm = {f: np.sort(np.asarray(v)) for f, v in bm.items()}

    cell, _ = fm._skill_lookup()
    rows = []
    for (fi, mi), s in cell.items():
        f = fm.fencer_ids[fi]
        yr = by.get(f)
        months = bm.get(f)
        if not yr or months is None:
            continue
        cum = int(np.searchsorted(months, mi, side="right"))
        rows.append((f, int(mi), float(s),
                     _month_decimal(fm.months[mi]) - yr, cum, int(mi - months[0])))
    p = pd.DataFrame(rows, columns=["fencer_id", "month_idx", "s", "age", "cum_bouts", "tenure"])
    p["age2"] = p["age"] ** 2
    p["log_bouts"] = np.log1p(p["cum_bouts"])
    return p


def _designX(p: pd.DataFrame, feats, means=None, stds=None):
    X = p[feats].to_numpy(float)
    if means is None:
        means = X.mean(0)
        stds = X.std(0) + 1e-9
    return (X - means) / stds, means, stds


def fit_growth(p: pd.DataFrame, feats=ALL_FEATURES, lam_alpha=10.0, lam_beta=1e-3) -> dict:
    """Partial-pooling least squares: ridge lam_alpha on the per-fencer intercepts,
    lam_beta on the (standardized) population slopes. Alpha is profiled out via a Schur
    complement so we only solve a small (k+1)x(k+1) system."""
    Xs, means, stds = _designX(p, feats)
    y = p["s"].to_numpy(float)
    N = len(y)
    Xa = np.column_stack([np.ones(N), Xs])          # col 0 = global intercept
    kb = Xa.shape[1]
    _, inv = np.unique(p["fencer_id"].to_numpy(), return_inverse=True)
    n_i = np.bincount(inv).astype(float)
    Sx = np.column_stack([np.bincount(inv, Xa[:, c]) for c in range(kb)])
    Sy = np.bincount(inv, y)
    d = 1.0 / (n_i + lam_alpha)
    Pb = np.diag([0.0] + [lam_beta] * (kb - 1))     # never penalize the intercept
    Mmat = Xa.T @ Xa + Pb - (Sx.T * d) @ Sx
    vvec = Xa.T @ y - (Sx.T * d) @ Sy
    beta = np.linalg.solve(Mmat, vvec)
    alpha = d * (Sy - Sx @ beta)
    uf = np.unique(p["fencer_id"].to_numpy())
    resid = y - (Xa @ beta + alpha[inv])
    return dict(feats=list(feats), beta=beta, alpha=dict(zip(uf.tolist(), alpha.tolist())),
                means=means, stds=stds, resid_std=float(resid.std()), y_std=float(y.std()),
                lam_alpha=lam_alpha)


def predict_panel(fitg: dict, p: pd.DataFrame) -> np.ndarray:
    Xs, _, _ = _designX(p, fitg["feats"], fitg["means"], fitg["stds"])
    base = fitg["beta"][0] + Xs @ fitg["beta"][1:]
    a = p["fencer_id"].map(fitg["alpha"]).fillna(0.0).to_numpy()
    return base + a


def beta_in_natural_units(fitg: dict) -> dict:
    """Standardized slopes -> per-natural-unit effects (e.g. skill per year of age)."""
    return {f: float(b / s) for f, b, s in
            zip(fitg["feats"], fitg["beta"][1:], fitg["stds"])}
