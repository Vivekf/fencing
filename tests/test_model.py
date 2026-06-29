"""Correctness tests for the rating-model fitter: analytic gradient vs finite
differences, and parameter recovery on synthetic data.

Run: python -m pytest tests/test_model.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.features import Dataset
from analytics import model as M


def _toy_dataset(seed=0, n_fencers=8, n_months=4, n_bouts=120):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_bouts):
        i, j = rng.choice(n_fencers, size=2, replace=False)
        m = int(rng.integers(n_months))
        de = bool(rng.integers(2))
        rows.append(dict(
            event_id=0, month=f"2024-{m+1:02d}", month_idx=m,
            fencer_a_id=int(i), fencer_b_id=int(j), a_idx=int(i), b_idx=int(j),
            z=float(rng.standard_normal() * 0.3), target=15 if de else 5,
            de=de, sigma_type="de" if de else "pool",
            age_a=10.0, age_b=9.0, age_diff=float(rng.standard_normal()),
            club_a=int(i % 3), club_b=int(j % 3),
            club_pop_a=int(i % 4), club_pop_b=int(j % 4), winner_id=int(i),
        ))
    df = pd.DataFrame(rows)
    return Dataset(bouts=df, months=[f"2024-{m+1:02d}" for m in range(n_months)],
                   fencer_ids=list(range(n_fencers)), club_names=["A", "B", "C"],
                   popular_clubs=["c0", "c1", "c2"])


def test_gradient_matches_finite_difference():
    ds = _toy_dataset()
    cfg = M.ModelConfig(rank=3, lam_time=0.7, lam_s=0.5, lam_u=0.3, lam_v=0.4,
                        lam_c=0.2, lam_beta=0.1, use_club_pair=True, lam_d=0.6,
                        use_de_delta=True, lam_delta=0.5)
    A = M._build_arrays(ds, cfg)
    assert A["npairs"] > 0   # exercise the club-pair gradient
    rng = np.random.default_rng(1)
    params = {
        "s": rng.standard_normal(A["Ns"]),
        "U": rng.standard_normal((A["F"], cfg.rank)),
        "V": rng.standard_normal((A["F"], cfg.rank)),
        "c": rng.standard_normal(A["C"]),
        "d": rng.standard_normal(A["npairs"]),
        "delta": rng.standard_normal(A["F"]),
        "beta_age": np.array(0.4), "beta_de": np.array(-0.2),
    }
    w = np.where(A["de"] > 0.5, 1.0, 2.0)
    _, grads = M._value_and_grad(A, cfg, w, params)

    eps = 1e-6
    for key in params:
        flat = np.atleast_1d(params[key]).ravel()
        idxs = range(len(flat)) if len(flat) <= 6 else rng.choice(len(flat), 6, replace=False)
        for idx in idxs:
            up = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in params.items()}
            dn = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in params.items()}
            up[key] = np.atleast_1d(up[key]).astype(float).copy()
            dn[key] = np.atleast_1d(dn[key]).astype(float).copy()
            up[key].ravel()[idx] += eps
            dn[key].ravel()[idx] -= eps
            up[key] = up[key].reshape(np.shape(params[key]))
            dn[key] = dn[key].reshape(np.shape(params[key]))
            num = (M._value_and_grad(A, cfg, w, up)[0] - M._value_and_grad(A, cfg, w, dn)[0]) / (2 * eps)
            ana = np.atleast_1d(grads[key]).ravel()[idx]
            assert abs(num - ana) < 1e-4 * (1 + abs(ana)), f"{key}[{idx}]: num={num} ana={ana}"


def test_recovers_signal_on_synthetic_data():
    """Generate Z from a known mean model + noise; fit; check in-sample predictions
    track the truth far better than a constant baseline."""
    rng = np.random.default_rng(3)
    F, Mn, k = 12, 5, 2
    s_true = rng.standard_normal((F, Mn)) * 0.5
    U = rng.standard_normal((F, k)) * 0.3
    V = rng.standard_normal((F, k)) * 0.3
    rows = []
    for _ in range(2000):
        i, j = rng.choice(F, 2, replace=False)
        m = int(rng.integers(Mn))
        de = bool(rng.integers(2))
        mu = (s_true[i, m] - s_true[j, m]) + (U[i] @ V[j] - U[j] @ V[i])
        z = mu + rng.standard_normal() * 0.15
        rows.append(dict(
            event_id=0, month=f"2024-{m+1:02d}", month_idx=m,
            fencer_a_id=i, fencer_b_id=j, a_idx=int(i), b_idx=int(j),
            z=float(z), target=15 if de else 5, de=de, sigma_type="de" if de else "pool",
            age_a=10.0, age_b=10.0, age_diff=0.0, club_a=-1, club_b=-1, winner_id=int(i),
        ))
    df = pd.DataFrame(rows)
    ds = Dataset(bouts=df, months=[f"2024-{m+1:02d}" for m in range(Mn)],
                 fencer_ids=list(range(F)), club_names=[])
    cfg = M.ModelConfig(rank=2, lam_time=0.1, lam_s=0.05, lam_u=0.05, lam_v=0.05,
                        lam_c=0.0, lam_beta=0.0, max_iter=3000, outer_iters=1)
    fm = M.fit(ds, cfg)

    A = M._build_arrays(ds, cfg)
    cfull = np.concatenate([fm.c, [0.0]])
    # rebuild s vector aligned to A's cells from fitted model (same construction)
    mu_hat = M._predict_mu(None, A, fm.s, fm.U, fm.V, cfull, fm.beta_age, fm.beta_de)
    z = df["z"].to_numpy()
    rmse_model = np.sqrt(np.mean((z - mu_hat) ** 2))
    rmse_const = np.sqrt(np.mean((z - z.mean()) ** 2))
    # Should explain most of the structured variance; noise floor is 0.15.
    assert rmse_model < 0.6 * rmse_const, f"model {rmse_model:.3f} vs const {rmse_const:.3f}"
    assert rmse_model < 0.30, f"rmse_model {rmse_model:.3f} too high vs 0.15 noise floor"


def test_club_pair_is_active_when_enabled():
    """Guard against silently-disabled club-pair (e.g. popular_clubs dropped in a subset)."""
    ds = _toy_dataset()
    fm = M.fit(ds, M.ModelConfig(rank=0, use_club_pair=True, lam_d=1.0, max_iter=300))
    assert fm.n_club_cat >= 2
    assert np.abs(fm.d).sum() > 0
    # and prediction actually uses it
    mu, _ = fm.predict(ds.bouts)
    assert np.isfinite(mu).all()


def test_subset_preserves_popular_clubs():
    from analytics import validate as V
    ds = _toy_dataset()
    sub = V._subset(ds, ds.bouts["month_idx"] >= 0)
    assert sub.popular_clubs == ds.popular_clubs


if __name__ == "__main__":
    test_gradient_matches_finite_difference()
    print("OK: gradient matches finite differences")
    test_recovers_signal_on_synthetic_data()
    print("OK: recovers signal on synthetic data")
    test_club_pair_is_active_when_enabled()
    print("OK: club-pair active when enabled")
    test_subset_preserves_popular_clubs()
    print("OK: subset preserves popular clubs")
    print("\nAll model tests passed.")
