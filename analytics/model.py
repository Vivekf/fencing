"""Fit the dynamic low-rank fencing rating model (see analytics.tex).

Mean model for a bout between i (=fencer_a) and j (=fencer_b) in month m:

    mu = s[i,m] - s[j,m]
       + (u_i . v_j - u_j . v_i)          # antisymmetric low-rank matchup
       + beta_age (a_im - a_jm)
       + c[club_i] - c[club_j]
       + beta_de * 1{DE}

Objective (penalized least squares; bout-type variances enter as weights):

    sum_b w_b (mu_b - Z_b)^2
  + lam_time sum over consecutive active months (s_t - s_{t-1})^2 / gap
  + lam_s ||s||^2 + lam_u ||U||^2 + lam_v ||V||^2 + lam_c ||c||^2 + lam_beta ||beta||^2

with w_b = 1 / (2 sigma_type^2). Solved by full-batch Adam on analytic gradients; the
sigma's are re-estimated from residuals and the model refit (a couple of outer passes).

Identifiability: skill is centered per month (sum_i s_im = 0) and club effects are
centered (sum_l c_l = 0) after fitting — the model is invariant to these gauges, which
the ridge already nearly fixes; we enforce them exactly for interpretability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .features import Dataset

def _pair_ids(ca, cb, P):
    """Antisymmetric club-pair encoding. Returns (pair_id, sign, npairs) where pair_id
    indexes the upper-triangular vector of P-category pairs (sentinel = npairs for
    same-category / unknown), and sign is +1 if ca<cb, -1 if ca>cb, 0 otherwise."""
    ca = np.asarray(ca); cb = np.asarray(cb)
    npairs = P * (P - 1) // 2
    valid = (ca >= 0) & (cb >= 0) & (ca != cb)
    lo = np.minimum(ca, cb); hi = np.maximum(ca, cb)
    idx = (lo * (2 * P - lo - 1)) // 2 + (hi - lo - 1)
    pid = np.where(valid, idx, npairs).astype(np.int64)
    sign = np.where(valid, np.where(ca < cb, 1.0, -1.0), 0.0)
    return pid, sign, npairs


def _norm_cdf(x):
    """Vectorized standard normal CDF via the Abramowitz-Stegun erf approximation
    (max abs error ~1.5e-7) — array-native and far faster than np.vectorize(math.erf)."""
    x = np.asarray(x, dtype=np.float64) / math.sqrt(2.0)  # erf argument
    s = np.sign(x); ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-ax * ax)
    return 0.5 * (1.0 + s * y)


@dataclass
class ModelConfig:
    rank: int = 4
    lam_time: float = 5.0
    lam_s: float = 1.0
    lam_u: float = 1.0
    lam_v: float = 1.0
    lam_c: float = 1.0
    lam_beta: float = 0.1
    # Hierarchical club: instead of an additive fixed effect, treat the club as a PRIOR
    # MEAN that each fencer's (absolute) skill is shrunk toward. Credibility w=N/(N+K)
    # emerges with K proportional to lam_s, so a high-N fencer escapes the club entirely
    # (club "wears off") while a low-N / unseen fencer defaults to their club mean.
    hier_club: bool = False
    lam_cm: float = 1.0           # ridge on club means toward the global mean
    use_club_pair: bool = False   # antisymmetric club-pair interaction D[club_i,club_j]
    lam_d: float = 1.0
    use_de_delta: bool = False    # per-fencer DE-vs-pool skill delta (delta_i - delta_j) on DE bouts
    lam_delta: float = 1.0
    lr: float = 0.05
    max_iter: int = 4000
    tol: float = 1e-7          # rel. loss improvement (over a window) to stop
    outer_iters: int = 2       # refits to update sigma_pool / sigma_de
    seed: int = 0


@dataclass
class FittedModel:
    config: ModelConfig
    fencer_ids: list[int]
    months: list[str]
    club_names: list[str]
    # parameters
    s: np.ndarray              # skill at active cells
    cell_fencer: np.ndarray    # cell -> fencer index
    cell_month: np.ndarray     # cell -> month index
    U: np.ndarray              # (F, k)
    V: np.ndarray              # (F, k)
    c: np.ndarray              # (C,)  additive club effect (empty when hier_club)
    beta_age: float
    beta_de: float
    sigma_pool: float
    sigma_de: float
    cm: np.ndarray = field(default_factory=lambda: np.zeros(0))          # (C,) club means (hier_club)
    delta: np.ndarray = field(default_factory=lambda: np.zeros(0))       # per-fencer DE-vs-pool skill delta
    d: np.ndarray = field(default_factory=lambda: np.zeros(0))           # club-pair interaction (antisymmetric)
    n_club_cat: int = 0                                                  # popular clubs + 'other'
    last_skill: np.ndarray = field(default_factory=lambda: np.zeros(0))  # carry-forward skill per global fencer idx
    seen: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # had >=1 training bout
    loss_history: list[float] = field(default_factory=list)

    def predict(self, df):
        """Out-of-sample mean and win-prob for bouts (oriented a vs b). Uses each
        fencer's carry-forward (last-known) skill — the random-walk forecast — so it's
        the right call for predicting a future event. Unseen fencers contribute 0 skill
        and 0 matchup (cold start)."""
        a = df["a_idx"].to_numpy(); b = df["b_idx"].to_numpy()
        hier = self.config.hier_club
        C = len(self.cm) if hier else len(self.c)
        ca = df["club_a"].to_numpy().copy(); cb = df["club_b"].to_numpy().copy()
        ca[ca < 0] = C; cb[cb < 0] = C
        if hier:
            # Skill is absolute; an unseen fencer defaults to their club mean (the prior).
            cmf = np.concatenate([self.cm, [0.0]])
            sa = np.where(self.seen[a], self.last_skill[a], cmf[ca])
            sb = np.where(self.seen[b], self.last_skill[b], cmf[cb])
            mu = sa - sb
        else:
            mu = self.last_skill[a] - self.last_skill[b]
        mu = mu + (np.einsum("bk,bk->b", self.U[a], self.V[b])
                   - np.einsum("bk,bk->b", self.U[b], self.V[a]))
        if not hier:                                     # additive club fixed effect
            cfull = np.concatenate([self.c, [0.0]])
            mu = mu + cfull[ca] - cfull[cb]
        if len(self.d) > 0 and self.n_club_cat >= 2:
            pid, sign, _ = _pair_ids(df["club_pop_a"].to_numpy(),
                                     df["club_pop_b"].to_numpy(), self.n_club_cat)
            dfull = np.concatenate([self.d, [0.0]])
            mu = mu + sign * dfull[pid]
        mu = mu + self.beta_age * np.nan_to_num(df["age_diff"].to_numpy(), nan=0.0)
        de = df["de"].to_numpy().astype(float)
        mu = mu + self.beta_de * de
        if len(self.delta) > 0:
            mu = mu + de * (self.delta[a] - self.delta[b])
        sigma = np.where(de > 0.5, self.sigma_de, self.sigma_pool)
        return mu, _norm_cdf(mu / sigma)

    def both_seen(self, df):
        return self.seen[df["a_idx"].to_numpy()] & self.seen[df["b_idx"].to_numpy()]

    # --- lookups -------------------------------------------------------------
    def _skill_lookup(self) -> dict:
        """(fencer_idx, month_idx) -> skill, plus per-fencer sorted (month, skill)."""
        cell = {}
        traj: dict[int, list[tuple[int, float]]] = {}
        for k in range(len(self.s)):
            fi, mi = int(self.cell_fencer[k]), int(self.cell_month[k])
            cell[(fi, mi)] = self.s[k]
            traj.setdefault(fi, []).append((mi, self.s[k]))
        for fi in traj:
            traj[fi].sort()
        return cell, traj

    def skill_at(self, cell, traj, fencer_idx: int, month_idx: int) -> float:
        """Skill of a fencer in a month; carry the last known skill forward for months
        with no bout (the random-walk forecast). 0.0 if the fencer is unseen."""
        v = cell.get((fencer_idx, month_idx))
        if v is not None:
            return v
        t = traj.get(fencer_idx)
        if not t:
            return 0.0
        prev = 0.0
        for mi, s in t:
            if mi <= month_idx:
                prev = s
            else:
                break
        return prev


def _build_arrays(ds: Dataset, cfg: ModelConfig):
    b = ds.bouts
    ai = b["a_idx"].to_numpy(np.int64)
    bi = b["b_idx"].to_numpy(np.int64)
    mi = b["month_idx"].to_numpy(np.int64)
    z = b["z"].to_numpy(np.float64)
    de = b["de"].to_numpy(np.float64)
    agediff = np.nan_to_num(b["age_diff"].to_numpy(np.float64), nan=0.0)

    # Active skill cells (fencer, month)
    F = len(ds.fencer_ids)
    keyA = ai * (len(ds.months)) + mi
    keyB = bi * (len(ds.months)) + mi
    keys = np.unique(np.concatenate([keyA, keyB]))
    cell_of = {int(k): idx for idx, k in enumerate(keys)}
    cellA = np.array([cell_of[int(k)] for k in keyA], dtype=np.int64)
    cellB = np.array([cell_of[int(k)] for k in keyB], dtype=np.int64)
    cell_fencer = (keys // len(ds.months)).astype(np.int64)
    cell_month = (keys % len(ds.months)).astype(np.int64)

    # Smoothness edges: consecutive active months per fencer, weight 1/gap (Brownian)
    elo, ehi, ew = [], [], []
    by_fencer: dict[int, list[tuple[int, int]]] = {}
    for k in range(len(keys)):
        by_fencer.setdefault(int(cell_fencer[k]), []).append((int(cell_month[k]), k))
    for cells in by_fencer.values():
        cells.sort()
        for (m0, c0), (m1, c1) in zip(cells, cells[1:]):
            gap = max(1, m1 - m0)
            elo.append(c0); ehi.append(c1); ew.append(1.0 / gap)
    elo = np.array(elo, np.int64); ehi = np.array(ehi, np.int64); ew = np.array(ew, np.float64)

    # Clubs: map -1 (unknown) to a sentinel slot C with fixed 0 effect
    C = len(ds.club_names)
    clubA = b["club_a"].to_numpy(np.int64).copy(); clubA[clubA < 0] = C
    clubB = b["club_b"].to_numpy(np.int64).copy(); clubB[clubB < 0] = C

    # Per-fencer club (stationary) and per-cell club, for the hierarchical prior anchor.
    fencer_club = np.full(F, C, np.int64)
    fencer_club[ai] = clubA; fencer_club[bi] = clubB
    cell_club = fencer_club[cell_fencer]

    # Antisymmetric club-pair interaction (optional)
    P = (len(ds.popular_clubs) + 1) if cfg.use_club_pair else 0
    if P >= 2:
        pair_id, pair_sign, npairs = _pair_ids(
            b["club_pop_a"].to_numpy(), b["club_pop_b"].to_numpy(), P)
    else:
        pair_id = np.zeros(len(b), np.int64); pair_sign = np.zeros(len(b)); npairs = 0

    return dict(
        ai=ai, bi=bi, mi=mi, z=z, de=de, agediff=agediff,
        cellA=cellA, cellB=cellB, Ns=len(keys), cell_fencer=cell_fencer, cell_month=cell_month,
        elo=elo, ehi=ehi, ew=ew, clubA=clubA, clubB=clubB, F=F, C=C,
        fencer_club=fencer_club, cell_club=cell_club,
        pair_id=pair_id, pair_sign=pair_sign, npairs=npairs, P=P,
    )


def _predict_mu(P, A, s, U, V, cfull, beta_age, beta_de, delta=None, use_club=True):
    match = np.einsum("bk,bk->b", U[A["ai"]], V[A["bi"]]) - np.einsum("bk,bk->b", U[A["bi"]], V[A["ai"]])
    mu = s[A["cellA"]] - s[A["cellB"]] + match + beta_age * A["agediff"] + beta_de * A["de"]
    if use_club:                      # additive club fixed effect (skipped under hier_club)
        mu = mu + (cfull[A["clubA"]] - cfull[A["clubB"]])
    if delta is not None and len(delta) > 0:
        mu = mu + A["de"] * (delta[A["ai"]] - delta[A["bi"]])
    return mu


def _value_and_grad(A, cfg, w, params):
    """Penalized objective and its gradient w.r.t. every parameter block."""
    s, U, V, c, d, delta = (params["s"], params["U"], params["V"], params["c"],
                            params["d"], params["delta"])
    beta_age = float(params["beta_age"]); beta_de = float(params["beta_de"])
    Ns, F, C, k = A["Ns"], A["F"], A["C"], cfg.rank
    ai, bi = A["ai"], A["bi"]; cellA, cellB = A["cellA"], A["cellB"]
    clubA, clubB = A["clubA"], A["clubB"]; z, de, agediff = A["z"], A["de"], A["agediff"]
    elo, ehi, ew = A["elo"], A["ehi"], A["ew"]
    npairs = A["npairs"]; has_delta = len(delta) > 0

    hier = cfg.hier_club
    cm = params["cm"] if hier else None
    cell_club = A["cell_club"]

    cfull = np.concatenate([c, [0.0]])
    mu = _predict_mu(None, A, s, U, V, cfull, beta_age, beta_de,
                     delta if has_delta else None, use_club=not hier)
    if npairs > 0:
        dfull = np.concatenate([d, [0.0]])
        mu = mu + A["pair_sign"] * dfull[A["pair_id"]]
    r = mu - z
    g = 2.0 * w * r

    sm = s[ehi] - s[elo]
    # Skill ridge target: 0 (fixed-effect club) or the fencer's club mean (hierarchical).
    if hier:
        cm_full = np.concatenate([cm, [0.0]])
        sdiff = s - cm_full[cell_club]                 # deviation from club mean
        skill_pen = cfg.lam_s * float(sdiff @ sdiff) + cfg.lam_cm * float(cm @ cm)
        club_pen = 0.0
    else:
        sdiff = s
        skill_pen = cfg.lam_s * float(s @ s)
        club_pen = cfg.lam_c * float(c @ c)

    loss = (float(np.sum(w * r * r))
            + cfg.lam_time * float(np.sum(ew * sm * sm))
            + skill_pen + club_pen + cfg.lam_u * float(np.sum(U * U))
            + cfg.lam_v * float(np.sum(V * V))
            + cfg.lam_beta * (beta_age ** 2 + beta_de ** 2)
            + (cfg.lam_d * float(d @ d) if npairs > 0 else 0.0)
            + (cfg.lam_delta * float(delta @ delta) if has_delta else 0.0))

    gs = (np.bincount(cellA, g, Ns) - np.bincount(cellB, g, Ns)) + 2 * cfg.lam_s * sdiff
    sm_g = 2 * cfg.lam_time * ew * sm
    gs += np.bincount(ehi, sm_g, Ns) - np.bincount(elo, sm_g, Ns)

    gU = np.empty_like(U); gV = np.empty_like(V)
    Ua, Ub, Va, Vb = U[ai], U[bi], V[ai], V[bi]
    for dd in range(k):
        gU[:, dd] = np.bincount(ai, g * Vb[:, dd], F) - np.bincount(bi, g * Va[:, dd], F)
        gV[:, dd] = np.bincount(bi, g * Ua[:, dd], F) - np.bincount(ai, g * Ub[:, dd], F)
    gU += 2 * cfg.lam_u * U
    gV += 2 * cfg.lam_v * V

    if hier:
        # Club not in mu; club means are pulled toward the skills anchored to them.
        gc = 2 * cfg.lam_c * c                          # inert (c unused, stays ~0)
        gcm = (-2 * cfg.lam_s * np.bincount(cell_club, sdiff, C + 1)[:C]
               + 2 * cfg.lam_cm * cm)
    else:
        gc_full = np.bincount(clubA, g, C + 1) - np.bincount(clubB, g, C + 1)
        gc = gc_full[:C] + 2 * cfg.lam_c * c

    gba = float(np.sum(g * agediff)) + 2 * cfg.lam_beta * beta_age
    gbd = float(np.sum(g * de)) + 2 * cfg.lam_beta * beta_de

    if npairs > 0:
        gd_full = np.bincount(A["pair_id"], A["pair_sign"] * g, npairs + 1)
        gd = gd_full[:npairs] + 2 * cfg.lam_d * d
    else:
        gd = d  # empty

    if has_delta:
        gde = g * de
        gdelta = (np.bincount(ai, gde, F) - np.bincount(bi, gde, F)) + 2 * cfg.lam_delta * delta
    else:
        gdelta = delta  # empty

    grads = {"s": gs, "U": gU, "V": gV, "c": gc, "d": gd, "delta": gdelta,
             "beta_age": np.array(gba), "beta_de": np.array(gbd)}
    if hier:
        grads["cm"] = gcm
    return loss, grads


def _fit_once(A, cfg, w, init=None):
    rng = np.random.default_rng(cfg.seed)
    Ns, F, k = A["Ns"], A["F"], cfg.rank
    ndelta = F if cfg.use_de_delta else 0
    if init is None:
        s = np.zeros(Ns); U = 0.01 * rng.standard_normal((F, k)); V = 0.01 * rng.standard_normal((F, k))
        c = np.zeros(A["C"]); d = np.zeros(A["npairs"]); delta = np.zeros(ndelta)
        beta_age = 0.0; beta_de = 0.0
        cm = np.zeros(A["C"]) if cfg.hier_club else None
    else:
        s, U, V, c, d, delta, beta_age, beta_de = (init[x].copy() if hasattr(init[x], "copy") else init[x]
                                                   for x in ("s", "U", "V", "c", "d", "delta", "beta_age", "beta_de"))
        cm = init["cm"].copy() if cfg.hier_club else None

    # Adam state
    params = {"s": s, "U": U, "V": V, "c": c, "d": d, "delta": delta,
              "beta_age": np.array(beta_age), "beta_de": np.array(beta_de)}
    if cfg.hier_club:
        params["cm"] = cm
    mom = {key: np.zeros_like(val) for key, val in params.items()}
    vel = {key: np.zeros_like(val) for key, val in params.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    history = []

    for it in range(1, cfg.max_iter + 1):
        loss, grads = _value_and_grad(A, cfg, w, params)
        history.append(loss)

        # Adam step
        for key in params:
            mom[key] = b1 * mom[key] + (1 - b1) * grads[key]
            vel[key] = b2 * vel[key] + (1 - b2) * grads[key] ** 2
            mhat = mom[key] / (1 - b1 ** it)
            vhat = vel[key] / (1 - b2 ** it)
            params[key] = params[key] - cfg.lr * mhat / (np.sqrt(vhat) + eps)

        if it > 50 and it % 25 == 0:
            recent = history[-25]
            if recent - loss < cfg.tol * max(1.0, recent):
                break

    return params, history


def fit(ds: Dataset, cfg: Optional[ModelConfig] = None, *, center_skill: bool = True) -> FittedModel:
    cfg = cfg or ModelConfig()
    A = _build_arrays(ds, cfg)

    sigma_pool = sigma_de = 1.0
    is_de = A["de"] > 0.5
    params, history = None, []
    for outer in range(cfg.outer_iters):
        w = np.where(is_de, 1.0 / (2 * sigma_de ** 2), 1.0 / (2 * sigma_pool ** 2))
        init = params  # warm start from previous outer pass
        params, hist = _fit_once(A, cfg, w, init=init)
        history += hist
        # re-estimate sigmas from residuals
        cfull = np.concatenate([params["c"], [0.0]])
        mu = _predict_mu(None, A, params["s"], params["U"], params["V"], cfull,
                         float(params["beta_age"]), float(params["beta_de"]),
                         use_club=not cfg.hier_club)
        r = mu - A["z"]
        sigma_pool = float(np.sqrt(np.mean(r[~is_de] ** 2))) if (~is_de).any() else 1.0
        sigma_de = float(np.sqrt(np.mean(r[is_de] ** 2))) if is_de.any() else 1.0

    # Gauge handling. Per-month centering (sum_i s_im=0) makes skill comparable only
    # WITHIN a month — it removes the cross-month level the smoothness prior + ridge
    # establish, which forecasting (carry-forward to a future month) needs. So default
    # to a single global centering (sum over all cells = 0), which keeps the random-walk
    # levels comparable across time. `center_skill=False` leaves the raw fitted gauge.
    s = params["s"]; cf, cmo = A["cell_fencer"], A["cell_month"]
    smean = float(s.mean()) if center_skill else 0.0
    if center_skill:
        s = s - smean
    c = params["c"] - params["c"].mean() if len(params["c"]) else params["c"]
    # Shift club means by the same gauge so seen skills and unseen club-mean fallbacks align.
    club_mean = (params["cm"] - smean) if cfg.hier_club else np.zeros(0)

    # Carry-forward (last-known) skill per global fencer, and a "seen in training" mask.
    F = len(ds.fencer_ids)
    last_skill = np.zeros(F)
    last_month = np.full(F, -1)
    seen = np.zeros(F, bool)
    for idx in range(len(s)):
        fi, mi = int(cf[idx]), int(cmo[idx])
        seen[fi] = True
        if mi > last_month[fi]:
            last_month[fi] = mi
            last_skill[fi] = s[idx]
    # Cold-start fencers (no training bouts) keep random-init factors; zero them so they
    # contribute nothing at prediction time.
    U = params["U"].copy(); V = params["V"].copy()
    U[~seen] = 0.0; V[~seen] = 0.0
    delta = params["delta"].copy()
    if len(delta):
        delta[~seen] = 0.0

    return FittedModel(
        config=cfg, fencer_ids=ds.fencer_ids, months=ds.months, club_names=ds.club_names,
        s=s, cell_fencer=cf, cell_month=cmo, U=U, V=V, c=c, cm=club_mean,
        beta_age=float(params["beta_age"]), beta_de=float(params["beta_de"]),
        sigma_pool=sigma_pool, sigma_de=sigma_de,
        delta=delta, d=params["d"], n_club_cat=A["P"],
        last_skill=last_skill, seen=seen, loss_history=history,
    )
