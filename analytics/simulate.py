"""Monte-Carlo a USA Fencing youth event to get a placement distribution.

Format modelled (Y10):
  1. Seed entrants by effective strength g (best = seed 1).
  2. Pools of 6-7 (as close as possible; >=5), snake-seeded across pools; round-robin
     to 5 touches. Rank after pools by win-ratio, then indicator (TS-TR), then TS.
  3. 100%-promotion single-elimination DE on a standard seeded tableau (byes to top
     seeds), bouts to 10 (Y10/Y8) or 15. Final placement from elimination round, ties
     within a round broken by entering seed.

A bout i vs j draws Z ~ N(g_i - g_j, sigma_type^2): winner reaches the target, loser
gets target - round(|Z|*target). Pool and DE use their own sigma and target, so both
the pool (to-5) and DE (to-10) models are used.
"""

from __future__ import annotations

import numpy as np


def pool_sizes(n: int) -> list[int]:
    """Pool sizes as close to 6-7 as possible (>=5 where feasible), summing to n."""
    if n <= 7:
        return [n]
    p = -(-n // 7)            # ceil(n/7) -> largest pool <= 7
    base, rem = divmod(n, p)  # rem pools of base+1, rest of base
    return [base + 1] * rem + [base] * (p - rem)


def _snake_pools(seed_order: np.ndarray, p: int) -> list[list[int]]:
    """Serpentine-assign seeds (best-first fencer indices) across p pools."""
    pools: list[list[int]] = [[] for _ in range(p)]
    idx, direction = 0, 1
    for f in seed_order:
        pools[idx].append(int(f))
        idx += direction
        if idx == p:
            idx, direction = p - 1, -1
        elif idx < 0:
            idx, direction = 0, 1
    return pools


def bracket_order(n: int) -> list[int]:
    """Standard single-elim seeding: seeds 1..n in tableau-position order (n a power of 2)."""
    r = [1]
    while len(r) < n:
        m = len(r) * 2 + 1
        r = [x for a in r for x in (a, m - a)]
    return r


def _simulate_de(de_order, g, sigma, target, rng) -> dict:
    """Single-elim, 100% promotion. de_order = fencer indices best-seed first.
    Returns {fencer_index: placement (1-based)}."""
    n = len(de_order)
    T = 1
    while T < n:
        T *= 2
    seed_rank = {f: i for i, f in enumerate(de_order)}   # 0 = top seed
    cur = [de_order[s - 1] if s <= n else None for s in bracket_order(T)]
    place: dict[int, int] = {}
    size = T
    while size > 1:
        nxt, losers = [], []
        for k in range(0, size, 2):
            a, b = cur[k], cur[k + 1]
            if a is None:
                nxt.append(b); continue
            if b is None:
                nxt.append(a); continue
            z = rng.normal(g[a] - g[b], sigma)
            win, lose = (a, b) if z > 0 else (b, a)
            nxt.append(win); losers.append(lose)
        for i, f in enumerate(sorted(losers, key=lambda f: seed_rank[f])):
            place[f] = size // 2 + 1 + i      # losers of this round share [size/2+1 .. size]
        cur, size = nxt, size // 2
    place[cur[0]] = 1
    return place


def _run(g, *, sigma_pool, sigma_de, pool_target, de_target, n_sims, seed):
    """Simulate n_sims events; return an (n_sims x n) matrix of every fencer's placement.
    Seeding/pools are deterministic from g; only bout outcomes vary."""
    g = np.asarray(g, float)
    n = len(g)
    rng = np.random.default_rng(seed)
    seed_order = np.argsort(-g)                      # fencer indices, best first

    pools = _snake_pools(seed_order, len(pool_sizes(n)))
    pi, pj = [], []
    for pm in pools:
        for a in range(len(pm)):
            for b in range(a + 1, len(pm)):
                pi.append(pm[a]); pj.append(pm[b])
    pi, pj = np.array(pi), np.array(pj)
    mu = g[pi] - g[pj]

    out = np.empty((n_sims, n), int)
    for s in range(n_sims):
        z = rng.normal(mu, sigma_pool)
        iwin = z > 0
        margin = np.clip(np.rint(np.abs(z) * pool_target).astype(int), 1, pool_target)
        w = np.where(iwin, pi, pj); l = np.where(iwin, pj, pi)
        ws = np.full(len(z), pool_target); ls = pool_target - margin
        V = np.zeros(n); TS = np.zeros(n); TR = np.zeros(n); M = np.zeros(n)
        np.add.at(V, w, 1.0)
        np.add.at(TS, w, ws); np.add.at(TR, w, ls)
        np.add.at(TS, l, ls); np.add.at(TR, l, ws)
        np.add.at(M, pi, 1.0); np.add.at(M, pj, 1.0)
        vr = np.where(M > 0, V / M, 0.0); ind = TS - TR
        de_order = sorted(range(n), key=lambda x: (-vr[x], -ind[x], -TS[x]))
        place = _simulate_de(de_order, g, sigma_de, de_target, rng)
        row = out[s]
        for f, pl in place.items():
            row[f] = pl
    return out


def simulate_placements(g, focal, *, sigma_pool, sigma_de, pool_target=5, de_target=10,
                        n_sims=3000, seed=0):
    """The focal fencer's final-placement array across n_sims event runs."""
    return _run(g, sigma_pool=sigma_pool, sigma_de=sigma_de, pool_target=pool_target,
                de_target=de_target, n_sims=n_sims, seed=seed)[:, focal]


def simulate_all_placements(g, *, sigma_pool, sigma_de, pool_target=5, de_target=10,
                            n_sims=3000, seed=0):
    """(n_sims x n) matrix of every fencer's final placement — for field-wide views."""
    return _run(g, sigma_pool=sigma_pool, sigma_de=sigma_de, pool_target=pool_target,
                de_target=de_target, n_sims=n_sims, seed=seed)
