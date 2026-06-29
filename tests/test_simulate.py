"""Tests for the competition simulator."""

from __future__ import annotations

import numpy as np

from analytics import simulate as S


def test_pool_sizes_close_to_6_7():
    # n>=10 is the range where splitting into pools of 5-7 is feasible
    for n in range(10, 130):
        sizes = S.pool_sizes(n)
        assert sum(sizes) == n
        assert min(sizes) >= 5 and max(sizes) <= 7, (n, sizes)
        assert max(sizes) - min(sizes) <= 1   # balanced


def test_pool_sizes_known():
    assert sum(S.pool_sizes(107)) == 107 and set(S.pool_sizes(107)) <= {6, 7}
    assert S.pool_sizes(6) == [6]


def test_bracket_order_valid():
    for T in (2, 4, 8, 16, 32):
        order = S.bracket_order(T)
        assert sorted(order) == list(range(1, T + 1))           # a permutation
        assert all(order[k] + order[k + 1] == T + 1 for k in range(0, T, 2))  # seed pairs


def test_strong_fencer_usually_wins():
    # one clearly dominant fencer among 32
    g = np.zeros(32); g[7] = 5.0
    places = S.simulate_placements(g, focal=7, sigma_pool=0.3, sigma_de=0.2, n_sims=300)
    assert (places == 1).mean() > 0.9


def test_weakest_never_wins_and_places_valid():
    g = np.linspace(2, -2, 24)   # fencer 0 best, 23 worst
    places = S.simulate_placements(g, focal=23, sigma_pool=0.3, sigma_de=0.2, n_sims=200)
    assert places.min() >= 1 and places.max() <= 24
    assert (places == 1).mean() < 0.05   # worst fencer essentially never wins


if __name__ == "__main__":
    test_pool_sizes_close_to_6_7(); print("OK: pool sizes 5-7")
    test_pool_sizes_known(); print("OK: pool sizes known")
    test_bracket_order_valid(); print("OK: bracket order")
    test_strong_fencer_usually_wins(); print("OK: strong fencer wins")
    test_weakest_never_wins_and_places_valid(); print("OK: weak fencer / valid places")
    print("\nAll simulate tests passed.")
