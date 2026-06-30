"""Shared statistical helpers for proxy-vs-ASD analysis."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import permutation_test, spearmanr


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    q = np.full_like(p, np.nan, dtype=np.float64)
    finite = np.isfinite(p)
    if not finite.any():
        return q

    finite_indices = np.flatnonzero(finite)
    finite_p = p[finite]
    order = np.argsort(finite_p)
    ranked = finite_p[order]
    m = len(ranked)
    ranked_q = np.empty(m, dtype=np.float64)

    previous = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        previous = min(previous, ranked[i] * m / rank)
        ranked_q[i] = previous

    out = np.empty(m, dtype=np.float64)
    out[order] = np.minimum(ranked_q, 1.0)
    q[finite_indices] = out
    return q


def exact_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3:
        return math.nan, math.nan

    rho = float(spearmanr(x, y).correlation)

    def statistic_func(values: np.ndarray) -> float:
        return float(spearmanr(values, y).correlation)

    result = permutation_test(
        (x,),
        statistic_func,
        permutation_type="pairings",
        alternative="two-sided",
        n_resamples=np.inf,
    )
    return rho, float(result.pvalue)
