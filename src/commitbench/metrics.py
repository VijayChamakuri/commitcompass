"""Ranking metrics, token-budget recall and bootstrap confidence intervals."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

BUDGETS = (2000, 4000, 8000, 16000)


def rank_order(scores: np.ndarray) -> np.ndarray:
    """Indices sorted best-first; ties keep candidate (alphabetical path) order."""
    return np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")


def gold_ranks(order: np.ndarray, gold_mask: np.ndarray) -> list[int]:
    """1-based ranks of the gold candidates."""
    pos = np.empty(len(order), dtype=np.int64)
    pos[order] = np.arange(1, len(order) + 1)
    return sorted(int(r) for r in pos[gold_mask])


def recall_at(ranks: Sequence[int], n_gold: int, k: int) -> float:
    return sum(1 for r in ranks if r <= k) / n_gold


def reciprocal_rank(ranks: Sequence[int]) -> float:
    return 1.0 / min(ranks) if ranks else 0.0


def ndcg_at(ranks: Sequence[int], n_gold: int, k: int) -> float:
    dcg = sum(1.0 / math.log2(r + 1) for r in ranks if r <= k)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(n_gold, k)))
    return dcg / ideal if ideal else 0.0


def budget_recall(
    order: np.ndarray,
    gold_mask: np.ndarray,
    chunk_top: list[list[tuple[float, int]]],
    budget: int,
    max_files: int = 50,
) -> float:
    """Fraction of gold files with >=1 chunk packed into a context of `budget` tokens.

    Files are visited in ranked order; each contributes its best chunks (by chunk
    BM25) while they fit. Mirrors what an agent can actually put in its prompt.
    """
    used = 0
    covered = 0
    for idx in order[:max_files]:
        hit = False
        for _, tok in chunk_top[idx]:
            if used + tok <= budget:
                used += tok
                hit = True
        if hit and gold_mask[idx]:
            covered += 1
        if used >= budget:
            break
    return covered / max(1, int(gold_mask.sum()))


def bootstrap_ci(values: Sequence[float] | np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_delta_ci(a: Sequence[float], b: Sequence[float], n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) over queries with a paired bootstrap 95% CI."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    lo, hi = bootstrap_ci(d, n, seed)
    return float(d.mean()), lo, hi
