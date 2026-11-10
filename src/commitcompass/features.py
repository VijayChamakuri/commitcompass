"""Per-(query, candidate file) features and the non-learned retrievers built from them."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .history import History
from .index import RepoIndex
from .textproc import file_stems, raw_words, tokenize

FEATURES = [
    "bm25_file",
    "bm25_file_norm",
    "bm25_file_rrank",
    "bm25_chunk",
    "bm25_chunk_norm",
    "path_overlap",
    "path_frac",
    "stem_mentioned",
    "symbol_exact",
    "symbol_part",
    "churn_log",
    "recency_log_days",
    "never_touched",
    "cochange",
    "is_test",
    "query_mentions_test",
    "test_match",
    "log_lines",
]
F = {name: i for i, name in enumerate(FEATURES)}
CHUNKS_PER_FILE = 3


@dataclass
class QueryFeatures:
    paths: list[str]
    X: np.ndarray  # (n_candidates, n_features) float32
    # For each candidate: its top chunks as (score, approx_tokens), best first.
    chunk_top: list[list[tuple[float, int]]] | None = None


def compute(index: RepoIndex, history: History, text: str, ts: int, with_chunks: bool) -> QueryFeatures:
    qtok = tokenize(text)
    qset = set(qtok)
    words = raw_words(text)
    fs = index.file_bm25.score(qtok)
    cs = index.chunk_bm25.score(qtok)
    chunk_max: dict[str, float] = {}
    for cid, s in cs.items():
        p = cid.rsplit("#", 1)[0]
        if s > chunk_max.get(p, 0.0):
            chunk_max[p] = s

    paths = sorted(index.files)
    fmax = max(fs.values(), default=0.0) or 1.0
    cmax = max(chunk_max.values(), default=0.0) or 1.0
    ranked = sorted((p for p in paths if fs.get(p, 0) > 0), key=lambda p: -fs[p])
    rrank = {p: 1.0 / (1 + r) for r, p in enumerate(ranked)}
    named = [p for p in paths if file_stems(p) & words]
    q_test = 1.0 if ({"test", "tests", "testing"} & words) else 0.0

    X = np.zeros((len(paths), len(FEATURES)), dtype=np.float32)
    chunk_top: list[list[tuple[float, int]]] | None = [] if with_chunks else None
    for i, p in enumerate(paths):
        e = index.files[p]
        row = X[i]
        row[F["bm25_file"]] = fs.get(p, 0.0)
        row[F["bm25_file_norm"]] = fs.get(p, 0.0) / fmax
        row[F["bm25_file_rrank"]] = rrank.get(p, 0.0)
        row[F["bm25_chunk"]] = chunk_max.get(p, 0.0)
        row[F["bm25_chunk_norm"]] = chunk_max.get(p, 0.0) / cmax
        ov = len(e.path_tokens & qset)
        row[F["path_overlap"]] = ov
        row[F["path_frac"]] = ov / max(1, len(e.path_tokens))
        row[F["stem_mentioned"]] = 1.0 if file_stems(p) & words else 0.0
        row[F["symbol_exact"]] = len(e.symbols & words)
        row[F["symbol_part"]] = len(e.symbol_tokens & qset) / math.sqrt(1 + len(e.symbol_tokens))
        churn = history.churn.get(p, 0)
        row[F["churn_log"]] = math.log1p(churn)
        if p in history.last_ts:
            row[F["recency_log_days"]] = math.log1p(max(0, ts - history.last_ts[p]) / 86400)
        else:
            row[F["recency_log_days"]] = 10.0
            row[F["never_touched"]] = 1.0
        cc = history.cochange.get(p)
        if cc and named:
            row[F["cochange"]] = sum(cc.get(n, 0) / (1 + history.churn.get(n, 0)) for n in named if n != p)
        row[F["is_test"]] = 1.0 if e.is_test else 0.0
        row[F["query_mentions_test"]] = q_test
        row[F["test_match"]] = q_test * row[F["is_test"]]
        row[F["log_lines"]] = math.log1p(e.n_lines)
        if chunk_top is not None:
            pairs = zip(e.chunk_ids, e.chunk_tokens, strict=True)
            scored = [(cs.get(cid, 0.0), j, tok) for j, (cid, tok) in enumerate(pairs)]
            scored.sort(key=lambda t: (-t[0], t[1]))
            chunk_top.append([(s, tok) for s, _, tok in scored[:CHUNKS_PER_FILE]])
    return QueryFeatures(paths=paths, X=X, chunk_top=chunk_top)


# ---- non-learned retrievers: each maps a feature matrix to scores ---------------


def _tiebreak(X: np.ndarray) -> np.ndarray:
    return 1e-6 * X[:, F["bm25_file_norm"]]


def score_bm25_file(X: np.ndarray) -> np.ndarray:
    return X[:, F["bm25_file"]]


def score_bm25_chunk(X: np.ndarray) -> np.ndarray:
    return X[:, F["bm25_chunk"]]


def score_path(X: np.ndarray) -> np.ndarray:
    return 2.0 * X[:, F["stem_mentioned"]] + X[:, F["path_frac"]] + _tiebreak(X)


def score_symbol(X: np.ndarray) -> np.ndarray:
    return X[:, F["symbol_exact"]] + 0.5 * X[:, F["symbol_part"]] + _tiebreak(X)


def score_history(X: np.ndarray) -> np.ndarray:
    """Pure prior: recently and frequently edited files, plus co-change with named files."""
    return 0.5 * X[:, F["churn_log"]] - 0.3 * X[:, F["recency_log_days"]] + X[:, F["cochange"]] + _tiebreak(X)


HYBRID_WEIGHTS = {
    "bm25_file_norm": 1.0,
    "bm25_chunk_norm": 0.6,
    "stem_mentioned": 1.0,
    "path_frac": 0.5,
    "symbol_exact": 0.3,
    "symbol_part": 0.3,
    "churn_log": 0.08,
    "recency_log_days": -0.05,
    "cochange": 0.4,
    "test_match": 0.3,
}


def score_hybrid(X: np.ndarray) -> np.ndarray:
    s = np.zeros(len(X), dtype=np.float64)
    for name, w in HYBRID_WEIGHTS.items():
        s += w * X[:, F[name]]
    return s


BASELINES = {
    "bm25_file": score_bm25_file,
    "bm25_chunk": score_bm25_chunk,
    "path_match": score_path,
    "symbol_match": score_symbol,
    "history_prior": score_history,
    "hybrid_hand_tuned": score_hybrid,
}
