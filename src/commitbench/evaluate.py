"""Replay each repo's history in time order, score every query, train and test rankers."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import metrics as M
from .features import BASELINES, FEATURES, F, QueryFeatures, compute
from .gitutil import BlobReader, ls_tree_py, read_history
from .history import History
from .index import RepoIndex
from .mine import load_queries
from .textproc import file_stems, mask_query, raw_words

TRAIN_FRACTION = 0.7
VARIANTS = ("as_is", "masked")
LEARNED = ("learned_lr", "learned_gbt")
METHODS = (*BASELINES, *LEARNED)
KS = (1, 5, 10)
FULL_REBUILD_SAMPLES = 15


@dataclass
class Record:
    repo: str
    sha: str
    order: int
    split: str
    text: dict[str, str]
    gold: list[str]
    feats: dict[str, QueryFeatures] = field(default_factory=dict)
    gold_mask: np.ndarray | None = None


def temporal_split(queries: list[dict], frac: float = TRAIN_FRACTION) -> dict[tuple[str, str], str]:
    """Earliest `frac` of each repo's queries (by history order) train; the rest test."""
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for q in queries:
        by_repo[q["repo"]].append(q)
    split: dict[tuple[str, str], str] = {}
    for repo, qs in by_repo.items():
        qs.sort(key=lambda q: q["order"])
        cut = int(len(qs) * frac)
        for i, q in enumerate(qs):
            split[(repo, q["sha"])] = "train" if i < cut else "test"
    return split


def replay_repo(
    repo: str,
    repo_dir: Path,
    cutoff: str,
    queries: list[dict],
    split: dict[tuple[str, str], str],
    full_rebuild_samples: int = FULL_REBUILD_SAMPLES,
) -> tuple[list[Record], dict]:
    """Walk history oldest->newest. For each query commit: move the index to the
    parent snapshot (incrementally), score candidates, THEN absorb the commit into
    the history priors. So priors only ever see strictly earlier commits."""
    commits = read_history(repo_dir, cutoff)
    qmap = {q["sha"]: q for q in queries}
    sample_every = max(1, len(qmap) // max(1, full_rebuild_samples))
    index = RepoIndex()
    hist = History()
    records: list[Record] = []
    timing: dict[str, Any] = {
        "incremental_seconds": 0.0,
        "incremental_updates": 0,
        "files_reindexed": 0,
        "first_build_seconds": 0.0,
        "full_rebuild_seconds": [],
        "full_rebuild_files": [],
        "verified_equal": 0,
        "verified_total": 0,
        "dropped_gold_missing": 0,
        "merkle_nodes_visited": 0,
        "merkle_nodes_total": 0,
    }
    n_seen = 0
    with BlobReader(repo_dir) as reader:
        for c in commits:
            q = qmap.get(c.sha)
            if q is not None:
                snap = ls_tree_py(repo_dir, q["parent"])
                first = index.tree is None
                t0 = time.perf_counter()
                d = index.update(snap, reader)
                dt = time.perf_counter() - t0
                if first:
                    timing["first_build_seconds"] = dt
                else:
                    timing["incremental_seconds"] += dt
                    timing["incremental_updates"] += 1
                    timing["files_reindexed"] += len(d.added) + len(d.modified)
                    timing["merkle_nodes_visited"] += d.nodes_visited
                    timing["merkle_nodes_total"] += len(snap) + _n_dirs(snap)
                if n_seen % sample_every == 0 and not first:
                    t0 = time.perf_counter()
                    full = RepoIndex.build_full(snap, reader)
                    timing["full_rebuild_seconds"].append(time.perf_counter() - t0)
                    timing["full_rebuild_files"].append(len(snap))
                    timing["verified_total"] += 1
                    timing["verified_equal"] += int(full.state() == index.state())
                n_seen += 1
                gold = [g for g in q["gold"] if g in index.files]
                if len(gold) != len(q["gold"]):
                    timing["dropped_gold_missing"] += 1
                if gold:
                    sp = split[(repo, c.sha)]
                    text = {"as_is": q["query"], "masked": mask_query(q["query"], gold, {repo})}
                    rec = Record(repo, c.sha, q["order"], sp, text, gold)
                    for v in VARIANTS:
                        rec.feats[v] = compute(index, hist, text[v], c.ts, with_chunks=(sp == "test"))
                    paths = rec.feats["as_is"].paths
                    gs = set(gold)
                    rec.gold_mask = np.array([p in gs for p in paths])
                    records.append(rec)
            hist.update(c)
    fr = timing.pop("full_rebuild_seconds")
    ff = timing.pop("full_rebuild_files")
    timing["full_rebuild_mean_seconds"] = float(np.mean(fr)) if fr else None
    timing["full_rebuild_mean_files"] = float(np.mean(ff)) if ff else None
    n = timing["incremental_updates"]
    timing["incremental_mean_seconds"] = timing["incremental_seconds"] / n if n else None
    timing["incremental_mean_files_reindexed"] = timing["files_reindexed"] / n if n else None
    if timing["full_rebuild_mean_seconds"] and timing["incremental_mean_seconds"]:
        timing["speedup"] = timing["full_rebuild_mean_seconds"] / timing["incremental_mean_seconds"]
    return records, timing


def _n_dirs(snap: dict[str, str]) -> int:
    dirs = {""}
    for p in snap:
        parts = p.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            dirs.add("/".join(parts[:i]))
    return len(dirs)


def _stack(recs: list[Record], variant: str) -> tuple[np.ndarray, np.ndarray]:
    X = np.vstack([r.feats[variant].X for r in recs])
    y = np.concatenate([r.gold_mask for r in recs])
    return X, y


def fit_models(train: list[Record], variant: str) -> dict:
    X, y = _stack(train, variant)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    gbt = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0
    )
    lr.fit(X, y)
    gbt.fit(X, y)
    return {"learned_lr": lr, "learned_gbt": gbt}


def score_method(method: str, X: np.ndarray, models: dict) -> np.ndarray:
    if method in BASELINES:
        return BASELINES[method](X)
    return models[method].predict_proba(X)[:, 1]


def evaluate_record(rec: Record, variant: str, models: dict) -> dict:
    qf = rec.feats[variant]
    assert rec.gold_mask is not None and qf.chunk_top is not None
    n_gold = int(rec.gold_mask.sum())
    out: dict[str, dict[str, Any]] = {}
    tops: dict[str, list[str]] = {}
    for m in METHODS:
        order = M.rank_order(score_method(m, qf.X, models))
        ranks = M.gold_ranks(order, rec.gold_mask)
        res: dict[str, Any] = {f"recall@{k}": M.recall_at(ranks, n_gold, k) for k in KS}
        res["mrr"] = M.reciprocal_rank(ranks)
        res["ndcg@10"] = M.ndcg_at(ranks, n_gold, 10)
        for b in M.BUDGETS:
            res[f"budget_recall@{b // 1000}k"] = M.budget_recall(order, rec.gold_mask, qf.chunk_top, b)
        res["gold_ranks"] = ranks
        out[m] = res
        if m in ("learned_gbt", "bm25_file", "hybrid_hand_tuned"):
            tops[m] = [qf.paths[i] for i in order[:5]]
    gold_idx = np.flatnonzero(rec.gold_mask)
    return {
        "repo": rec.repo,
        "sha": rec.sha,
        "order": rec.order,
        "variant": variant,
        "query": rec.text[variant],
        "gold": rec.gold,
        "n_candidates": len(qf.paths),
        "gold_churn": [int(round(math.expm1(qf.X[i, F["churn_log"]]))) for i in gold_idx],
        "gold_is_test": [bool(qf.X[i, F["is_test"]]) for i in gold_idx],
        "gold_stem_mentioned": bool(any(file_stems(g) & raw_words(rec.text["as_is"]) for g in rec.gold)),
        "query_words": len(rec.text[variant].split()),
        "methods": out,
        "top5": tops,
    }


METRIC_KEYS = (
    "recall@1",
    "recall@5",
    "recall@10",
    "mrr",
    "ndcg@10",
    *(f"budget_recall@{b // 1000}k" for b in M.BUDGETS),
)


def aggregate(rows: list[dict]) -> dict:
    agg: dict[str, dict[str, Any]] = {}
    for m in METHODS:
        agg[m] = {}
        for k in METRIC_KEYS:
            vals = [r["methods"][m][k] for r in rows]
            lo, hi = M.bootstrap_ci(vals)
            agg[m][k] = {"mean": float(np.mean(vals)), "ci95": [lo, hi]}
    return agg


def run(data_dir: Path, results_dir: Path) -> dict:
    t_start = time.perf_counter()
    meta = json.loads((data_dir / "mine_stats.json").read_text())
    queries = load_queries(data_dir)
    split = temporal_split(queries)
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for q in queries:
        by_repo[q["repo"]].append(q)

    records: list[Record] = []
    timing: dict[str, dict] = {}
    for repo in meta["repos"]:
        t0 = time.perf_counter()
        recs, tm = replay_repo(repo, data_dir / "repos" / repo, meta["cutoff"], by_repo[repo], split)
        tm["replay_seconds"] = time.perf_counter() - t0
        records.extend(recs)
        timing[repo] = tm
        print(f"[eval] {repo}: {len(recs)} queries replayed in {tm['replay_seconds']:.1f}s")

    train = [r for r in records if r.split == "train"]
    test = [r for r in records if r.split == "test"]
    _check_no_temporal_leak(train, test)

    results_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "cutoff": meta["cutoff"],
        "n_train": len(train),
        "n_test": len(test),
        "features": FEATURES,
        "timing": timing,
        "mine": meta["stats"],
        "variants": {},
    }
    with open(results_dir / "per_query.jsonl", "w") as f:
        for v in VARIANTS:
            models = fit_models(train, v)
            rows = [evaluate_record(r, v, models) for r in test]
            for row in rows:
                f.write(json.dumps(row) + "\n")
            vs: dict[str, Any] = {"pooled": aggregate(rows)}
            for repo in meta["repos"]:
                rr = [r for r in rows if r["repo"] == repo]
                if rr:
                    vs[repo] = aggregate(rr)
            vs["n"] = {"pooled": len(rows), **{repo: sum(r["repo"] == repo for r in rows) for repo in meta["repos"]}}
            vs["deltas"] = {}
            for base in ("bm25_file", "hybrid_hand_tuned"):
                for key in ("recall@10", "mrr", "budget_recall@8k"):
                    a = [r["methods"]["learned_gbt"][key] for r in rows]
                    b = [r["methods"][base][key] for r in rows]
                    mean, lo, hi = M.paired_delta_ci(a, b)
                    vs["deltas"][f"learned_gbt_minus_{base}:{key}"] = {"mean": mean, "ci95": [lo, hi]}
            Xt, yt = _stack(test, v)
            pi = permutation_importance(
                models["learned_gbt"], Xt, yt, scoring="average_precision", n_repeats=3, random_state=0
            )
            vs["permutation_importance"] = dict(
                sorted(zip(FEATURES, map(float, pi.importances_mean), strict=True), key=lambda t: -t[1])
            )
            summary["variants"][v] = vs
    summary["leakage_rate_gold_stem_in_message"] = float(
        np.mean([any(file_stems(g) & raw_words(r.text["as_is"]) for g in r.gold) for r in test])
    )
    summary["eval_seconds"] = time.perf_counter() - t_start
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _check_no_temporal_leak(train: list[Record], test: list[Record]) -> None:
    for repo in {r.repo for r in train} | {r.repo for r in test}:
        tr = [r.order for r in train if r.repo == repo]
        te = [r.order for r in test if r.repo == repo]
        if tr and te:
            assert max(tr) < min(te), f"temporal leak in {repo}"
