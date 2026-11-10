"""Score every method on SWE-bench Lite and compare with published numbers.

Training data for the learned rerankers is *only* commit history (message -> modified files)
from commits made before each repository's first SWE-bench issue; SWE-bench labels are used
exclusively for evaluation. Two training regimes:

  pooled  one model trained on the commit-history queries of all 12 repositories
  loro    leave-one-repo-out: the model scoring repo R never saw R's commits
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import metrics as M
from .features import BASELINES, FEATURES, F
from .swe_replay import EventRecord
from .textproc import file_stems, raw_words

KS = (1, 3, 5, 10)

# File-level localization on SWE-bench Lite, reported in LocAgent (Chen et al., 2025,
# arXiv:2503.09089, Table 4). Acc@k = all gold files in the top k. Not re-run here.
PUBLISHED = [
    {"method": "BM25", "kind": "sparse retrieval", "acc@1": 38.69, "acc@3": 51.82, "acc@5": 61.68},
    {"method": "E5-base-v2", "kind": "dense retrieval", "acc@1": 49.64, "acc@3": 74.45, "acc@5": 80.29},
    {"method": "Jina-Code-v2", "kind": "dense retrieval", "acc@1": 43.43, "acc@3": 71.17, "acc@5": 80.29},
    {"method": "Codesage-large-v2", "kind": "dense retrieval", "acc@1": 47.81, "acc@3": 69.34, "acc@5": 78.10},
    {"method": "CodeRankEmbed", "kind": "dense retrieval", "acc@1": 52.55, "acc@3": 77.74, "acc@5": 84.67},
    {"method": "Agentless (GPT-4o)", "kind": "LLM pipeline", "acc@1": 67.15, "acc@3": 74.45, "acc@5": 74.45},
    {"method": "Agentless (Claude-3.5)", "kind": "LLM pipeline", "acc@1": 72.63, "acc@3": 79.20, "acc@5": 79.56},
    {"method": "SWE-agent (Claude-3.5)", "kind": "LLM agent", "acc@1": 77.37, "acc@3": 87.23, "acc@5": 90.15},
    {"method": "OpenHands (Claude-3.5)", "kind": "LLM agent", "acc@1": 76.28, "acc@3": 89.78, "acc@5": 90.15},
    {"method": "MoatlessTools (Claude-3.5)", "kind": "LLM agent", "acc@1": 72.63, "acc@3": 85.77, "acc@5": 86.13},
    {"method": "LocAgent (Claude-3.5)", "kind": "LLM agent", "acc@1": 77.74, "acc@3": 91.97, "acc@5": 94.16},
]
PUBLISHED_SOURCE = "LocAgent: Graph-Guided LLM Agents for Code Localization (arXiv:2503.09089), Table 4"


def _stack(recs: list[EventRecord]) -> tuple[np.ndarray, np.ndarray]:
    return np.vstack([r.feats.X for r in recs]), np.concatenate([r.gold_mask for r in recs])


def fit_gbt(recs: list[EventRecord], seed: int = 0) -> HistGradientBoostingClassifier:
    X, y = _stack(recs)
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed
    ).fit(X, y)


def rrf(*score_arrays: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion."""
    out = np.zeros(len(score_arrays[0]))
    for s in score_arrays:
        order = M.rank_order(s)
        ranks = np.empty(len(s))
        ranks[order] = np.arange(1, len(s) + 1)
        out += 1.0 / (k + ranks)
    return out


def method_scores(rec: EventRecord, models: dict) -> dict[str, np.ndarray]:
    X = rec.feats.X
    s = {name: fn(X) for name, fn in BASELINES.items()}
    s["commit_reranker"] = models["pooled"].predict_proba(X)[:, 1]
    s["commit_reranker_loro"] = models["loro"][rec.repo].predict_proba(X)[:, 1]
    if "dense_config" in rec.extra:
        d = rec.extra["dense"].astype(np.float64)
        s["dense_jina_code"] = d
        s["rrf_bm25_dense"] = rrf(s["bm25_file"], d)
        s["rrf_reranker_dense"] = rrf(s["commit_reranker"], d)
    return s


def check_no_label_leak(train: list[EventRecord], test: list[EventRecord], issue_ts: dict[str, float]) -> None:
    """Every training commit must predate the creation of every issue in its repository."""
    for repo in {r.repo for r in test}:
        tr = [r.ts for r in train if r.repo == repo]
        first_issue = min(issue_ts[r.key] for r in test if r.repo == repo)
        if tr and max(tr) >= first_issue:
            raise AssertionError(f"training commit after first issue in {repo}")


def evaluate(records: list[EventRecord], issue_ts: dict[str, float], log=print) -> dict:
    t0 = time.time()
    train = [r for r in records if r.kind == "train"]
    test = sorted([r for r in records if r.kind == "test"], key=lambda r: r.key)
    check_no_label_leak(train, test, issue_ts)
    repos = sorted({r.repo for r in test})
    models = {"pooled": fit_gbt(train), "loro": {}}
    for repo in repos:
        models["loro"][repo] = fit_gbt([r for r in train if r.repo != repo])
    log(f"[eval] trained rerankers on {len(train)} commit queries in {time.time() - t0:.0f}s")

    rows = []
    for rec in test:
        scores = method_scores(rec, models)
        words = raw_words(rec.query)
        gi = int(np.flatnonzero(rec.gold_mask)[0])
        row: dict[str, Any] = {
            "instance_id": rec.key,
            "repo": rec.repo,
            "gold": rec.gold,
            "n_candidates": len(rec.feats.paths),
            "gold_named_in_issue": bool(file_stems(rec.gold[0]) & words),
            "gold_symbol_in_issue": bool(rec.feats.X[gi, F["symbol_exact"]] > 0),
            "gold_is_test": bool(rec.feats.X[gi, F["is_test"]] > 0),
            "methods": {},
        }
        for m, sc in scores.items():
            order = M.rank_order(sc)
            ranks = M.gold_ranks(order, rec.gold_mask)
            res: dict[str, Any] = {f"acc@{k}": 1.0 if ranks and max(ranks) <= k else 0.0 for k in KS}
            res["mrr"] = M.reciprocal_rank(ranks)
            res["rank"] = int(ranks[0]) if ranks else None
            res["budget_recall@8k"] = M.budget_recall(order, rec.gold_mask, rec.feats.chunk_top or [], 8000)
            res["top10"] = [rec.feats.paths[i] for i in order[:10]]
            row["methods"][m] = res
        rows.append(row)
    methods = list(rows[0]["methods"])
    agg: dict[str, dict[str, Any]] = {}
    for m in methods:
        rs = [r for r in rows if m in r["methods"]]
        agg[m] = {"n": len(rs)}
        for key in [f"acc@{k}" for k in KS] + ["mrr", "budget_recall@8k"]:
            v = [r["methods"][m][key] for r in rs]
            lo, hi = M.bootstrap_ci(v, n=2000)
            agg[m][key] = {"mean": float(np.mean(v)), "ci95": [lo, hi]}
    deltas: dict[str, Any] = {}
    for a, b in [
        ("commit_reranker", "bm25_file"),
        ("commit_reranker", "hybrid_hand_tuned"),
        ("commit_reranker_loro", "bm25_file"),
        ("rrf_reranker_dense", "dense_jina_code"),
        ("rrf_reranker_dense", "commit_reranker"),
        ("dense_jina_code", "bm25_file"),
    ]:
        if a in agg and b in agg:
            for key in ("acc@1", "acc@5", "mrr"):
                va = [r["methods"][a][key] for r in rows]
                vb = [r["methods"][b][key] for r in rows]
                mean, lo, hi = M.paired_delta_ci(va, vb, n=2000)
                deltas[f"{a} - {b} : {key}"] = {"mean": mean, "ci95": [lo, hi]}
    per_repo = {
        repo: {m: float(np.mean([r["methods"][m]["acc@5"] for r in rows if r["repo"] == repo])) for m in methods}
        | {"n": sum(r["repo"] == repo for r in rows)}
        for repo in repos
    }
    from sklearn.inspection import permutation_importance

    Xt, yt = _stack(test)
    pi = permutation_importance(models["pooled"], Xt, yt, scoring="average_precision", n_repeats=3, random_state=0)
    importance = dict(sorted(zip(FEATURES, map(float, pi.importances_mean), strict=True), key=lambda t: -t[1]))
    return {
        "n_test": len(test),
        "n_train_commit_queries": len(train),
        "train_queries_per_repo": {
            repo: sum(r.repo == repo for r in train) for repo in sorted({r.repo for r in train})
        },
        "aggregate": agg,
        "paired_deltas": deltas,
        "per_repo_acc@5": per_repo,
        "feature_importance": importance,
        "pool_recall": float(np.mean([r.extra["pool_has_gold"] for r in test if "pool_has_gold" in r.extra]))
        if any("pool_has_gold" in r.extra for r in test)
        else None,
        "published": PUBLISHED,
        "published_source": PUBLISHED_SOURCE,
        "rows": rows,
        "seconds": time.time() - t0,
    }


def categorize_misses(rows: list[dict], method: str, k: int = 5) -> dict:
    """Why did `method` miss the gold file at k? Mutually exclusive, checked in order."""
    cats: dict[str, list[str]] = {}
    for r in rows:
        res = r["methods"][method]
        if res[f"acc@{k}"]:
            continue
        bm = r["methods"]["bm25_file"]["rank"] or 10**6
        if r["gold_named_in_issue"]:
            c = "gold file named in issue, still missed"
        elif r["gold_symbol_in_issue"]:
            c = "issue names a symbol defined in the gold file"
        elif bm > 50:
            c = "vocabulary gap (gold not in BM25 top 50)"
        elif (res["rank"] or 10**6) <= 10:
            c = "near miss (ranked 6-10)"
        else:
            c = "lexically reachable but out-ranked"
        cats.setdefault(c, []).append(r["instance_id"])
    return {c: {"count": len(v), "examples": v[:4]} for c, v in sorted(cats.items(), key=lambda t: -len(t[1]))}


def write(results: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = results.pop("rows")
    best = max((m for m in results["aggregate"]), key=lambda m: results["aggregate"][m]["acc@5"]["mean"])
    results["best_method"] = best
    results["miss_categories"] = categorize_misses(rows, best)
    (out / "swebench_lite.json").write_text(json.dumps(results, indent=2))
    with (out / "swebench_lite_per_instance.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
