"""Driver: replay all 12 SWE-bench Lite repos (sparse features), then add dense scores."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import metrics as M
from .dense import POOL, DenseScorer
from .features import BASELINES
from .gitutil import BlobReader, ls_tree_py
from .swe_replay import EventRecord, replay
from .swebench import REPOS, fetch, load

DATA = Path("data")
REPO_DIR = DATA / "swe_repos"
CACHE = DATA / "swe_cache"


def repo_path(repo: str) -> Path:
    return REPO_DIR / repo.replace("/", "_")


def replay_all(log=print) -> dict[str, tuple[list[EventRecord], dict]]:
    meta = fetch(DATA / "swebench_lite.jsonl")
    instances = load(DATA / "swebench_lite.jsonl")
    by_repo = defaultdict(list)
    for i in instances:
        by_repo[i.repo].append(i)
    out = {}
    for repo in sorted(REPOS, key=lambda r: len(by_repo[r])):
        t0 = time.time()
        out[repo] = replay(repo, repo_path(repo), by_repo[repo], CACHE / f"{repo.replace('/', '_')}.pkl", log=log)
        log(f"[replay] {repo}: {out[repo][1]['records']} records in {time.time() - t0:.0f}s")
    (CACHE / "dataset.json").write_text(json.dumps(meta, indent=2))
    return out


def candidate_pool(rec: EventRecord, size: int = POOL) -> np.ndarray:
    """Union of the top-`size` files by BM25 (file) and by the hand-tuned hybrid (non-learned)."""
    X = rec.feats.X
    a = M.rank_order(BASELINES["bm25_file"](X))[:size]
    b = M.rank_order(BASELINES["hybrid_hand_tuned"](X))[:size]
    return np.unique(np.concatenate([a, b]))


def add_dense(replayed: dict, scorer: DenseScorer, log=print) -> None:
    """Dense scores for every test record over its candidate pool (cached in the record pickle).
    Blobs were fetched during replay, so no checkout is needed here."""
    import pickle

    inst = {i.instance_id: i for i in load(DATA / "swebench_lite.jsonl")}
    for repo, (records, timing) in replayed.items():
        tests = [r for r in records if r.kind == "test" and r.extra.get("dense_config") != scorer.config]
        if not tests:
            continue
        scorer.load_repo(repo)
        path = repo_path(repo)
        t0 = time.time()
        with BlobReader(path) as reader:
            for n, r in enumerate(sorted(tests, key=lambda r: r.ts)):
                snap = ls_tree_py(path, inst[r.key].base_commit)
                pool = candidate_pool(r)
                r.extra["dense"] = scorer.file_scores(snap, reader, r.feats.paths, pool, r.query)
                r.extra["pool_has_gold"] = bool(r.gold_mask[pool].any())
                r.extra["dense_config"] = scorer.config
                if (n + 1) % 10 == 0:
                    scorer.save_repo(repo)
                    log(
                        f"[dense] {repo} {n + 1}/{len(tests)} ({time.time() - t0:.0f}s, {scorer.embedded} files embedded)"
                    )
        scorer.save_repo(repo)
        timing["dense_seconds"] = time.time() - t0
        timing["dense_files_embedded"] = scorer.embedded
        with (CACHE / f"{repo.replace('/', '_')}.pkl").open("wb") as f:
            pickle.dump((records, timing), f)
        log(f"[dense] {repo} done in {time.time() - t0:.0f}s")
