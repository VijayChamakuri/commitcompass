"""Replay each SWE-bench Lite repository through the incremental index.

For every repository, two kinds of *events* are processed in time order on one index:

  train  commit-history queries (commit message -> modified .py files) from commits made
         strictly before that repository's first SWE-bench issue. These are the only labels
         the learned reranker ever sees, so no SWE-bench label is used for training.
  test   the SWE-bench Lite issues, scored against the codebase at `base_commit`.

Snapshots are materialised with a sparse (*.py) checkout of a blob-less partial clone, so
only Python sources are downloaded. History priors (churn, recency, co-change) are absorbed
only from commits older than the event being scored.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import merkle
from .features import QueryFeatures, compute
from .gitutil import BlobReader, Commit, git, ls_tree_py
from .history import History
from .index import RepoIndex
from .mine import classify, clean_message
from .swebench import Instance

TRAIN_PER_REPO = 250


@dataclass
class Event:
    kind: str  # "train" | "test"
    repo: str
    key: str  # instance_id or commit sha
    snapshot_commit: str
    ts: int
    query: str
    gold: list[str]


@dataclass
class EventRecord:
    kind: str
    repo: str
    key: str
    ts: int
    query: str
    gold: list[str]
    feats: QueryFeatures
    gold_mask: np.ndarray
    extra: dict = field(default_factory=dict)


def read_history_fast(repo: Path, rev: str) -> list[Commit]:
    """Non-merge history reachable from `rev`, oldest first, without rename detection
    (rename detection needs blob contents, which a blob-less clone does not have)."""
    # multi-character sentinels: some commit messages contain ASCII control separators
    rs, fs, end = "@@CB-REC@@", "@@CB-FS@@", "@@CB-END@@"
    out = git(
        repo,
        "log",
        "--no-merges",
        "--date-order",
        "--reverse",
        "--no-renames",
        "--name-status",
        f"--format={rs}%H{fs}%P{fs}%ct{fs}%an{fs}%ae{fs}%B{end}",
        rev,
    )
    commits = []
    for rec in out.split(rs):
        if not rec.strip():
            continue
        header, _, rest = rec.partition(end)
        fields = header.split(fs, 5)
        if len(fields) != 6:  # malformed record; skip rather than guess
            continue
        sha, parents, ts, an, ae, body = fields
        changes: list[tuple[str, str, str | None]] = []
        for line in rest.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[0]:
                changes.append((parts[0][0], parts[-1], None))
        commits.append(
            Commit(sha, parents.split()[0] if parents.strip() else None, int(ts), an, ae, body.strip(), changes)
        )
    return commits


def commit_ts(repo: Path, sha: str) -> int:
    return int(git(repo, "show", "-s", "--format=%ct", sha).strip())


def materialise(repo: Path, commit: str) -> dict[str, str]:
    """Sparse-checkout the .py files of `commit` (batch-fetching missing blobs), return path->blob."""
    git(repo, "-c", "advice.detachedHead=false", "checkout", "-q", "--force", commit)
    return ls_tree_py(repo, commit)


def ensure_sparse(repo: Path) -> None:
    git(repo, "config", "core.sparseCheckout", "true")
    git(repo, "sparse-checkout", "set", "--no-cone", "*.py")


def build_events(
    repo_name: str, repo: Path, instances: list[Instance], n_train: int = TRAIN_PER_REPO
) -> tuple[list[Event], list[Commit]]:
    latest = max(instances, key=lambda i: i.created_at).base_commit
    history = read_history_fast(repo, latest)
    first_issue = min(datetime.fromisoformat(i.created_at.replace("Z", "+00:00")).timestamp() for i in instances)
    train_commits = [c for c in history if c.ts < first_issue and c.parent is not None and classify(c).keep]
    events = [
        Event("train", repo_name, c.sha, c.parent or "", c.ts, clean_message(c.message), classify(c).gold)
        for c in train_commits[-n_train:]
    ]
    for inst in instances:
        events.append(
            Event(
                "test",
                repo_name,
                inst.instance_id,
                inst.base_commit,
                commit_ts(repo, inst.base_commit),
                inst.problem_statement,
                inst.gold,
            )
        )
    events.sort(key=lambda e: (e.ts, e.kind != "train"))
    return events, history


def replay(
    repo_name: str,
    repo: Path,
    instances: list[Instance],
    cache: Path,
    n_train: int = TRAIN_PER_REPO,
    dense=None,
    log=print,
) -> tuple[list[EventRecord], dict]:
    if cache.exists():
        with cache.open("rb") as f:
            return pickle.load(f)
    ensure_sparse(repo)
    if dense is not None:
        dense.load_repo(repo_name)
    events, history = build_events(repo_name, repo, instances, n_train)
    hist, h_i = History(), 0
    index = RepoIndex()
    records: list[EventRecord] = []
    timing: dict[str, Any] = {
        "checkout_s": 0.0,
        "first_build_s": None,
        "incremental_s": 0.0,
        "incremental_updates": 0,
        "files_reindexed": 0,
        "features_s": 0.0,
        "dense_s": 0.0,
        "verified_equal": 0,
        "verified_total": 0,
        "snapshot_files": [],
    }
    t_repo = time.perf_counter()
    with BlobReader(repo) as reader:
        for n, ev in enumerate(events):
            while h_i < len(history) and history[h_i].ts < ev.ts:
                hist.update(history[h_i])
                h_i += 1
            t0 = time.perf_counter()
            snap = materialise(repo, ev.snapshot_commit)
            timing["checkout_s"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            if index.tree is None:
                index = RepoIndex.build_full(snap, reader)
                timing["first_build_s"] = time.perf_counter() - t0
            else:
                d: merkle.Diff = index.update(snap, reader)
                timing["incremental_s"] += time.perf_counter() - t0
                timing["incremental_updates"] += 1
                timing["files_reindexed"] += len(d.added) + len(d.modified)
                if ev.kind == "test" and n % 25 == 0:  # spot-check incremental == full rebuild
                    full = RepoIndex.build_full(snap, reader)
                    timing["verified_total"] += 1
                    timing["verified_equal"] += int(full.state() == index.state())
            timing["snapshot_files"].append(len(snap))
            gold = [g for g in ev.gold if g in index.files]
            if not gold:
                continue
            t0 = time.perf_counter()
            qf = compute(index, hist, ev.query, ev.ts, with_chunks=True)
            timing["features_s"] += time.perf_counter() - t0
            mask = np.array([p in set(gold) for p in qf.paths])
            rec = EventRecord(ev.kind, repo_name, ev.key, ev.ts, ev.query, gold, qf, mask)
            if dense is not None and ev.kind == "test":  # dense is fused at test time only
                t0 = time.perf_counter()
                rec.extra["dense"] = dense.file_scores(snap, reader, qf.paths, ev.query)
                timing["dense_s"] += time.perf_counter() - t0
            records.append(rec)
            if (n + 1) % 50 == 0:
                log(f"[{repo_name}] {n + 1}/{len(events)} events, {time.perf_counter() - t_repo:.0f}s")
    if dense is not None:
        dense.save_repo(repo_name)
        timing["dense_chunks_embedded"] = dense.embedded_chunks
    timing["events"] = len(events)
    timing["records"] = len(records)
    timing["train_records"] = sum(r.kind == "train" for r in records)
    timing["test_records"] = sum(r.kind == "test" for r in records)
    timing["mean_snapshot_files"] = float(np.mean(timing.pop("snapshot_files")))
    timing["repo_seconds"] = time.perf_counter() - t_repo
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
        pickle.dump((records, timing), f)
    return records, timing
