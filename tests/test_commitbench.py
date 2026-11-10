from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from commitcompass import merkle
from commitcompass import metrics as M
from commitcompass.chunk import chunk_source, symbols
from commitcompass.evaluate import replay_repo, temporal_split
from commitcompass.features import F
from commitcompass.gitutil import BlobReader, ls_tree_py, read_history
from commitcompass.history import History
from commitcompass.index import RepoIndex
from commitcompass.mine import classify, mine_repo
from commitcompass.textproc import mask_query, split_identifier, tokenize

CUTOFF = "2030-01-01"


# ---- mining --------------------------------------------------------------------


def test_mining_filters_and_gold(toy_repo: Path) -> None:
    queries, stats = mine_repo("toy", toy_repo, CUTOFF)
    msgs = [q["query"] for q in queries]
    reasons = stats["drop_reasons"]
    assert reasons["root_commit"] == 1
    assert reasons["short_message"] == 1  # "fix typo"
    assert reasons["typo_or_format"] == 1  # black reformat
    assert reasons["bot_author"] == 1  # dependabot
    assert reasons["no_modified_py"] == 1  # rename-only commit
    assert msgs == [
        "Add timeout argument when sending a session request",
        "Ignore empty cookie values and pass cookies through the session",
        "Allow choosing the encoding used for native string conversion",
    ]
    cookie_q = queries[1]
    # New file adapters.py is excluded from gold; only pre-existing files count.
    assert cookie_q["gold"] == ["pkg/cookies.py", "pkg/sessions.py"]
    assert stats["new_py_files_excluded_from_gold"] >= 1
    # Every gold file must exist at the parent snapshot.
    for q in queries:
        assert set(q["gold"]) <= set(ls_tree_py(toy_repo, q["parent"]))


def test_too_many_files_dropped(toy_repo: Path) -> None:
    c = read_history(toy_repo, CUTOFF)[1]
    c.changes = [("M", f"pkg/m{i}.py", None) for i in range(9)]
    assert classify(c).reason == "too_many_files"


# ---- text processing ------------------------------------------------------------


def test_tokenizer_splits_code_identifiers() -> None:
    assert split_identifier("HTTPAdapter") == ["http", "adapter"]
    assert split_identifier("send_request") == ["send", "request"]
    toks = tokenize("def getResponseHeaders(self): return self.raw_headers")
    assert "response" in toks and "header" in toks  # camelCase split + light plural stem
    assert "getresponseheader" in toks  # whole identifier kept too
    assert "raw" in toks
    assert "self" not in toks and "def" not in toks and "return" not in toks


def test_mask_query_removes_only_file_names() -> None:
    q = "sessions.py: make Session.send_request honour timeout (see test_sessions)"
    masked = mask_query(q, ["pkg/sessions.py", "tests/test_sessions.py"], {"pkg"})
    assert "sessions.py" not in masked and "test_sessions" not in masked
    assert "Session.send_request" in masked  # symbols survive
    assert "timeout" in masked


def test_chunking_and_symbols() -> None:
    src = "import os\n\nX = 1\n\ndef a():\n    return 1\n\nclass B:\n    def c(self):\n        pass\n"
    spans = chunk_source(src)
    names = [s.name for s in spans]
    assert "a" in names and "B" in names and "<module>" in names
    assert symbols(src) == {"a", "b", "c"}
    # Python 2 code still chunks via the window fallback.
    assert chunk_source("print 'hello'\n")[0].name == "<window>"


# ---- Merkle tree ----------------------------------------------------------------


def test_merkle_diff() -> None:
    a = merkle.build({"pkg/a.py": "1", "pkg/b.py": "2", "pkg/sub/c.py": "3", "top.py": "4"})
    b = merkle.build({"pkg/a.py": "1", "pkg/b.py": "2x", "pkg/sub/d.py": "5", "top.py": "4"})
    d = merkle.diff(a, b)
    assert d.modified == ["pkg/b.py"]
    assert d.added == ["pkg/sub/d.py"]
    assert d.removed == ["pkg/sub/c.py"]
    assert merkle.diff(a, a).changed == 0
    assert merkle.diff(a, a).nodes_visited == 1  # identical root: nothing below is visited
    assert merkle.diff(None, a).added == ["pkg/a.py", "pkg/b.py", "pkg/sub/c.py", "top.py"]


def test_merkle_skips_unchanged_subtrees() -> None:
    leaves = {f"big/f{i}.py": str(i) for i in range(50)} | {"small/x.py": "1"}
    changed = dict(leaves) | {"small/x.py": "2"}
    d = merkle.diff(merkle.build(leaves), merkle.build(changed))
    assert d.modified == ["small/x.py"]
    assert d.nodes_visited < 10  # the 50-file 'big' directory is skipped by hash


def test_leaves_from_dir(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "notes.txt").write_text("y")
    assert list(merkle.leaves_from_dir(tmp_path)) == ["a.py"]


# ---- incremental index ------------------------------------------------------------


def test_incremental_index_equals_full_rebuild(toy_repo: Path) -> None:
    commits = read_history(toy_repo, CUTOFF)
    idx = RepoIndex()
    with BlobReader(toy_repo) as reader:
        for c in commits:
            snap = ls_tree_py(toy_repo, c.sha)
            idx.update(snap, reader)
            full = RepoIndex.build_full(snap, reader)
            assert idx.state() == full.state(), f"diverged at {c.sha[:8]}"
        # Last step: utils.py -> helpers.py, so the index must have dropped utils.py.
        assert "pkg/utils.py" not in idx.files and "pkg/helpers.py" in idx.files


def test_bm25_ranks_relevant_file_first(toy_repo: Path) -> None:
    with BlobReader(toy_repo) as reader:
        idx = RepoIndex.build_full(ls_tree_py(toy_repo, "HEAD"), reader)
    scores = idx.file_bm25.score(tokenize("cookie header parsing"))
    assert max(scores, key=scores.get) == "pkg/cookies.py"


# ---- metrics ------------------------------------------------------------------------


def test_metrics_math() -> None:
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    gold = np.array([True, False, True, False])
    order = M.rank_order(scores)
    assert order.tolist() == [1, 3, 2, 0]
    ranks = M.gold_ranks(order, gold)
    assert ranks == [3, 4]
    assert M.recall_at(ranks, 2, 1) == 0.0
    assert M.recall_at(ranks, 2, 3) == 0.5
    assert M.reciprocal_rank(ranks) == pytest.approx(1 / 3)
    ideal = 1 + 1 / np.log2(3)
    assert M.ndcg_at(ranks, 2, 10) == pytest.approx((1 / np.log2(4) + 1 / np.log2(5)) / ideal)
    assert M.ndcg_at([1, 2], 2, 10) == pytest.approx(1.0)


def test_rank_ties_are_deterministic() -> None:
    assert M.rank_order(np.array([1.0, 1.0, 2.0])).tolist() == [2, 0, 1]


def test_budget_recall() -> None:
    order = np.array([0, 1, 2])
    gold = np.array([False, False, True])
    chunks = [[(1.0, 3000)], [(1.0, 3000)], [(1.0, 3000)]]
    assert M.budget_recall(order, gold, chunks, 8000) == 0.0  # third file doesn't fit
    assert M.budget_recall(order, gold, chunks, 9000) == 1.0


def test_bootstrap_ci_brackets_mean() -> None:
    vals = [0.0, 1.0] * 50
    lo, hi = M.bootstrap_ci(vals)
    assert lo < 0.5 < hi
    mean, dlo, dhi = M.paired_delta_ci([1.0] * 10, [0.0] * 10)
    assert mean == dlo == dhi == 1.0


# ---- leakage ----------------------------------------------------------------------------


def test_temporal_split_has_no_leakage(toy_repo: Path) -> None:
    queries, _ = mine_repo("toy", toy_repo, CUTOFF)
    split = temporal_split(queries, frac=0.67)
    train = [q["order"] for q in queries if split[("toy", q["sha"])] == "train"]
    test = [q["order"] for q in queries if split[("toy", q["sha"])] == "test"]
    assert train and test and max(train) < min(test)


def test_history_priors_exclude_query_and_future(toy_repo: Path) -> None:
    """Features for a query must be computed before that commit enters the priors."""
    queries, _ = mine_repo("toy", toy_repo, CUTOFF)
    split = temporal_split(queries)
    records, timing = replay_repo("toy", toy_repo, CUTOFF, queries, split, full_rebuild_samples=10)
    assert timing["verified_equal"] == timing["verified_total"]
    first = next(r for r in records if r.text["as_is"].startswith("Add timeout"))
    qf = first.feats["as_is"]
    churn = {p: round(float(np.expm1(qf.X[i, F["churn_log"]]))) for i, p in enumerate(qf.paths)}
    # Only the initial import touched sessions.py before this query; the query commit itself
    # and the later cookie commit must not be counted.
    assert churn["pkg/sessions.py"] == 1


def test_history_follows_renames(toy_repo: Path) -> None:
    h = History()
    for c in read_history(toy_repo, CUTOFF):
        h.update(c)
    assert "pkg/utils.py" not in h.churn
    assert h.churn["pkg/helpers.py"] >= 3


def test_queries_jsonl_roundtrip(toy_repo: Path, tmp_path: Path) -> None:
    queries, _ = mine_repo("toy", toy_repo, CUTOFF)
    p = tmp_path / "q.jsonl"
    p.write_text("".join(json.dumps(q) + "\n" for q in queries))
    assert [json.loads(line)["sha"] for line in p.read_text().splitlines()] == [q["sha"] for q in queries]
