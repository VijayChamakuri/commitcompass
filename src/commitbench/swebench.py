"""SWE-bench Lite file-level localization: data loading and gold extraction.

SWE-bench Lite (princeton-nlp/SWE-bench_Lite, test split) has 300 real GitHub issues from
12 Python repositories. For each instance the query is the issue text (`problem_statement`)
and the gold files are the files modified by the reference fix (`patch`), which is the
definition used by published localization work (e.g. LocAgent, arXiv:2503.09089, Table 4).

Files the reference patch *creates* cannot be retrieved from the base commit, so they are
excluded from gold and counted. Acc@k follows the published definition: an instance counts
as correct at k only if *every* gold file is in the top k.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset=princeton-nlp%2FSWE-bench_Lite"
    "&config=default&split=test&offset={offset}&length=100"
)
N_INSTANCES = 300
REPOS = [
    "astropy/astropy",
    "django/django",
    "matplotlib/matplotlib",
    "mwaskom/seaborn",
    "pallets/flask",
    "psf/requests",
    "pydata/xarray",
    "pylint-dev/pylint",
    "pytest-dev/pytest",
    "scikit-learn/scikit-learn",
    "sphinx-doc/sphinx",
    "sympy/sympy",
]

_DIFF_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)$")


@dataclass
class Instance:
    instance_id: str
    repo: str
    base_commit: str
    created_at: str
    problem_statement: str
    gold: list[str]  # modified, pre-existing .py files
    created_files: list[str]
    non_py_files: list[str]


def patch_files(patch: str) -> tuple[list[str], list[str], list[str]]:
    """Parse a unified diff into (modified .py files, created files, non-.py modified files)."""
    modified, created, non_py = [], [], []
    current: str | None = None
    is_new = False

    def flush() -> None:
        if current is None:
            return
        if is_new:
            created.append(current)
        elif current.endswith(".py"):
            modified.append(current)
        else:
            non_py.append(current)

    for line in patch.splitlines():
        m = _DIFF_HEADER.match(line)
        if m:
            flush()
            current, is_new = m.group(2), False
        elif current is not None and line.startswith("new file mode"):
            is_new = True
    flush()
    return sorted(set(modified)), sorted(set(created)), sorted(set(non_py))


def fetch(out: Path) -> dict:
    """Download all 300 instances once; returns {path, sha256}. Idempotent."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        rows = []
        for offset in range(0, N_INSTANCES, 100):
            for attempt in range(4):
                try:
                    req = urllib.request.Request(ROWS_URL.format(offset=offset), headers={"User-Agent": "commitbench"})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        rows += [x["row"] for x in json.loads(r.read())["rows"]]
                    break
                except Exception:  # noqa: BLE001 - transient network errors
                    if attempt == 3:
                        raise
                    time.sleep(3 * (attempt + 1))
        rows.sort(key=lambda r: r["instance_id"])
        with out.open("w") as f:
            for r in rows:
                f.write(
                    json.dumps(
                        {
                            k: r[k]
                            for k in ("instance_id", "repo", "base_commit", "created_at", "problem_statement", "patch")
                        }
                    )
                    + "\n"
                )
    data = out.read_bytes()
    return {"path": str(out), "instances": data.count(b"\n"), "sha256": hashlib.sha256(data).hexdigest()}


def load(path: Path) -> list[Instance]:
    out = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        modified, created, non_py = patch_files(r["patch"])
        out.append(
            Instance(
                r["instance_id"],
                r["repo"],
                r["base_commit"],
                r["created_at"],
                r["problem_statement"],
                modified,
                created,
                non_py,
            )
        )
    return out


def acc_at(gold_ranks: list[int], n_gold: int, k: int) -> float:
    """Published file-level Acc@k: 1 only if all gold files rank within the top k."""
    return 1.0 if n_gold and len(gold_ranks) == n_gold and max(gold_ranks) <= k else 0.0
