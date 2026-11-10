"""Mine (query, gold files) pairs from real git history.

query = the commit message, gold = the existing .py files the commit modified.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gitutil import Commit, read_history

# Fixed cutoff keeps the benchmark deterministic even as upstream repos move.
CUTOFF = "2026-10-01"
REPOS: dict[str, str] = {
    "flask": "https://github.com/pallets/flask",
    "requests": "https://github.com/psf/requests",
    "httpx": "https://github.com/encode/httpx",
}
MIN_WORDS = 5
MAX_GOLD = 8

_TRAILER = re.compile(r"^(signed-off-by|co-authored-by|reviewed-by|change-id|cc):", re.I)
_BOT = re.compile(r"\[bot\]|dependabot|pre-commit-ci|github-actions|renovate", re.I)
_VERSION = re.compile(
    r"^(bump|release|prepare|version)\b|\bbump(ed)?\b.*\bversion\b|^v?\d+(\.\d+)+\S*$"
    r"|\brelease\s+v?\d|\bchangelog\b|\bchanges\.rst\b",
    re.I,
)
_HYGIENE = re.compile(
    r"\b(typos?|spelling|whitespace|reformat(ted|ting)?|formatting|black|isort|flake8|lint(ing)?"
    r"|pep\s?8|ruff|pre-commit|pyupgrade|codespell|copyright|license)\b",
    re.I,
)


@dataclass
class Decision:
    keep: bool
    reason: str
    gold: list[str]
    new_files_excluded: int


def clean_message(msg: str) -> str:
    lines = [ln for ln in msg.splitlines() if not _TRAILER.match(ln.strip())]
    return "\n".join(lines).strip()[:1500]


def classify(commit: Commit) -> Decision:
    """Apply benchmark filters to one commit. Order matters only for reason labels."""
    py = [(s, p, old) for s, p, old in commit.changes if p.endswith(".py")]
    modified = sorted({p for s, p, _ in py if s == "M"})
    new = sum(1 for s, _, _ in py if s in "ARC")
    msg = clean_message(commit.message)
    if commit.parent is None:
        return Decision(False, "root_commit", [], new)
    if _BOT.search(commit.author) or _BOT.search(commit.email):
        return Decision(False, "bot_author", [], new)
    if not modified:
        return Decision(False, "no_modified_py", [], new)
    if len(modified) > MAX_GOLD:
        return Decision(False, "too_many_files", [], new)
    if len(msg.split()) < MIN_WORDS:
        return Decision(False, "short_message", [], new)
    if _VERSION.search(commit.subject):
        return Decision(False, "version_or_release", [], new)
    if _HYGIENE.search(commit.subject):
        return Decision(False, "typo_or_format", [], new)
    return Decision(True, "kept", modified, new)


def clone(name: str, url: str, repos_dir: Path) -> Path:
    dest = repos_dir / name
    if not dest.exists():
        repos_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", url, str(dest)], check=True)
    return dest


def mine_repo(name: str, repo: Path, cutoff: str = CUTOFF) -> tuple[list[dict], dict]:
    commits = read_history(repo, cutoff)
    queries: list[dict] = []
    reasons: Counter[str] = Counter()
    new_excluded = 0
    for order, c in enumerate(commits):
        d = classify(c)
        reasons[d.reason] += 1
        if d.keep:
            new_excluded += d.new_files_excluded
            queries.append(
                {
                    "repo": name,
                    "sha": c.sha,
                    "parent": c.parent,
                    "ts": c.ts,
                    "order": order,
                    "query": clean_message(c.message),
                    "gold": d.gold,
                }
            )
    stats: dict[str, Any] = {
        "repo": name,
        "commits_scanned": len(commits),
        "queries": len(queries),
        "kept_fraction": round(len(queries) / max(1, len(commits)), 4),
        "drop_reasons": dict(reasons),
        "new_py_files_excluded_from_gold": new_excluded,
        "mean_gold_size": round(sum(len(q["gold"]) for q in queries) / max(1, len(queries)), 3),
    }
    return queries, stats


def mine_all(data_dir: Path, repos: dict[str, str] | None = None, cutoff: str = CUTOFF) -> dict:
    repos = repos or REPOS
    all_q: list[dict] = []
    all_stats: dict[str, Any] = {}
    for name, url in repos.items():
        path = clone(name, url, data_dir / "repos")
        q, s = mine_repo(name, path, cutoff)
        all_q.extend(q)
        all_stats[name] = s
        print(f"[mine] {name}: {s['queries']} queries from {s['commits_scanned']} commits")
    with open(data_dir / "queries.jsonl", "w") as f:
        for row in all_q:
            f.write(json.dumps(row) + "\n")
    meta = {"cutoff": cutoff, "repos": repos, "stats": all_stats}
    (data_dir / "mine_stats.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_queries(data_dir: Path) -> list[dict]:
    with open(data_dir / "queries.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]
