"""History priors, updated strictly after each query is scored (no future leakage)."""

from __future__ import annotations

from collections import Counter

from .gitutil import Commit

MAX_COCHANGE_FILES = 20  # skip huge sweeping commits for co-change counts


class History:
    def __init__(self) -> None:
        self.churn: Counter[str] = Counter()
        self.last_ts: dict[str, int] = {}
        self.cochange: dict[str, Counter[str]] = {}
        self.seen: set[str] = set()  # commit shas absorbed so far (for leakage tests)

    def update(self, c: Commit) -> None:
        paths: list[str] = []
        for status, path, old in c.changes:
            if status == "R" and old and old.endswith(".py"):
                # Follow renames so priors survive file moves.
                self.churn[path] += self.churn.pop(old, 0)
                if old in self.last_ts:
                    self.last_ts[path] = self.last_ts.pop(old)
                if old in self.cochange:
                    self.cochange[path] = self.cochange.pop(old)
            if path.endswith(".py") and status != "D":
                paths.append(path)
        for p in paths:
            self.churn[p] += 1
            self.last_ts[p] = c.ts
        if 1 < len(paths) <= MAX_COCHANGE_FILES:
            for p in paths:
                cc = self.cochange.setdefault(p, Counter())
                for q in paths:
                    if q != p:
                        cc[q] += 1
        self.seen.add(c.sha)
