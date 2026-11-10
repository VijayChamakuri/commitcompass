"""Thin wrappers over the git CLI (no third-party git library)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

_RS, _FS, _END = "\x1e", "\x1f", "\x1d"


@dataclass
class Commit:
    sha: str
    parent: str | None
    ts: int
    author: str
    email: str
    message: str
    # (status letter, path) — for renames/copies, path is the *new* path and old_path is set
    changes: list[tuple[str, str, str | None]] = field(default_factory=list)

    @property
    def subject(self) -> str:
        return self.message.split("\n", 1)[0]


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.quotepath=false", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout


def read_history(repo: Path, cutoff: str) -> list[Commit]:
    """All non-merge commits before `cutoff`, oldest first (parents before children)."""
    fmt = f"{_RS}%H{_FS}%P{_FS}%ct{_FS}%an{_FS}%ae{_FS}%B{_END}"
    out = git(
        repo,
        "log",
        "--no-merges",
        "--date-order",
        "--reverse",
        f"--before={cutoff}",
        "--find-renames",
        "--name-status",
        f"--format={fmt}",
        "HEAD",
    )
    commits: list[Commit] = []
    for rec in out.split(_RS):
        if not rec.strip():
            continue
        header, _, rest = rec.partition(_END)
        sha, parents, ts, an, ae, body = header.split(_FS, 5)
        changes: list[tuple[str, str, str | None]] = []
        for line in rest.splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            status = parts[0][0]
            if status in "RC" and len(parts) == 3:
                changes.append((status, parts[2], parts[1]))
            elif len(parts) >= 2:
                changes.append((status, parts[1], None))
        commits.append(
            Commit(
                sha=sha,
                parent=parents.split()[0] if parents.strip() else None,
                ts=int(ts),
                author=an,
                email=ae,
                message=body.strip(),
                changes=changes,
            )
        )
    return commits


def ls_tree_py(repo: Path, commit: str) -> dict[str, str]:
    """Map of .py path -> git blob id at `commit`. Blob ids are content hashes."""
    out = git(repo, "ls-tree", "-r", "-z", commit)
    snap: dict[str, str] = {}
    for entry in out.split("\0"):
        if not entry:
            continue
        meta, _, path = entry.partition("\t")
        mode, kind, blob = meta.split()
        if kind == "blob" and path.endswith(".py") and mode != "120000":
            snap[path] = blob
    return snap


class BlobReader:
    """Persistent `git cat-file --batch` process for fast blob reads."""

    def __init__(self, repo: Path):
        self.proc = subprocess.Popen(
            ["git", "-C", str(repo), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self.reads = 0

    def read(self, blob: str) -> str:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(f"{blob}\n".encode())
        self.proc.stdin.flush()
        header = self.proc.stdout.readline().decode().split()
        if len(header) < 3 or header[1] == "missing":
            raise KeyError(blob)
        size = int(header[2])
        data = self.proc.stdout.read(size)
        self.proc.stdout.read(1)  # trailing newline
        self.reads += 1
        return data.decode("utf-8", errors="replace")

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait()

    def __enter__(self) -> BlobReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
