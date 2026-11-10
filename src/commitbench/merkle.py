"""Merkle tree over a file tree, so re-indexing only touches what changed.

Leaves hash file contents; a directory's hash is the hash of its children's
(name, hash) pairs. Two snapshots are diffed top-down and identical subtrees
are skipped without being visited.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Node:
    hash: str = ""
    children: dict[str, Node] | None = None  # None => leaf (file)


@dataclass
class Diff:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    nodes_visited: int = 0

    @property
    def changed(self) -> int:
        return len(self.added) + len(self.removed) + len(self.modified)


def build(leaves: dict[str, str]) -> Node:
    """Build a tree from {path: content_hash}."""
    root = Node(children={})
    for path, h in leaves.items():
        node = root
        *dirs, name = path.split("/")
        for d in dirs:
            assert node.children is not None
            node = node.children.setdefault(d, Node(children={}))
        assert node.children is not None
        node.children[name] = Node(hash=h)
    _rehash(root)
    return root


def _rehash(node: Node) -> str:
    if node.children is None:
        return node.hash
    h = hashlib.sha256()
    for name in sorted(node.children):
        h.update(f"{name}\0{_rehash(node.children[name])}\n".encode())
    node.hash = h.hexdigest()
    return node.hash


def _all_leaves(node: Node, prefix: str) -> list[str]:
    if node.children is None:
        return [prefix]
    out: list[str] = []
    for name, child in node.children.items():
        out.extend(_all_leaves(child, f"{prefix}/{name}" if prefix else name))
    return out


def diff(old: Node | None, new: Node | None) -> Diff:
    d = Diff()
    _diff(old, new, "", d)
    for lst in (d.added, d.removed, d.modified):
        lst.sort()
    return d


def _diff(a: Node | None, b: Node | None, prefix: str, d: Diff) -> None:
    d.nodes_visited += 1
    if a is None and b is None:
        return
    if a is None:
        assert b is not None
        d.added.extend(_all_leaves(b, prefix))
        return
    if b is None:
        d.removed.extend(_all_leaves(a, prefix))
        return
    if a.hash == b.hash:
        return  # identical subtree: skip entirely
    if a.children is None and b.children is None:
        d.modified.append(prefix)
        return
    if a.children is None or b.children is None:  # file <-> directory
        d.removed.extend(_all_leaves(a, prefix))
        d.added.extend(_all_leaves(b, prefix))
        return
    for name in sorted(a.children.keys() | b.children.keys()):
        _diff(
            a.children.get(name),
            b.children.get(name),
            f"{prefix}/{name}" if prefix else name,
            d,
        )


def leaves_from_dir(root: Path, suffix: str = ".py") -> dict[str, str]:
    """Hash files in a working directory (for indexing a checkout, not a git tree)."""
    out: dict[str, str] = {}
    for p in sorted(root.rglob(f"*{suffix}")):
        if p.is_file() and ".git" not in p.parts:
            out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out
