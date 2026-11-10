"""Split Python source into retrievable chunks along def/class boundaries."""

from __future__ import annotations

import ast
import re
import warnings
from dataclasses import dataclass

MAX_LINES = 150
WINDOW = 60
_DEF_RE = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)", re.M)


@dataclass(frozen=True)
class Span:
    start: int  # 1-based, inclusive
    end: int  # inclusive
    name: str


def _node_start(node: ast.AST) -> int:
    decos = [d.lineno for d in getattr(node, "decorator_list", [])]
    return min([node.lineno, *decos])  # type: ignore[attr-defined]


def _spans_for(node: ast.AST, prefix: str = "") -> list[Span]:
    name = prefix + node.name  # type: ignore[attr-defined]
    start, end = _node_start(node), node.end_lineno  # type: ignore[attr-defined]
    if end - start + 1 <= MAX_LINES:
        return [Span(start, end, name)]
    kids = [
        n for n in getattr(node, "body", []) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if not kids:
        return _windows(start, end, name)
    spans: list[Span] = []
    for k in kids:
        spans.extend(_spans_for(k, name + "."))
    # Fill the class header / non-def statements between methods.
    return spans + _gaps(spans, start, end, name)


def _windows(start: int, end: int, name: str) -> list[Span]:
    return [Span(s, min(s + WINDOW - 1, end), name) for s in range(start, end + 1, WINDOW)]


def _gaps(spans: list[Span], start: int, end: int, name: str) -> list[Span]:
    covered: set[int] = set()
    for s in spans:
        covered.update(range(s.start, s.end + 1))
    out: list[Span] = []
    run_start = None
    for ln in range(start, end + 2):
        free = ln <= end and ln not in covered
        if free and run_start is None:
            run_start = ln
        elif not free and run_start is not None:
            out.extend(_windows(run_start, ln - 1, name))
            run_start = None
    return out


def chunk_source(src: str) -> list[Span]:
    """Top-level defs/classes (large classes split per method) + module-level gaps.

    Falls back to fixed windows for code that does not parse (e.g. Python 2 files
    in old history).
    """
    n_lines = src.count("\n") + 1
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return _windows(1, n_lines, "<window>")
    spans: list[Span] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            spans.extend(_spans_for(node))
    spans.extend(_gaps(spans, 1, n_lines, "<module>"))
    return sorted(spans, key=lambda s: (s.start, s.end))


def symbols(src: str) -> set[str]:
    """Every def/class name in the file (regex so it also works on Python 2 code)."""
    return {m.group(1).lower() for m in _DEF_RE.finditer(src)}


def span_text(lines: list[str], span: Span) -> str:
    return "\n".join(lines[span.start - 1 : span.end])
