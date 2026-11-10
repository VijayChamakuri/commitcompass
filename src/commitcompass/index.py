"""Incrementally maintained BM25 index over files and chunks of one repo snapshot."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from . import merkle
from .chunk import chunk_source, span_text, symbols
from .gitutil import BlobReader
from .textproc import is_test_path, path_tokens, tokenize


class BM25:
    """BM25 with an inverted index that supports add/remove of documents."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, dict[str, int]] = {}
        self.doc_terms: dict[str, Counter[str]] = {}
        self.doc_len: dict[str, int] = {}
        self.total_len = 0

    def __len__(self) -> int:
        return len(self.doc_len)

    def add(self, doc_id: str, tokens: list[str]) -> None:
        if doc_id in self.doc_len:
            self.remove(doc_id)
        tf = Counter(tokens)
        self.doc_terms[doc_id] = tf
        self.doc_len[doc_id] = len(tokens)
        self.total_len += len(tokens)
        for term, n in tf.items():
            self.postings.setdefault(term, {})[doc_id] = n

    def remove(self, doc_id: str) -> None:
        tf = self.doc_terms.pop(doc_id, None)
        if tf is None:
            return
        self.total_len -= self.doc_len.pop(doc_id)
        for term in tf:
            plist = self.postings[term]
            del plist[doc_id]
            if not plist:
                del self.postings[term]

    def score(self, query_terms: list[str]) -> dict[str, float]:
        n = len(self.doc_len)
        if n == 0:
            return {}
        avgdl = self.total_len / n or 1.0
        scores: dict[str, float] = {}
        for term, qtf in Counter(query_terms).items():
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for doc, tf in plist.items():
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[doc] / avgdl)
                scores[doc] = scores.get(doc, 0.0) + qtf * idf * tf * (self.k1 + 1) / denom
        return scores

    def state(self) -> tuple:
        """Canonical, comparable view of the index (used to verify incremental == full)."""
        return (
            {t: dict(sorted(p.items())) for t, p in sorted(self.postings.items())},
            dict(sorted(self.doc_len.items())),
            self.total_len,
        )


@dataclass
class FileEntry:
    blob: str
    n_chars: int
    n_lines: int
    is_test: bool
    path_tokens: set[str]
    symbols: set[str]
    symbol_tokens: set[str]
    chunk_ids: list[str] = field(default_factory=list)
    chunk_tokens: list[int] = field(default_factory=list)  # approx LLM tokens per chunk


class RepoIndex:
    def __init__(self) -> None:
        self.files: dict[str, FileEntry] = {}
        self.file_bm25 = BM25()
        self.chunk_bm25 = BM25()
        self.tree: merkle.Node | None = None

    # ---- building -------------------------------------------------------
    def _add_file(self, path: str, blob: str, src: str) -> None:
        lines = src.split("\n")
        syms = symbols(src)
        entry = FileEntry(
            blob=blob,
            n_chars=len(src),
            n_lines=len(lines),
            is_test=is_test_path(path),
            path_tokens=path_tokens(path),
            symbols=syms,
            symbol_tokens=set(tokenize(" ".join(sorted(syms)))),
        )
        ptoks = list(entry.path_tokens)
        self.file_bm25.add(path, tokenize(src) + ptoks)
        for i, span in enumerate(chunk_source(src)):
            text = span_text(lines, span)
            if not text.strip():
                continue
            cid = f"{path}#{i}"
            self.chunk_bm25.add(cid, tokenize(text) + tokenize(span.name) + ptoks)
            entry.chunk_ids.append(cid)
            entry.chunk_tokens.append(max(1, len(text) // 4))
        self.files[path] = entry

    def _remove_file(self, path: str) -> None:
        entry = self.files.pop(path)
        self.file_bm25.remove(path)
        for cid in entry.chunk_ids:
            self.chunk_bm25.remove(cid)

    @classmethod
    def build_full(cls, snapshot: dict[str, str], reader: BlobReader) -> RepoIndex:
        idx = cls()
        for path in sorted(snapshot):
            idx._add_file(path, snapshot[path], reader.read(snapshot[path]))
        idx.tree = merkle.build(snapshot)
        return idx

    def update(self, snapshot: dict[str, str], reader: BlobReader) -> merkle.Diff:
        """Move the index to `snapshot`, re-reading only files the Merkle diff flags."""
        new_tree = merkle.build(snapshot)
        d = merkle.diff(self.tree, new_tree)
        for path in d.removed + d.modified:
            self._remove_file(path)
        for path in d.added + d.modified:
            self._add_file(path, snapshot[path], reader.read(snapshot[path]))
        self.tree = new_tree
        return d

    def state(self) -> tuple:
        files = {
            p: (e.blob, e.n_chars, tuple(e.chunk_ids), tuple(e.chunk_tokens), tuple(sorted(e.symbols)))
            for p, e in sorted(self.files.items())
        }
        return files, self.file_bm25.state(), self.chunk_bm25.state()
