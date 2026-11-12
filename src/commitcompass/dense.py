"""Dense reranking with a local code-embedding model (no paid model service).

Model: `jinaai/jina-embeddings-v2-base-code` via fastembed, on CPU. This is the model
family reported as "Jina-Code-v2" in published SWE-bench Lite localization tables.

Full-corpus chunk embedding of the file versions behind the 300 SWE-bench Lite base
commits was not run. Dense reranks a candidate pool instead:

  pool   top-30 files by BM25 (file) and top-30 by the hand-tuned hybrid (both non-learned)
  file   one vector per file version: its path plus a skeleton of class/def signatures and
         the first line of each docstring, truncated to `max_tokens`

The share of issues whose gold file is in the pool is reported, so the ceiling is explicit.
Vectors are cached per git blob id.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

from .gitutil import BlobReader

MODEL = "jinaai/jina-embeddings-v2-base-code"
POOL = 30
_SIG = re.compile(r"^\s*(?:async\s+def|def|class)\s+\w+.*$")


def skeleton(path: str, src: str, max_chars: int) -> str:
    """Path + signatures + first docstring line under each signature."""
    out = [f"# {path}"]
    lines = src.split("\n")
    for i, line in enumerate(lines):
        if _SIG.match(line):
            out.append(line.rstrip())
            for nxt in lines[i + 1 : i + 4]:
                s = nxt.strip()
                if s.startswith(('"""', "'''")):
                    doc = s.strip("\"' ")
                    if doc:
                        out.append(" " * (len(line) - len(line.lstrip()) + 4) + doc)
                    break
        if sum(len(x) + 1 for x in out) > max_chars:
            break
    return "\n".join(out)[:max_chars]


class DenseScorer:
    def __init__(
        self,
        cache_dir: Path,
        model: str = MODEL,
        max_tokens: int = 256,
        query_chars: int = 2000,
        threads: int | None = None,
    ):
        self.model_name, self.max_tokens, self.query_chars, self.threads = model, max_tokens, query_chars, threads
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self.blob_vecs: dict[str, np.ndarray] = {}
        self.embedded = 0
        self.embed_seconds = 0.0

    @property
    def config(self) -> str:
        return f"{self.model_name}|skeleton{self.max_tokens}|pool{POOL}"

    def _path(self, repo: str) -> Path:
        return (
            self.cache_dir / f"{repo.replace('/', '_')}__{self.model_name.replace('/', '_')}_skel{self.max_tokens}.npz"
        )

    def load_repo(self, repo: str) -> None:
        p = self._path(repo)
        self.blob_vecs = dict(np.load(p, allow_pickle=True)["vecs"].item()) if p.exists() else {}

    def save_repo(self, repo: str) -> None:
        np.savez(self._path(repo), vecs=np.array(self.blob_vecs, dtype=object))

    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            kw: dict = {"max_length": self.max_tokens}
            if self.threads:
                kw["threads"] = self.threads
            self._model = TextEmbedding(self.model_name, **kw)
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        t0 = time.perf_counter()
        v = np.stack(list(self.model().embed(texts, batch_size=16))).astype(np.float32)
        self.embed_seconds += time.perf_counter() - t0
        self.embedded += len(texts)
        return v / np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-9)

    def file_scores(
        self, blobs: dict[str, str], reader: BlobReader, paths: list[str], pool: np.ndarray, query: str
    ) -> np.ndarray:
        """Cosine similarity for pooled candidates; -inf elsewhere. `blobs`: path -> blob id."""
        todo = sorted({blobs[paths[i]] for i in pool} - set(self.blob_vecs))
        if todo:
            by_blob = {blobs[paths[i]]: paths[i] for i in pool}
            texts = [skeleton(by_blob[b], reader.read(b), self.max_tokens * 4) for b in todo]
            for b, v in zip(todo, self._embed(texts), strict=True):
                self.blob_vecs[b] = v.astype(np.float16)
        q = self._embed([query[: self.query_chars]])[0]
        out = np.full(len(paths), -np.inf, dtype=np.float32)
        for i in pool:
            out[i] = float(self.blob_vecs[blobs[paths[i]]].astype(np.float32) @ q)
        return out
