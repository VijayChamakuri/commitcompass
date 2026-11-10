"""Code-aware tokenization shared by queries and source files."""

from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_RAW_WORD = re.compile(r"[A-Za-z0-9_]+")

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "elif",
        "when",
        "while",
        "for",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "we",
        "you",
        "they",
        "he",
        "she",
        "i",
        "me",
        "my",
        "our",
        "your",
        "their",
        "them",
        "not",
        "no",
        "nor",
        "so",
        "than",
        "too",
        "very",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "do",
        "does",
        "did",
        "done",
        "has",
        "have",
        "had",
        "having",
        "into",
        "onto",
        "out",
        "up",
        "down",
        "over",
        "under",
        "again",
        "further",
        "once",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "just",
        "also",
        "via",
        "per",
        "about",
        "after",
        "before",
        "between",
        "through",
        "during",
        "above",
        "below",
        "off",
        "why",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "where",
        "whose",
        "def",
        "self",
        "cls",
        "return",
        "import",
        "none",
        "true",
        "false",
        "elif",
        "try",
        "except",
        "raise",
        "pass",
        "lambda",
        "yield",
        "async",
        "await",
        "args",
        "kwargs",
        "py",
    ]
)


def _light_stem(tok: str) -> str:
    """Strip a plural 's' so 'sessions' and 'session' match; same rule on both sides."""
    if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def split_identifier(word: str) -> list[str]:
    """Split snake_case and camelCase identifiers into lowercase parts."""
    parts: list[str] = []
    for piece in word.split("_"):
        if piece:
            parts.extend(m.lower() for m in _CAMEL.findall(piece))
    return parts


def tokenize(text: str) -> list[str]:
    """Tokens for BM25: whole identifiers plus their sub-parts, stopwords dropped."""
    out: list[str] = []
    for m in _WORD.finditer(text):
        word = m.group(0)
        parts = split_identifier(word)
        whole = word.lower().strip("_")
        if len(parts) > 1 and len(whole) > 1 and whole not in STOPWORDS:
            out.append(_light_stem(whole))
        for p in parts:
            if len(p) > 1 and p not in STOPWORDS and not p.isdigit():
                out.append(_light_stem(p))
    return out


def raw_words(text: str) -> set[str]:
    """Lowercased raw words (identifiers kept whole, dotted names split on '.')."""
    return {w.lower() for w in _RAW_WORD.findall(text)}


def path_tokens(path: str) -> set[str]:
    return set(tokenize(re.sub(r"[/.\-]", " ", path)))


def file_stems(path: str) -> set[str]:
    """Names a human would use to refer to a file: 'sessions', 'test_sessions'."""
    name = path.rsplit("/", 1)[-1]
    stem = name[:-3] if name.endswith(".py") else name
    stems = {stem.lower()}
    if stem.startswith("test_"):
        stems.add(stem[5:].lower())
    if stem.endswith("_test"):
        stems.add(stem[:-5].lower())
    stems.discard("__init__")
    stems.discard("")
    return stems


def is_test_path(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
        or "/tests/" in f"/{path}"
        or "/testing/" in f"/{path}"
    )


def mask_query(text: str, gold: list[str], package_names: set[str]) -> str:
    """Remove words that literally name a gold file (stem or non-package parent dir).

    This removes the easiest leakage path: commit messages that say 'sessions.py: ...'.
    """
    banned: set[str] = set()
    for path in gold:
        banned |= file_stems(path)
        parts = path.split("/")
        if len(parts) >= 2:
            parent = parts[-2].lower()
            if parent not in package_names and parent not in {"tests", "test", "src"}:
                banned.add(parent)

    def repl(m: re.Match[str]) -> str:
        w = m.group(0).lower()
        base = w[:-3] if w.endswith(".py") else w
        return "" if base in banned else m.group(0)

    masked = re.sub(r"[A-Za-z0-9_]+(?:\.py)?", repl, text)
    return re.sub(r"[ \t]+", " ", masked).strip()
