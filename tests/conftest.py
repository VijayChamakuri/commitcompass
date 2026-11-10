"""A tiny synthetic git repo with known history, built offline in tmp_path."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


def _git(repo: Path, *args: str, date: str | None = None, author: str = "Dev") -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": author,
        "GIT_AUTHOR_EMAIL": f"{author.lower().replace(' ', '')}@example.com",
        "GIT_COMMITTER_NAME": author,
        "GIT_COMMITTER_EMAIL": f"{author.lower().replace(' ', '')}@example.com",
    }
    if date:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = date
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env).stdout


def commit(repo: Path, files: dict[str, str | None], msg: str, date: str, author: str = "Dev") -> str:
    for path, content in files.items():
        p = repo / path
        if content is None:
            _git(repo, "rm", "-q", path)
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        _git(repo, "add", path)
    _git(repo, "commit", "-q", "-m", msg, date=date, author=author)
    return _git(repo, "rev-parse", "HEAD").strip()


@pytest.fixture
def toy_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "toy"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    commit(
        repo,
        {
            "pkg/__init__.py": "",
            "pkg/sessions.py": "class Session:\n    def send_request(self, url):\n        return url\n",
            "pkg/cookies.py": "def parse_cookie_header(value):\n    return value.split(';')\n",
            "pkg/utils.py": "def to_native_string(s):\n    return str(s)\n",
            "tests/test_sessions.py": "def test_send():\n    assert True\n",
        },
        "Initial import of the toy http package",
        "2024-01-01T00:00:00",
    )
    commit(
        repo,
        {"pkg/sessions.py": "class Session:\n    def send_request(self, url, timeout=None):\n        return url\n"},
        "Add timeout argument when sending a session request",
        "2024-01-02T00:00:00",
    )
    commit(
        repo,
        {"pkg/cookies.py": "def parse_cookie_header(value):\n    return [v.strip() for v in value.split(';')]\n"},
        "fix typo",  # short message -> dropped
        "2024-01-03T00:00:00",
    )
    commit(
        repo,
        {"pkg/utils.py": "def to_native_string(s):\n    return str(s)  # reformatted with black\n"},
        "Reformat utils module with black for consistency",  # hygiene -> dropped
        "2024-01-04T00:00:00",
    )
    commit(
        repo,
        {"pkg/cookies.py": "def parse_cookie_header(value):\n    return value\n"},
        "Update dependency pins for the cookie parser module",
        "2024-01-05T00:00:00",
        author="dependabot[bot]",  # bot -> dropped
    )
    commit(
        repo,
        {
            "pkg/cookies.py": (
                "def parse_cookie_header(value):\n    return [v.strip() for v in value.split(';') if v]\n"
            ),
            "pkg/sessions.py": (
                "class Session:\n    def send_request(self, url, timeout=None, cookies=None):\n        return url\n"
            ),
            "pkg/adapters.py": "class HTTPAdapter:\n    pass\n",  # new file -> excluded from gold
        },
        "Ignore empty cookie values and pass cookies through the session",
        "2024-01-06T00:00:00",
    )
    commit(
        repo,
        {"pkg/utils.py": "def to_native_string(s, encoding='ascii'):\n    return str(s)\n"},
        "Allow choosing the encoding used for native string conversion",
        "2024-01-07T00:00:00",
    )
    commit(
        repo,
        {"pkg/utils.py": None, "pkg/helpers.py": "def to_native_string(s, encoding='ascii'):\n    return str(s)\n"},
        "Move string helpers out of utils into a dedicated helpers module",
        "2024-01-08T00:00:00",
    )
    return repo
