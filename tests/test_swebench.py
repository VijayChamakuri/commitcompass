import numpy as np
import pytest

from commitbench.swe_eval import check_no_label_leak, rrf
from commitbench.swebench import acc_at, patch_files

PATCH = """diff --git a/django/db/models/query.py b/django/db/models/query.py
index 1..2 100644
--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -1 +1 @@
-a
+b
diff --git a/django/utils/new_helper.py b/django/utils/new_helper.py
new file mode 100644
--- /dev/null
+++ b/django/utils/new_helper.py
diff --git a/docs/releases/4.2.txt b/docs/releases/4.2.txt
--- a/docs/releases/4.2.txt
+++ b/docs/releases/4.2.txt
"""


def test_patch_parsing_separates_modified_created_and_non_python():
    modified, created, non_py = patch_files(PATCH)
    assert modified == ["django/db/models/query.py"]
    assert created == ["django/utils/new_helper.py"]
    assert non_py == ["docs/releases/4.2.txt"]


def test_acc_at_requires_all_gold_files():
    assert acc_at([3], 1, 5) == 1.0
    assert acc_at([3, 7], 2, 5) == 0.0
    assert acc_at([3], 2, 5) == 0.0  # a gold file missing from candidates is a miss


def test_rrf_rewards_agreement():
    a = np.array([3.0, 2.0, 1.0])
    b = np.array([1.0, 3.0, 2.0])
    fused = rrf(a, b)
    assert fused.argmax() == 1  # ranked 2nd and 1st beats 1st and 3rd


class R:
    def __init__(self, kind, repo, key, ts):
        self.kind, self.repo, self.key, self.ts = kind, repo, key, ts


def test_label_leak_guard():
    train = [R("train", "a/b", "c1", 100.0)]
    ok = [R("test", "a/b", "i1", 50.0)]
    check_no_label_leak(train, ok, {"i1": 200.0})
    with pytest.raises(AssertionError):
        check_no_label_leak(train, ok, {"i1": 90.0})


def test_history_parser_survives_control_characters_in_messages(tmp_path):
    import subprocess

    from commitbench.swe_replay import read_history_fast

    def run(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)

    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    run("add", "a.py")
    run("commit", "-q", "-m", "first commit\x1e with a record separator\x1f inside")
    (tmp_path / "a.py").write_text("x = 2\n")
    run("commit", "-qam", "second commit")
    commits = read_history_fast(tmp_path, "HEAD")
    assert [c.message.split()[0] for c in commits] == ["first", "second"]
    assert commits[1].changes == [("M", "a.py", None)]
