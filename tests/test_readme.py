import pytest

from commitcompass.readme import ROOT, render, render_results

RESULTS = ROOT / "results"


@pytest.mark.skipif(not (RESULTS / "swebench_lite.json").exists(), reason="results not generated")
def test_readme_matches_results():
    assert (ROOT / "README.md").read_text() == render(), "README.md is stale: run `uv run commitcompass readme`"
    got = (ROOT / "docs" / "RESULTS.md").read_text()
    assert got == render_results(), "docs/RESULTS.md is stale: run `uv run commitcompass readme`"
