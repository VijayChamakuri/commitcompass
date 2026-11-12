# Contributing

```bash
uv sync --all-groups
pre-commit install
make test
```

- README numbers come from `results/`. Edit `docs/README.template.md` and `docs/RESULTS.template.md`, then run `uv run commitcompass readme`. CI fails on drift.
- Never train or tune on SWE-bench labels; `check_no_label_leak` guards the commit-history boundary.
- Changes that alter results ship with regenerated `results/` (`make reproduce`).
