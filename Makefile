.PHONY: install swebench history site readme reproduce test smoke

install:
	uv sync --all-groups

swebench:
	uv run commitbench swebench

history:
	uv run commitbench history

site:
	uv run commitbench site

readme:
	uv run commitbench readme

reproduce:
	uv run commitbench reproduce

test:
	uv run ruff check . && uv run ruff format --check .
	uv run mypy src
	uv run pytest -q --cov=commitbench --cov-report=term-missing
