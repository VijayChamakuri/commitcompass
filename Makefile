.PHONY: install swebench history site readme reproduce test smoke

install:
	uv sync --all-groups

swebench:
	uv run commitcompass swebench

history:
	uv run commitcompass history

site:
	uv run commitcompass site

readme:
	uv run commitcompass readme

reproduce:
	uv run commitcompass reproduce

test:
	uv run ruff check . && uv run ruff format --check .
	uv run mypy src
	uv run pytest -q --cov=commitcompass --cov-report=term-missing
