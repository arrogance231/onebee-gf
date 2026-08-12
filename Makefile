.PHONY: install install-gpu lint format typecheck test test-unit baseline eval figures paper

install:
	uv sync

install-gpu:
	uv sync --extra gpu

lint:
	uv run ruff check
	uv run black --check .

format:
	uv run ruff check --fix
	uv run black .

typecheck:
	uv run mypy src

test:
	uv run pytest

test-unit:
	uv run pytest tests/unit

baseline:
	echo "run scripts/model_bakeoff.py — requires GPU, not run in CI"

eval:
	echo "placeholder for eval targets"

figures:
	uv run python scripts/make_figures.py

paper:
	cd paper && bash build.sh
