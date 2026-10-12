.PHONY: sync test lint portal hygiene

sync:
	uv sync --extra portal --extra dev

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

portal:
	uv run rs-eval portal

hygiene:
	uv run python scripts/check_repo_hygiene.py
