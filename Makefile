.PHONY: verify
verify:
	uv run ruff check .
	uv run ruff format --check .
	uv run lint-imports
	uv run mypy
	uv run pytest
