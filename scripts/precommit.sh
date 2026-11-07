uv sync --locked --all-extras --dev
uv run mypy aatm/
uv run ruff check aatm/ tests/
uv run ruff format aatm/ tests/
uv run pytest