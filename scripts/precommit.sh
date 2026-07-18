uv sync --locked --all-extras --dev
uv run mypy aatm/
uv run ruff check aatm/
uv run ruff format aatm/
uv run pytest