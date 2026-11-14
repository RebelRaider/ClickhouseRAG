#!/usr/bin/env bash
set -euo pipefail

uv lock --check
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked --extra postgres ty check src tests examples benchmarks scripts
uv run --locked bandit -q -r src
uv run --locked zizmor --offline --pedantic .github/workflows
uv run --locked coverage run -m pytest -m "not integration" --ignore=tests/integration/test_postgres.py
uv run --locked coverage report
uv run --locked --group docs mkdocs build --strict
