#!/usr/bin/env bash
set -euo pipefail

repo="$(pwd)"
output="$(mktemp -d)"
trap 'rm -rf "$output"' EXIT
uv build --out-dir "$output"
uv run --locked twine check --strict "$output"/*
for artifact in "$output"/*.whl "$output"/*.tar.gz; do
    (
        cd "$output"
        uv run --no-project --isolated --python 3.14 --with "$artifact" python "$repo/examples/basic_usage.py"
    )
done
