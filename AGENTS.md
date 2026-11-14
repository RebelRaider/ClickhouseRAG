# ClickHouse RAG

Read [PLANS.md](PLANS.md), the [design](docs/maintainer/design.md), and the
[operation and release docs](docs/operations.md) before changing the public API,
storage schema, or workflows.

- Public types and search methods are exported from `src/clickhouserag`.
  Preserve namespace scope, query-time revision semantics and provenance.
  Keep SQL values bound and image versions visible in CI.
- Start with `uv sync --locked --group docs` and `bash scripts/check.sh`.
  Run the real ClickHouse tests on both versions from the CI matrix for SQL
  changes. Use an isolated test server and keep production credentials out
  of examples, logs, tests and source.
- Build and install wheel and sdist for packaging changes. Document behavior
  changes at the public entry point.
- Use Conventional Commits and follow the
  [release runbook](docs/maintainer/releases.md). Releases use exact source tags
  and checked build artifacts.
