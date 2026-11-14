# Contributing

Install uv 0.13.8 or later and CPython 3.14 for the default local checks.

```sh
uv sync --locked --group docs
bash scripts/check.sh
```

For changes to SQL, run `uv run --locked pytest -m integration` with the
connection variables from the documentation against both LTS servers. The
tests create and remove temporary tables and require permissions to stop
merges on those test tables. Use a dedicated test server.

Describe the use case and include reproducible evidence in your pull request.
Use Conventional Commits. A behavior change should include coverage at the
public interface. For retrieval performance changes, include query plans and
latency and ranking measurements where applicable.

See [maintainer release documentation](docs/maintainer/releases.md).
