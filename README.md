# ClickHouse RAG

ClickHouse RAG is a Python retrieval layer for teams that already run ClickHouse.
Store passages and embeddings together. Search by meaning, term match, or both,
with namespace and metadata filters and a source pointer on every result.

- **Use your own embeddings.** Pass vectors directly or supply a Python provider.
- **Keep the latest revision searchable.** The same identity and payload can be
  retried and higher revisions replace earlier passages.
- **Return evidence for any RAG stack.** Typed search results carry stable
  chunk IDs, document IDs, sources and metadata.

Version 1.0 replaces the old managers and requires a new table. See the
[migration guide](docs/migration.md) for an existing setup.

## Quickstart

The 1.x line targets CPython 3.11–3.15 and ClickHouse 26.3 or 26.8 LTS.
Follow the [quickstart](docs/quickstart.md) to start a local server and run hybrid
retrieval without a model service. To run the example from a checkout with an
existing local server:

```sh
uv sync --locked
uv run --locked python examples/basic_usage.py
```

See the [API reference](docs/api.md), [operations guide](docs/operations.md),
[benchmark](docs/benchmark.md) and [design](docs/maintainer/design.md).

[Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md) · [MIT license](LICENSE)
