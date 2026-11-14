# ClickHouse RAG

A Python retrieval layer for applications that already store data in ClickHouse.
Write passages with embeddings, search within a namespace and metadata scope,
and get the document and source for every result. Use your own embedding model
and feed the returned passages into any application or language model.

The 1.0 rewrite is in prerelease development. The published 0.2 package has a
different public API.

```bash
uv sync --locked
uv run python examples/basic_usage.py
```

The example uses a local ClickHouse HTTP endpoint and precomputed toy vectors.

See the [design](docs/maintainer/design.md) and [release plan](PLANS.md).
Released under the [MIT license](LICENSE).
