<img src="https://rebelraider.github.io/ClickhouseRAG/assets/clickhouserag-icon.png" width="96" height="96" alt="ClickHouse RAG logo">

# ClickHouse RAG

ClickHouse RAG is a Python framework for finding and citing evidence in
ClickHouse. Use one collection for document updates, hybrid text and vector
search, and relationship traversal. Compose local and hosted models for
embeddings, graph extraction, planning, reranking and cited answers.

- **Use your own embeddings.** Pass vectors directly or supply a Python provider.
- **Keep the latest revision searchable.** The same identity and payload can be
  retried and higher revisions replace earlier passages.
- **Return evidence for any RAG stack.** Typed search results carry stable
  chunk IDs, document IDs, sources and metadata.
- **Follow relationships across documents.** Graph search adds cited paths
  and corpus overviews. Typed knowledge queries can require multiple paths
  to reach the same answer.

Version 1.0 replaces the old managers and requires a new table. See the
[migration guide](https://rebelraider.github.io/ClickhouseRAG/migration/) for an existing setup.

[GitHub Releases](https://github.com/RebelRaider/ClickhouseRAG/releases/latest) ·
[Documentation](https://rebelraider.github.io/ClickhouseRAG/) ·
[PyPI](https://pypi.org/project/clickhouserag/)

## Quickstart

The 1.x line targets CPython 3.11–3.15 and ClickHouse 26.3 or 26.8 LTS.
Install the library with pip.

```sh
python -m pip install clickhouserag
```

Follow the [quickstart](https://rebelraider.github.io/ClickhouseRAG/quickstart/)
to start a local server and run hybrid retrieval without a model service.
To run the example from a checkout with an existing local server:

```sh
uv sync --locked
uv run --locked python examples/basic_usage.py
```

For document indexing, GraphRAG, KAG and evaluation, use the
[framework walkthrough](https://rebelraider.github.io/ClickhouseRAG/framework/).
The [model provider guide](https://rebelraider.github.io/ClickhouseRAG/providers/)
shows how to mix backends. The optional
[PostgreSQL add-on](https://rebelraider.github.io/ClickhouseRAG/postgres/)
delivers document revisions to ClickHouse.

See the [API reference](https://rebelraider.github.io/ClickhouseRAG/api/), [operations guide](https://rebelraider.github.io/ClickhouseRAG/operations/),
[benchmark](https://rebelraider.github.io/ClickhouseRAG/benchmark/) and [design](https://rebelraider.github.io/ClickhouseRAG/maintainer/design/).

[Contributing](https://github.com/RebelRaider/ClickhouseRAG/blob/main/CONTRIBUTING.md) · [Changelog](https://github.com/RebelRaider/ClickhouseRAG/blob/main/CHANGELOG.md) · [MIT license](https://github.com/RebelRaider/ClickhouseRAG/blob/main/LICENSE)
