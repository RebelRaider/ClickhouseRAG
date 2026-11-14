# Design for ClickHouse RAG 1.0

## Baseline

GitHub main and the last local upstream reference pointed to `219dc74` on
2026-11-14. PyPI lists 0.2.0. Main's 0.3.0 declaration is ahead of the published
version. The README points to modules that have since moved. The package exposes
a ClickHouse wrapper, arbitrary SQL, database backups and a vectorizer registry.
It interpolates schema and identifiers, and its bulk vectorizer silently truncates
mismatched outputs through `zip`. The similarity query has a broken projection,
reimplements cosine similarity, and raises the query-size setting substantially.
CI uses one Python version and an unpinned legacy image. The test fixtures patch
driver methods, so that service does not prove the retrieval SQL works. Releases
depend on a long-lived PyPI secret.

## Audience and value

Start with Python teams storing logs, events, catalog data or documents in ClickHouse
who want their RAG application to find evidence without operating a separate vector
database. An independently useful package should offer more than a wrapper for
`SELECT … ORDER BY cosineDistance(…) LIMIT …`. The useful combination is a typed
contract for batch indexing, filter-scoped semantic and lexical search, traceable
passages, and operational examples that users can run and measure.

[ClickHouse's own vector SQL](https://clickhouse.com/docs/reference/engines/table-engines/mergetree-family/annindexes)
is enough for a single customized query. [LlamaIndex's ClickHouseVectorStore](https://developers.llamaindex.ai/python/framework-api-reference/storage/vector_store/clickhouse/)
already handles common vector-store operations for LlamaIndex users. The proposed
advantage is control of retrieval quality and ClickHouse operations through a
small, framework-independent API. This is a positioning hypothesis. Reproducible
comparisons, usable docs, and feedback from actual users will test it.

## Retrieval choices

| Approach | Suitable workload | Decision |
| --- | --- | --- |
| Exact dense retrieval | Paraphrases and semantic similarity, filtered and small corpora, baseline for ANN | Stable baseline. Use the server's native cosine distance and store fixed-dimension `Array(Float32)`. |
| Lexical retrieval | Identifiers, error codes and rare terms | Use Unicode-normalized tokens and the production text index. Rank by query token coverage. Document that score separately from dense distance. |
| Hybrid retrieval | Queries that mix vocabulary and semantics | Fuse bounded candidate lists using [reciprocal rank fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf). Use the same scope and filters for each leg. |
| Approximate dense retrieval | Large collections with tight latency budgets | Treat HNSW as a separate query and index strategy. Compare individual filtering plans and Recall@k with the exact baseline before giving it a supported API. ClickHouse's ordering, candidate limits and filter strategy affect index use. |
| GraphRAG | Entity relationships, multi-hop evidence and questions across a corpus | Microsoft GraphRAG extracts entities and relationships and uses community summaries for global questions. Keep passage identities and sources available for a future graph adapter. Promote graph retrieval only with multi-hop and corpus-level tests. |
| KAG | Professional domain questions requiring structured facts, rules and logical reasoning | Keep a domain schema, entity linking, logical plans and reasoning in an adapter or separate project. Use typed passages and citations to supply text evidence. |

[Original RAG paper](https://arxiv.org/abs/2005.11401) ·
[GraphRAG paper](https://arxiv.org/abs/2404.16130) ·
[KAG paper](https://arxiv.org/abs/2409.13731) ·
[ClickHouse text indexes](https://clickhouse.com/docs/reference/engines/table-engines/mergetree-family/textindexes)

## Storage and API

- A store binds to one named table and one embedding model identity and dimension.
  Use a new table for a new model or schema. The app provides the ClickHouse Connect
  client and controls credentials, TLS settings and connection lifecycle.
- A stable namespace and chunk ID identify a passage. A document ID and source
  identify its origin. Store string metadata for exact filtering. The library
  accepts existing chunks and vectors, and offers a protocol for application-owned
  embedding providers. Ingestion does not require a generation provider.
- Group writes into bounded batches. Validate all rows in each batch before
  submitting it. Same-ID retries and replacements use explicit row revisions and
  documented read semantics. Tests cover replacement before merges and changes to
  metadata. Keep table creation and migration explicit.
- Use parameterized values and constrain identifiers to a documented grammar.
  Read under a ClickHouse user with only the appropriate SELECT grant. A namespace
  is an application filter. Use privileges and row policies for access control.
- Return ranked passages, score semantics and provenance. Keep raw text and
  embedding contents out of application logging inside the library.

[ClickHouse Connect driver API](https://clickhouse.com/docs/integrations/language-clients/python/driver-api)
· [ClickHouse bulk inserts](https://clickhouse.com/docs/optimize/bulk-inserts)
· [ReplacingMergeTree](https://clickhouse.com/docs/reference/engines/table-engines/mergetree-family/replacingmergetree)

## Python and delivery

Use one PEP 621 project definition, a `src/` layout, an inline typing marker and
uv's lockfile for reproducible development. Target supported stable CPython
3.11–3.15 as of November 2026. Exercise the exact selected driver and ClickHouse
versions in CI. Evaluate linting and formatting with Ruff, type checking with ty,
source scanning with Bandit, dependency auditing, install tests for built
artifacts, strict documentation builds and workflow linting.

Use Conventional Commits for SemVer, with a major release for the 0.x API
replacement. Build and check artifacts before publication. Use a protected
GitHub environment and PyPI Trusted Publishing with OIDC. Attach verified files
to GitHub Releases. Publish the documentation with GitHub Pages, and add
contribution and security guidance, issue forms and dependency update automation.

[Python lifecycle](https://devguide.python.org/versions/)
· [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
· [src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
· [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
· [Python Semantic Release](https://python-semantic-release.readthedocs.io/en/latest/)
· [PyPI release workflows](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
· [GitHub Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)

## Evaluation

Report Recall@k and MRR by query type, with a fixed labelled corpus and a documented
embedder. Add p50/p95 retrieval latency, row count, dimensions and cache/warmup
conditions. Include scope and replacement invariants in integration tests. Compare
exact and hybrid approaches and keep relevance claims limited to measured data.
