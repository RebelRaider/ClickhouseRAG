# Framework design

Users should be able to index a document and ask a grounded question using one
ClickHouse client and their existing model provider. Retain the passage store
for typed low-level access. Compose additional strategies around a common
evidence representation so each stage stays testable without a paid model.

- [LightRAG](https://arxiv.org/abs/2410.05779) combines entity and relationship
  signals with vector retrieval. Its [mode selection](https://github.com/HKUDS/LightRAG#selecting-query-modes)
  is a reference for letting callers combine passage and graph evidence.
- [Microsoft GraphRAG](https://arxiv.org/abs/2404.16130) uses communities and
  community summaries for corpus-wide questions. Local multi-hop retrieval is
  separate from corpus-wide summarization. Benchmark them on different queries.
- [OpenSPG KAG](https://arxiv.org/abs/2409.13731) uses structured knowledge and
  logical-form-guided reasoning. Support explicit domain schemas and a bounded
  query plan evaluated against cited facts. Model outputs should become typed
  requests, with no model-generated SQL execution.

## Graph storage

Store directed relationship assertions in a companion MergeTree-family table,
keyed by namespace, endpoints, predicate and passage identity/revision. Keep
multiple independent sources for the same relationship. Resolve visible passage
revisions before joining by chunk ID and revision. Apply caller metadata scope
to every supporting passage. A stale, absent, or filtered passage cannot serve
as an intermediate relationship.

The traversal takes explicit seeds or labels ranked by ClickHouse token overlap. Use a sorted ClickHouse
lookup per frontier with per-node and global limits. The query uses
[ClickHouse LIMIT BY](https://clickhouse.com/docs/reference/statements/select/limit-by)
and a [shared per-query snapshot](https://clickhouse.com/blog/clickhouse-release-25-06).
Traversal uses one snapshot per hop, so callers needing a fixed corpus across
all hops should read an immutable collection generation. Bound depth and
fanout, and suppress paths that revisit an entity. Estimate the tradeoff
against a single recursive CTE on larger, skewed graphs before changing it.

Changes to text or extracted assertions require a higher passage revision.
Writing the same chunk and assertion payload again is replayable. An assertion
becomes queryable only once a matching passage revision is present.

## Evaluation

Use labelled cases for exact lookup, paraphrases, multi-hop answers, questions
over a corpus and questions where evidence is absent. Record corpus/model
versions, mode, budgets and raw per-case results. Compare ranking and answer
quality separately. Do not infer gains in retrieval from the generation model's
ability to guess answers.

## Work still in progress

The initial implementation supports local directed traversal and a typed
predicate chain. Next, add document lifecycle management so a replacement
supersedes all of its former chunks together, including facts and shorter
documents. Scope that change around the same revision invariants and add
concurrency and retry tests. Compare graph query plans on graphs with
high-degree nodes. Evaluate corpus-wide questions with separately generated
community evidence and include a representative labelled corpus before
promoting the framework release.
