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

The initial implementation supports local directed traversal, a typed predicate
chain, and document revisions with logical deletion. Compare query plans on
graphs with high-degree nodes. Evaluate corpus-wide questions with community
evidence and include a representative labelled corpus before promoting the
framework release.


### Verification baseline

The bundled four-document synthetic example uses a controlled embedding,
extractor and generator. On the multi-hop question, dense, lexical and
hybrid retrieval returned no relevant evidence in their first two results.
Graph, mixed and explicitly planned KAG returned both supporting passages.
The example reports metrics by category and makes the synthetic setup
reproducible. Compare a representative corpus before using these results
to choose a mode for production.

A separate local smoke test used Ollama 0.40.16, EmbeddingGemma (768 output
dimensions), Qwen 2.5 1.5B, and ClickHouse 26.3.62.3. The example returned
`Vault` with both sources in mixed and model-planned KAG. Rephrasing the
question to mention the intermediate service explicitly caused that model to
produce an invalid plan and abstain. Treat the example as an end-to-end
integration check. The planner still needs paraphrase and ambiguity evaluation.

The expanded offline fixture adds a shared endpoint and a second community.
The two-hop lookup retrieves both references in graph and mixed mode and
one of the two with dense, lexical and hybrid at a two-passage budget. The
planned KAG intersection cites all four references. The corpus overview
finds both topics at a four-passage budget and retrieves two of the four
labelled passages. Different passages in the first topic compete for that
same budget. See the [benchmark guide](../benchmark.md) for the comparison
on public text.

### Document publication

The document facade uses a managed passage table. It stages immutable passages
and their extracted facts in the same MergeTree table, then inserts a small
publication record. A query selects the highest published document revision
before applying metadata filters. A deletion publishes a revision without
passages. This also hides assertions on removed chunks and metadata changed
by a replacement. The low-level passage and relation stores retain their
original layouts and operations.

A single ClickHouse statement shares its part snapshot across all references to
the managed table. In the facade, multi-hop and mixed retrieval also recheck
assembled evidence against the published revisions. An update during retrieval
may result in fewer returned passages or paths. Visibility is defined for
queries on the same ClickHouse server. Physical retention of superseded data
is a separate operation.


### Global retrieval

The global strategy groups current, scoped relation assertions into connected
components at query time. It ranks components by query token matches, distinct
documents and distinct facts. It samples supporting passages across groups
and source documents. This design draws on
[GraphRAG's global query model](https://microsoft.github.io/graphrag/query/global_search/).
It uses connected components with citations instead of hierarchical clustering
or persisted LLM summaries. The evidence budget makes this a bounded
overview. Results expose truncation rather than presenting a partial graph as
the entire corpus.
