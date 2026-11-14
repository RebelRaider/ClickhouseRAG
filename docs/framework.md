# Framework quickstart

The framework API uses one ClickHouse HTTP client. It creates a passage table
and, when configured for graph retrieval or reasoning, an adjacent relationship
table. The same collection and metadata filters apply to text search, graph
expansion, and reasoning.

Start a compatible ClickHouse server as in the [quickstart](quickstart.md).
In a checkout of this version, run the offline example. It uses deterministic
fixtures for embeddings, extraction and generation.

```sh
uv run --locked python examples/graph_rag.py
uv run --locked python examples/graph_rag.py --evaluate
```

The program indexes two documents, returns a two-hop answer with links to
both supporting documents, and runs a structured query against typed
relationships. The evaluation flag adds two distractors and compares dense,
lexical, hybrid, graph, and mixed search. KAG receives typed plans for the
multi-hop and no-answer cases. Tests on this synthetic corpus verify the
flow and metric behavior. They do not estimate quality on arbitrary text.

## Use your models

Pass an `EmbeddingProvider` and its vector dimension to `RAG`. Add a
`RelationExtractor` for documents that should populate the graph. The adapter
extracts triples from a specific passage and can attach endpoint types. The
caller supplies the model API, timeouts and credentials. Pass an
`AnswerProvider` to use `ask`; the provider gets a context and returns text
and evidence IDs or an explicit abstention. Unknown citations are rejected.
Citation validation checks that the references were supplied to the generator.
It does not judge whether the generated claim entails the evidence.

- `index(Document(...))` chunks a text snapshot by character count and overlap,
  batches its embeddings, and stores its passages and optional assertions.
  Identical input with deterministic providers can be retried. Changed input
  creates a new snapshot alongside the previous one. Use a new namespace
  for an exclusively current corpus; the current facade has no document
  replacement or deletion API.
- `retrieve(question)` defaults to mixed search. It selects matching entity
  labels and follows directed relationships, then fills the remaining
  context budget with hybrid text search. Use `seeds` for explicit entity IDs. Open a collection for reading with
  `graph=True` to use its existing relationships without an extractor.
  Without graph configuration mixed retrieval uses hybrid text search alone.
- `reason(KnowledgeQuery(...))` follows up to four predicates in order. Pass
  `schema=(Predicate(...), ...)` to enable it. It requires matching stored
  endpoint types and returns the endpoints and complete supporting paths.
  Empty evidence means no matching path was returned within the configured
  traversal bounds. Pass a `KnowledgePlanner` and call `ask_knowledge(question)`
  to translate a natural-language question to a typed plan before reasoning.
  Plans are validated against the schema and executed with bound parameters.
  Corpus-wide community summaries are described in the
  [framework design](maintainer/framework.md).

## Evaluate against your corpus

Use `EvalCase` to label each question with the set of relevant passage IDs,
an optional set of accepted exact answers and a category. An empty relevance
set labels an unanswerable case. `evaluate` takes a callable returning
`Context` or `Answer`; it reports Recall@k, MRR@k, binary nDCG@k, plus exact
answer matching and citation metrics when available. It preserves the
per-case category and reports wall-clock latency. Exact matching uses
case-folding and normalizes whitespace. For free-form answers use a separate
claim-level human or model-based judging process.

Keep model and corpus versions fixed across retrieval modes. Evaluate with a
representative distribution of lookups, paraphrases, multi-hop and unanswerable
questions. Measure project-appropriate p95 latency and report any truncated
paths, retries or provider failures.
