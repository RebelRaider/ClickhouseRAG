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

## Local models with Ollama

Use the supplied adapter to try arbitrary text with a local model server.
Choose installed embedding and chat models; the chat model must support
[structured outputs](https://docs.ollama.com/capabilities/structured-outputs).
Set the dimension to a size supported by the embedding model.

```python
import clickhouse_connect

from clickhouserag import RAG, Document, Predicate
from clickhouserag.ollama import Ollama

schema = (
    Predicate("calls", "service", "service"),
    Predicate("writes", "service", "database"),
)
model = Ollama(
    embedding_model="embeddinggemma", chat_model="qwen2.5:1.5b", dimensions=768, schema=schema
)
with clickhouse_connect.get_client(
    host="127.0.0.1",
    autogenerate_session_id=False,
    form_encode_query_params=True,
) as client:
    rag = RAG(
        client,
        model,
        dimensions=model.dimensions,
        table="service_runbooks",
        extractor=model,
        generator=model,
        schema=schema,
        planner=model,
    )
    rag.initialize()
    rag.index(Document("gateway", "Gateway calls Ledger.", "https://example.org/gateway"))
    rag.index(Document("ledger", "Ledger writes Vault.", "https://example.org/ledger"))
    answer = rag.ask_knowledge("Which database does Gateway write to through Ledger?")
    print(answer.text, [chunk.source for chunk in answer.citations])
```

The packaged example also accepts the ClickHouse connection variables from
the quickstart and `OLLAMA_URL` for a nondefault local endpoint.

```sh
ollama pull embeddinggemma
ollama pull qwen2.5:1.5b
uv run --locked python examples/ollama_rag.py
```

The adapter uses the configured Ollama origin directly and sends input text
to it. Use `embedding_version` to distinguish model revisions if reusing a tag.
Change the table when moving to a different embedding space. The local
offline example tests the RAG pipeline independently of Ollama. Model output
quality depends on the schema, corpus and selected models.
