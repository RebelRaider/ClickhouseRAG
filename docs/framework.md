# Framework quickstart

The framework API uses one ClickHouse HTTP client and provisions one managed
table for document revisions, passages, and their relationships. The same
collection and metadata filters apply to text search, graph expansion and
reasoning.

Start a compatible ClickHouse server as in the [quickstart](quickstart.md).
In a checkout of this version, run the offline example. It uses deterministic
fixtures for embeddings, extraction and generation.

```sh
uv run --locked python examples/graph_rag.py
uv run --locked python examples/graph_rag.py --evaluate
```

The program indexes two documents, returns a two-hop answer with links to
both supporting documents, and runs a structured query against typed
relationships. The evaluation flag adds distractors, a second service chain
to the same database, and an independent group. It compares dense, lexical,
hybrid, graph and mixed search, and exercises KAG intersections and corpus
overviews on their own labelled questions. The bundled evaluation uses
synthetic text and controlled providers. The [benchmark guide](benchmark.md)
describes the public HotpotQA comparison.

## Use your models

Pass an `EmbeddingProvider` and its vector dimension to `RAG`. Add a
`RelationExtractor` for documents that should populate the graph. The adapter
extracts triples from a specific passage and can attach endpoint types. The
caller supplies the model API, timeouts and credentials. Pass an
`AnswerProvider` to use `ask`; the provider gets a context and returns text
and evidence IDs or an explicit abstention. Unknown citations are rejected.
Citation validation checks that the references were supplied to the generator.
It does not judge whether the generated claim entails the evidence.

- `index(Document(...))` chunks a document by character count and overlap,
  batches the embeddings, and publishes the new passages and assertions
  together. Reusing the same document ID with changed input supersedes the
  entire previous revision. Repeating the current payload returns zero and
  skips the models. `delete(document_id)` hides the document and its facts.
- `retrieve(question)` defaults to hybrid text and vector search. Choose
  `mode="mixed"` to expand matching entity labels through directed
  relationships and fill the remaining budget with hybrid search. Use
  `seeds` in `mixed` or `graph` mode for explicit entity IDs. To use existing
  relationships without an extractor, open the collection with `graph=True`.
- `reason(KnowledgeQuery(...))` follows up to four predicates in order. Pass
  `schema=(Predicate(...), ...)` to enable it. It requires matching stored
  endpoint types and returns the endpoints and complete supporting paths.
  Empty evidence means no matching path was returned within the configured
  traversal bounds. Pass a `KnowledgePlanner` and call `ask_knowledge(question)`
  to translate a natural-language question to a typed plan before reasoning.
  Plans are validated against the schema and executed with bound parameters.
  An invalid chain proposed by a planner produces an abstention. If your
  application has a plan, pass it to `ask_knowledge(question, plan=plan)` for
  a cited answer or to `reason(plan)` for the endpoints and paths. An invalid
  caller-supplied plan raises `KnowledgePlanError`.
  Use `KnowledgeIntersection((path_a, path_b))` to find endpoints reachable
  from both paths. An answer must cite all passages in at least one complete
  proof for an endpoint. Raise the context limit if the proof spans more passages.
  The Ollama planner can propose either form. The framework validates types
  and paths. For questions that require precise semantics, supply a plan
  directly or evaluate the chosen model against your domain's questions.
- `retrieve(question, mode="global")` broadens the context beyond matching
  one entity. It groups the scoped facts into connected components and
  balances citations across groups and documents. `ask` can summarize
  those groups using the same answer provider. For bounded corpus reviews,
  `overview(question, max_facts=...)` exposes a `truncated` flag when the fact
  scan exhausts its budget. A corpus dominated by one connected component
  needs finer clustering for a useful topical split.

## Updates and deployment

Treat document IDs as stable within a namespace. For multiple writers or
replayed events, pass a monotonically increasing source revision as
`Document(..., revision=n)` or `delete(id, revision=n)`. An older explicit
revision raises `DocumentRevisionError`. Repeating the latest revision and
payload is a no-op and a conflicting payload at the same visible revision
raises an error. If two different payloads race before either head is published,
ClickHouse deterministically chooses one generation at that revision. A
same-revision deletion wins over a live generation. Supply unique increasing
source revisions for intentional ordering across concurrent writers.

Publication follows successful staging. A failed operation can leave hidden
parts that consume storage. Logical deletion keeps stored history; manage
physical retention separately. On a single ClickHouse server, a fresh read
can see the whole old or whole new document. Queries that span several
statements recheck candidate evidence, which can reduce recall if a revision
changes during retrieval. Replica-wide visibility depends on deployment and
read routing. Provision a new table for the managed API when coming from the
low-level passage schema or the framework prototype, then re-index source
documents. The low-level `ClickHouseStore` still reads its own tables.

## Evaluate against your corpus

Use `EvalCase` to label each question with the set of relevant passage IDs,
an optional set of accepted exact answers and a category. An empty relevance
set labels an unanswerable case. `evaluate` takes a callable returning
`Context` or `Answer`; it reports Recall@k, MRR@k, binary nDCG@k, plus exact
answer matching and citation metrics when available. It preserves the
per-case scores, reports aggregates by category, and measures wall-clock latency. Exact matching uses
case-folding and normalizes whitespace. For free-form answers use a separate
claim-level human or model-based judging process.

Keep model and corpus versions fixed across retrieval modes. Evaluate with a
representative distribution of lookups, paraphrases, multi-hop and unanswerable
questions. Measure project-appropriate p95 latency and report any truncated
paths, retries or provider failures.

## Model providers

See the [model provider guide](providers.md) for mixing local and hosted
embedding, extraction, planning and generation APIs.

## Local models with Ollama

Use the supplied adapter to try arbitrary text with a local model server.
Choose installed embedding and chat models; the chat model must support
[structured outputs](https://docs.ollama.com/capabilities/structured-outputs).
Set the dimension to a size supported by the embedding model.

```python
import clickhouse_connect

from clickhouserag import Document, Predicate
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
    rag = model.rag(client, table="service_runbooks")
    rag.initialize()
    rag.index(Document("gateway", "Gateway calls Ledger.", "https://example.org/gateway"))
    rag.index(Document("ledger", "Ledger writes Vault.", "https://example.org/ledger"))
    answer = rag.ask("What database is written to by a service that Gateway calls?", mode="mixed")
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
quality depends on the schema, corpus and selected models. The adapter requests
up to twelve relationships per passage by default. Set `max_relations` to
adjust extraction work for the density of your passages. When the model fails
to complete a response, the document stays at its previous published revision.

Use `reason(plan)` when the endpoint and its paths are sufficient. It does not
invoke the answer model. In the local example Qwen 2.5 1.5B returned the two-hop
answer in mixed mode. With full proof citation checks enabled, the generated
KAG answer abstained while the explicit plan still resolved the endpoint.

## Customize document preparation

Pass a `Chunker` to `RAG` when you need section-based, token-based or semantic
splitting. Its `split(text)` method yields nonblank passages in source order.
ClickHouse RAG assigns the document ID, source URI, metadata and revision to
each passage. With a custom chunker you must provide `index_revision`.
Set an `index_revision` when changing the extraction model or schema too.
This lets a changed indexing pipeline republish an otherwise identical
document when using automatic revision assignment. With explicit revisions,
enqueue a new revision for each affected source document.

## Combine several searches in one answer

Use `retrieve_many` for questions that benefit from multiple search terms or
strategies. The application can supply a known plan, user-selected searches,
or validated model-generated rewrites. The original question is passed to the
answer provider separately from the terms used for search.

```python
from clickhouserag import SearchQuery

searches = (
    SearchQuery("retention exceptions", mode="hybrid"),
    SearchQuery("ERR-1044", mode="lexical"),
    SearchQuery("archive service", mode="graph", seeds=("Archive",)),
)
answer = rag.ask_many(
    "When can we override the archive retention policy?",
    searches,
    metadata={"team": "operations"},
    candidates_per_query=12,
    limit=8,
)
```

Up to eight searches share one namespace and metadata scope. The framework
fuses the ranked passages, checks current document revisions across all
searches, and gives complete graph paths priority within the final budget.
Corpus-wide summaries and KAG proofs remain available through `overview`
and `reason` or `ask_knowledge`.

For model-driven search, pass `search_planner=StructuredLLM(backend)` and
call `rag.ask_planned(question)`. The planner sees the question and
available retrieval modes. Namespace and metadata come from the application.
The built-in model planner can propose up to four search terms. An empty
plan uses one hybrid search with the original question. Malformed plans
raise a model error. Each query result goes through the same revision and
citation checks as an explicitly composed plan.
