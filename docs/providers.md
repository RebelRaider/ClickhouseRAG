# Model providers

A collection has one embedding space. Choose independently which model
extracts relationships, proposes knowledge plans, or writes answers.
The library runs search and evidence validation in the application and
ClickHouse. Each adapter sends only the input required for its stage to
the configured model endpoint.

| API | Embeddings | Structured generation |
| --- | --- | --- |
| [Native Ollama](https://docs.ollama.com/capabilities/structured-outputs) | `OllamaEmbeddings` | `OllamaChat` |
| OpenAI-compatible HTTP, including [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/) and [TEI embeddings](https://huggingface.co/docs/text-embeddings-inference/en/quick_tour) | `OpenAICompatibleEmbeddings` | `OpenAICompatibleChat` or `OpenAICompatibleResponses`, where supported |
| [Claude Messages](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) | Pair with an embedder | `AnthropicMessages` |
| Gemini Developer API | [`GeminiEmbeddings`](https://ai.google.dev/api/embeddings) | [`GeminiChat`](https://ai.google.dev/api/generate-content) |
| Cohere v2 | [`CohereEmbeddings`](https://docs.cohere.com/reference/embed) | [`CohereChat`](https://docs.cohere.com/reference/chat) |
| [Voyage AI](https://docs.voyageai.com/reference/embeddings-api) | `VoyageEmbeddings` | Pair with a language model |

Chat adapters implement `JsonModel`. Wrap one with `StructuredLLM` to
use it for any combination of `extractor`, `planner` and `generator`.
Those roles also accept custom Python providers directly. Use an explicit
schema for typed extraction and KAG. The Ollama convenience adapter and
its original import remain available.

## Mix two or more providers

```python
import os
from clickhouserag import Predicate, RAG
from clickhouserag.providers import (
    GeminiChat,
    OllamaChat,
    OllamaEmbeddings,
    StructuredLLM,
)

schema = (
    Predicate("calls", "service", "service"),
    Predicate("writes", "service", "database"),
)
vectors = OllamaEmbeddings("embeddinggemma", 768)
local = StructuredLLM(OllamaChat("qwen2.5:1.5b"), schema=schema)
writer = StructuredLLM(
    GeminiChat(os.environ["GEMINI_RAG_MODEL"], api_key=os.environ["GEMINI_API_KEY"])
)

# client is an application-configured ClickHouse Connect client.
rag = RAG(
    client,
    vectors,
    dimensions=vectors.dimensions,
    table="runbooks",
    schema=schema,
    extractor=local,
    planner=local,
    generator=writer,
)
```

For a local example, start ClickHouse and Ollama and load EmbeddingGemma
and Qwen 2.5 1.5B as in the [framework quickstart](framework.md). Run
`uv run --locked python examples/composed_models.py`. It uses native
Ollama embeddings and independently configured Responses generation over
the same ClickHouse collection. It creates and drops a temporary demo table.
Set `RAG_EMBEDDING_URL` to an OpenAI-compatible base URL to replace the
embedding server independently. The example was verified with llama.cpp
serving EmbeddingGemma and Ollama serving Qwen via Responses.

## Configure the model and endpoint

- Pass the API base URL, including its version or proxy prefix where needed.
  For OpenAI-compatible APIs this is typically a URL ending in `/v1`.
  Use the provider's assigned model ID. Confirm that its specific endpoint
  and model support the requested operation and structured output format.
- Use `OpenAICompatibleChat` for Chat Completions. Set
  `json_mode="object"` on compatible servers that implement JSON mode only,
  and `token_parameter="max_tokens"` on servers using the older field.
  `OpenAICompatibleResponses` requires a Responses endpoint that supports
  structured output. Requests include `store=False`.
- For [Azure OpenAI v1](https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle),
  use the resource URL ending in `/openai/v1` and the deployment name as
  `model`. Set `auth="api-key"` for the Azure key header, or use the default
  `bearer` mode and pass a callable returning the current Entra token as
  `api_key`. The callable runs per request. This also allows token renewal
  for compatible gateways that accept bearer credentials.
- All adapters validate returned dimensions. Enable `request_dimensions=True`
  only for embedding models that let the caller choose the output dimension.
  Use `query_prefix` and `document_prefix` for embedding models that require
  different instructions for search and indexing. Gemini's `use_task_type`
  is intended for models with the corresponding retrieval task support.
  For a default TEI server set `batch_size=32` to match its client limit.
- Model identity for the new embedders includes the endpoint, model,
  dimensions and encoding settings. Set `revision` to pin a version or
  deployment revision. Use a new table when changing the embedding space.
  If switching an existing Ollama collection to a different embedding
  adapter, initialize a new table and re-index its source documents.
- Pass credentials from your application environment or secret manager.
  Authenticated HTTP endpoints outside loopback require HTTPS. Adapter
  errors omit provider response bodies. HTTP errors expose the status and
  `Retry-After` header. Retry behavior belongs to the calling application.
- A refusal, truncated response, invalid vector or invalid citation fails
  the current model operation. ClickHouse keeps the previously published
  revision if indexing fails before publication.

## Rerank search candidates

Pass `reranker=` to `RAG` with `CohereReranker`, `VoyageReranker`,
`TEIReranker`, or your own `Reranker`. Set `rerank_candidates` to the
number of scoped candidates to request from ClickHouse before taking the
final `limit`. The default candidate count is 32, and the actual request
uses at least the final limit. Check the chosen endpoint's batch limit.
In mixed mode, complete graph paths keep priority and reranking orders
the text candidates that can fill the remaining budget. Global and KAG
queries use their existing group and proof selection.

The reranker sees only candidates that passed namespace and metadata
filters. Built-in HTTP rerankers send the query and passage text. They
return input indices. ClickHouse remains the source of passage text,
identity and provenance. Revisions are checked again after reranking so
that a document removed during the model call cannot appear in the result.

## Drivers and capabilities

Use [OpenAI's structured output guide](https://developers.openai.com/api/docs/guides/structured-outputs)
for model capabilities across Chat Completions and Responses. The Ollama
native API and its OpenAI-compatible endpoints have separate feature sets.
The same applies to gateways and Azure or self-hosted deployments. Use the
contract supported by the selected engine. `StructuredLLM` also accepts
custom transports through the `JsonModel` protocol, including SDK-backed
authentication, or application providers can implement the RAG roles directly.
