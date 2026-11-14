"""Independent native Ollama embedding and chat components."""

from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

from clickhouse_connect.driver.client import Client

from clickhouserag._structured import ModelResponseError, StructuredLLM
from clickhouserag._validation import integer, nonblank
from clickhouserag.knowledge import Predicate
from clickhouserag.providers._chat import model_text
from clickhouserag.providers._embeddings import TextEmbeddings
from clickhouserag.providers._http import JsonEndpoint
from clickhouserag.rag import RAG

_DEFAULT_URL = "http://127.0.0.1:11434"


class OllamaEmbeddings(TextEmbeddings):
    """Embed text with the native Ollama API."""

    def __init__(
        self,
        model: str,
        dimensions: int,
        *,
        base_url: str = _DEFAULT_URL,
        api_key: str | None = None,
        revision: str | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 64,
        timeout: int = 120,
    ) -> None:
        self._endpoint = JsonEndpoint(base_url, timeout=timeout, key=api_key)
        super().__init__(
            "ollama-native",
            model,
            dimensions,
            binding=(self._endpoint.url, query_prefix, document_prefix),
            revision=revision,
            batch_size=batch_size,
            max_batch=256,
        )
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix

    def _request_vectors(self, texts: list[str], *, query: bool) -> Any:
        prefix = self._query_prefix if query else self._document_prefix
        return self._endpoint.post(
            "/api/embed",
            {
                "model": self.model,
                "input": [prefix + text for text in texts],
                "dimensions": self.dimensions,
                "truncate": False,
            },
        ).get("embeddings")


class OllamaChat:
    """Generate structured output on a local Ollama server."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = _DEFAULT_URL,
        api_key: str | None = None,
        max_tokens: int = 2048,
        timeout: int = 120,
    ) -> None:
        nonblank(model, "model")
        integer(max_tokens, "max_tokens", 2**31 - 1)
        self._endpoint = JsonEndpoint(base_url, timeout=timeout, key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def complete_json(self, instructions: str, input_text: str, schema: dict[str, Any]) -> str:
        """Return JSON text from a complete native chat response."""
        response = self._endpoint.post(
            "/api/chat",
            {
                "model": self._model,
                "stream": False,
                "format": schema,
                "options": {"temperature": 0, "num_predict": self._max_tokens},
                "messages": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": input_text},
                ],
            },
        )
        message = response.get("message")
        if (
            response.get("done") is not True
            or response.get("done_reason") == "length"
            or not isinstance(message, dict)
        ):
            raise ModelResponseError("Ollama did not complete the requested JSON response")
        return model_text(message.get("content"))


class Ollama(StructuredLLM):
    """Configure all RAG model stages on one native Ollama server."""

    def __init__(
        self,
        *,
        embedding_model: str,
        chat_model: str,
        dimensions: int,
        url: str = _DEFAULT_URL,
        schema: tuple[Predicate, ...] = (),
        embedding_version: str | None = None,
        max_relations: int = 12,
        timeout: int = 120,
    ) -> None:
        if urlsplit(url).path not in ("", "/"):
            raise ValueError("Ollama URL must be an HTTP(S) origin without a path")
        super().__init__(
            OllamaChat(chat_model, base_url=url, timeout=timeout),
            schema=schema,
            max_relations=max_relations,
        )
        self._embedder = OllamaEmbeddings(
            embedding_model, dimensions, base_url=url, revision=embedding_version, timeout=timeout
        )
        self.dimensions = dimensions
        # Keep the collection manifest identity used by the original public facade.
        self.model_id = (
            f"ollama:{embedding_model}:{embedding_version or 'unversioned'}:{dimensions}"
        )

    def rag(
        self,
        client: Client,
        *,
        table: str = "rag_passages",
        database: str = "default",
        namespace: str = "default",
        chunk_size: int = 1600,
        chunk_overlap: int = 160,
    ) -> RAG:
        """Configure one collection with this adapter for all enabled RAG stages."""
        schema = tuple(self._schema.values()) or None
        return RAG(
            client,
            self,
            dimensions=self.dimensions,
            table=table,
            database=database,
            namespace=namespace,
            extractor=self,
            generator=self,
            schema=schema,
            planner=self if schema is not None else None,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document batches with the configured output dimension."""
        return self._embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed one query in the same space as the documents."""
        return self._embedder.embed_query(text)
