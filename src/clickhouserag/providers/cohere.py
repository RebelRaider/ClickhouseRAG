"""Cohere Embed and Chat v2."""

from typing import Any

from clickhouserag._structured import ModelResponseError
from clickhouserag._validation import integer, nonblank
from clickhouserag.providers._chat import portable_schema, text_blocks
from clickhouserag.providers._embeddings import TextEmbeddings
from clickhouserag.providers._http import JsonEndpoint

_BASE_URL = "https://api.cohere.com/v2"


class CohereEmbeddings(TextEmbeddings):
    """Use Cohere retrieval-specific document and query embeddings."""

    def __init__(
        self,
        model: str,
        dimensions: int,
        *,
        api_key: str,
        base_url: str = _BASE_URL,
        revision: str | None = None,
        request_dimensions: bool = False,
        batch_size: int = 64,
        timeout: int = 120,
    ) -> None:
        nonblank(api_key, "api_key")
        self._endpoint = JsonEndpoint(base_url, timeout=timeout, key=api_key)
        super().__init__(
            "cohere",
            model,
            dimensions,
            binding=(self._endpoint.url, request_dimensions),
            revision=revision,
            batch_size=batch_size,
            max_batch=96,
        )
        self._request_dimensions = request_dimensions

    def _request_vectors(self, texts: list[str], *, query: bool) -> Any:
        body = {
            "model": self.model,
            "texts": texts,
            "embedding_types": ["float"],
            "input_type": "search_query" if query else "search_document",
            "truncate": "NONE",
        }
        if self._request_dimensions:
            body["output_dimension"] = self.dimensions
        result = self._endpoint.post("/embed", body).get("embeddings")
        return result.get("float") if isinstance(result, dict) else None


class CohereChat:
    """Generate structured output with Cohere Chat v2."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = _BASE_URL,
        max_tokens: int = 4096,
        timeout: int = 120,
    ) -> None:
        nonblank(model, "model")
        nonblank(api_key, "api_key")
        integer(max_tokens, "max_tokens", 2**31 - 1)
        self._endpoint = JsonEndpoint(base_url, timeout=timeout, key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def complete_json(self, instructions: str, input_text: str, schema: dict[str, Any]) -> str:
        """Return text from a completed Cohere turn."""
        result = self._endpoint.post(
            "/chat",
            {
                "model": self._model,
                "max_tokens": self._max_tokens,
                "messages": [
                    {"role": "system", "content": instructions + "\nReturn a JSON object."},
                    {"role": "user", "content": input_text},
                ],
                "response_format": {"type": "json_object", "schema": portable_schema(schema)},
            },
        )
        message = result.get("message")
        if result.get("finish_reason") != "COMPLETE" or not isinstance(message, dict):
            raise ModelResponseError("Cohere did not complete the requested JSON response")
        return text_blocks(message.get("content"))
