"""Voyage AI text embeddings."""

from typing import Any

from clickhouserag._validation import nonblank
from clickhouserag.providers._embeddings import TextEmbeddings, indexed_vectors
from clickhouserag.providers._http import JsonEndpoint


class VoyageEmbeddings(TextEmbeddings):
    """Use Voyage retrieval-specific query and document embeddings."""

    def __init__(
        self,
        model: str,
        dimensions: int,
        *,
        api_key: str,
        base_url: str = "https://api.voyageai.com/v1",
        revision: str | None = None,
        request_dimensions: bool = False,
        batch_size: int = 64,
        timeout: int = 120,
    ) -> None:
        nonblank(api_key, "api_key")
        self._endpoint = JsonEndpoint(base_url, timeout=timeout, key=api_key)
        super().__init__(
            "voyage",
            model,
            dimensions,
            binding=(self._endpoint.url, request_dimensions),
            revision=revision,
            batch_size=batch_size,
            max_batch=1000,
        )
        self._request_dimensions = request_dimensions

    def _request_vectors(self, texts: list[str], *, query: bool) -> list[Any]:
        body = {
            "model": self.model,
            "input": texts,
            "input_type": "query" if query else "document",
            "truncation": False,
        }
        if self._request_dimensions:
            body["output_dimension"] = self.dimensions
        result = self._endpoint.post("/embeddings", body)
        return indexed_vectors(result.get("data"), len(texts))
