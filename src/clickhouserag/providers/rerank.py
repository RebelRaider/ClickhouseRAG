"""Reranking adapters for Cohere-compatible servers, Voyage AI and TEI."""

import math
from collections.abc import Sequence
from typing import Any

from clickhouserag._structured import ModelResponseError
from clickhouserag._validation import nonblank
from clickhouserag.models import Chunk
from clickhouserag.providers._http import JsonEndpoint


def _ranked_indices(rows: Any, count: int, score_field: str) -> list[int]:
    if not isinstance(rows, list) or len(rows) != count:
        raise ModelResponseError("Reranker returned the wrong number of results")
    scores = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ModelResponseError("Reranker returned an invalid result")
        index, score = row.get("index"), row.get(score_field)
        if type(index) is not int or not 0 <= index < count or index in scores:
            raise ModelResponseError("Reranker returned an invalid or repeated index")
        try:
            valid_score = type(score) in (int, float) and math.isfinite(score)
        except OverflowError:
            valid_score = False
        if not valid_score:
            raise ModelResponseError("Reranker returned an invalid score")
        scores[index] = score
    return sorted(scores, key=lambda index: (-scores[index], index))


class CohereReranker:
    """Rerank with Cohere v2 or a compatible server such as vLLM."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.cohere.com/v2",
        timeout: int = 120,
    ) -> None:
        nonblank(model, "model")
        self._endpoint = JsonEndpoint(base_url, timeout=timeout, key=api_key)
        self._model = model

    def rank(self, query: str, passages: Sequence[Chunk]) -> list[int]:
        """Return candidate indexes sorted by the model's relevance score."""
        if not passages:
            return []
        result = self._endpoint.post(
            "/rerank",
            {
                "model": self._model,
                "query": query,
                "documents": [passage.text for passage in passages],
                "top_n": len(passages),
            },
        )
        return _ranked_indices(result.get("results"), len(passages), "relevance_score")


class VoyageReranker:
    """Rerank text using Voyage AI."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = "https://api.voyageai.com/v1",
        timeout: int = 120,
    ) -> None:
        nonblank(model, "model")
        nonblank(api_key, "api_key")
        self._endpoint = JsonEndpoint(base_url, timeout=timeout, key=api_key)
        self._model = model

    def rank(self, query: str, passages: Sequence[Chunk]) -> list[int]:
        """Return candidate indexes without returning document data from the server."""
        if not passages:
            return []
        result = self._endpoint.post(
            "/rerank",
            {
                "model": self._model,
                "query": query,
                "documents": [passage.text for passage in passages],
                "top_k": len(passages),
                "truncation": False,
                "return_documents": False,
            },
        )
        return _ranked_indices(result.get("data"), len(passages), "relevance_score")


class TEIReranker:
    """Rerank with the native Hugging Face Text Embeddings Inference endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: int = 120,
    ) -> None:
        self._endpoint = JsonEndpoint(base_url, timeout=timeout, key=api_key)

    def rank(self, query: str, passages: Sequence[Chunk]) -> list[int]:
        """Return candidate indexes from the server's single deployed model."""
        if not passages:
            return []
        rows = self._endpoint.post_array(
            "/rerank",
            {
                "query": query,
                "texts": [passage.text for passage in passages],
                "truncate": False,
                "return_text": False,
            },
        )
        return _ranked_indices(rows, len(passages), "score")
