"""Common validation for first-party text embedding adapters."""

import hashlib
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Any

from clickhouserag._structured import ModelResponseError
from clickhouserag._validation import _MAX_DIMENSIONS, integer, nonblank


class TextEmbeddings(ABC):
    def __init__(
        self,
        family: str,
        model: str,
        dimensions: int,
        *,
        binding: tuple[object, ...],
        revision: str | None,
        batch_size: int,
        max_batch: int,
    ) -> None:
        nonblank(model, "model")
        integer(dimensions, "dimensions", _MAX_DIMENSIONS)
        integer(batch_size, "batch_size", max_batch)
        if revision is not None:
            nonblank(revision, "revision")
        self.model = model
        self.dimensions = dimensions
        self._batch_size = batch_size
        fingerprint = hashlib.sha256(json.dumps((binding, revision)).encode()).hexdigest()[:24]
        self.model_id = f"{family}:{model}:{dimensions}:{fingerprint}"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents in input order using bounded batches."""
        if isinstance(texts, str):
            raise ValueError("texts must be a sequence of document strings")
        return [
            vector
            for batch in _batches(texts, self._batch_size)
            for vector in self._embed(batch, query=False)
        ]

    def embed_query(self, text: str) -> list[float]:
        """Embed one query with the provider's retrieval settings."""
        return self._embed([text], query=True)[0]

    def _embed(self, texts: list[str], *, query: bool) -> list[list[float]]:
        for text in texts:
            nonblank(text, "embedding input")
        result = self._request_vectors(texts, query=query)
        if not isinstance(result, list) or len(result) != len(texts):
            raise ModelResponseError("Model returned the wrong number of embedding vectors")
        return [_vector(vector, self.dimensions) for vector in result]

    @abstractmethod
    def _request_vectors(self, texts: list[str], *, query: bool) -> Any: ...


def _batches(texts: Sequence[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(texts), size):
        yield list(texts[start : start + size])


def _vector(vector: Any, dimensions: int) -> list[float]:
    if not isinstance(vector, list) or len(vector) != dimensions:
        raise ModelResponseError(f"Model embedding must have {dimensions} dimensions")
    try:
        numbers = [float(value) for value in vector if type(value) in (int, float)]
    except OverflowError as exc:
        raise ModelResponseError("Model returned non-finite embedding values") from exc
    if len(numbers) != dimensions or not all(map(math.isfinite, numbers)) or not any(numbers):
        raise ModelResponseError("Model returned an invalid or zero embedding vector")
    return numbers


def indexed_vectors(rows: Any, count: int) -> list[Any]:
    """Restore input order and reject ambiguous or incomplete indexed responses."""
    if not isinstance(rows, list) or len(rows) != count:
        raise ModelResponseError("Model returned the wrong number of embedding rows")
    indexed: dict[int, Any] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ModelResponseError("Model returned an invalid embedding row")
        index = row.get("index")
        if type(index) is not int or not 0 <= index < count or index in indexed:
            raise ModelResponseError("Model returned an invalid or repeated embedding index")
        indexed[index] = row.get("embedding")
    return [indexed[index] for index in range(count)]
