"""Public types for passages and search results."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


class SchemaError(RuntimeError):
    """The configured ClickHouse table is missing or incompatible."""


@dataclass(frozen=True, slots=True)
class Chunk:
    """A passage with a stable identity, provenance and a write revision."""

    chunk_id: str
    document_id: str
    text: str
    source: str
    revision: int = 1
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """A passage paired with its document embedding."""

    chunk: Chunk
    embedding: Sequence[float]


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A passage and its score, ordered by the selected search strategy."""

    chunk: Chunk
    score: float
