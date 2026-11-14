"""Bridge an application-owned embedding model to a ClickHouse store."""

from collections.abc import Mapping, Sequence
from typing import Protocol

from clickhouserag._validation import _MAX_BATCH, chunk_fields, nonblank
from clickhouserag.models import Chunk, EmbeddedChunk, SearchHit
from clickhouserag.store import ClickHouseStore


class EmbeddingProvider(Protocol):
    """Application-supplied encoder with separate document and query methods."""

    @property
    def model_id(self) -> str:
        """Identity of the model and encoding configuration."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Encode the passages in input order."""
        ...

    def embed_query(self, text: str) -> Sequence[float]:
        """Encode a search query in the same vector space."""
        ...


class Retriever:
    """Encode and retrieve passages using a specific embedding provider."""

    def __init__(self, store: ClickHouseStore, provider: EmbeddingProvider) -> None:
        if store.embedding_model != provider.model_id:
            raise ValueError("provider model_id must match the store embedding_model")
        self._store = store
        self._provider = provider

    def index(self, chunks: Sequence[Chunk], *, namespace: str = "default") -> int:
        """Embed and insert a batch of passages."""
        nonblank(namespace, "namespace")
        if not chunks:
            return 0
        if len(chunks) > _MAX_BATCH:
            raise ValueError(f"index accepts at most {_MAX_BATCH} chunks per batch")
        for chunk in chunks:
            chunk_fields(chunk)
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("chunk IDs must be unique within a batch")
        vectors = self._provider.embed_documents([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("embedding provider returned the wrong number of document vectors")
        return self._store.insert(
            (EmbeddedChunk(chunk, vector) for chunk, vector in zip(chunks, vectors, strict=True)),
            namespace=namespace,
        )

    def search(
        self,
        query: str,
        *,
        namespace: str = "default",
        metadata: Mapping[str, str] | None = None,
        limit: int = 10,
        hybrid: bool = True,
    ) -> list[SearchHit]:
        """Embed a query and perform hybrid or exact vector retrieval."""
        nonblank(query, "query")
        embedding = self._provider.embed_query(query)
        if hybrid:
            return self._store.search_hybrid(
                query, embedding, namespace=namespace, metadata=metadata, limit=limit
            )
        return self._store.search_vector(
            embedding, namespace=namespace, metadata=metadata, limit=limit
        )
