"""ClickHouse storage and exact semantic retrieval."""

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from clickhouse_connect.driver.client import Client

from clickhouserag._lexical import _MAX_QUERY_TERMS, terms
from clickhouserag._ranking import reciprocal_rank_fusion
from clickhouserag._validation import (
    _MAX_BATCH,
    _MAX_DIMENSIONS,
    _MAX_LIMIT,
    chunk_fields,
    identifier,
    integer,
    nonblank,
    string_map,
    unit_vector,
)
from clickhouserag.models import Chunk, EmbeddedChunk, SchemaError, SearchHit

_INSERT_COLUMNS = (
    "namespace",
    "chunk_id",
    "document_id",
    "text",
    "source",
    "revision",
    "metadata",
    "embedding",
    "lexical_terms",
)
_SELECT_SETTINGS = {
    "optimize_move_to_prewhere_if_final": 0,
    "use_skip_indexes_if_final_exact_mode": 1,
    "enable_shared_storage_snapshot_in_query": 1,
}
_PassageRow = tuple[str, str, str, str, str, int, dict[str, str], list[float], list[str]]


class ClickHouseStore:
    """A table of versioned passages using an application-owned ClickHouse client."""

    def __init__(
        self,
        client: Client,
        *,
        table: str,
        embedding_model: str,
        dimensions: int,
        database: str = "default",
    ) -> None:
        integer(dimensions, "dimensions", _MAX_DIMENSIONS)
        nonblank(embedding_model, "embedding_model")
        self._client = client
        self._table = f"{identifier(database)}.{identifier(table)}"
        self._location = {"database": database, "table": table}
        self._dimensions = dimensions
        self._embedding_model = embedding_model
        self._manifest = "clickhouserag:" + json.dumps(
            {"schema": 2, "model": embedding_model, "dimensions": dimensions},
            sort_keys=True,
            separators=(",", ":"),
        )
        self._validated = False

    @property
    def dimensions(self) -> int:
        """Expected number of components in each vector."""
        return self._dimensions

    @property
    def embedding_model(self) -> str:
        """Identifier of the embedding model associated with this table."""
        return self._embedding_model

    def initialize(self) -> None:
        """Create an empty table if absent, then validate its schema identity."""
        # The qualified name uses the validated identifier grammar. The comment
        # and dimension are passed through driver binding.
        self._client.command(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                namespace String,
                chunk_id String,
                document_id String,
                text String,
                source String,
                revision UInt64,
                metadata Map(String, String),
                embedding Array(Float32),
                lexical_terms Array(String),
                INDEX lexical_idx(lexical_terms) TYPE text(tokenizer = 'array'),
                CONSTRAINT embedding_dimensions CHECK length(embedding) = %(dimensions)s
            )
            ENGINE = ReplacingMergeTree(revision)
            ORDER BY (namespace, chunk_id)
            COMMENT %(manifest)s
            """,
            parameters={"dimensions": self._dimensions, "manifest": self._manifest},
        )
        self.validate_schema()

    def validate_schema(self) -> None:
        """Verify that the existing table belongs to this schema and embedding model."""
        self._validated = False
        result = self._client.query(
            """
            SELECT comment
            FROM system.tables
            WHERE database = {database:String} AND name = {table:String}
            """,
            parameters=self._location,
        )
        if result.result_rows != [(self._manifest,)]:
            raise SchemaError(
                "ClickHouse table is missing or does not match the store schema and embedding model"
            )
        self._validated = True

    def insert(self, chunks: Iterable[EmbeddedChunk], *, namespace: str = "default") -> int:
        """Insert one validated batch of up to 10,000 embedded passages.

        Repeat the same payload at the same revision to retry a failed write. Use
        a strictly greater revision for replacements of a given namespace/chunk ID.
        """
        return self._insert_rows(self._prepare_rows(chunks, namespace))

    def _prepare_rows(self, chunks: Iterable[EmbeddedChunk], namespace: str) -> list[_PassageRow]:
        nonblank(namespace, "namespace")
        rows = []
        ids = set()
        for item in chunks:
            if len(rows) >= _MAX_BATCH:
                raise ValueError(f"insert accepts at most {_MAX_BATCH} chunks per batch")
            chunk = item.chunk
            metadata = chunk_fields(chunk)
            if chunk.chunk_id in ids:
                raise ValueError("chunk IDs must be unique within a batch")
            ids.add(chunk.chunk_id)
            rows.append(
                (
                    namespace,
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.text,
                    chunk.source,
                    chunk.revision,
                    metadata,
                    unit_vector(item.embedding, self.dimensions),
                    terms(chunk.text),
                )
            )
        return rows

    def _insert_rows(
        self,
        rows: list[_PassageRow],
    ) -> int:
        if not rows:
            return 0
        self._ensure_schema()
        self._client.insert(
            self._table,
            rows,
            column_names=_INSERT_COLUMNS,
            settings={"async_insert": 0},
        )
        return len(rows)

    def search_vector(
        self,
        embedding: Sequence[float],
        *,
        namespace: str = "default",
        metadata: Mapping[str, str] | None = None,
        limit: int = 10,
    ) -> list[SearchHit]:
        """Find the nearest passages by exact cosine similarity (higher is closer)."""
        parameters = _scope(namespace, metadata, limit)
        parameters.update(self._location)
        parameters["embedding"] = unit_vector(embedding, self.dimensions)
        return self._search(
            """
            SELECT chunk_id, document_id, text, source, revision, metadata,
                1 - cosineDistance(embedding, {embedding:Array(Float32)}) AS score
            FROM {database:Identifier}.{table:Identifier} FINAL
            WHERE namespace = {namespace:String}
                AND arrayAll(pair -> mapContains(metadata, pair.1)
                    AND metadata[pair.1] = pair.2, {filters:Array(Tuple(String, String))})
            ORDER BY score DESC, chunk_id ASC
            LIMIT {limit:UInt32}
            """,
            parameters,
        )

    def search_lexical(
        self,
        query: str,
        *,
        namespace: str = "default",
        metadata: Mapping[str, str] | None = None,
        limit: int = 10,
    ) -> list[SearchHit]:
        """Rank matching passages by distinct query token coverage."""
        parameters = _scope(namespace, metadata, limit)
        query_terms = _query_terms(query)
        if not query_terms:
            return []
        parameters.update(self._location, terms=query_terms)
        return self._search(
            """
            SELECT chunk_id, document_id, text, source, revision, metadata,
                length(arrayIntersect(lexical_terms, {terms:Array(String)})) AS score
            FROM {database:Identifier}.{table:Identifier} FINAL
            WHERE namespace = {namespace:String}
                AND arrayAll(pair -> mapContains(metadata, pair.1)
                    AND metadata[pair.1] = pair.2, {filters:Array(Tuple(String, String))})
                AND hasAnyTokens(lexical_terms, {terms:Array(String)})
            ORDER BY score DESC, chunk_id ASC
            LIMIT {limit:UInt32}
            """,
            parameters,
        )

    def search_hybrid(
        self,
        query: str,
        embedding: Sequence[float],
        *,
        namespace: str = "default",
        metadata: Mapping[str, str] | None = None,
        limit: int = 10,
        candidates: int | None = None,
    ) -> list[SearchHit]:
        """Fuse independently ranked vector and lexical candidates."""
        integer(limit, "limit", _MAX_LIMIT)
        if candidates is None:
            candidates = min(limit * 4, _MAX_LIMIT)
        integer(candidates, "candidates", _MAX_LIMIT)
        if candidates < limit:
            raise ValueError("candidates must be at least limit")
        parameters = _scope(namespace, metadata, candidates)
        query_terms = _query_terms(query)
        if not query_terms:
            dense = self.search_vector(
                embedding, namespace=namespace, metadata=metadata, limit=limit
            )
            return reciprocal_rank_fusion(dense, [], limit)
        parameters.update(
            self._location, terms=query_terms, embedding=unit_vector(embedding, self.dimensions)
        )
        self._ensure_schema()
        result = self._client.query(
            """
            WITH scoped AS (
                SELECT chunk_id, document_id, text, source, revision, metadata,
                    embedding, lexical_terms
                FROM {database:Identifier}.{table:Identifier} FINAL
                WHERE namespace = {namespace:String}
                    AND arrayAll(pair -> mapContains(metadata, pair.1)
                        AND metadata[pair.1] = pair.2, {filters:Array(Tuple(String, String))})
            ),
            dense AS (
                SELECT chunk_id, document_id, text, source, revision, metadata,
                    1 - cosineDistance(embedding, {embedding:Array(Float32)}) AS score, 0 AS lane
                FROM scoped
                ORDER BY score DESC, chunk_id ASC
                LIMIT {limit:UInt32}
            ),
            lexical AS (
                SELECT chunk_id, document_id, text, source, revision, metadata,
                    toFloat64(length(arrayIntersect(lexical_terms, {terms:Array(String)})))
                        AS score, 1 AS lane
                FROM scoped
                WHERE hasAnyTokens(lexical_terms, {terms:Array(String)})
                ORDER BY score DESC, chunk_id ASC
                LIMIT {limit:UInt32}
            )
            SELECT * FROM dense
            UNION ALL
            SELECT * FROM lexical
            """,
            parameters=parameters,
            settings=_SELECT_SETTINGS,
        )
        lanes: tuple[list[SearchHit], list[SearchHit]] = ([], [])
        for *row, lane in result.result_rows:
            lanes[lane].append(_hit(row))
        for hits in lanes:
            hits.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return reciprocal_rank_fusion(*lanes, limit)

    def _ensure_schema(self) -> None:
        if not self._validated:
            self.validate_schema()

    def _search(self, query: str, parameters: dict[str, Any]) -> list[SearchHit]:
        self._ensure_schema()
        rows = self._client.query(query, parameters=parameters, settings=_SELECT_SETTINGS)
        return [_hit(row) for row in rows.result_rows]


def _hit(row: Sequence[Any]) -> SearchHit:
    chunk_id, document_id, text, source, revision, metadata, score = row
    return SearchHit(Chunk(chunk_id, document_id, text, source, revision, metadata), float(score))


def _query_terms(query: str) -> list[str]:
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    result = terms(query)
    if len(result) > _MAX_QUERY_TERMS:
        raise ValueError(f"query must have at most {_MAX_QUERY_TERMS} distinct tokens")
    return result


def _scope(namespace: str, metadata: Mapping[str, str] | None, limit: int) -> dict[str, Any]:
    nonblank(namespace, "namespace")
    integer(limit, "limit", _MAX_LIMIT)
    filters = list(string_map(metadata).items()) if metadata is not None else []
    return {"namespace": namespace, "limit": limit, "filters": filters}
