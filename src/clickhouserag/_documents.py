"""Storage and publication of complete document revisions."""

import hashlib
import json
from collections.abc import Iterable, Sequence

from clickhouse_connect.driver.client import Client

from clickhouserag._validation import integer
from clickhouserag.models import Chunk, EmbeddedChunk
from clickhouserag.store import _INSERT_COLUMNS, _SELECT_SETTINGS, ClickHouseStore, _PassageRow

# A tombstone has an empty generation. At the same revision it wins any tie.
# Competing live generations have a deterministic winner even if both were staged.
_ACTIVE_PASSAGES = """
    (SELECT namespace, chunk_id, document_id, text, source, revision, metadata,
        embedding, lexical_terms, facts, entity_terms
    FROM {database:Identifier}.{table:Identifier} FINAL
    WHERE namespace = {namespace:String} AND kind = 0
        AND (document_id, revision, generation) IN (
            SELECT document_id, head.1, head.2 FROM (
                SELECT document_id, argMax(
                    tuple(revision, generation), tuple(revision, generation = '', generation)
                ) AS head
                FROM {database:Identifier}.{table:Identifier} FINAL
                WHERE namespace = {namespace:String} AND kind = 1
                GROUP BY document_id
            )
        ))
"""
_VISIBLE_CHUNKS = """
    SELECT chunk_id, revision FROM @passages@ WHERE chunk_id IN {ids:Array(String)}
"""

_Fact = tuple[str, str, str, str, str, list[str], list[str]]
_MAX_REVISION = 2**64 - 1


class DocumentRevisionError(ValueError):
    """An explicit revision conflicts with a published operation."""


class DocumentStore(ClickHouseStore):
    """Internal passage store with visibility controlled by document heads."""

    _passage_source = _ACTIVE_PASSAGES

    def __init__(
        self, client: Client, *, table: str, embedding_model: str, dimensions: int, database: str
    ) -> None:
        super().__init__(
            client,
            table=table,
            embedding_model=embedding_model,
            dimensions=dimensions,
            database=database,
        )
        self._manifest = self._manifest.replace('"schema":2', '"schema":3')

    def initialize(self) -> None:
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
                kind UInt8,
                generation String,
                identity String,
                facts Array(Tuple(String, String, String, String, String,
                                  Array(String), Array(String))),
                entity_terms Array(String) MATERIALIZED arrayFlatten(
                    arrayMap(fact -> arrayConcat(fact.6, fact.7), facts)
                ),
                INDEX lexical_idx(lexical_terms) TYPE text(tokenizer = 'array'),
                INDEX entity_idx(entity_terms) TYPE text(tokenizer = 'array'),
                CONSTRAINT row_kind CHECK kind IN (0, 1),
                CONSTRAINT embedding_dimensions CHECK
                    (kind = 0 AND length(embedding) = %(dimensions)s)
                    OR (kind = 1 AND empty(embedding))
            )
            ENGINE = ReplacingMergeTree
            ORDER BY (namespace, kind, document_id, revision, generation, chunk_id)
            COMMENT %(manifest)s
            """,
            parameters={"dimensions": self.dimensions, "manifest": self._manifest},
        )
        self.validate_schema()

    def insert(self, chunks: Iterable[EmbeddedChunk], *, namespace: str = "default") -> int:
        raise ValueError("index a Document through RAG to write this managed collection")

    def next_revision(
        self, namespace: str, document_id: str, identity: str, requested: int | None
    ) -> int | None:
        """Return the revision to publish, or None for an already applied operation."""
        if requested is not None:
            integer(requested, "revision", _MAX_REVISION)
        self._ensure_schema()
        result = self._client.query(
            """
            SELECT revision, identity
            FROM {database:Identifier}.{table:Identifier} FINAL
            WHERE namespace = {namespace:String} AND kind = 1
                AND document_id = {document:String}
            ORDER BY revision DESC, (generation = '') DESC, generation DESC
            LIMIT 1
            """,
            parameters={**self._location, "namespace": namespace, "document": document_id},
            settings=_SELECT_SETTINGS,
        )
        current, previous = result.result_rows[0] if result.result_rows else (0, None)
        if requested is None:
            if previous == identity:
                return None
            integer(current + 1, "revision", _MAX_REVISION)
            return current + 1
        if requested < current:
            raise DocumentRevisionError("revision is older than the published document")
        if requested == current:
            if previous == identity:
                return None
            raise DocumentRevisionError(
                "revision already published with a different document payload"
            )
        return requested

    def publish(
        self,
        namespace: str,
        document_id: str,
        revision: int,
        identity: str,
        rows: list[_PassageRow],
        facts: Sequence[list[_Fact]],
    ) -> None:
        generation = ""
        if rows:
            # Include staged facts and vectors so independent writers cannot mix
            # content under a common generation after a partial failure.
            digest = hashlib.sha256()
            for row, relations in zip(rows, facts, strict=True):
                digest.update(
                    json.dumps((row, relations), ensure_ascii=False, sort_keys=True).encode()
                )
                digest.update(b"\n")
            generation = digest.hexdigest()
            self._client.insert(
                self._table,
                [
                    (row[0], f"{row[1]}:{generation}", *row[2:], 0, generation, "", relations)
                    for row, relations in zip(rows, facts, strict=True)
                ],
                column_names=(*_INSERT_COLUMNS, "kind", "generation", "identity", "facts"),
                settings={"async_insert": 0},
            )
        self._client.insert(
            self._table,
            [(namespace, document_id, revision, 1, generation, identity)],
            column_names=("namespace", "document_id", "revision", "kind", "generation", "identity"),
            settings={"async_insert": 0},
        )

    def visible_chunks(self, namespace: str, chunks: Sequence[Chunk]) -> set[tuple[str, int]]:
        if not chunks:
            return set()
        result = self._client.query(
            _VISIBLE_CHUNKS.replace("@passages@", self._passage_source),
            parameters={
                **self._location,
                "namespace": namespace,
                "ids": list(dict.fromkeys(chunk.chunk_id for chunk in chunks)),
            },
            settings=_SELECT_SETTINGS,
        )
        return {(chunk_id, revision) for chunk_id, revision in result.result_rows}
