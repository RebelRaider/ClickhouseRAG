"""Optional PostgreSQL document outbox feeding one ClickHouse collection."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

from psycopg import Connection, sql
from psycopg.pq import TransactionStatus
from psycopg.rows import tuple_row
from psycopg.types.json import Jsonb

from clickhouserag._validation import integer, nonblank, string_map
from clickhouserag.rag import RAG, Document


@dataclass(frozen=True, slots=True)
class SyncResult:
    """One acknowledged document revision."""

    document_id: str
    revision: int
    deleted: bool
    passages: int


class PostgresOutbox:
    """Maintain source revisions in PostgreSQL and publish them to ClickHouse.

    Keep one outbox table per target RAG collection. Producers can enqueue
    changes in their own PostgreSQL transaction. Workers use dedicated idle
    connections and acknowledge a row only after publication succeeds.
    """

    def __init__(self, *, schema: str = "public", table: str = "rag_document_outbox") -> None:
        for name in (schema, table):
            nonblank(name, "PostgreSQL identifier")
            if len(name.encode()) > 63:
                raise ValueError("PostgreSQL identifiers must fit in 63 UTF-8 bytes")
        self._table = sql.Identifier(schema, table)
        digest = hashlib.sha256(table.encode()).hexdigest()[:24]
        self._pending_index = sql.Identifier(f"chrag_pending_{digest}")

    def initialize(self, connection: Connection) -> None:
        """Create the outbox during provisioning in the caller's transaction."""
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("""
                    CREATE TABLE IF NOT EXISTS {} (
                        document_id text PRIMARY KEY,
                        revision bigint NOT NULL CHECK (revision > 0),
                        published bigint NOT NULL DEFAULT 0
                            CHECK (published >= 0 AND published <= revision),
                        body text,
                        source text,
                        metadata jsonb NOT NULL,
                        CHECK ((body IS NULL) = (source IS NULL))
                    )
                """).format(self._table)
            )
            cursor.execute(
                sql.SQL("""
                    CREATE INDEX IF NOT EXISTS {} ON {} (document_id)
                        WHERE revision > published
                """).format(self._pending_index, self._table)
            )

    def enqueue(
        self,
        connection: Connection,
        document_id: str,
        text: str,
        source: str,
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> int:
        """Record the latest source document in the caller's transaction."""
        nonblank(text, "document text")
        nonblank(source, "source")
        return self._write(
            connection,
            document_id,
            text,
            source,
            string_map(metadata) if metadata is not None else {},
        )

    def enqueue_delete(self, connection: Connection, document_id: str) -> int:
        """Record a deletion with the next revision in the caller's transaction."""
        return self._write(connection, document_id, None, None, {})

    def _write(
        self,
        connection: Connection,
        document_id: str,
        text: str | None,
        source: str | None,
        metadata: dict[str, str],
    ) -> int:
        nonblank(document_id, "document_id")
        with connection.cursor(row_factory=tuple_row) as cursor:
            cursor.execute(
                sql.SQL("""
                    INSERT INTO {} AS previous (document_id, revision, body, source, metadata)
                    VALUES (%s, 1, %s, %s, %s)
                    ON CONFLICT (document_id) DO UPDATE
                    SET revision = previous.revision + 1, body = EXCLUDED.body,
                        source = EXCLUDED.source, metadata = EXCLUDED.metadata
                    RETURNING revision
                """).format(self._table),
                (document_id, text, source, Jsonb(metadata)),
            )
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError("PostgreSQL did not return the queued revision")
            return row[0]

    def sync(self, connection: Connection, rag: RAG, *, limit: int = 100) -> list[SyncResult]:
        """Publish up to limit pending documents, committing each acknowledgment."""
        integer(limit, "limit", 1000)
        if connection.info.transaction_status != TransactionStatus.IDLE:
            raise ValueError(
                "sync requires a dedicated PostgreSQL connection outside a transaction"
            )
        delivered = []
        for _ in range(limit):
            item = self._publish_one(connection, rag)
            if item is None:
                break
            delivered.append(item)
        return delivered

    def _publish_one(self, connection: Connection, rag: RAG) -> SyncResult | None:
        with connection.transaction(), connection.cursor(row_factory=tuple_row) as cursor:
            cursor.execute(
                sql.SQL("""
                    SELECT document_id, revision, body, source, metadata
                    FROM {}
                    WHERE revision > published
                    ORDER BY document_id
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                """).format(self._table)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            document_id, revision, body, source, metadata = row
            if body is None:
                rag.delete(document_id, revision=revision)
                passages = 0
            else:
                passages = rag.index(Document(document_id, body, source, metadata, revision))
            cursor.execute(
                sql.SQL("UPDATE {} SET published = %s WHERE document_id = %s").format(self._table),
                (revision, document_id),
            )
            return SyncResult(document_id, revision, body is None, passages)
