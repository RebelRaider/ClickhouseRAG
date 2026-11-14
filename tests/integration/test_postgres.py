import os
from collections.abc import Iterator
from unittest.mock import patch
from uuid import uuid4

import pytest
from clickhouse_connect.driver.client import Client

from clickhouserag import RAG

psycopg = pytest.importorskip("psycopg")
from clickhouserag.postgres import PostgresOutbox  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


class Embed:
    model_id = "outbox-test"

    def embed_documents(self, texts) -> list[list[float]]:
        return [[1, 0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1, 0]


@pytest.fixture
def source() -> Iterator:
    dsn = os.getenv("CH_RAG_POSTGRES_DSN")
    if not dsn:
        pytest.skip("Set CH_RAG_POSTGRES_DSN for PostgreSQL outbox tests")
    with psycopg.connect(dsn, autocommit=True) as conn:
        yield conn


@pytest.fixture
def outbox(source) -> Iterator[PostgresOutbox]:
    table = f"RAG Outbox {uuid4().hex}"
    store = PostgresOutbox(table=table)
    try:
        store.initialize(source)
        yield store
    finally:
        source.execute(
            psycopg.sql.SQL("DROP TABLE IF EXISTS {}").format(psycopg.sql.Identifier(table))
        )


@pytest.fixture
def collection(database_client: Client, table: str) -> RAG:
    rag = RAG(database_client, Embed(), dimensions=2, table=table)
    rag.initialize()
    return rag


def test_outbox_delivers_committed_replacements_and_deletions(source, outbox, collection) -> None:
    with pytest.raises(RuntimeError, match="rollback"), source.transaction():
        outbox.enqueue(source, "doc", "Old text", "ref:rollback")
        raise RuntimeError("rollback")
    assert outbox.sync(source, collection) == []

    with source.transaction():
        assert outbox.enqueue(source, "doc", "Old text", "ref:old") == 1
        assert (
            outbox.enqueue(source, "doc", "Current text", "ref:current", metadata={"team": "ops"})
            == 2
        )
        with pytest.raises(ValueError, match="dedicated"):
            outbox.sync(source, collection)
    (change,) = outbox.sync(source, collection)
    assert (change.document_id, change.revision, change.deleted, change.passages) == (
        "doc",
        2,
        False,
        1,
    )
    assert collection.retrieve("Old", mode="lexical").passages == ()
    (chunk,) = collection.retrieve("Current", mode="lexical", metadata={"team": "ops"}).passages
    assert chunk.source == "ref:current" and chunk.revision == 2

    assert outbox.enqueue_delete(source, "doc") == 3
    (deleted,) = outbox.sync(source, collection)
    assert deleted.deleted and deleted.revision == 3 and deleted.passages == 0
    assert collection.retrieve("Current", mode="lexical").passages == ()
    assert outbox.sync(source, collection) == []
    assert outbox.enqueue(source, "doc", "Restored text", "ref:restored") == 4
    assert outbox.sync(source, collection)[0].revision == 4


def test_outbox_replays_after_clickhouse_succeeds_before_ack(source, outbox, collection) -> None:
    assert outbox.enqueue(source, "doc", "Ledger writes Vault", "ref:doc") == 1
    index = collection.index

    def disconnect_after_write(document) -> int:
        assert index(document) == 1
        raise ConnectionError("after ClickHouse publication")

    with (
        patch.object(collection, "index", side_effect=disconnect_after_write),
        pytest.raises(ConnectionError, match="after ClickHouse"),
    ):
        outbox.sync(source, collection)
    assert len(collection.retrieve("Ledger", mode="lexical").passages) == 1
    (replay,) = outbox.sync(source, collection)
    assert replay.revision == 1 and replay.passages == 0
    assert outbox.sync(source, collection) == []


def test_workers_skip_another_workers_locked_document(source, outbox, collection) -> None:
    outbox.enqueue(source, "alpha", "Alpha text", "ref:a")
    outbox.enqueue(source, "beta", "Beta text", "ref:b")
    index = collection.index
    dsn = os.environ["CH_RAG_POSTGRES_DSN"]
    with psycopg.connect(dsn, autocommit=True) as other:

        def publish(document) -> int:
            if document.document_id == "alpha":
                (result,) = outbox.sync(other, collection, limit=1)
                assert result.document_id == "beta"
            return index(document)

        with patch.object(collection, "index", side_effect=publish):
            (result,) = outbox.sync(source, collection)
        assert result.document_id == "alpha"
        assert outbox.sync(other, collection) == []
