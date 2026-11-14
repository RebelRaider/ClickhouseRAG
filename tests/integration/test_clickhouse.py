import pytest
from clickhouse_connect.driver.client import Client

from clickhouserag import Chunk, ClickHouseStore, EmbeddedChunk, SchemaError

pytestmark = pytest.mark.integration


def test_retrieval_orders_and_scopes_passages(store: ClickHouseStore) -> None:
    origin = "https://example.com/документ"
    first = Chunk("a", "doc", "Высокая доступность", origin, metadata={"project": "O'Reilly"})
    second = Chunk("b", "doc", "Different", origin, metadata={"project": ""})
    third = Chunk("c", "doc", "Missing key", origin)
    store.insert(
        [EmbeddedChunk(first, [4, 0]), EmbeddedChunk(second, [0, 1]), EmbeddedChunk(third, [1, 1])],
        namespace="production",
    )
    store.insert([EmbeddedChunk(second, [1, 0])], namespace="other")

    hits = store.search_vector([1, 0], namespace="production", limit=2)
    assert [hit.chunk.chunk_id for hit in hits] == ["a", "c"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].chunk == first
    for chunk in (first, second):
        scoped = store.search_vector([1, 0], namespace="production", metadata=chunk.metadata)
        assert [h.chunk for h in scoped] == [chunk]
    assert store.search_vector([1, 0], namespace="absent") == []


def test_replacement_is_visible_before_merges(
    store: ClickHouseStore, database_client: Client, table: str
) -> None:
    database_client.command("SYSTEM STOP MERGES {table:Identifier}", {"table": table})
    older = Chunk("id", "doc", "Original", "source", metadata={"state": "old"})
    newer = Chunk("id", "doc", "Updated", "source", revision=2, metadata={"state": "new"})
    original = EmbeddedChunk(older, [1, 0])
    update = EmbeddedChunk(newer, [0, 1])
    store.insert([original])
    store.insert([update])
    store.insert([update])
    store.insert([original])

    assert store.search_vector([1, 0], metadata={"state": "old"}) == []
    hits = store.search_vector([0, 1], metadata={"state": "new"})
    assert [h.chunk for h in hits] == [newer]


def test_model_and_dimensions_are_part_of_table_identity(
    database_client: Client, table: str, store: ClickHouseStore
) -> None:
    store.initialize()
    incompatible = ClickHouseStore(
        database_client, table=table, embedding_model="different", dimensions=2
    )
    with pytest.raises(SchemaError, match="schema and embedding model"):
        incompatible.initialize()


def test_identifier_keywords_and_quoted_model(database_client: Client, table: str) -> None:
    store = ClickHouseStore(
        database_client, table=table, embedding_model="model's path", dimensions=2
    )
    store.initialize()
    store.validate_schema()
