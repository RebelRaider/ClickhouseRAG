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


def test_lexical_and_hybrid_use_a_shared_scope(store: ClickHouseStore) -> None:
    shared = Chunk("a", "doc", "Альфа RFC_9110", "a", metadata={"lang": "ru"})
    semantic = Chunk("b", "doc", "Гамма", "b", metadata={"lang": "ru"})
    lexical = Chunk("c", "doc", "Альфа Альфа", "c", metadata={"lang": "ru"})
    filtered = Chunk("d", "doc", "Альфа RFC_9110", "d", metadata={"lang": "en"})
    store.insert(
        [
            EmbeddedChunk(shared, [0.9, 0.1]),
            EmbeddedChunk(semantic, [1, 0]),
            EmbeddedChunk(lexical, [0, 1]),
            EmbeddedChunk(filtered, [1, 0]),
        ],
        namespace="docs",
    )
    store.insert([EmbeddedChunk(filtered, [1, 0])], namespace="other")

    scope = {"lang": "ru"}
    phrase = "АЛЬФА RFC 9110"
    hits = store.search_lexical(phrase, namespace="docs", metadata=scope)
    assert [(hit.chunk.chunk_id, hit.score) for hit in hits] == [("a", 3.0), ("c", 1.0)]
    hits = store.search_hybrid(phrase, [1, 0], namespace="docs", metadata=scope, limit=2)
    assert [hit.chunk.chunk_id for hit in hits] == ["a", "c"]
    assert store.search_lexical("??", namespace="docs") == []
    fallback = store.search_hybrid("??", [1, 0], namespace="docs", metadata=scope)
    exact = store.search_vector([1, 0], namespace="docs", metadata=scope)
    assert [h.chunk for h in fallback] == [h.chunk for h in exact]
    assert fallback[0].score == pytest.approx(1 / 61)


def test_text_index_respects_replacements_before_merges(
    store: ClickHouseStore, database_client: Client, table: str
) -> None:
    database_client.command("SYSTEM STOP MERGES {table:Identifier}", {"table": table})
    store.insert([EmbeddedChunk(Chunk("a", "doc", "tokenold", "source"), [1, 0])])
    store.insert([EmbeddedChunk(Chunk("a", "doc", "tokennew", "source", revision=2), [0, 1])])
    assert store.search_lexical("tokenold") == []
    assert [h.chunk.revision for h in store.search_lexical("tokennew")] == [2]
    assert [h.chunk.revision for h in store.search_hybrid("tokenold", [1, 0])] == [2]
