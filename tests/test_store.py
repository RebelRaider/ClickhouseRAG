from unittest.mock import Mock

import pytest
from clickhouse_connect.driver.client import Client
from hypothesis import given
from hypothesis import strategies as st

from clickhouserag import Chunk, ClickHouseStore, EmbeddedChunk, SchemaError
from clickhouserag._validation import unit_vector


@pytest.fixture
def client() -> Mock:
    return Mock(spec=Client)


@pytest.fixture
def store(client: Mock) -> ClickHouseStore:
    return ClickHouseStore(client, table="passages", embedding_model="test", dimensions=2)


def test_missing_or_incompatible_table(store: ClickHouseStore, client: Mock) -> None:
    client.query.return_value.result_rows = []
    with pytest.raises(SchemaError, match="missing"):
        store.search_vector([1, 2])
    client.insert.assert_not_called()


@pytest.mark.parametrize("name", ["", "db.table", "white space", "bad`name", "таблица"])
def test_table_identifier_validation(client: Mock, name: str) -> None:
    with pytest.raises(ValueError, match="identifiers"):
        ClickHouseStore(client, table=name, embedding_model="test", dimensions=2)
    client.command.assert_not_called()


@pytest.mark.parametrize("vector", [[0, 0], [1], [1, float("nan")], [1, float("inf")]])
def test_invalid_vector_fails_before_network(
    store: ClickHouseStore, client: Mock, vector: list[float]
) -> None:
    with pytest.raises(ValueError, match="embedding"):
        store.search_vector(vector)
    client.query.assert_not_called()


def test_batch_validation_precedes_insert(store: ClickHouseStore, client: Mock) -> None:
    chunk = Chunk("a", "doc", "Text", "source")
    rows = [EmbeddedChunk(chunk, [1, 2]), EmbeddedChunk(chunk, [2, 1])]
    with pytest.raises(ValueError, match="unique"):
        store.insert(iter(rows))
    client.insert.assert_not_called()
    client.query.assert_not_called()


def test_empty_batch(store: ClickHouseStore, client: Mock) -> None:
    assert store.insert([]) == 0
    client.query.assert_not_called()


@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2, max_size=20))
def test_normalized_vector_has_unit_norm(vector: list[float]) -> None:
    if any(vector):
        result = unit_vector(vector, len(vector))
        assert len(result) == len(vector)
        assert sum(value**2 for value in result) == pytest.approx(1)
    else:
        with pytest.raises(ValueError, match="nonzero"):
            unit_vector(vector, len(vector))
