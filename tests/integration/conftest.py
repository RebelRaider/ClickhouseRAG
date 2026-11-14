import os
from collections.abc import Iterator
from uuid import uuid4

import clickhouse_connect
import pytest
from clickhouse_connect.driver.client import Client

from clickhouserag import ClickHouseStore


@pytest.fixture(scope="session")
def database_client() -> Iterator[Client]:
    port = os.environ.get("CLICKHOUSE_PORT")
    if not port:
        pytest.skip("Set CLICKHOUSE_PORT to run tests against an actual ClickHouse server")
    with clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST", "127.0.0.1"),
        port=int(port),
        username=os.getenv("CLICKHOUSE_USER", "default"),
        password=os.getenv("CLICKHOUSE_PASSWORD", ""),
        autogenerate_session_id=False,
    ) as client:
        yield client


@pytest.fixture
def table(database_client: Client) -> Iterator[str]:
    name = f"chrag_test_{uuid4().hex}"
    try:
        yield name
    finally:
        database_client.command("DROP TABLE IF EXISTS {table:Identifier}", {"table": name})


@pytest.fixture
def store(database_client: Client, table: str) -> ClickHouseStore:
    store = ClickHouseStore(database_client, table=table, embedding_model="test", dimensions=2)
    store.initialize()
    return store
