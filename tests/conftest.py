import os
import time
from unittest.mock import MagicMock, patch

import pytest

from clickhouserag.data_access.clickhouse_client import ClickhouseConnectClient
from clickhouserag.data_access.clickhouse_table import ClickhouseTableManager
from clickhouserag.rag.manager import RAGManager


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(pytestconfig.rootdir, "clickhouse/docker-compose.yml")

@pytest.fixture(scope="session")
def _clickhouse_service(docker_services):
    """Ensure that Clickhouse service is up and responsive."""
    docker_services.start("clickhouse")
    time.sleep(10)  # Give some time for Clickhouse to be ready

    yield

    docker_services.stop("clickhouse")
@pytest.fixture()
def clickhouse_client():
    client = ClickhouseConnectClient("localhost", 9000, "test_user", "test_password", "test_db")
    with patch.object(client, "connect", MagicMock()), \
         patch.object(client, "execute_query", MagicMock()), \
         patch.object(client, "fetch_one", MagicMock()), \
         patch.object(client, "fetch_all", MagicMock()), \
         patch.object(client, "close", MagicMock()):
        yield client

@pytest.fixture()
def table_manager(clickhouse_client):
    return ClickhouseTableManager(clickhouse_client, "test_table")

@pytest.fixture()
def rag_manager(clickhouse_client):
    return RAGManager(clickhouse_client, "test_table", {"id": "String", "title": "String", "vector": "Array(Float32)"})

@pytest.fixture()
def sample_data():
    return {"id": "1", "title": "Test Title", "vector": [0.1, 0.2, 0.3]}
