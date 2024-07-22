"""Clickhouse client module for Clickhouse data access."""

import logging
from typing import Any, Dict, List, Optional

from clickhouse_driver import Client

from clickhouserag.data_access.abstract_client import ClickhouseClient


class ClickhouseConnectClient(ClickhouseClient):
    """Clickhouse client implementation using clickhouse-driver."""

    def __init__(self, host: str, port: int, username: str, password: str, database: str):
        """Initialize ClickhouseConnectClient.

        Args:
        ----
            host (str): The host of the Clickhouse server.
            port (int): The port of the Clickhouse server.
            username (str): The username for Clickhouse authentication.
            password (str): The password for Clickhouse authentication.
            database (str): The database to connect to.

        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.client = None
        self.logger = logging.getLogger(__name__)

    def connect(self) -> None:
        """Connect to the Clickhouse database."""
        try:
            self.client = Client(host=self.host, port=self.port, user=self.username, password=self.password, database=self.database)
        except Exception as e:
            self.logger.error(f"Failed to connect to Clickhouse: {e}")
            raise

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a query in the Clickhouse database."""
        try:
            if params:
                return self.client.execute(query, params)
            return self.client.execute(query)
        except Exception as e:
            self.logger.error(f"Failed to execute query: {e}")
            raise

    def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Fetch one result from the Clickhouse database."""
        try:
            result = self.execute_query(query, params)
            return result[0] if result else None
        except Exception as e:
            self.logger.error(f"Failed to fetch one: {e}")
            raise

    def fetch_all(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Fetch all results from the Clickhouse database."""
        try:
            return self.execute_query(query, params)
        except Exception as e:
            self.logger.error(f"Failed to fetch all: {e}")
            raise

    def fetch_column(self, query: str, column: int = 0, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Fetch a specific column from the Clickhouse database."""
        try:
            result = self.execute_query(query, params)
            return [row[column] for row in result]
        except Exception as e:
            self.logger.error(f"Failed to fetch column: {e}")
            raise

    def close(self) -> None:
        """Close the connection to the Clickhouse database."""
        try:
            if self.client:
                self.client.disconnect()
        except Exception as e:
            self.logger.error(f"Failed to close connection: {e}")
            raise
