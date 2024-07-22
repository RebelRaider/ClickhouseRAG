"""Clickhouse client module for Clickhouse data access."""

import functools
import logging
from typing import Any, Dict, List, Optional, Tuple

from clickhouse_driver import Client, errors

from clickhouserag.data_access.abstract_client import ClickhouseClient


def ensure_connection(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.client:
            raise ConnectionError("Client is not connected. Call `connect` first.")
        return method(self, *args, **kwargs)
    return wrapper


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
        self.client: Optional[Client] = None
        self.logger = logging.getLogger(__name__)

    def connect(self) -> None:
        """Connect to the Clickhouse database."""
        try:
            self.client = Client(host=self.host, port=self.port, user=self.username, password=self.password, database=self.database)
            self.ping()
        except errors.Error as err:
            self.logger.error(f"Failed to connect to Clickhouse: {err}")
            raise ConnectionError(f"Failed to connect to Clickhouse: {err}") from err

    def ping(self) -> None:
        """Ping the Clickhouse server to check the connection."""
        try:
            self.client.execute("SELECT 1")
        except errors.Error as err:
            self.logger.error(f"Ping failed: {err}")
            raise ConnectionError(f"Ping to Clickhouse failed: {err}") from err

    @ensure_connection
    def _execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Tuple]:
        """Execute a query in the Clickhouse database with error handling."""
        try:
            if params:
                return self.client.execute(query, params)
            return self.client.execute(query)
        except errors.Error as err:
            self.logger.error(f"Failed to execute query: {err}")
            raise RuntimeError(f"Failed to execute query: {err}") from err

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Tuple]:
        """Execute a query in the Clickhouse database."""
        return self._execute(query, params)

    def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Fetch one result from the Clickhouse database."""
        result = self._execute(query, params)
        return result[0] if result else None

    def fetch_all(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Fetch all results from the Clickhouse database."""
        return [dict(row) for row in self._execute(query, params)]

    def fetch_column(self, query: str, column: int = 0, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Fetch a specific column from the Clickhouse database."""
        result = self._execute(query, params)
        try:
            return [row[column] for row in result]
        except IndexError as err:
            self.logger.error(f"Column index out of range: {err}")
            raise IndexError(f"Column index out of range: {err}") from None

    def close(self) -> None:
        """Close the connection to the Clickhouse database."""
        try:
            if self.client:
                self.client.disconnect()
        except errors.Error as err:
            self.logger.error(f"Failed to close connection: {err}")
            raise RuntimeError(f"Failed to close connection: {err}") from err

    def __enter__(self) -> "ClickhouseConnectClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
