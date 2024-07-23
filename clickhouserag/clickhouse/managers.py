"""Clickhouse table management module."""

import logging
from typing import Any, Dict, List, Optional

from clickhouserag.clickhouse.base import ClickhouseTable
from clickhouserag.clickhouse.clients import ClickhouseConnectClient


class ClickhouseTableManager(ClickhouseTable):
    """Clickhouse table manager implementation."""

    def __init__(self, client: ClickhouseConnectClient, table_name: str) -> None:
        """Initialize ClickhouseTableManager.

        Args:
        ----
            client (ClickhouseConnectClient): The Clickhouse client.
            table_name (str): The name of the table.
        """
        super().__init__(client, table_name)
        self.logger = logging.getLogger(__name__)

    def insert(self, values: List[Dict[str, Any]]) -> None:
        """Insert values into the table."""
        query = f"INSERT INTO {self.table_name} VALUES"
        self._execute_query(query, values, "insert")

    def update(self, values: Dict[str, Any], conditions: Dict[str, Any]) -> None:
        """Update values in the table based on conditions."""
        set_clause = ", ".join([f"{key} = %({key})s" for key in values.keys()])
        condition_clause = " AND ".join([f"{key} = %({key})s" for key in conditions.keys()])
        query = f"ALTER TABLE {self.table_name} UPDATE {set_clause} WHERE {condition_clause}"
        self._execute_query(query, {**values, **conditions}, "update")

    def delete(self, conditions: Dict[str, Any]) -> None:
        """Delete values from the table based on conditions."""
        condition_clause = " AND ".join([f"{key} = %(cond_{key})s" for key in conditions.keys()])
        query = f"DELETE FROM {self.table_name} WHERE {condition_clause}"
        params = {f"cond_{key}": value for key, value in conditions.items()}
        self._execute_query(query, params, "delete")

    def search(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search the table based on a query."""
        return self._fetch_results(query, params, "search")

    def fetch_all(self, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Fetch all values from the table."""
        query = f"SELECT * FROM {self.table_name}"
        return self._fetch_results(query, params, "fetch all")

    def reset_table(self) -> None:
        """Reset the table."""
        query = f"TRUNCATE TABLE {self.table_name}"
        self._execute_query(query, None, "truncate")

    def _execute_query(self, query: str, params: Optional[Dict[str, Any]], operation: str) -> None:
        """Execute a query in the Clickhouse database with error handling."""
        try:
            self.client.execute_query(query, params)
            self.logger.info(f"{operation.capitalize()} operation successful on {self.table_name}")
        except Exception as err:
            self.logger.error(f"Failed to {operation} in {self.table_name}: {err}")
            raise RuntimeError(f"Failed to {operation} in {self.table_name}") from err

    def _fetch_results(self, query: str, params: Optional[Dict[str, Any]], operation: str) -> List[Dict[str, Any]]:
        """Fetch results from the Clickhouse database with error handling."""
        try:
            results = self.client.fetch_all(query, params)
            self.logger.info(f"{operation.capitalize()} query executed on {self.table_name}")
            return results
        except Exception as err:
            self.logger.error(f"Failed to execute {operation} query on {self.table_name}: {err}")
            raise RuntimeError(f"Failed to execute {operation} query on {self.table_name}") from err
