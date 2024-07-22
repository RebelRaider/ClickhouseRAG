"""Clickhouse table management module."""

from typing import Any, Dict, List, Optional

from clickhouserag.data_access.abstract_table import ClickhouseTable
from clickhouserag.data_access.clickhouse_client import ClickhouseConnectClient


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

    def insert(self, values: List[Dict[str, Any]]) -> None:
        """Insert values into the table."""
        query = f"INSERT INTO {self.table_name} VALUES"
        self.client.execute_query(query, values)

    def update(self, values: Dict[str, Any], conditions: Dict[str, Any]) -> None:
        """Update values in the table based on conditions."""
        set_clause = ", ".join([f"{key} = %({key})s" for key in values.keys()])
        condition_clause = " AND ".join([f"{key} = %({key})s" for key in conditions.keys()])
        query = f"ALTER TABLE {self.table_name} UPDATE {set_clause} WHERE {condition_clause}"
        self.client.execute_query(query, {**values, **conditions})

    def delete(self, conditions: Dict[str, Any]) -> None:
        """Delete values from the table based on conditions."""
        condition_clause = " AND ".join([f"{key} = %(cond_{key})s" for key in conditions.keys()])
        query = f"DELETE FROM {self.table_name} WHERE {condition_clause}"
        params = {f"cond_{key}": value for key, value in conditions.items()}
        self.client.execute_query(query, params)

    def search(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search the table based on a query."""
        return self.client.fetch_all(query, params)

    def fetch_all(self, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Fetch all values from the table."""
        query = f"SELECT * FROM {self.table_name}"
        return self.client.fetch_all(query, params)

    def reset_table(self) -> None:
        """Reset the table."""
        query = f"TRUNCATE TABLE {self.table_name}"
        self.client.execute_query(query)
