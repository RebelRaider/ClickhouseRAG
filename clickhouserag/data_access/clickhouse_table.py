"""Clickhouse table management module."""

import logging
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
        self.logger = logging.getLogger(__name__)

    def insert(self, values: List[Dict[str, Any]]) -> None:
        """Insert values into the table."""
        query = f"INSERT INTO {self.table_name} VALUES"
        try:
            self.client.execute_query(query, values)
            self.logger.info(f"Inserted values into {self.table_name}")
        except Exception as err:
            self.logger.error(f"Failed to insert values into {self.table_name}: {err}")
            raise RuntimeError(f"Failed to insert values into {self.table_name}") from err

    def update(self, values: Dict[str, Any], conditions: Dict[str, Any]) -> None:
        """Update values in the table based on conditions."""
        set_clause = ", ".join([f"{key} = %({key})s" for key in values.keys()])
        condition_clause = " AND ".join([f"{key} = %({key})s" for key in conditions.keys()])
        query = f"ALTER TABLE {self.table_name} UPDATE {set_clause} WHERE {condition_clause}"
        try:
            self.client.execute_query(query, {**values, **conditions})
            self.logger.info(f"Updated values in {self.table_name} where {conditions}")
        except Exception as err:
            self.logger.error(f"Failed to update values in {self.table_name}: {err}")
            raise RuntimeError(f"Failed to update values in {self.table_name}") from err

    def delete(self, conditions: Dict[str, Any]) -> None:
        """Delete values from the table based on conditions."""
        condition_clause = " AND ".join([f"{key} = %(cond_{key})s" for key in conditions.keys()])
        query = f"DELETE FROM {self.table_name} WHERE {condition_clause}"
        params = {f"cond_{key}": value for key, value in conditions.items()}
        try:
            self.client.execute_query(query, params)
            self.logger.info(f"Deleted from {self.table_name} where {conditions}")
        except Exception as err:
            self.logger.error(f"Failed to delete from {self.table_name}: {err}")
            raise RuntimeError(f"Failed to delete from {self.table_name}") from err

    def search(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search the table based on a query."""
        try:
            results = self.client.fetch_all(query, params)
            self.logger.info(f"Executed search query on {self.table_name}")
            return results
        except Exception as err:
            self.logger.error(f"Failed to execute search query on {self.table_name}: {err}")
            raise RuntimeError(f"Failed to execute search query on {self.table_name}") from err

    def fetch_all(self, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Fetch all values from the table."""
        query = f"SELECT * FROM {self.table_name}"
        try:
            results = self.client.fetch_all(query, params)
            self.logger.info(f"Fetched all rows from {self.table_name}")
            return results
        except Exception as err:
            self.logger.error(f"Failed to fetch all rows from {self.table_name}: {err}")
            raise RuntimeError(f"Failed to fetch all rows from {self.table_name}") from err

    def reset_table(self) -> None:
        """Reset the table."""
        query = f"TRUNCATE TABLE {self.table_name}"
        try:
            self.client.execute_query(query)
            self.logger.info(f"Truncated table {self.table_name}")
        except Exception as err:
            self.logger.error(f"Failed to truncate table {self.table_name}: {err}")
            raise RuntimeError(f"Failed to truncate table {self.table_name}") from err
