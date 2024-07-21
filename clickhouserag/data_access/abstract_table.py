"""Abstract table module for Clickhouse data access."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from clickhouserag.data_access.abstract_client import ClickhouseClient


class ClickhouseTable(ABC):
    """Abstract class for Clickhouse table management."""

    def __init__(self, client: ClickhouseClient, table_name: str) -> None:
        """Initialize ClickhouseTable.

        Args:
        ----
            client (ClickhouseClient): The Clickhouse client.
            table_name (str): The name of the table.

        """
        self.client = client
        self.table_name = table_name

    @abstractmethod
    def insert(self, values: List[Dict[str, Any]]) -> None:
        """Insert values into the table."""
        pass

    @abstractmethod
    def update(self, values: Dict[str, Any], conditions: Dict[str, Any]) -> None:
        """Update values in the table based on conditions."""
        pass

    @abstractmethod
    def delete(self, conditions: Dict[str, Any]) -> None:
        """Delete values from the table based on conditions."""
        pass

    @abstractmethod
    def search(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search the table based on a query."""
        pass

    @abstractmethod
    def fetch_all(
        self, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch all values from the table."""
        pass

    @abstractmethod
    def reset_table(self) -> None:
        """Reset the table."""
        pass
