"""Abstract client module for Clickhouse."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class ClickhouseClient(ABC):
    """Abstract class for Clickhouse client."""

    @abstractmethod
    def connect(self) -> None:
        """Connect to the Clickhouse database."""
        pass

    @abstractmethod
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Tuple]:
        """Execute a query in the Clickhouse database."""
        pass

    @abstractmethod
    def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Fetch one result from the Clickhouse database."""
        pass

    @abstractmethod
    def fetch_all(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Fetch all results from the Clickhouse database."""
        pass

    @abstractmethod
    def fetch_column(self, query: str, column: int = 0, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Fetch a specific column from the Clickhouse database."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the connection to the Clickhouse database."""
        pass


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
