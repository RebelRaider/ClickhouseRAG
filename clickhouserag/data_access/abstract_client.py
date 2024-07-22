"""Abstract client module for Clickhouse data access."""

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
