from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class ClickhouseClient(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the Clickhouse database."""
        pass

    @abstractmethod
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a given query and return the results."""
        pass

    @abstractmethod
    def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Fetch a single row from the result of the given query."""
        pass

    @abstractmethod
    def fetch_all(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Fetch all rows from the result of the given query."""
        pass

    @abstractmethod
    def fetch_column(self, query: str, column: int = 0, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Fetch a single column from the result of the given query."""
        pass

    @abstractmethod
    def insert(self, table: str, values: List[Dict[str, Any]], params: Optional[Dict[str, Any]] = None) -> None:
        """Insert multiple rows into a specified table."""
        pass

    @abstractmethod
    def update(self, table: str, values: Dict[str, Any], conditions: Dict[str, Any]) -> None:
        """Update rows in a specified table."""
        pass

    @abstractmethod
    def delete(self, table: str, conditions: Dict[str, Any]) -> None:
        """Delete rows from a specified table."""
        pass

    @abstractmethod
    def reset_database(self) -> None:
        """Reset the Clickhouse database by dropping and recreating it."""
        pass

    @abstractmethod
    def start_transaction(self) -> None:
        """Start a new transaction."""
        pass

    @abstractmethod
    def commit_transaction(self) -> None:
        """Commit the current transaction."""
        pass

    @abstractmethod
    def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the connection to the Clickhouse database."""
        pass
