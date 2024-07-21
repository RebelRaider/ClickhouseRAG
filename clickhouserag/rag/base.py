"""Base module for RAG management."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class RAGBase(ABC):
    """Abstract base class for RAG management."""

    @abstractmethod
    def add_data(self, data: Dict[str, Any]) -> None:
        """Add data to the RAG."""
        pass

    @abstractmethod
    def delete_data(self, data_id: str) -> None:
        """Delete data from the RAG."""
        pass

    @abstractmethod
    def update_data(self, data_id: str, new_data: Dict[str, Any]) -> None:
        """Update data in the RAG."""
        pass

    @abstractmethod
    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search the RAG."""
        pass

    @abstractmethod
    def reset_database(self) -> None:
        """Reset the RAG database."""
        pass
