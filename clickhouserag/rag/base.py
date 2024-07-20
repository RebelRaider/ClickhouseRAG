from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class RAGBase(ABC):
    @abstractmethod
    def add_data(self, data: Dict[str, Any]) -> None:
        """Add new data to the RAG database."""
        pass

    @abstractmethod
    def delete_data(self, data_id: str) -> None:
        """Delete a specific data entry from the RAG database."""
        pass

    @abstractmethod
    def update_data(self, data_id: str, new_data: Dict[str, Any]) -> None:
        """Update a specific data entry in the RAG database."""
        pass

    @abstractmethod
    def search(self, query: str, similarity: bool = False, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search for data in the RAG database based on a query."""
        pass

    @abstractmethod
    def compute_cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Compute the cosine similarity between two vectors."""
        pass

    @abstractmethod
    def vectorize_data(self, data: Dict[str, Any]) -> List[float]:
        """Convert data into a vector representation."""
        pass

    @abstractmethod
    def reset_database(self) -> None:
        """Reset the RAG database, deleting all data."""
        pass

    @abstractmethod
    def backup_database(self, path: str) -> None:
        """Create a backup of the RAG database to a specified file."""
        pass

    @abstractmethod
    def restore_database(self, path: str) -> None:
        """Restore the RAG database from a backup file."""
        pass

    @abstractmethod
    def save_to_file(self, path: str) -> None:
        """Save the current state of the RAG to a file."""
        pass

    @abstractmethod
    def load_from_file(self, path: str) -> None:
        """Load the state of the RAG from a file."""
        pass

    @abstractmethod
    def validate_data(self, data: Dict[str, Any]) -> None:
        """Validate a data entry before adding or updating."""
        pass
