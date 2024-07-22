"""Manager module for vectorizers in ClickhouseRAG."""

from typing import Any, Dict, List, Optional

from clickhouserag.vectorizers.base import VectorizerBase


class VectorizerManager:
    """Manager class for handling multiple vectorizers."""

    def __init__(self):
        """Initialize VectorizerManager."""
        self.vectorizers: Dict[str, VectorizerBase] = {}

    def add_vectorizer(self, name: str, vectorizer: VectorizerBase) -> None:
        """Add a vectorizer to the manager."""
        self.vectorizers[name] = vectorizer

    def remove_vectorizer(self, name: str) -> None:
        """Remove a vectorizer from the manager."""
        if name in self.vectorizers:
            del self.vectorizers[name]

    def get_vectorizer(self, name: str) -> Optional[VectorizerBase]:
        """Get a vectorizer by name."""
        return self.vectorizers.get(name)

    def vectorize(self, name: str, data: Any) -> List[float]:
        """Vectorize data using the specified vectorizer."""
        vectorizer = self.get_vectorizer(name)
        if vectorizer:
            return vectorizer.vectorize(data)
        else:
            raise ValueError(f"Vectorizer '{name}' not found.")
