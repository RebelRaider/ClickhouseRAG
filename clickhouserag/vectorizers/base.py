"""Base module for vectorizers in ClickhouseRAG."""

from abc import ABC, abstractmethod
from typing import Any, List


class VectorizerBase(ABC):
    """Abstract base class for vectorizers."""

    @abstractmethod
    def vectorize(self, data: Any) -> List[float]:
        """Convert data into a vector representation."""
        pass

    @abstractmethod
    def bulk_vectorize(self, data: Any) -> List[List[float]]:
        """Convert multiple data into a vector representation."""
        pass
