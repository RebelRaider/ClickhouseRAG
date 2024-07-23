from typing import List


def compute_cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
        """Compute the cosine similarity between two vectors."""
        dot_product = sum(p * q for p, q in zip(vector1, vector2))
        magnitude1 = sum(p ** 2 for p in vector1) ** 0.5
        magnitude2 = sum(q ** 2 for q in vector2) ** 0.5
        if magnitude1 * magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)
