"""Rank fusion for retrieval lists with different score scales."""

from collections.abc import Iterable, Sequence
from dataclasses import replace

from clickhouserag.models import SearchHit


def reciprocal_rank_fusion(
    dense: list[SearchHit], lexical: list[SearchHit], limit: int
) -> list[SearchHit]:
    candidates: dict[str, SearchHit] = {}
    ranks = []
    for hits in (dense, lexical):
        ranks.append([hit.chunk.chunk_id for hit in hits])
        for hit in hits:
            candidates.setdefault(hit.chunk.chunk_id, hit)
    return [replace(candidates[key], score=score) for key, score in fused_ranks(ranks)[:limit]]


def fused_ranks(lanes: Iterable[Sequence[str]]) -> list[tuple[str, float]]:
    """Fuse candidate IDs across ranked inputs with a deterministic tie rule."""
    scores: dict[str, float] = {}
    for lane in lanes:
        for rank, chunk_id in enumerate(lane, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)
    return sorted(scores.items(), key=lambda entry: (-entry[1], entry[0]))
