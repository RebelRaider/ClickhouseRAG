"""Rank fusion for retrieval lists with different score scales."""

from dataclasses import replace

from clickhouserag.models import SearchHit


def reciprocal_rank_fusion(
    dense: list[SearchHit], lexical: list[SearchHit], limit: int
) -> list[SearchHit]:
    candidates: dict[str, SearchHit] = {}
    scores: dict[str, float] = {}
    for hits in (dense, lexical):
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk.chunk_id
            candidates.setdefault(chunk_id, hit)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)
    ranked = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:limit]
    return [replace(candidates[chunk_id], score=scores[chunk_id]) for chunk_id in ranked]
