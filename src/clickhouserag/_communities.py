"""Deterministic connected components and evidence coverage for global search."""

from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

from clickhouserag._lexical import terms
from clickhouserag.graph import GraphStep
from clickhouserag.models import Chunk

_DEFAULT_EDGE_BUDGET = 5_000
_MAX_EDGE_BUDGET = 25_000


@dataclass(frozen=True, slots=True)
class Community:
    """A connected group of entities and its selected supporting passages."""

    entities: tuple[str, ...]
    passages: tuple[Chunk, ...]
    document_count: int
    fact_count: int


def select_communities(query: str, facts: list[GraphStep], *, limit: int) -> tuple[Community, ...]:
    adjacency: defaultdict[str, set[str]] = defaultdict(set)
    for step in facts:
        source, target = step.relation.subject, step.relation.object
        adjacency[source].add(target)
        adjacency[target].add(source)

    group_by_entity = _components(adjacency)
    grouped: defaultdict[str, list[GraphStep]] = defaultdict(list)
    for step in facts:
        grouped[group_by_entity[step.relation.subject]].append(step)
    entities: defaultdict[str, list[str]] = defaultdict(list)
    for name, group in sorted(group_by_entity.items()):
        entities[group].append(name)

    query_terms = set(terms(query))
    candidates = []
    for group, steps in grouped.items():
        labels = tuple(entities[group])
        docs = {step.chunk.document_id for step in steps}
        triples = {
            (step.relation.subject, step.relation.predicate, step.relation.object) for step in steps
        }
        predicates = {predicate for _, predicate, _ in triples}
        overlap = len(query_terms.intersection(terms(" ".join((*labels, *sorted(predicates))))))
        rank = (-overlap, -len(docs), -len(triples), labels)
        candidates.append(
            (rank, Community(labels, _representatives(steps), len(docs), len(triples)))
        )

    ranked = [item for _, item in sorted(candidates, key=lambda entry: entry[0])[:limit]]
    selected: list[list[Chunk]] = [[] for _ in ranked]
    passage_ids: set[str] = set()
    for index, passage in _round_robin(item.passages for item in ranked):
        if passage.chunk_id not in passage_ids and len(passage_ids) == limit:
            continue
        passage_ids.add(passage.chunk_id)
        selected[index].append(passage)

    return tuple(
        Community(item.entities, tuple(passages), item.document_count, item.fact_count)
        for item, passages in zip(ranked, selected, strict=True)
        if passages
    )


def _components(adjacency: Mapping[str, set[str]]) -> dict[str, str]:
    groups: dict[str, str] = {}
    for root in sorted(adjacency):
        if root in groups:
            continue
        frontier = [root]
        groups[root] = root
        while frontier:
            for neighbor in adjacency[frontier.pop()]:
                if neighbor not in groups:
                    groups[neighbor] = root
                    frontier.append(neighbor)
    return groups


def _representatives(steps: Iterable[GraphStep]) -> tuple[Chunk, ...]:
    # Interleave documents so a prolific source cannot fill the whole context.
    by_document: defaultdict[str, dict[str, Chunk]] = defaultdict(dict)
    for step in steps:
        by_document[step.chunk.document_id].setdefault(step.chunk.chunk_id, step.chunk)
    return tuple(
        chunk
        for _, chunk in _round_robin(values.values() for _, values in sorted(by_document.items()))
    )


def _round_robin(groups: Iterable[Iterable[Chunk]]) -> Iterator[tuple[int, Chunk]]:
    pending = deque((position, iter(items)) for position, items in enumerate(groups))
    while pending:
        position, items = pending.popleft()
        item = next(items, None)
        if item is not None:
            yield position, item
            pending.append((position, items))
