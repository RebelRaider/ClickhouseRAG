"""Schema-checked, cited reasoning over directed ClickHouse relationships."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from clickhouserag._validation import _MAX_LIMIT, integer
from clickhouserag.graph import GraphPath, GraphStore, canonical_entity


@dataclass(frozen=True, slots=True)
class Predicate:
    """Allowed directed relation and the types of its endpoints."""

    name: str
    subject_type: str
    object_type: str


@dataclass(frozen=True, slots=True)
class KnowledgeQuery:
    """Follow a typed predicate chain from a named entity."""

    entity: str
    entity_type: str
    predicates: tuple[str, ...]


class KnowledgePlanner(Protocol):
    """Translate a question into a schema-aware plan or decline to plan it."""

    def plan(self, question: str, schema: tuple[Predicate, ...]) -> KnowledgeQuery | None:
        """Return a typed relation chain. The framework validates it before querying."""
        ...


@dataclass(frozen=True, slots=True)
class KnowledgeResult:
    """Matched endpoints and the paths that justify each one."""

    entities: tuple[str, ...]
    evidence: tuple[GraphPath, ...]


class Knowledge:
    """Resolve bounded conjunctive path queries against a declared relation schema.

    Stored assertions must carry endpoint types that agree with the schema.
    """

    def __init__(self, graph: GraphStore, schema: tuple[Predicate, ...]) -> None:
        self._graph = graph
        self._schema = {}
        for item in schema:
            name = canonical_entity(item.name)
            if name in self._schema:
                raise ValueError(f"duplicate predicate {name!r}")
            self._schema[name] = (
                canonical_entity(item.subject_type),
                canonical_entity(item.object_type),
            )
        if not self._schema:
            raise ValueError("knowledge schema requires at least one predicate")

    def solve(
        self,
        query: KnowledgeQuery,
        *,
        namespace: str = "default",
        metadata: Mapping[str, str] | None = None,
        limit: int = 10,
    ) -> KnowledgeResult:
        """Find cited paths matching every predicate in the validated chain."""
        integer(limit, "limit", _MAX_LIMIT)
        predicates, entity_types = self._validate_chain(query)
        paths = self._graph.expand(
            [query.entity],
            predicates=predicates,
            entity_types=entity_types,
            depth=len(predicates),
            namespace=namespace,
            metadata=metadata,
            fanout=32,
            limit=_MAX_LIMIT,
        )
        evidence = []
        endpoints: dict[str, None] = {}
        for path in paths:
            key = path.target
            if key not in endpoints:
                if len(endpoints) == limit:
                    continue
                endpoints[key] = None
            evidence.append(path)
        return KnowledgeResult(tuple(endpoints), tuple(evidence))

    def _validate_chain(self, query: KnowledgeQuery) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if not 1 <= len(query.predicates) <= 4:
            raise ValueError("knowledge queries require one to four predicates")
        entity_type = canonical_entity(query.entity_type)
        types = [entity_type]
        predicates = tuple(map(canonical_entity, query.predicates))
        for name in predicates:
            if name not in self._schema:
                raise ValueError(f"unknown predicate {name!r}")
            source, target = self._schema[name]
            if source != entity_type:
                raise ValueError(f"predicate {name!r} does not accept type {entity_type!r}")
            entity_type = target
            types.append(target)
        return predicates, tuple(types)
