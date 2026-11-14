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


@dataclass(frozen=True, slots=True)
class KnowledgeIntersection:
    """Find endpoint entities reached by every supplied typed path."""

    paths: tuple[KnowledgeQuery, ...]


class KnowledgePlanError(ValueError):
    """A predicate chain cannot be executed against the declared schema."""


class KnowledgePlanner(Protocol):
    """Translate a question into a schema-aware plan or decline to plan it."""

    def plan(
        self, question: str, schema: tuple[Predicate, ...]
    ) -> KnowledgeQuery | KnowledgeIntersection | None:
        """Return a typed relation chain. The framework validates it before querying."""
        ...


@dataclass(frozen=True, slots=True)
class KnowledgeResult:
    """Matched endpoints and the paths that justify each one."""

    entities: tuple[str, ...]
    evidence: tuple[GraphPath, ...]
    evidence_legs: tuple[int, ...]
    required_legs: int


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
        query: KnowledgeQuery | KnowledgeIntersection,
        *,
        namespace: str = "default",
        metadata: Mapping[str, str] | None = None,
        limit: int = 10,
    ) -> KnowledgeResult:
        """Find cited paths matching every predicate in the validated chain."""
        integer(limit, "limit", _MAX_LIMIT)
        queries = query.paths if isinstance(query, KnowledgeIntersection) else (query,)
        if not 1 <= len(queries) <= 8:
            raise KnowledgePlanError("knowledge intersections require one to eight paths")
        try:
            chains = [self._validate_chain(path) for path in queries]
        except ValueError as exc:
            if isinstance(exc, KnowledgePlanError):
                raise
            raise KnowledgePlanError("knowledge query contains an invalid label") from exc
        if len({entity_types[-1] for _, entity_types in chains}) != 1:
            raise KnowledgePlanError("intersected paths must have the same endpoint type")
        bundles: list[dict[str, list[GraphPath]]] = []
        for path, (predicates, entity_types) in zip(queries, chains, strict=True):
            matches = self._graph.expand(
                [path.entity],
                predicates=predicates,
                entity_types=entity_types,
                depth=len(predicates),
                namespace=namespace,
                metadata=metadata,
                fanout=32,
                limit=_MAX_LIMIT,
            )
            grouped: dict[str, list[GraphPath]] = {}
            for match in matches:
                grouped.setdefault(match.target, []).append(match)
            bundles.append(grouped)
        targets = [name for name in bundles[0] if all(name in group for group in bundles[1:])][
            :limit
        ]
        evidence = [
            (leg, path)
            for name in targets
            for leg, group in enumerate(bundles)
            for path in group[name]
        ]
        return KnowledgeResult(
            tuple(targets),
            tuple(path for _, path in evidence),
            tuple(leg for leg, _ in evidence),
            len(bundles),
        )

    def _validate_chain(self, query: KnowledgeQuery) -> tuple[tuple[str, ...], tuple[str, ...]]:
        canonical_entity(query.entity)
        if not 1 <= len(query.predicates) <= 4:
            raise KnowledgePlanError("knowledge queries require one to four predicates")
        entity_type = canonical_entity(query.entity_type)
        types = [entity_type]
        predicates = tuple(map(canonical_entity, query.predicates))
        for name in predicates:
            if name not in self._schema:
                raise KnowledgePlanError(f"unknown predicate {name!r}")
            source, target = self._schema[name]
            if source != entity_type:
                raise KnowledgePlanError(f"predicate {name!r} does not accept type {entity_type!r}")
            entity_type = target
            types.append(target)
        return predicates, tuple(types)
