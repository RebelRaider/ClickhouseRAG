"""Versioned graph evidence and bounded directed traversal in ClickHouse."""

import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from clickhouserag._lexical import terms
from clickhouserag._validation import _MAX_BATCH, identifier, integer, nonblank
from clickhouserag.models import Chunk, SchemaError
from clickhouserag.store import _SELECT_SETTINGS, ClickHouseStore, _scope

_GRAPH_SCHEMA = "clickhouserag:graph:3"
_MAX_DEPTH = 4
_MAX_FRONTIER = 128
_MAX_FANOUT = 64


@dataclass(frozen=True, slots=True)
class Relation:
    """A directed fact asserted by one specific passage revision."""

    subject: str
    predicate: str
    object: str
    chunk_id: str
    revision: int = 1
    subject_type: str | None = None
    object_type: str | None = None


@dataclass(frozen=True, slots=True)
class GraphStep:
    """A relationship and the current passage supporting it."""

    relation: Relation
    chunk: Chunk


@dataclass(frozen=True, slots=True)
class GraphPath:
    """An ordered sequence of cited relationships from one seed."""

    seed: str
    steps: tuple[GraphStep, ...]

    @property
    def target(self) -> str:
        """Canonical entity ID at the end of this path."""
        return self.steps[-1].relation.object if self.steps else self.seed


class GraphStore:
    """Store cited relationships alongside a ClickHouse passage store."""

    def __init__(self, passages: ClickHouseStore, *, table: str | None = None) -> None:
        self._passages = passages
        self._client = passages._client
        database = passages._location["database"]
        passage_table = passages._location["table"]
        if table is None:
            table = f"{passage_table}_relations"
        self._qualified_table = f"{identifier(database)}.{identifier(table)}"
        self._location = {"database": database, "table": table, "passage_table": passage_table}
        self._manifest = f"{_GRAPH_SCHEMA}:{passage_table}"
        self._validated = False

    def initialize(self) -> None:
        """Provision the companion table after the passage table has been created."""
        self._passages.validate_schema()
        self._client.command(
            f"""
            CREATE TABLE IF NOT EXISTS {self._qualified_table} (
                namespace String,
                subject String,
                predicate String,
                object String,
                chunk_id String,
                revision UInt64,
                subject_type String,
                object_type String,
                entity_terms Array(String),
                INDEX entity_idx(entity_terms) TYPE text(tokenizer = 'array')
            )
            ENGINE = ReplacingMergeTree
            ORDER BY (namespace, subject, predicate, object, chunk_id, revision,
                subject_type, object_type)
            COMMENT %(manifest)s
            """,
            parameters={"manifest": self._manifest},
        )
        self.validate_schema()

    def validate_schema(self) -> None:
        """Verify the companion table belongs to the configured passage table."""
        self._validated = False
        self._passages.validate_schema()
        result = self._client.query(
            """
            SELECT comment FROM system.tables
            WHERE database = {database:String} AND name = {table:String}
            """,
            parameters=self._location,
        )
        if result.result_rows != [(self._manifest,)]:
            raise SchemaError("ClickHouse graph table is missing or belongs to another store")
        self._validated = True

    def insert(self, relations: Iterable[Relation], *, namespace: str = "default") -> int:
        """Validate and append a batch of up to 10,000 cited relationships."""
        nonblank(namespace, "namespace")
        rows = {}
        for count, relation in enumerate(relations, start=1):
            if count > _MAX_BATCH:
                raise ValueError(f"insert accepts at most {_MAX_BATCH} relations per batch")
            row = _relation_row(namespace, relation)
            rows[row] = None
        if not rows:
            return 0
        self._ensure_schema()
        self._client.insert(
            self._qualified_table,
            [(*row, terms(f"{row[1]} {row[3]}")) for row in rows],
            column_names=(
                "namespace",
                "subject",
                "predicate",
                "object",
                "chunk_id",
                "revision",
                "subject_type",
                "object_type",
                "entity_terms",
            ),
            settings={"async_insert": 0},
        )
        return len(rows)

    def expand(
        self,
        seeds: Sequence[str],
        *,
        namespace: str = "default",
        metadata: Mapping[str, str] | None = None,
        depth: int = 2,
        fanout: int = 8,
        limit: int = 64,
        predicates: Sequence[str] | None = None,
        entity_types: Sequence[str] | None = None,
    ) -> list[GraphPath]:
        """Return bounded simple paths, shortest first or matching a full predicate chain."""
        integer(depth, "depth", _MAX_DEPTH)
        integer(fanout, "fanout", _MAX_FANOUT)
        parameters = _scope(namespace, metadata, limit)
        ids = _entities(seeds, "seeds")
        chain = _labels_for_hops(predicates, depth, "predicates")
        types = _labels_for_hops(entity_types, depth + 1, "entity_types")
        frontier = [(GraphPath(seed, ()), frozenset((seed,))) for seed in ids]
        paths: list[GraphPath] = []
        for hop in range(depth):
            if not frontier:
                break
            constraints = {
                "predicate": chain[hop] if chain else "",
                "subject_type": types[hop] if types else "",
                "object_type": types[hop + 1] if types else "",
            }
            grouped = self._neighbors(
                {path.target for path, _ in frontier}, parameters, constraints, fanout
            )
            next_frontier = []
            for path, visited in frontier:
                for step in grouped[path.target]:
                    target = step.relation.object
                    if target in visited or any(
                        previous.chunk.chunk_id == step.chunk.chunk_id
                        and previous.chunk.revision != step.chunk.revision
                        for previous in path.steps
                    ):
                        continue
                    next_path = GraphPath(path.seed, (*path.steps, step))
                    if not chain or hop == depth - 1:
                        paths.append(next_path)
                        if len(paths) == limit:
                            return paths
                    if hop < depth - 1 and len(next_frontier) < _MAX_FRONTIER:
                        next_frontier.append((next_path, visited | {target}))
            frontier = next_frontier
        return paths

    def seed_entities(
        self,
        query: str,
        *,
        namespace: str = "default",
        metadata: Mapping[str, str] | None = None,
        limit: int = 8,
    ) -> list[str]:
        """Rank entity labels matching the query, using only visible cited facts."""
        integer(limit, "limit", _MAX_FRONTIER)
        params = _scope(namespace, metadata, limit)
        nonblank(query, "query")
        query_terms = terms(query)[:128]
        if not query_terms:
            return []
        self._ensure_schema()
        result = self._client.query(
            """
            WITH matches AS (
                SELECT subject, object, chunk_id, revision
                FROM {database:Identifier}.{table:Identifier} FINAL
                WHERE namespace = {namespace:String}
                    AND hasAnyTokens(entity_terms, {terms:Array(String)})
            ), supporting AS (
                SELECT chunk_id, revision
                FROM {database:Identifier}.{passage_table:Identifier} FINAL
                WHERE namespace = {namespace:String}
                    AND chunk_id IN (SELECT chunk_id FROM matches)
                    AND arrayAll(pair -> mapContains(metadata, pair.1)
                        AND metadata[pair.1] = pair.2, {filters:Array(Tuple(String, String))})
            )
            SELECT DISTINCT m.subject, m.object
            FROM matches AS m
            INNER JOIN supporting AS p ON m.chunk_id = p.chunk_id AND m.revision = p.revision
            ORDER BY m.subject, m.object
            LIMIT 4096
            """,
            parameters={**params, **self._location, "terms": query_terms},
            settings=_SELECT_SETTINGS,
        )
        tokens = set(query_terms)
        labels = {label for row in result.result_rows for label in row}
        scored = []
        for label in labels:
            label_terms = set(terms(label))
            if matches := len(label_terms & tokens):
                scored.append((label, matches, matches / len(label_terms)))
        scored.sort(key=lambda item: (-item[1], -item[2], item[0]))
        return [label for label, _, _ in scored[:limit]]

    def _ensure_schema(self) -> None:
        if not self._validated:
            self.validate_schema()

    def _neighbors(
        self, subjects: set[str], scope: dict[str, Any], constraints: dict[str, str], fanout: int
    ) -> defaultdict[str, list[GraphStep]]:
        self._ensure_schema()
        result = self._client.query(
            """
            WITH edges AS (
                SELECT subject, predicate, object, chunk_id, revision, subject_type, object_type
                FROM {database:Identifier}.{table:Identifier} FINAL
                WHERE namespace = {namespace:String}
                    AND subject IN {subjects:Array(String)}
                    AND ({predicate:String} = '' OR predicate = {predicate:String})
                    AND ({subject_type:String} = '' OR subject_type = {subject_type:String})
                    AND ({object_type:String} = '' OR object_type = {object_type:String})
            ), supporting AS (
                SELECT chunk_id, document_id, text, source, revision, metadata
                FROM {database:Identifier}.{passage_table:Identifier} FINAL
                WHERE namespace = {namespace:String}
                    AND chunk_id IN (SELECT chunk_id FROM edges)
                    AND arrayAll(pair -> mapContains(metadata, pair.1)
                        AND metadata[pair.1] = pair.2, {filters:Array(Tuple(String, String))})
            )
            SELECT e.subject, e.predicate, e.object, e.chunk_id, e.revision,
                p.document_id, p.text, p.source, p.metadata, e.subject_type, e.object_type
            FROM edges AS e
            INNER JOIN supporting AS p ON e.chunk_id = p.chunk_id AND e.revision = p.revision
            ORDER BY e.subject, e.predicate, e.object, e.chunk_id, e.subject_type, e.object_type
            LIMIT {fanout:UInt32} BY e.subject
            """,
            parameters={
                **scope,
                **self._location,
                "subjects": sorted(subjects),
                **constraints,
                "fanout": fanout,
            },
            settings=_SELECT_SETTINGS,
        )
        grouped: defaultdict[str, list[GraphStep]] = defaultdict(list)
        for (
            subject,
            predicate,
            obj,
            chunk_id,
            revision,
            document_id,
            text,
            source,
            metadata,
            subject_type,
            object_type,
        ) in result.result_rows:
            grouped[subject].append(
                GraphStep(
                    Relation(
                        subject,
                        predicate,
                        obj,
                        chunk_id,
                        revision,
                        subject_type or None,
                        object_type or None,
                    ),
                    Chunk(chunk_id, document_id, text, source, revision, metadata),
                )
            )
        return grouped


def canonical_entity(value: str) -> str:
    """Normalize entity and predicate labels for matching and stored IDs."""
    nonblank(value, "entity or predicate")
    result = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    if len(result) > 512:
        raise ValueError("entity and predicate labels must have at most 512 characters")
    return result


def _entities(values: Sequence[str], name: str) -> list[str]:
    if isinstance(values, str) or len(values) > _MAX_FRONTIER:
        raise ValueError(f"{name} must be a sequence of up to {_MAX_FRONTIER} entity labels")
    return list(dict.fromkeys(canonical_entity(value) for value in values))


def _labels_for_hops(values: Sequence[str] | None, count: int, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str) or len(values) != count:
        raise ValueError(f"{name} must contain {count} labels")
    return tuple(map(canonical_entity, values))


def _relation_row(
    namespace: str, relation: Relation
) -> tuple[str, str, str, str, str, int, str, str]:
    nonblank(relation.chunk_id, "chunk_id")
    integer(relation.revision, "revision", 2**64 - 1)
    if (relation.subject_type is None) != (relation.object_type is None):
        raise ValueError("relation must supply both endpoint types or neither")
    return (
        namespace,
        canonical_entity(relation.subject),
        canonical_entity(relation.predicate),
        canonical_entity(relation.object),
        relation.chunk_id,
        relation.revision,
        canonical_entity(relation.subject_type) if relation.subject_type is not None else "",
        canonical_entity(relation.object_type) if relation.object_type is not None else "",
    )
