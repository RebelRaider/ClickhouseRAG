"""Document indexing, mixed retrieval and grounded answer orchestration."""

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Literal, Protocol, TypedDict

from clickhouse_connect.driver.client import Client

from clickhouserag._communities import (
    _DEFAULT_EDGE_BUDGET,
    _MAX_EDGE_BUDGET,
    Community,
    select_communities,
)
from clickhouserag._documents import DocumentStore, _Fact
from clickhouserag._validation import _MAX_BATCH, _MAX_LIMIT, integer, nonblank, string_map
from clickhouserag.graph import GraphPath, GraphStore, Relation, _document_fact
from clickhouserag.knowledge import (
    Knowledge,
    KnowledgeIntersection,
    KnowledgePlanError,
    KnowledgePlanner,
    KnowledgeQuery,
    KnowledgeResult,
    Predicate,
)
from clickhouserag.models import Chunk, SearchHit
from clickhouserag.retriever import EmbeddingProvider, Retriever


@dataclass(frozen=True, slots=True)
class Document:
    """A document snapshot identified within a collection."""

    document_id: str
    text: str
    source: str
    metadata: Mapping[str, str] = field(default_factory=dict)
    revision: int | None = None


@dataclass(frozen=True, slots=True)
class ExtractedRelation:
    """A structured directed relationship extracted from a passage."""

    subject: str
    predicate: str
    object: str
    subject_type: str | None = None
    object_type: str | None = None


class RelationExtractor(Protocol):
    """An application-provided relation extractor, called once per passage."""

    def extract(self, chunk: Chunk) -> Sequence[ExtractedRelation]:
        """Extract directed facts supported by the exact input passage."""
        ...


class Chunker(Protocol):
    """Split source text into an ordered sequence of passages."""

    def split(self, text: str) -> Iterable[str]:
        """Yield source text or application-preprocessed passages in document order."""
        ...


@dataclass(frozen=True, slots=True)
class Context:
    """Ordered source passages and paths whose every step has a citation."""

    passages: tuple[Chunk, ...]
    paths: tuple[GraphPath, ...] = ()
    communities: tuple[Community, ...] = ()
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """A provider response using IDs of supplied evidence passages."""

    text: str | None
    citation_ids: Sequence[str] = ()


class AnswerProvider(Protocol):
    """An application-provided generator that selects supporting passage IDs."""

    def generate(self, question: str, context: Context) -> GeneratedAnswer:
        """Generate an answer grounded in the supplied context."""
        ...


class Reranker(Protocol):
    """Return a permutation of the ClickHouse candidates, best match first."""

    def rank(self, query: str, passages: Sequence[Chunk]) -> Sequence[int]:
        """Return only indices into the input sequence, with each index used once."""
        ...


@dataclass(frozen=True, slots=True)
class Answer:
    """A generated answer and its verified references, or a no-evidence result."""

    text: str | None
    citations: tuple[Chunk, ...]
    context: Context


class _Scope(TypedDict):
    namespace: str
    metadata: Mapping[str, str] | None


_Mode = Literal["dense", "lexical", "hybrid", "graph", "mixed", "global"]


class RAG:
    """Index text documents, retrieve cited evidence and optionally ask a model."""

    def __init__(
        self,
        client: Client,
        provider: EmbeddingProvider,
        *,
        dimensions: int,
        table: str = "rag_passages",
        database: str = "default",
        namespace: str = "default",
        extractor: RelationExtractor | None = None,
        generator: AnswerProvider | None = None,
        graph: bool = False,
        schema: tuple[Predicate, ...] | None = None,
        planner: KnowledgePlanner | None = None,
        reranker: Reranker | None = None,
        rerank_candidates: int = 32,
        chunker: Chunker | None = None,
        index_revision: str | None = None,
        chunk_size: int = 1600,
        chunk_overlap: int = 160,
    ) -> None:
        nonblank(namespace, "namespace")
        integer(chunk_size, "chunk_size", 100_000)
        integer(rerank_candidates, "rerank_candidates", _MAX_LIMIT)
        if type(chunk_overlap) is not int or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be a nonnegative integer less than chunk_size")
        if chunker is not None and index_revision is None:
            raise ValueError("a custom chunker requires an index_revision")
        if index_revision is not None:
            nonblank(index_revision, "index_revision")
        self.passages = DocumentStore(
            client,
            table=table,
            database=database,
            embedding_model=provider.model_id,
            dimensions=dimensions,
        )
        self.graph = (
            GraphStore(self.passages)
            if graph or extractor is not None or schema is not None
            else None
        )
        self._knowledge = (
            Knowledge(self.graph, schema) if self.graph is not None and schema is not None else None
        )
        self._retriever = Retriever(self.passages, provider)
        self._namespace = namespace
        if planner is not None and schema is None:
            raise ValueError("a knowledge planner requires a schema")
        self._schema = schema or ()
        self._planner = planner
        self._extractor = extractor
        self._generator = generator
        self._reranker = reranker
        self._rerank_candidates = rerank_candidates
        self._chunker = chunker
        self._index_revision = index_revision
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def initialize(self) -> None:
        """Provision the managed passage and fact table."""
        self.passages.initialize()

    def index(self, document: Document) -> int:
        """Publish this document's full contents; return the number of staged passages."""
        nonblank(document.document_id, "document_id")
        nonblank(document.source, "source")
        nonblank(document.text, "text")
        metadata = string_map(document.metadata)
        text = document.text.strip()
        config = [
            self._chunk_size,
            self._chunk_overlap,
            self._extractor is not None,
        ]
        if self._index_revision is not None:
            config.extend((self._index_revision, self._chunker is not None))
        identity = hashlib.sha256(
            json.dumps(
                [
                    text,
                    document.source,
                    metadata,
                    *config,
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        revision = self.passages.next_revision(
            self._namespace, document.document_id, identity, document.revision
        )
        if revision is None:
            return 0
        chunks = self._chunks(document, text, metadata, revision)
        rows = self.passages._prepare_rows(self._retriever._embed(chunks), self._namespace)
        facts = [self._facts(chunk) for chunk in chunks]
        self.passages.publish(
            self._namespace, document.document_id, revision, identity, rows, facts
        )
        return len(rows)

    def delete(self, document_id: str, *, revision: int | None = None) -> bool:
        """Hide a document and its facts by publishing a tombstone."""
        nonblank(document_id, "document_id")
        chosen = self.passages.next_revision(self._namespace, document_id, "", revision)
        if chosen is None:
            return False
        self.passages.publish(self._namespace, document_id, chosen, "", [], [])
        return True

    def _facts(self, chunk: Chunk) -> list[_Fact]:
        if self._extractor is None:
            return []
        extracted = self._extractor.extract(chunk)
        if len(extracted) > _MAX_BATCH:
            raise ValueError(f"a passage may contain at most {_MAX_BATCH} relations")
        return sorted(
            _document_fact(
                Relation(
                    item.subject,
                    item.predicate,
                    item.object,
                    chunk.chunk_id,
                    chunk.revision,
                    item.subject_type,
                    item.object_type,
                )
            )
            for item in extracted
        )

    def retrieve(
        self,
        query: str,
        *,
        mode: _Mode = "hybrid",
        metadata: Mapping[str, str] | None = None,
        seeds: Sequence[str] | None = None,
        limit: int = 10,
        depth: int = 2,
    ) -> Context:
        """Retrieve passages and whole cited paths within the passage limit."""
        nonblank(query, "query")
        integer(limit, "limit", _MAX_LIMIT)
        if mode not in ("dense", "lexical", "hybrid", "graph", "mixed", "global"):
            raise ValueError("unknown retrieval mode")
        if seeds is not None and mode not in ("graph", "mixed"):
            raise ValueError("explicit seeds require graph or mixed mode")
        if mode == "global":
            return self.overview(query, metadata=metadata, limit=limit)
        if mode == "graph" and self.graph is None:
            raise ValueError("graph mode requires graph configuration")
        filters = string_map(metadata) if metadata is not None else None
        scope: _Scope = {"namespace": self._namespace, "metadata": filters}
        candidate_limit = (
            max(limit, self._rerank_candidates) if self._reranker is not None else limit
        )
        hits: list[SearchHit] = []
        if mode == "lexical":
            hits = self.passages.search_lexical(query, limit=candidate_limit, **scope)
        elif mode != "graph":
            hits = self._retriever.search(
                query, limit=candidate_limit, hybrid=mode != "dense", **scope
            )
        paths = []
        if self.graph is not None and mode in ("graph", "mixed"):
            selected = self.graph.seed_entities(query, **scope) if seeds is None else seeds
            paths = self.graph.expand(selected, depth=depth, limit=min(limit * 4, 1000), **scope)
            hits, paths = self._visible_evidence(hits, paths)
        if self._reranker is not None and len(hits) > 1:
            order = tuple(self._reranker.rank(query, tuple(hit.chunk for hit in hits)))
            if (
                len(order) != len(hits)
                or any(type(index) is not int for index in order)
                or set(order) != set(range(len(hits)))
            ):
                raise ValueError("reranker must return a permutation of the candidate indices")
            hits, paths = self._visible_evidence([hits[index] for index in order], paths)
        return _context(hits, paths, limit)

    def overview(
        self,
        query: str,
        *,
        metadata: Mapping[str, str] | None = None,
        limit: int = 10,
        max_facts: int = _DEFAULT_EDGE_BUDGET,
    ) -> Context:
        """Return cited representatives of connected groups across the scoped corpus."""
        nonblank(query, "query")
        integer(limit, "limit", _MAX_LIMIT)
        integer(max_facts, "max_facts", _MAX_EDGE_BUDGET)
        if self.graph is None:
            raise ValueError("global mode requires graph configuration")
        facts = self.graph._corpus_facts(self._namespace, metadata, max_facts + 1)
        communities = select_communities(query, facts[:max_facts], limit=limit)
        passages = {
            chunk.chunk_id: chunk
            for row in zip_longest(*(item.passages for item in communities))
            for chunk in row
            if chunk is not None
        }
        return Context(
            tuple(passages.values()), communities=communities, truncated=len(facts) > max_facts
        )

    def reason(
        self,
        query: KnowledgeQuery | KnowledgeIntersection,
        *,
        metadata: Mapping[str, str] | None = None,
        limit: int = 10,
    ) -> KnowledgeResult:
        """Resolve a typed knowledge plan in the same collection as text retrieval."""
        if self._knowledge is None:
            raise ValueError("reason requires a knowledge schema")
        result = self._knowledge.solve(
            query, namespace=self._namespace, metadata=metadata, limit=limit
        )
        chunks = [step.chunk for path in result.evidence for step in path.steps]
        visible = self.passages.visible_chunks(self._namespace, chunks)
        evidence = [
            (leg, path)
            for leg, path in zip(result.evidence_legs, result.evidence, strict=True)
            if all((step.chunk.chunk_id, step.chunk.revision) in visible for step in path.steps)
        ]
        proof: defaultdict[str, set[int]] = defaultdict(set)
        for leg, path in evidence:
            proof[path.target].add(leg)
        targets = tuple(
            target for target in result.entities if len(proof[target]) == result.required_legs
        )
        target_set = set(targets)
        return KnowledgeResult(
            targets,
            tuple(path for _, path in evidence if path.target in target_set),
            tuple(leg for leg, path in evidence if path.target in target_set),
            result.required_legs,
        )

    def ask_knowledge(
        self,
        question: str,
        *,
        plan: KnowledgeQuery | KnowledgeIntersection | None = None,
        metadata: Mapping[str, str] | None = None,
        limit: int = 10,
    ) -> Answer:
        """Execute typed reasoning and answer with cited proof, optionally using a supplied plan."""
        nonblank(question, "question")
        integer(limit, "limit", _MAX_LIMIT)
        if self._generator is None:
            raise ValueError("ask_knowledge requires an answer provider")
        filters = string_map(metadata) if metadata is not None else None
        if plan is None:
            if self._planner is None:
                raise ValueError("ask_knowledge requires a knowledge planner or an explicit plan")
            proposed = self._planner.plan(question, self._schema)
            try:
                result = self.reason(proposed, metadata=filters, limit=limit) if proposed else None
            except KnowledgePlanError:
                result = None
            if result is None:
                return Answer(None, (), Context(()))
        else:
            result = self.reason(plan, metadata=filters, limit=limit)
        context = _knowledge_context(result, limit)
        answer = self._answer(question, context)
        if answer.text is None:
            return answer
        cited = {chunk.chunk_id for chunk in answer.citations}
        covered: defaultdict[str, set[int]] = defaultdict(set)
        for leg, path in zip(result.evidence_legs, result.evidence, strict=True):
            if all(step.chunk.chunk_id in cited for step in path.steps):
                covered[path.target].add(leg)
        if any(len(proof) == result.required_legs for proof in covered.values()):
            return answer
        return Answer(None, (), context)

    def ask(
        self,
        question: str,
        *,
        mode: _Mode = "hybrid",
        metadata: Mapping[str, str] | None = None,
        seeds: Sequence[str] | None = None,
        limit: int = 10,
        depth: int = 2,
    ) -> Answer:
        """Generate with evidence and reject citation IDs outside the context."""
        if self._generator is None:
            raise ValueError("ask requires an answer provider")
        context = self.retrieve(
            question, mode=mode, metadata=metadata, seeds=seeds, limit=limit, depth=depth
        )
        return self._answer(question, context)

    def _answer(self, question: str, context: Context) -> Answer:
        if self._generator is None:
            raise ValueError("asking a question requires an answer provider")
        if not context.passages:
            return Answer(None, (), context)
        result = self._generator.generate(question, context)
        if result.text is None:
            if result.citation_ids:
                raise ValueError("an abstention must not have citations")
            return Answer(None, (), context)
        nonblank(result.text, "answer")
        known = {chunk.chunk_id: chunk for chunk in context.passages}
        if (
            isinstance(result.citation_ids, str)
            or not result.citation_ids
            or any(key not in known for key in result.citation_ids)
        ):
            raise ValueError("answer provider must cite IDs from the supplied context")
        return Answer(
            result.text, tuple(known[key] for key in dict.fromkeys(result.citation_ids)), context
        )

    def _visible_evidence(
        self, hits: list[SearchHit], paths: Sequence[GraphPath]
    ) -> tuple[list[SearchHit], list[GraphPath]]:
        chunks = [hit.chunk for hit in hits]
        chunks.extend(step.chunk for path in paths for step in path.steps)
        visible = self.passages.visible_chunks(self._namespace, chunks)
        return (
            [hit for hit in hits if (hit.chunk.chunk_id, hit.chunk.revision) in visible],
            [
                path
                for path in paths
                if all((step.chunk.chunk_id, step.chunk.revision) in visible for step in path.steps)
            ],
        )

    def _chunks(
        self, document: Document, text: str, metadata: Mapping[str, str], revision: int
    ) -> list[Chunk]:
        if self._chunker is None:
            located = self._character_chunks(text)
        else:
            parts = self._chunker.split(text)
            if isinstance(parts, (str, bytes)):
                raise ValueError("chunker must return an iterable of passages, not a string")
            located = enumerate(parts)
        chunks = []
        for offset, part in located:
            nonblank(part, "chunk text")
            if len(chunks) == _MAX_BATCH:
                raise ValueError(f"document produces more than {_MAX_BATCH} chunks")
            key = hashlib.sha256(
                f"{document.document_id}\0{revision}\0{offset}".encode()
            ).hexdigest()
            chunks.append(
                Chunk(key, document.document_id, part, document.source, revision, metadata)
            )
        if not chunks:
            raise ValueError("chunker must produce at least one passage")
        return chunks

    def _character_chunks(self, text: str) -> Iterable[tuple[int, str]]:
        for start in range(0, len(text), self._chunk_size - self._chunk_overlap):
            part = text[start : start + self._chunk_size].strip()
            if part:
                yield start, part
            if start + self._chunk_size >= len(text):
                break


def _context(hits: list[SearchHit], paths: list[GraphPath], limit: int) -> Context:
    passages: dict[str, Chunk] = {}
    accepted = []
    # Prefer complete paths, then fill the remaining budget with ranked passages.
    for path in sorted(paths, key=lambda current: -len(current.steps)):
        cite = {step.chunk.chunk_id: step.chunk for step in path.steps}
        if any(
            key in passages and passages[key].revision != chunk.revision
            for key, chunk in cite.items()
        ):
            continue
        if len(passages.keys() | cite.keys()) <= limit:
            passages.update(cite)
            accepted.append(path)
    for hit in hits:
        if len(passages) == limit:
            break
        passages.setdefault(hit.chunk.chunk_id, hit.chunk)
    return Context(tuple(passages.values()), tuple(accepted))


def _knowledge_context(result: KnowledgeResult, limit: int) -> Context:
    if result.required_legs == 1:
        return _context([], list(result.evidence), limit)
    proofs: defaultdict[str, dict[int, GraphPath]] = defaultdict(dict)
    for leg, path in zip(result.evidence_legs, result.evidence, strict=True):
        proofs[path.target].setdefault(leg, path)
    selected: list[GraphPath] = []
    known: set[str] = set()
    for target in result.entities:
        paths = tuple(proofs[target].values())
        needed = {step.chunk.chunk_id for path in paths for step in path.steps}
        if len(known | needed) <= limit:
            selected.extend(paths)
            known.update(needed)
    return _context([], selected, limit)
