"""Framework-independent retrieval for ClickHouse-backed RAG applications."""

from clickhouserag._communities import Community
from clickhouserag._documents import DocumentRevisionError
from clickhouserag.evaluation import CaseScore, EvalCase, Evaluation, Metrics, evaluate
from clickhouserag.graph import GraphPath, GraphStep, GraphStore, Relation
from clickhouserag.knowledge import (
    Knowledge,
    KnowledgeIntersection,
    KnowledgePlanError,
    KnowledgePlanner,
    KnowledgeQuery,
    KnowledgeResult,
    Predicate,
)
from clickhouserag.models import Chunk, EmbeddedChunk, SchemaError, SearchHit
from clickhouserag.rag import (
    RAG,
    Answer,
    AnswerProvider,
    Context,
    Document,
    ExtractedRelation,
    GeneratedAnswer,
    RelationExtractor,
    Reranker,
)
from clickhouserag.retriever import EmbeddingProvider, Retriever
from clickhouserag.store import ClickHouseStore

__all__ = [
    "RAG",
    "Answer",
    "AnswerProvider",
    "CaseScore",
    "Chunk",
    "ClickHouseStore",
    "Community",
    "Context",
    "Document",
    "DocumentRevisionError",
    "EmbeddedChunk",
    "EmbeddingProvider",
    "EvalCase",
    "Evaluation",
    "ExtractedRelation",
    "GeneratedAnswer",
    "GraphPath",
    "GraphStep",
    "GraphStore",
    "Knowledge",
    "KnowledgeIntersection",
    "KnowledgePlanError",
    "KnowledgePlanner",
    "KnowledgeQuery",
    "KnowledgeResult",
    "Metrics",
    "Predicate",
    "Relation",
    "RelationExtractor",
    "Reranker",
    "Retriever",
    "SchemaError",
    "SearchHit",
    "evaluate",
]
