from collections.abc import Iterator, Sequence

import pytest
from clickhouse_connect.driver.client import Client

from clickhouserag import (
    RAG,
    Chunk,
    Context,
    Document,
    ExtractedRelation,
    GeneratedAnswer,
    KnowledgeQuery,
    Predicate,
)

pytestmark = pytest.mark.integration


class ToyProvider:
    model_id = "toy-v1"

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[1, 0] if text.startswith("Gateway") else [0, 1] for text in texts]

    def embed_query(self, text: str) -> Sequence[float]:
        return [1, 0]

    def extract(self, chunk: Chunk) -> Sequence[ExtractedRelation]:
        subject, predicate, obj = chunk.text.strip(".").split(" ", maxsplit=2)
        return [
            ExtractedRelation(
                subject,
                predicate,
                obj,
                "service",
                "database" if predicate == "writes" else "service",
            )
        ]

    def plan(self, question: str, schema: tuple[Predicate, ...]) -> KnowledgeQuery | None:
        if question == "Invalid":
            return KnowledgeQuery("gateway", "service", ("missing",))
        return (
            KnowledgeQuery("gateway", "service", ("calls", "writes"))
            if "Gateway" in question
            else None
        )

    def generate(self, question: str, context: Context) -> GeneratedAnswer:
        path = next(path for path in context.paths if len(path.steps) == 2)
        return GeneratedAnswer(path.target, [step.chunk.chunk_id for step in path.steps])


@pytest.fixture
def rag(database_client: Client, table: str) -> Iterator[RAG]:
    provider = ToyProvider()
    rag = RAG(
        database_client,
        provider,
        table=table,
        dimensions=2,
        extractor=provider,
        generator=provider,
        planner=provider,
        schema=(
            Predicate("calls", "service", "service"),
            Predicate("writes", "service", "database"),
        ),
    )
    rag.initialize()
    try:
        yield rag
    finally:
        database_client.command(
            "DROP TABLE IF EXISTS {table:Identifier}", {"table": f"{table}_relations"}
        )


def test_document_to_cited_answer(rag: RAG) -> None:
    one = Document("gateway", "Gateway calls Ledger.", "ref:gateway", {"team": "ops"})
    two = Document("ledger", "Ledger writes Vault.", "ref:ledger", {"team": "ops"})
    assert rag.index(one) == 1
    assert rag.index(two) == 1
    rag.index(one)

    query = "Gateway storage?"
    context = rag.retrieve(query, mode="graph", limit=2, metadata={"team": "ops"})
    assert [chunk.document_id for chunk in context.passages] == ["gateway", "ledger"]
    assert any(path.target == "vault" for path in context.paths)
    reasoning = rag.reason(
        KnowledgeQuery("Gateway", "service", ("calls", "writes")), metadata={"team": "ops"}
    )
    assert reasoning.entities == ("vault",)
    answer = rag.ask(query, limit=2, metadata={"team": "ops"})
    assert answer.text == "vault"
    assert rag.ask_knowledge(query, limit=2).citations == answer.citations
    assert rag.ask_knowledge("Who owns Jupiter?").text is None
    assert rag.ask_knowledge("Invalid").text is None
    assert [chunk.source for chunk in answer.citations] == ["ref:gateway", "ref:ledger"]

    empty = rag.ask(query, mode="graph", metadata={"team": "other"})
    assert empty.text is None
    assert not empty.citations


def test_graph_evidence_survives_reopen(rag: RAG, database_client: Client, table: str) -> None:
    rag.index(Document("a", "Gateway calls Ledger.", "ref:a"))
    provider = ToyProvider()
    reader = RAG(database_client, provider, table=table, dimensions=2, graph=True)
    assert [path.target for path in reader.retrieve("Gateway", mode="graph").paths] == ["ledger"]


def test_generator_cannot_invent_citations(
    rag: RAG,
    database_client: Client,
    table: str,
) -> None:
    class BadModel(ToyProvider):
        def generate(self, question: str, context: Context) -> GeneratedAnswer:
            return GeneratedAnswer("Unsupported", ["missing"])

    rag.index(Document("a", "Gateway calls Ledger.", "ref:a"))
    model = BadModel()
    reader = RAG(database_client, model, table=table, dimensions=2, generator=model)
    with pytest.raises(ValueError, match="supplied context"):
        reader.ask("Gateway", mode="hybrid")


def test_changed_document_is_indexed_as_a_new_snapshot(rag: RAG) -> None:
    rag.index(Document("a", "Gateway calls Ledger.", "ref:a"))
    rag.index(Document("a", "Gateway calls Beta.", "ref:a"))
    assert {p.target for p in rag.retrieve("Gateway", mode="graph").paths} == {"ledger", "beta"}


def test_invalid_embedding_is_rejected_before_extraction(rag: RAG) -> None:
    class Invalid(ToyProvider):
        def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
            return [[0, 0]] * len(texts)

        def extract(self, chunk: Chunk) -> Sequence[ExtractedRelation]:
            pytest.fail("do not extract facts for invalid embeddings")

    provider = Invalid()
    writer = RAG(rag.passages._client, provider, dimensions=2, extractor=provider)
    with pytest.raises(ValueError, match="nonzero"):
        writer.index(Document("a", "Gateway calls Ledger.", "ref:a"))
