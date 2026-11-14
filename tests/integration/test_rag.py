from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch

import pytest
from clickhouse_connect.driver.client import Client

from clickhouserag import (
    RAG,
    Chunk,
    Context,
    Document,
    DocumentRevisionError,
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
def rag(database_client: Client, table: str) -> RAG:
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
    return rag


def test_document_to_cited_answer(rag: RAG) -> None:
    one = Document("gateway", "Gateway calls Ledger.", "ref:gateway", {"team": "ops"})
    two = Document("ledger", "Ledger writes Vault.", "ref:ledger", {"team": "ops"})
    assert rag.index(one) == 1
    assert rag.index(two) == 1
    assert rag.index(one) == 0

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
    assert {p.target for p in rag.retrieve("Gateway", mode="graph").paths} == {"beta"}


def test_invalid_embedding_is_rejected_before_extraction(rag: RAG, table: str) -> None:
    class Invalid(ToyProvider):
        def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
            return [[0, 0]] * len(texts)

        def extract(self, chunk: Chunk) -> Sequence[ExtractedRelation]:
            pytest.fail("do not extract facts for invalid embeddings")

    provider = Invalid()
    writer = RAG(rag.passages._client, provider, table=table, dimensions=2, extractor=provider)
    with pytest.raises(ValueError, match="nonzero"):
        writer.index(Document("a", "Gateway calls Ledger.", "ref:a"))


def test_document_revision_filters_and_delete(
    rag: RAG, database_client: Client, table: str
) -> None:
    database_client.command("SYSTEM STOP MERGES {table:Identifier}", {"table": table})
    original = Document("gateway", "Gateway calls Ledger.", "ref:old", {"team": "ops"}, 7)
    replacement = Document("gateway", "Gateway calls Beta.", "ref:new", {"team": "other"}, 8)
    ledger = Document("ledger", "Ledger writes Vault.", "ref:ledger", {"team": "ops"})
    rag.index(original)
    rag.index(ledger)
    assert rag.reason(KnowledgeQuery("Gateway", "service", ("calls", "writes"))).entities == (
        "vault",
    )
    rag.index(replacement)
    assert rag.index(replacement) == 0
    with pytest.raises(DocumentRevisionError, match="older"):
        rag.index(original)
    with pytest.raises(DocumentRevisionError, match="different"):
        rag.index(Document("gateway", "Gateway calls Unrelated.", "ref:other", revision=8))

    assert rag.retrieve("Gateway", mode="lexical", metadata={"team": "ops"}).passages == ()
    assert [path.target for path in rag.retrieve("Gateway", mode="graph").paths] == ["beta"]
    assert rag.reason(KnowledgeQuery("Gateway", "service", ("calls", "writes"))).entities == ()
    assert rag.delete("gateway", revision=9)
    assert not rag.delete("gateway", revision=9)
    assert rag.retrieve("Gateway", mode="lexical").passages == ()
    assert rag.retrieve("Gateway", mode="graph").paths == ()
    assert [c.document_id for c in rag.retrieve("Vault", mode="dense").passages] == ["ledger"]


def test_failed_publication_leaves_old_passages_and_facts_visible(
    rag: RAG, database_client: Client
) -> None:
    rag.index(Document("gateway", "Gateway calls Ledger.", "ref:old"))
    insert = database_client.insert

    def fail_marker(*args, **kwargs):
        if "facts" not in kwargs["column_names"]:
            assert [path.target for path in rag.retrieve("Gateway", mode="graph").paths] == [
                "ledger"
            ]
            raise RuntimeError("publication interrupted")
        return insert(*args, **kwargs)

    replacement = Document("gateway", "Gateway calls Beta.", "ref:new")
    with (
        patch.object(database_client, "insert", side_effect=fail_marker),
        pytest.raises(RuntimeError, match="interrupted"),
    ):
        rag.index(replacement)
    assert [chunk.source for chunk in rag.retrieve("Gateway", mode="lexical").passages] == [
        "ref:old"
    ]
    rag.index(replacement)
    assert [path.target for path in rag.retrieve("Gateway", mode="graph").paths] == ["beta"]


def test_concurrent_generations_do_not_mix(rag: RAG, database_client: Client, table: str) -> None:
    barrier = Barrier(2)

    class ConcurrentProvider(ToyProvider):
        def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
            barrier.wait(timeout=10)
            return super().embed_documents(texts)

    provider = ConcurrentProvider()
    writer = RAG(database_client, provider, table=table, dimensions=2, extractor=provider)
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(
            executor.map(
                writer.index,
                [
                    Document("gateway", "Gateway calls Ledger.", "ref:ledger", revision=1),
                    Document("gateway", "Gateway calls Beta.", "ref:beta", revision=1),
                ],
            )
        )
    assert outcomes == [1, 1]
    result = rag.retrieve("Gateway", mode="mixed")
    assert len(result.paths) == len(result.passages) == 1
    assert result.paths[0].steps[0].chunk == result.passages[0]
    assert result.paths[0].target in {"beta", "ledger"}


def test_shrinking_a_document_and_deleting_respects_namespace(
    database_client: Client, table: str
) -> None:
    provider = ToyProvider()
    current = RAG(
        database_client,
        provider,
        table=table,
        dimensions=2,
        chunk_size=22,
        chunk_overlap=0,
    )
    other = RAG(
        database_client,
        provider,
        table=table,
        dimensions=2,
        namespace="other",
    )
    current.initialize()
    assert (
        current.index(Document("service", "Gateway calls Ledger. Vault writes reports.", "source"))
        == 2
    )
    assert other.index(Document("service", "Gateway calls Ledger.", "other")) == 1
    assert current.index(Document("service", "Gateway calls Beta.", "source")) == 1
    assert current.retrieve("reports", mode="lexical").passages == ()
    assert len(current.retrieve("Gateway", mode="dense").passages) == 1
    assert current.delete("service")
    assert current.retrieve("Gateway", mode="dense").passages == ()
    assert [c.source for c in other.retrieve("Gateway", mode="dense").passages] == ["other"]
    assert current.index(Document("service", "Gateway calls Beta.", "source")) == 1
    assert current.retrieve("Gateway", mode="lexical").passages[0].revision == 4


def test_mixed_retrieval_drops_evidence_replaced_between_stages(rag: RAG) -> None:
    old = Document("gateway", "Gateway calls Ledger.", "ref:old")
    new = Document("gateway", "Gateway calls Beta.", "ref:new")
    rag.index(old)
    assert rag.graph is not None
    expand = rag.graph.expand

    def replace_before_expand(*args, **kwargs):
        rag.index(new)
        return expand(*args, **kwargs)

    with patch.object(rag.graph, "expand", side_effect=replace_before_expand):
        result = rag.retrieve("Gateway", mode="mixed")
    assert [p.target for p in result.paths] == ["beta"]
    assert [c.source for c in result.passages] == ["ref:new"]


def test_global_coverage_uses_current_scope_and_signals_truncation(rag: RAG) -> None:
    for document in [
        Document("gateway", "Gateway calls Ledger.", "ref:gateway", {"team": "ops"}),
        Document("ledger", "Ledger writes Vault.", "ref:ledger", {"team": "ops"}),
        Document("mailer", "Mailer calls Queue.", "ref:mailer", {"team": "ops"}),
        Document("queue", "Queue writes Archive.", "ref:queue", {"team": "ops"}),
        Document("hidden", "Ledger calls Mailer.", "ref:hidden", {"team": "other"}),
    ]:
        rag.index(document)

    scope = {"team": "ops"}
    result = rag.retrieve("What are the major themes?", mode="global", metadata=scope, limit=2)
    assert len(result.passages) == len(result.communities) == 2
    assert {community.entities for community in result.communities} == {
        ("gateway", "ledger", "vault"),
        ("archive", "mailer", "queue"),
    }
    assert all(c.document_count == c.fact_count == 2 for c in result.communities)
    assert not result.truncated
    assert len(rag.overview("What are the major themes?").communities) == 1
    assert rag.overview("What are the major themes?", metadata=scope, max_facts=2).truncated
    rag.delete("ledger")
    result = rag.overview("storage", metadata=scope)
    assert "vault" not in {entity for row in result.communities for entity in row.entities}
    assert "ref:ledger" not in {chunk.source for chunk in result.passages}


def test_global_answer_passes_grouped_evidence_to_provider(
    rag: RAG,
    database_client: Client,
    table: str,
) -> None:
    class Summarizer(ToyProvider):
        def generate(self, question: str, context: Context) -> GeneratedAnswer:
            assert len(context.communities) == 2
            return GeneratedAnswer("Two themes", [chunk.chunk_id for chunk in context.passages])

    rag.index(Document("gateway", "Gateway calls Ledger.", "ref:a"))
    rag.index(Document("mailer", "Mailer calls Queue.", "ref:b"))
    provider = Summarizer()
    reader = RAG(
        database_client, provider, table=table, dimensions=2, graph=True, generator=provider
    )
    result = reader.ask("Which separate workflows are represented?", mode="global")
    assert result.text == "Two themes"
    assert {chunk.source for chunk in result.citations} == {"ref:a", "ref:b"}
