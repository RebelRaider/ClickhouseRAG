"""Index documents and ask a cited multi-hop question on a local ClickHouse server."""

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from typing import Literal

import clickhouse_connect

from clickhouserag import (
    RAG,
    Chunk,
    Context,
    Document,
    EvalCase,
    Evaluation,
    ExtractedRelation,
    GeneratedAnswer,
    KnowledgeIntersection,
    KnowledgeQuery,
    Predicate,
    evaluate,
)


class DemoModel:
    """Offline fixtures for a corpus made of simple service relationship sentences."""

    model_id = "service-demo-v1"

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[1.0, 0.0] if text.startswith("Gateway ") else [0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> Sequence[float]:
        return [1.0, 0.0]

    def extract(self, chunk: Chunk) -> Sequence[ExtractedRelation]:
        subject, verb, target = chunk.text.rstrip(".").split(maxsplit=2)
        kind = "database" if verb == "writes" else "service"
        return [ExtractedRelation(subject, verb, target, "service", kind)]

    def plan(
        self, question: str, schema: tuple[Predicate, ...]
    ) -> KnowledgeQuery | KnowledgeIntersection | None:
        gateway = KnowledgeQuery("Gateway", "service", ("calls", "writes"))
        if question == "Gateway":
            return KnowledgeQuery("Gateway", "service", ("calls",))
        if question.startswith("Which database do Gateway and Worker"):
            return KnowledgeIntersection(
                (gateway, KnowledgeQuery("Worker", "service", ("calls", "writes")))
            )
        if question.startswith("Where does Gateway"):
            return gateway
        return None

    def generate(self, question: str, context: Context) -> GeneratedAnswer:
        if question == "Summarize the independent service groups.":
            labels = {label for group in context.communities for label in group.entities}
            if {"gateway", "billing"} <= labels:
                return GeneratedAnswer(
                    "Gateway and Billing", [c.chunk_id for c in context.passages]
                )
            return GeneratedAnswer(None)
        if question.startswith("Which database do Gateway and Worker"):
            required = {"gateway", "ledger", "worker", "audit"}
            evidence = {c.document_id: c.chunk_id for c in context.passages}
            if required <= evidence.keys():
                return GeneratedAnswer("Vault", [evidence[key] for key in sorted(required)])
            return GeneratedAnswer(None)
        facts = {
            (fact.subject, fact.predicate): (fact, chunk.chunk_id)
            for chunk in context.passages
            for fact in self.extract(chunk)
        }
        first = facts.get(("Gateway", "calls"))
        if first is None:
            return GeneratedAnswer(None)
        if question == "Gateway":
            return GeneratedAnswer(first[0].object, [first[1]])
        second = facts.get((first[0].object, "writes"))
        if question.startswith("Where does Gateway") and second is not None:
            return GeneratedAnswer(second[0].object, [first[1], second[1]])
        return GeneratedAnswer(None)


def run_evaluation(rag: RAG) -> None:
    """Compare strategies on held-out questions with manually labelled evidence."""
    for document in (
        Document("detour", "Gateway deploys Proxy.", "https://example.org/proxy"),
        Document("monitor", "Gateway monitors Queue.", "https://example.org/queue"),
        Document("worker", "Worker calls Audit.", "https://example.org/worker"),
        Document("audit", "Audit writes Vault.", "https://example.org/audit"),
        Document("billing", "Billing calls Invoice.", "https://example.org/billing"),
        Document("invoice", "Invoice writes Archive.", "https://example.org/invoice"),
    ):
        rag.index(document)
    indexed = {
        chunk.document_id: chunk.chunk_id
        for chunk in rag.retrieve("Gateway records", mode="dense", limit=10).passages
    }
    cases = (
        EvalCase(
            "service", "Gateway", frozenset({indexed["gateway"]}), ("Ledger",), "entity lookup"
        ),
        EvalCase(
            "storage",
            "Where does Gateway record data?",
            frozenset({indexed["gateway"], indexed["ledger"]}),
            ("vault",),
            "multi-hop",
        ),
        EvalCase("absent", "Who owns Jupiter?", frozenset(), category="unanswerable"),
    )
    modes: tuple[Literal["dense", "lexical", "hybrid", "graph", "mixed"], ...] = (
        "dense",
        "lexical",
        "hybrid",
        "graph",
        "mixed",
    )
    results: dict[str, Evaluation] = {
        mode: evaluate(cases, lambda case, mode=mode: rag.ask(case.query, mode=mode, limit=2), k=2)
        for mode in modes
    }
    plan = KnowledgeQuery("Gateway", "service", ("calls", "writes"))

    def run_knowledge(case: EvalCase) -> Context:
        # This benchmark supplies the plan explicitly to measure the reasoner.
        current = (
            plan
            if case.case_id == "storage"
            else KnowledgeQuery("Jupiter", "service", ("calls", "writes"))
        )
        solved = rag.reason(current)
        passages = {
            step.chunk.chunk_id: step.chunk for path in solved.evidence for step in path.steps
        }
        return Context(tuple(passages.values()), solved.evidence)

    results["kag_explicit"] = evaluate(cases[1:], run_knowledge, k=2)
    shared = EvalCase(
        "intersection",
        "Which database do Gateway and Worker both reach?",
        frozenset(indexed[key] for key in ("gateway", "ledger", "worker", "audit")),
        ("vault",),
        "conjunctive query",
    )
    overview = EvalCase(
        "overview",
        "Summarize the independent service groups.",
        frozenset(indexed[key] for key in ("gateway", "ledger", "billing", "invoice")),
        ("Gateway and Billing",),
        "corpus overview",
    )
    results["kag_planned"] = evaluate(
        (*cases, shared), lambda case: rag.ask_knowledge(case.query, limit=4), k=4
    )
    results["global"] = evaluate(
        (overview,), lambda case: rag.ask(case.query, mode="global", limit=4), k=4
    )
    print(json.dumps({mode: asdict(run) for mode, run in results.items()}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true", help="run the synthetic evaluation")
    args = parser.parse_args()
    with clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST", "127.0.0.1"),
        port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        username=os.getenv("CLICKHOUSE_USER", "default"),
        password=os.getenv("CLICKHOUSE_PASSWORD", ""),
        autogenerate_session_id=False,
        form_encode_query_params=True,
    ) as client:
        model = DemoModel()
        rag = RAG(
            client,
            model,
            table="rag_graph_demo",
            dimensions=2,
            extractor=model,
            generator=model,
            planner=model,
            schema=(
                Predicate("calls", "service", "service"),
                Predicate("writes", "service", "database"),
            ),
        )
        rag.initialize()
        rag.index(Document("gateway", "Gateway calls Ledger.", "https://example.org/gateway"))
        rag.index(Document("ledger", "Ledger writes Vault.", "https://example.org/ledger"))

        answer = rag.ask("Where does Gateway record data?", mode="mixed")
        assert answer.text == "Vault"
        print(answer.text, [item.source for item in answer.citations])
        result = rag.reason(KnowledgeQuery("Gateway", "service", ("calls", "writes")))
        assert result.entities == ("vault",)
        print("Verified path:", [step.relation.predicate for step in result.evidence[0].steps])
        if args.evaluate:
            run_evaluation(rag)


if __name__ == "__main__":
    main()
