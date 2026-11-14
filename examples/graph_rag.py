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

    def generate(self, question: str, context: Context) -> GeneratedAnswer:
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

    results["kag"] = evaluate(cases[1:], run_knowledge, k=2)
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
            schema=(
                Predicate("calls", "service", "service"),
                Predicate("writes", "service", "database"),
            ),
        )
        rag.initialize()
        rag.index(Document("gateway", "Gateway calls Ledger.", "https://example.org/gateway"))
        rag.index(Document("ledger", "Ledger writes Vault.", "https://example.org/ledger"))

        answer = rag.ask("Where does Gateway record data?")
        assert answer.text == "Vault"
        print(answer.text, [item.source for item in answer.citations])
        result = rag.reason(KnowledgeQuery("Gateway", "service", ("calls", "writes")))
        assert result.entities == ("vault",)
        print("Verified path:", [step.relation.predicate for step in result.evidence[0].steps])
        if args.evaluate:
            run_evaluation(rag)


if __name__ == "__main__":
    main()
