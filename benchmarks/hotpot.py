"""Evaluate ClickHouse RAG on a pinned HotpotQA distractor sample."""

import argparse
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote
from uuid import uuid4

import clickhouse_connect

from clickhouserag import (
    RAG,
    Answer,
    Chunk,
    Context,
    Document,
    EvalCase,
    ExtractedRelation,
    GeneratedAnswer,
    evaluate,
)
from clickhouserag.ollama import ModelResponseError, Ollama

Mode = Literal["dense", "lexical", "hybrid", "graph", "mixed", "global"]
_MODES: tuple[Mode, ...] = ("dense", "lexical", "hybrid", "graph", "mixed", "global")


class RecordedModel:
    """Count model failures while retaining the evidence available for scoring."""

    def __init__(self, model: Ollama) -> None:
        self.model = model
        self.calls = 0
        self.facts = 0
        self.extraction_failures = 0
        self.answer_failures = 0

    def extract(self, chunk: Chunk) -> list[ExtractedRelation]:
        self.calls += 1
        try:
            facts = self.model.extract(chunk)
        except ModelResponseError:
            self.extraction_failures += 1
            return []
        self.facts += len(facts)
        return facts

    def generate(self, question: str, context: Context) -> GeneratedAnswer:
        try:
            return self.model.generate(question, context)
        except ModelResponseError:
            self.answer_failures += 1
            return GeneratedAnswer(None)


def document_ids(context: Context) -> Context:
    return Context(tuple(replace(chunk, chunk_id=chunk.document_id) for chunk in context.passages))


def read_case(row: dict[str, Any]) -> EvalCase:
    return EvalCase(
        row["id"],
        row["question"],
        frozenset(row["support_titles"]),
        (row["answer"],),
        row["level"],
    )


def run_case(rag: RAG, case: EvalCase, mode: Mode, k: int, *, answers: bool) -> Context | Answer:
    if not answers:
        return document_ids(rag.retrieve(case.query, mode=mode, limit=k))
    answer = rag.ask(case.query, mode=mode, limit=k)
    return Answer(
        answer.text,
        tuple(replace(chunk, chunk_id=chunk.document_id) for chunk in answer.citations),
        document_ids(answer.context),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="HotpotQA sample in JSONEachRow format")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max-relations", type=int, default=8)
    parser.add_argument(
        "--answers", action="store_true", help="also run model answer generation for each mode"
    )
    args = parser.parse_args()
    payload = args.dataset.read_bytes()
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    cases = [read_case(row) for row in rows]
    if not cases or len({case.case_id for case in cases}) != len(cases):
        parser.error("the sample must contain unique case IDs")
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:1.5b")
    model = Ollama(
        embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma"),
        embedding_version=os.getenv("OLLAMA_EMBEDDING_VERSION"),
        chat_model=chat_model,
        dimensions=int(os.getenv("OLLAMA_EMBEDDING_DIMENSIONS", "768")),
        url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        max_relations=args.max_relations,
    )
    recorded = RecordedModel(model)
    table = f"chrag_hotpot_{uuid4().hex}"
    with clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST", "127.0.0.1"),
        port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        username=os.getenv("CLICKHOUSE_USER", "default"),
        password=os.getenv("CLICKHOUSE_PASSWORD", ""),
        autogenerate_session_id=False,
        form_encode_query_params=True,
    ) as client:
        collections: dict[str, RAG] = {}
        try:
            for case, row in zip(cases, rows, strict=True):
                documents = dict(zip(row["titles"], row["sentences"], strict=True))
                if not case.relevant <= documents.keys():
                    raise ValueError(f"case {case.case_id} is missing supporting documents")
                rag = RAG(
                    client,
                    model,
                    table=table,
                    namespace=case.case_id,
                    dimensions=model.dimensions,
                    extractor=recorded,
                    generator=recorded,
                    chunk_size=3500,
                )
                rag.initialize()
                collections[case.case_id] = rag
                print(f"Indexing case {case.case_id}", flush=True)
                for title, sentences in documents.items():
                    rag.index(
                        Document(
                            title,
                            f"{title}\n{''.join(sentences)}",
                            "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_")),
                        )
                    )
            results = {}
            for mode in _MODES:
                print(f"Scoring {mode}", flush=True)
                for case in cases:
                    collections[case.case_id].retrieve(case.query, mode=mode, limit=args.k)
                recorded.answer_failures = 0
                result = evaluate(
                    cases,
                    lambda case, mode=mode: run_case(
                        collections[case.case_id],
                        case,
                        mode,
                        args.k,
                        answers=args.answers,
                    ),
                    k=args.k,
                )
                score = asdict(result)
                # Keep numeric scores and case IDs without redistributing article titles.
                for case_score in score["cases"]:
                    case_score.pop("retrieved")
                score["answer_failures"] = recorded.answer_failures
                results[mode] = score
        finally:
            client.command("DROP TABLE IF EXISTS {table:Identifier}", {"table": table})
        output = {
            "dataset_sha256": hashlib.sha256(payload).hexdigest(),
            "clickhouse_version": client.server_version,
            "embedding_model": model.model_id,
            "chat_model": chat_model,
            "chat_version": os.getenv("OLLAMA_CHAT_VERSION"),
            "server_version": os.getenv("OLLAMA_SERVER_VERSION"),
            "answer_generation": args.answers,
            "n_cases": len(cases),
            "max_relations_per_chunk": args.max_relations,
            "extraction_calls": recorded.calls,
            "extraction_failures": recorded.extraction_failures,
            "extracted_facts": recorded.facts,
            "results": results,
        }
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
