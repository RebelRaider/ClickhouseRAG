"""Measure graph retrieval cost on a deterministic shared-hub graph."""

import json
import os
from collections.abc import Sequence
from math import ceil
from statistics import median
from time import perf_counter
from typing import Literal
from uuid import uuid4

import clickhouse_connect

from clickhouserag import RAG, Chunk, Document, ExtractedRelation

Mode = Literal["hybrid", "graph", "mixed", "global"]
_MODES: tuple[Mode, ...] = ("hybrid", "graph", "mixed", "global")


class Fixture:
    model_id = "graph-scale-v1"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def extract(self, chunk: Chunk) -> list[ExtractedRelation]:
        group = int(chunk.document_id)
        return [
            ExtractedRelation("Common hub", "routes", f"Topic {group}"),
            *(
                ExtractedRelation(f"Topic {group}", "feeds", f"Endpoint {group} {endpoint}")
                for endpoint in range(99)
            ),
        ]


def time_requests(rag: RAG, mode: Mode, repetitions: int) -> dict[str, float | int | bool]:
    result = rag.retrieve("Common hub", mode=mode, limit=10)
    samples = []
    for _ in range(repetitions):
        start = perf_counter()
        rag.retrieve("Common hub", mode=mode, limit=10)
        samples.append((perf_counter() - start) * 1000)
    return {
        "p50_ms": round(median(samples), 2),
        "p95_ms": round(sorted(samples)[ceil(0.95 * repetitions) - 1], 2),
        "paths": len(result.paths),
        "passages": len(result.passages),
        "communities": len(result.communities),
        "truncated": result.truncated,
    }


def main() -> None:
    with clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST", "127.0.0.1"),
        port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        username=os.getenv("CLICKHOUSE_USER", "default"),
        password=os.getenv("CLICKHOUSE_PASSWORD", ""),
        autogenerate_session_id=False,
        form_encode_query_params=True,
    ) as client:
        table = f"chrag_scale_{uuid4().hex}"
        fixture = Fixture()
        rag = RAG(client, fixture, table=table, dimensions=2, extractor=fixture)
        try:
            rag.initialize()
            for start, end in ((0, 10), (10, 100)):
                for group in range(start, end):
                    rag.index(
                        Document(str(group), f"Common hub routes to topic {group}.", f"ref:{group}")
                    )
                print(
                    json.dumps(
                        {
                            "clickhouse_version": client.server_version,
                            "documents": end,
                            "facts": end * 100,
                            "repetitions": 20,
                            "modes": {mode: time_requests(rag, mode, 20) for mode in _MODES},
                        }
                    ),
                    flush=True,
                )
        finally:
            client.command("DROP TABLE IF EXISTS {table:Identifier}", {"table": table})


if __name__ == "__main__":
    main()
