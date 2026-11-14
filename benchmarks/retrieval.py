"""Reproducible synthetic ranking and latency probe for a local ClickHouse server."""

import argparse
import json
import math
import os
import random
import statistics
import time
from collections.abc import Callable
from uuid import uuid4

import clickhouse_connect

from clickhouserag import Chunk, ClickHouseStore, EmbeddedChunk, SearchHit


def evaluate(
    search: Callable[[str, list[float]], list[SearchHit]],
    queries: list[tuple[str, list[float], str]],
    repetitions: int,
) -> dict[str, float]:
    for text, vector, _ in queries:
        search(text, vector)
    ranking: list[int] = []
    latencies: list[float] = []
    for _ in range(repetitions):
        for text, vector, relevant in queries:
            start = time.perf_counter()
            hits = search(text, vector)
            latencies.append((time.perf_counter() - start) * 1000)
            rank = next((i for i, hit in enumerate(hits, 1) if hit.chunk.chunk_id == relevant), 0)
            ranking.append(rank)
    return {
        "recall_at_k": sum(bool(rank) for rank in ranking) / len(ranking),
        "mrr_at_k": statistics.mean(1 / rank if rank else 0 for rank in ranking),
        "p50_ms": statistics.median(latencies),
        "p95_ms": sorted(latencies)[math.ceil(len(latencies) * 0.95) - 1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    if args.rows < 100 or args.repetitions < 1:
        parser.error("--rows must be at least 100 and --repetitions must be positive")
    dimensions, seed, top_k = 32, 42, 5
    rng = random.Random(seed)
    with clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST", "127.0.0.1"),
        port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        username=os.getenv("CLICKHOUSE_USER", "default"),
        password=os.getenv("CLICKHOUSE_PASSWORD", ""),
        autogenerate_session_id=False,
    ) as client:
        table = f"chrag_bench_{uuid4().hex}"
        store = ClickHouseStore(
            client, table=table, embedding_model="synthetic", dimensions=dimensions
        )
        try:
            store.initialize()
            unit_a = [1.0] + [0.0] * (dimensions - 1)
            unit_b = [0.0, 1.0] + [0.0] * (dimensions - 2)
            store.insert(
                [
                    EmbeddedChunk(Chunk("keyword", "a", "Diagnostic ERR531", "a"), unit_a),
                    EmbeddedChunk(Chunk("semantic", "b", "Traffic arrives gradually", "b"), unit_b),
                ]
            )
            for start in range(2, args.rows, 1000):
                store.insert(
                    EmbeddedChunk(
                        Chunk(f"noise{i}", f"d{i}", f"noise record {i}", f"d{i}"),
                        [rng.gauss(0, 1) for _ in range(dimensions)],
                    )
                    for i in range(start, min(start + 1000, args.rows))
                )
            queries = [("ERR531", unit_b, "keyword"), ("rate smoothing", unit_b, "semantic")]
            methods = {
                "dense": lambda _text, vector: store.search_vector(vector, limit=top_k),
                "lexical": lambda text, _vector: store.search_lexical(text, limit=top_k),
                "hybrid": lambda text, vector: store.search_hybrid(text, vector, limit=top_k),
            }
            metrics = {
                query[2]: {
                    mode: evaluate(search, [query], args.repetitions)
                    for mode, search in methods.items()
                }
                for query in queries
            }
            print(
                json.dumps(
                    {
                        "version": client.server_version,
                        "rows": args.rows,
                        "dimensions": dimensions,
                        "seed": seed,
                        "top_k": top_k,
                        "repetitions": args.repetitions,
                        "warmup_per_query": 1,
                        "metrics_by_query": metrics,
                    },
                    indent=2,
                )
            )
        finally:
            client.command("DROP TABLE IF EXISTS {table:Identifier}", {"table": table})


if __name__ == "__main__":
    main()
