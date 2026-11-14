# Synthetic retrieval check

Two deliberately contrasting labelled queries, 10,000 passages and 32-dimensional precomputed vectors. Single client and local ClickHouse 26.8.36.3 via Docker Desktop on macOS ARM64. One warm-up and 20 timed requests per strategy and query, without embedding latency.

| Query | Strategy | Recall@5 | MRR@5 | p50 (ms) | p95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| keyword | dense | 0.0 | 0 | 3.54 | 5.69 |
| keyword | lexical | 1.0 | 1.0 | 3.67 | 5.27 |
| keyword | hybrid | 1.0 | 1.0 | 7.28 | 9.41 |
| semantic | dense | 1.0 | 1.0 | 3.56 | 4.24 |
| semantic | lexical | 0.0 | 0 | 2.25 | 2.74 |
| semantic | hybrid | 1.0 | 1.0 | 5.32 | 6.02 |

Hybrid recovered both labelled passages. The keyword was constructed to fail dense search and the paraphrase to fail lexical search. These two queries verify expected behavior on a controlled corpus. Estimate quality and performance on your documents, embeddings, query distribution and hardware.

Run from a fresh checkout with a local test server and the connection environment variables used by the quickstart.

```sh
uv run --locked python benchmarks/retrieval.py --rows 10000 --repetitions 20
```
