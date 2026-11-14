# Retrieval and answer benchmarks

## Synthetic retrieval check

Two deliberately contrasting labelled queries, 10,000 passages and 32-dimensional precomputed vectors. Single client and local ClickHouse 26.8.36.3 via Docker Desktop on macOS ARM64 (Mac15,12, 8 logical CPUs, 16 GiB host memory). One warm-up and 20 timed requests per strategy and query, without embedding latency.

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

## HotpotQA bridge questions

The [HotpotQA distractor validation set](https://hotpotqa.github.io/) presents
ten Wikipedia excerpts per question and identifies the supporting sentences.
The harness indexes one namespace per question and evaluates at article-title
granularity. It compares six retrieval modes with the same input, embedding,
chunking and passage budget. The `--answers` flag adds answer and citation
scores and includes generation in wall time. Each retrieval mode is warmed up
once per question. Answer generation itself can still include model startup.

Download the validation Parquet file from the pinned
[dataset revision](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab/distractor)
and save it as `hotpot_dev.parquet`. The file SHA-256 is
`c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`.
Run the following with a ClickHouse local binary from the 26.8 LTS line.

```sh
clickhouse local --query "
    SELECT id, question, answer, type, level,
           supporting_facts.title AS support_titles,
           context.title AS titles, context.sentences AS sentences
    FROM file('hotpot_dev.parquet', Parquet)
    WHERE type = 'bridge'
    ORDER BY cityHash64(concat(id, ':chrag-20261114'))
    LIMIT 12
    FORMAT JSONEachRow
" > bridge12.jsonl
```

The sample SHA-256 is
`9eba08057e9fc7d75be768d09f552f3be34056ba5f77c05ae92a85de5c1b8b49`.
Set the test ClickHouse and Ollama connection variables from the
[framework quickstart](framework.md). Set Ollama version fields in the report
using the model digests from `ollama list` and the server version from
`ollama --version`.

```sh
uv run --locked python benchmarks/hotpot.py bridge12.jsonl \
    --max-relations 8 --k 4 --answers --output hotpot-results.json
```

The harness creates and drops a temporary benchmark table. Failed relation
extractions leave text searchable and count towards `extraction_failures`.
Failed answer calls count towards `answer_failures` and score as abstentions.
The report records case IDs, metrics, timings and model identifiers. The small
bridge sample tests multi-hop lookup, while the scoped corpus overview and
typed KAG require other labelled questions and a domain schema.
