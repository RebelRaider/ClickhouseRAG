# Quickstart

The 1.x line targets CPython 3.11–3.15 and ClickHouse 26.3 and 26.8 LTS.
Use an existing server or start an isolated local one.

```sh
export CLICKHOUSE_USER=chrag
export CLICKHOUSE_PASSWORD="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
docker run --detach --rm --name chrag-demo \
  --publish 127.0.0.1:8123:8123 \
  --env CLICKHOUSE_USER --env CLICKHOUSE_PASSWORD \
  clickhouse/clickhouse-server:26.8.36.3

python -m pip install 'clickhouserag>=1,<2'
```

If the server is still starting, wait until `curl -fsS http://127.0.0.1:8123/ping`
returns `Ok.` Run this example in the same shell.

```python
import os

import clickhouse_connect

from clickhouserag import Chunk, ClickHouseStore, EmbeddedChunk

with clickhouse_connect.get_client(
    host="127.0.0.1",
    port=8123,
    username=os.environ["CLICKHOUSE_USER"],
    password=os.environ["CLICKHOUSE_PASSWORD"],
    autogenerate_session_id=False,
    form_encode_query_params=True,
) as client:
    store = ClickHouseStore(client, table="demo_passages", embedding_model="toy-v1", dimensions=2)
    store.initialize()
    store.insert(
        [
            EmbeddedChunk(
                Chunk(
                    chunk_id="retention-1",
                    document_id="runbook",
                    text="ERR531 indicates that retention exceeded the configured window.",
                    source="https://example.com/runbook#retention",
                    metadata={"product": "storage"},
                ),
                [1.0, 0.0],
            )
        ],
        namespace="runbooks",
    )
    for hit in store.search_hybrid(
        "ERR531 retention",
        [1.0, 0.0],
        namespace="runbooks",
        metadata={"product": "storage"},
    ):
        print(hit.chunk.text, hit.chunk.source)
```

Run `docker stop chrag-demo` to remove the temporary server.

For a model-backed application, implement the
[`EmbeddingProvider` protocol](api.md#clickhouserag.EmbeddingProvider) and
pass it to `Retriever`. Include the model version and the relevant encoding
configuration in `model_id`. Provide separate document and query encoders
when the model expects different prompts or prefixes. The caller owns document
chunking and generation. Store stable chunk IDs and resolvable source pointers.

Pass previously chunked data to `store.insert` or `Retriever.index` in batches.
For local development from a clone, install [uv](https://docs.astral.sh/uv/),
then run `uv sync --locked` and `uv run --locked python examples/basic_usage.py`.
