"""Run a complete dense retrieval example against a local ClickHouse server."""

import os

import clickhouse_connect

from clickhouserag import Chunk, ClickHouseStore, EmbeddedChunk


def main() -> None:
    with clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST", "localhost"),
        port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        username=os.getenv("CLICKHOUSE_USER", "default"),
        password=os.getenv("CLICKHOUSE_PASSWORD", ""),
        autogenerate_session_id=False,
    ) as client:
        store = ClickHouseStore(
            client, table="chrag_quickstart", embedding_model="toy-v1", dimensions=2
        )
        store.initialize()
        store.insert(
            [
                EmbeddedChunk(
                    Chunk("1", "docs", "A replica keeps a copy.", "https://example.com/replica"),
                    [1.0, 0.0],
                ),
                EmbeddedChunk(
                    Chunk("2", "docs", "A partition groups rows.", "https://example.com/partition"),
                    [0.0, 1.0],
                ),
            ],
            namespace="quickstart",
        )
        for hit in store.search_vector([1.0, 0.0], namespace="quickstart", limit=1):
            print(f"{hit.score:.3f}  {hit.chunk.text}  {hit.chunk.source}")


if __name__ == "__main__":
    main()
