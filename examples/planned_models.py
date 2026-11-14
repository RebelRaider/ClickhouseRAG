"""Plan scoped searches with Ollama and answer using a separate server."""

import os
from uuid import uuid4

import clickhouse_connect

from clickhouserag import RAG, Document
from clickhouserag.providers import (
    OllamaChat,
    OpenAICompatibleChat,
    OpenAICompatibleEmbeddings,
    StructuredLLM,
)


def main() -> None:
    ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    embedding_url = os.environ["RAG_EMBEDDING_URL"]
    answer_url = os.environ["RAG_ANSWER_URL"]
    embedder = OpenAICompatibleEmbeddings(
        os.getenv("RAG_EMBEDDING_MODEL", "embeddinggemma"),
        int(os.getenv("RAG_EMBEDDING_DIMENSIONS", "768")),
        base_url=embedding_url,
    )
    planner = StructuredLLM(
        OllamaChat(
            os.getenv("RAG_PLANNER_MODEL", "qwen2.5:1.5b"), base_url=ollama_url, max_tokens=256
        )
    )
    writer = StructuredLLM(
        OpenAICompatibleChat(
            os.getenv("RAG_ANSWER_MODEL", "qwen2.5"),
            base_url=answer_url,
            max_tokens=128,
            token_parameter="max_tokens",
        )
    )
    with clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST", "127.0.0.1"),
        port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        username=os.getenv("CLICKHOUSE_USER", "default"),
        password=os.getenv("CLICKHOUSE_PASSWORD", ""),
        autogenerate_session_id=False,
        form_encode_query_params=True,
    ) as client:
        table = f"rag_planned_demo_{uuid4().hex}"
        try:
            rag = RAG(
                client,
                embedder,
                table=table,
                dimensions=embedder.dimensions,
                search_planner=planner,
                generator=writer,
            )
            rag.initialize()
            rag.index(
                Document(
                    "inventory",
                    "Restart the inventory service to recover from error ERR-42.",
                    "https://example.org/inventory",
                    {"team": "ops"},
                )
            )
            rag.index(
                Document(
                    "different-team",
                    "ERR-42 is resolved by removing the test record.",
                    "https://example.org/test",
                    {"team": "test"},
                )
            )
            answer = rag.ask_planned(
                "What action recovers the inventory service from error ERR-42?",
                metadata={"team": "ops"},
                limit=1,
            )
            print(answer.text, [chunk.source for chunk in answer.citations])
        finally:
            client.command("DROP TABLE IF EXISTS {table:Identifier}", {"table": table})


if __name__ == "__main__":
    main()
