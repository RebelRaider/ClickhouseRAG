"""Use independent model APIs over one ClickHouse evidence collection."""

import os
from uuid import uuid4

import clickhouse_connect

from clickhouserag import RAG, Document
from clickhouserag.providers import (
    OllamaEmbeddings,
    OpenAICompatibleEmbeddings,
    OpenAICompatibleResponses,
    StructuredLLM,
)


def main() -> None:
    ollama = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("RAG_EMBEDDING_MODEL", "embeddinggemma")
    dimensions = int(os.getenv("RAG_EMBEDDING_DIMENSIONS", "768"))
    revision = os.getenv("RAG_EMBEDDING_REVISION")
    embedding_url = os.getenv("RAG_EMBEDDING_URL")
    if embedding_url:
        embeddings = OpenAICompatibleEmbeddings(
            model,
            dimensions,
            base_url=embedding_url,
            revision=revision,
            api_key=os.getenv("RAG_EMBEDDING_API_KEY"),
        )
    else:
        embeddings = OllamaEmbeddings(model, dimensions, base_url=ollama, revision=revision)
    writer = StructuredLLM(
        OpenAICompatibleResponses(
            os.getenv("RAG_ANSWER_MODEL", "qwen2.5:1.5b"),
            base_url=os.getenv("RAG_ANSWER_URL", f"{ollama}/v1"),
            api_key=os.getenv("RAG_ANSWER_API_KEY"),
            max_tokens=int(os.getenv("RAG_ANSWER_MAX_TOKENS", "128")),
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
        table = f"rag_models_demo_{uuid4().hex}"
        try:
            rag = RAG(
                client, embeddings, dimensions=embeddings.dimensions, table=table, generator=writer
            )
            rag.initialize()
            rag.index(
                Document(
                    "inventory",
                    "The inventory system is called Warehouse.",
                    "https://example.org/inventory",
                )
            )
            answer = rag.ask("What is the inventory system called?", limit=1)
            print(answer.text, [source.source for source in answer.citations])
        finally:
            client.command("DROP TABLE IF EXISTS {table:Identifier}", {"table": table})


if __name__ == "__main__":
    main()
