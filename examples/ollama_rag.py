"""Run document indexing and grounded questions with local Ollama models."""

import os

import clickhouse_connect

from clickhouserag import Document, Predicate
from clickhouserag.ollama import Ollama


def main() -> None:
    schema = (
        Predicate("calls", "service", "service"),
        Predicate("writes", "service", "database"),
    )
    model = Ollama(
        embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma"),
        chat_model=os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:1.5b"),
        dimensions=int(os.getenv("OLLAMA_EMBEDDING_DIMENSIONS", "768")),
        embedding_version=os.getenv("OLLAMA_EMBEDDING_VERSION"),
        url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        schema=schema,
    )
    with clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST", "127.0.0.1"),
        port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        username=os.getenv("CLICKHOUSE_USER", "default"),
        password=os.getenv("CLICKHOUSE_PASSWORD", ""),
        autogenerate_session_id=False,
        form_encode_query_params=True,
    ) as client:
        rag = model.rag(client, table="ollama_rag_demo")
        rag.initialize()
        rag.index(Document("gateway", "Gateway calls Ledger.", "https://example.org/gateway"))
        rag.index(Document("ledger", "Ledger writes Vault.", "https://example.org/ledger"))

        question = "Which database does Gateway write to through Ledger?"
        for answer in (rag.ask(question), rag.ask_knowledge(question)):
            print(answer.text, [item.source for item in answer.citations])


if __name__ == "__main__":
    main()
