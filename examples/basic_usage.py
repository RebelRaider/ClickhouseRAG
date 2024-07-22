from typing import Any, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from clickhouserag.data_access.clickhouse_client import ClickhouseConnectClient
from clickhouserag.rag.manager import RAGManager
from clickhouserag.vectorizers.base import VectorizerBase


class TransformersVectorizer(VectorizerBase):
    """Vectorizer that uses a Transformers model to convert text to vectors."""

    def __init__(self, model_name: str) -> None:
        """Initialize the TransformersVectorizer.

        Args:
        ----
            model_name (str): The name of the Transformers model to use.

        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def vectorize(self, data: Any) -> List[float]:
        """Convert text data into a vector representation using a Transformers model.

        Args:
        ----
            data (Any): The text data to vectorize.

        Returns:
        -------
            List[float]: The vector representation of the text data.

        """
        if not isinstance(data, str):
            raise ValueError("Data should be a string for text vectorization.")

        inputs = self.tokenizer(
            data, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the mean pooling of the last hidden state as the vector
            vector = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

        return vector


def main():
    # Создание клиента для подключения к Clickhouse
    client = ClickhouseConnectClient(
        host="localhost",
        port=9000,
        username="default",
        password="",
        database="default"
    )

    # Подключение к базе данных
    client.connect()

    # Определение структуры таблицы
    table_schema = {
        "id": "UInt32",
        "title": "String",
        "vector": "Array(Float64)"
    }

    # Создание экземпляра RAGManager с указанным движком и схемой таблицы
    rag_manager = RAGManager(client, "rag_table", table_schema, engine="MergeTree", order_by="id")

    # Создание и добавление векторизатора Transformers
    transformers_vectorizer = TransformersVectorizer(model_name="distilbert-base-uncased")
    rag_manager.add_vectorizer("transformers", transformers_vectorizer)

    # Пример добавления данных с векторизацией через Transformers
    data = {"id": 1, "title": "Sample text data for transformers"}
    rag_manager.add_data(data, vectorizer_name="transformers")

    # Пример массового добавления данных с векторизацией через Transformers
    bulk_data = [
        {"id": 2, "title": "Sample text data 1 for transformers"},
        {"id": 3, "title": "Sample text data 2 for transformers"},
        {"id": 4, "title": "Sample text data 3 for transformers"}
    ]
    rag_manager.add_bulk_data(bulk_data, vectorizer_name="transformers")

    # Получение данных по ID
    data = rag_manager.get_data(1)
    print("Data with ID 1:", data)

    # Обновление данных с векторизацией через Transformers
    updated_data = {"id": 1, "title": "Updated text data for transformers"}
    rag_manager.update_data(1, updated_data, vectorizer_name="transformers")

    # Выполнение поиска по тексту
    query = "SELECT * FROM rag_table WHERE title LIKE '%Sample%'"
    search_results = rag_manager.search(query)
    print("Search results:", search_results)

    # Выполнение поиска по косинусной схожести
    embedding = np.random.rand(768)  # Пример случайного вектора
    similarity_results = rag_manager.similarity_search(embedding, top_k=2, columns=["id", "title"])
    print("Similarity search results:", similarity_results)

    # Удаление данных
    rag_manager.delete_data(1)

    # Резервное копирование базы данных
    rag_manager.backup_database("backup.json")

    # Сброс и восстановление базы данных
    rag_manager.reset_database()
    rag_manager.restore_database("backup.json", table_schema=table_schema)

    # Закрытие подключения к базе данных
    client.close()

if __name__ == "__main__":
    main()
