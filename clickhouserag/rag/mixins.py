import logging
from typing import Any, Dict, List, Optional

import numpy as np

from clickhouserag.backup.managers import BackupManager
from clickhouserag.clickhouse.clients import ClickhouseClient
from clickhouserag.clickhouse.managers import ClickhouseTableManager
from clickhouserag.vectorizers.base import VectorizerBase
from clickhouserag.vectorizers.managers import VectorizerManager


class TableManagerMixin:
    def __init__(self, client: ClickhouseClient, table_name: str, table_schema: Optional[Dict[str, str]], engine: str, order_by: str) -> None:
        self.client = client
        self.table_manager = ClickhouseTableManager(client, table_name)
        self.table_name = table_name
        self.table_schema = table_schema
        self.engine = engine
        self.order_by = order_by
        self.logger = logging.getLogger(__name__)

        if table_schema:
            self._initialize_table(table_schema, engine, order_by)

    def _initialize_table(self, table_schema: Dict[str, str], engine: str, order_by: str) -> None:
        if not self._check_table_exists():
            self._create_table(table_schema, engine, order_by)
        else:
            self.logger.info(f"Table '{self.table_name}' already exists.")

    def _create_table(self, table_schema: Dict[str, str], engine: str, order_by: str) -> None:
        fields = ", ".join([f"{name} {dtype}" for name, dtype in table_schema.items()])
        query = f"CREATE TABLE {self.table_name} ({fields}) ENGINE = {engine} ORDER BY {order_by}"
        self.client.execute_query(query)
        self.logger.info(f"Table '{self.table_name}' created with schema: {table_schema}, engine: {engine}, order by: {order_by}")

    def _check_table_exists(self) -> bool:
        query = f"EXISTS TABLE {self.table_name}"
        result = self.client.execute_query(query)
        return result[0][0] == 1

    def reset_database(self) -> None:
        self.table_manager.reset_table()
        self.logger.info("RAG table reset")


class DataOperationsMixin:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def add_data(self, data: Dict[str, Any], vectorizer_name: Optional[str] = None) -> None:
        if vectorizer_name:
            data["vector"] = self.vectorize(data, vectorizer_name)
        self.validate_data(data)
        self.table_manager.insert([data])
        self.logger.info(f"Data added with id {data.get('id')}")

    def add_bulk_data(self, data_list: List[Dict[str, Any]], vectorizer_name: Optional[str] = None, vectorizer: VectorizerBase = None) -> None:
        if vectorizer_name and vectorizer:
            raise ValueError("Only one of vectorizer_name and vectorizer should be provided.")
        if not (vectorizer_name or vectorizer):
            raise ValueError("Either vectorizer_name or vectorizer should be provided.")

        if vectorizer_name:
            for data in data_list:
                data["vector"] = self.vectorize(data, vectorizer_name)
        elif vectorizer:
            vectors = vectorizer.bulk_vectorize([data["title"] for data in data_list])
            for data, vector in zip(data_list, vectors):
                data["vector"] = vector

        for data in data_list:
            self.validate_data(data)

        self.table_manager.insert(data_list)
        self.logger.info(f"Bulk data added with {len(data_list)} records")

    def delete_data(self, data_id: str) -> None:
        self.table_manager.delete({"id": data_id})
        self.logger.info(f"Data deleted with id {data_id}")

    def update_data(self, data_id: str, new_data: Dict[str, Any], vectorizer_name: Optional[str] = None) -> None:
        if vectorizer_name:
            new_data["vector"] = self.vectorize(new_data, vectorizer_name)
        self.validate_data(new_data)
        self.table_manager.update(new_data, {"id": data_id})
        self.logger.info(f"Data updated with id {data_id}")

    def search(self, query: str, similarity: bool = False, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if similarity:
            return self.similarity_search(query, top_k)
        results = self.table_manager.search(query)
        self.logger.info(f"Search executed with query '{query}', found {len(results)} results")
        return results

    def similarity_search(self, embedding: np.array, columns: List[str], top_k: Optional[int]) -> List[Dict[str, Any]]:
        columns_str = ", ".join(columns)
        query = f"""
        WITH %(embedding)s as query_vector
        SELECT {columns_str}
        arraySum(x -> x * x, vector) * arraySum(x -> x * x, query_vector) != 0
        ? arraySum((x, y) -> x * y, vector, query_vector) / sqrt(arraySum(x -> x * x, vector) * arraySum(x -> x * x, query_vector))
        : 0 AS cosine_distance
        FROM {self.table_name}
        WHERE length(query_vector) == length(vector)
        ORDER BY cosine_distance DESC
        LIMIT %(top_k)s
        """
        params = {"embedding": embedding.tolist(), "top_k": top_k}
        result = self.client.execute_query(query, params=params, settings={"max_query_size": "10000000000000"})
        self.logger.info(f"Similarity search executed with embedding, top {top_k} results found")
        return [{"id": row[0], "title": row[1], "cosine_distance": row[2]} for row in result]

    def validate_data(self, data: Dict[str, Any]) -> None:
        if "id" not in data:
            raise ValueError("Data must contain an 'id' field")

    def get_data(self, data_id: str) -> Optional[Dict[str, Any]]:
        query = f"SELECT * FROM {self.table_name} WHERE id = %(data_id)s"
        result = self.client.fetch_one(query, params={"data_id": data_id})
        if result:
            return dict(zip(self.table_manager.columns, result))
        return None

    def set_data(self, data: Dict[str, Any], vectorizer_name: Optional[str] = None) -> None:
        if "id" in data:
            if self.get_data(data["id"]):
                self.update_data(data["id"], data, vectorizer_name)
            else:
                self.add_data(data, vectorizer_name)
        else:
            raise ValueError("Data must contain an 'id' field")


class BackupMixin:
    def __init__(self, client: ClickhouseClient, table_manager: ClickhouseTableManager) -> None:
        self.backup_manager = BackupManager(client, table_manager)

    def backup_database(self, path: str) -> None:
        self.backup_manager.backup_to_file(path)

    def restore_database(self, path: str, table_schema: Optional[Dict[str, str]] = None, engine: str = "MergeTree", order_by: str = "id") -> None:
        self.backup_manager.restore_from_file(path, table_schema, engine, order_by)

    def load_from_file(self, path: str, table_schema: Optional[Dict[str, str]] = None, engine: str = "MergeTree", order_by: str = "id") -> None:
        self.restore_database(path, table_schema, engine, order_by)


class VectorizerMixin:
    def __init__(self) -> None:
        self.vectorizer_manager = VectorizerManager()

    def get_vectorizer(self, name: str) -> Optional[VectorizerBase]:
        return self.vectorizer_manager.get_vectorizer(name)

    def add_vectorizer(self, name: str, vectorizer: VectorizerBase) -> None:
        self.vectorizer_manager.add_vectorizer(name, vectorizer)

    def vectorize(self, data: Dict[str, Any], vectorizer_name: str) -> List[float]:
        vectorizer = self.get_vectorizer(vectorizer_name)
        if not vectorizer:
            raise ValueError(f"Vectorizer '{vectorizer_name}' not found")
        return vectorizer.vectorize(data["title"])

    def bulk_vectorize(self, data_list: List[Dict[str, Any]], vectorizer_name: str) -> List[List[float]]:
        vectorizer = self.get_vectorizer(vectorizer_name)
        if not vectorizer:
            raise ValueError(f"Vectorizer '{vectorizer_name}' not found")
        return vectorizer.bulk_vectorize([data["title"] for data in data_list])
