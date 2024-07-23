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
        try:
            if not self._check_table_exists():
                self._create_table(table_schema, engine, order_by)
            else:
                self.logger.info(f"Table '{self.table_name}' already exists.")
        except Exception as e:
            self.logger.exception(f"Error initializing table '{self.table_name}': {e}")

    def _create_table(self, table_schema: Dict[str, str], engine: str, order_by: str) -> None:
        try:
            fields = ", ".join([f"{name} {dtype}" for name, dtype in table_schema.items()])
            query = f"CREATE TABLE {self.table_name} ({fields}) ENGINE = {engine} ORDER BY {order_by}"
            self.client.execute_query(query)
            self.logger.info(f"Table '{self.table_name}' created with schema: {table_schema}, engine: {engine}, order by: {order_by}")
        except Exception as e:
            self.logger.error(f"Failed to create table '{self.table_name}': {e}")
            raise RuntimeError(f"Failed to create table '{self.table_name}'") from e

    def _check_table_exists(self) -> bool:
        query = f"EXISTS TABLE {self.table_name}"
        try:
            result = self.client.execute_query(query)
            return result[0][0] == 1
        except Exception as e:
            self.logger.error(f"Failed to check if table '{self.table_name}' exists: {e}")
            raise RuntimeError(f"Failed to check if table '{self.table_name}' exists") from e

    def reset_database(self) -> None:
        try:
            self.table_manager.reset_table()
            self.logger.info("RAG table reset")
        except Exception as e:
            self.logger.error(f"Failed to reset RAG table: {e}")
            raise RuntimeError("Failed to reset RAG table") from e


class DataOperationsMixin:
    def add_data(self, data: Dict[str, Any], vectorizer_name: Optional[str] = None) -> None:
        try:
            if vectorizer_name:
                data["vector"] = self.vectorize(data, vectorizer_name)
            self.validate_data(data)
            self.table_manager.insert([data])
            self.logger.info(f"Data added with id {data.get('id')}")
        except Exception as e:
            self.logger.exception(f"Failed to add data: {e}")
            raise RuntimeError("Failed to add data") from e

    def add_bulk_data(self, data_list: List[Dict[str, Any]], vectorizer_name: Optional[str] = None, vectorizer: VectorizerBase = None) -> None:
        try:
            if vectorizer_name and vectorizer:
                raise ValueError("Only one of vectorizer_name and vectorizer should be provided.")
            elif not (vectorizer_name or vectorizer):
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
        except Exception as e:
            self.logger.exception(f"Failed to add bulk data: {e}")
            raise RuntimeError("Failed to add bulk data") from e

    def delete_data(self, data_id: str) -> None:
        try:
            self.table_manager.delete({"id": data_id})
            self.logger.info(f"Data deleted with id {data_id}")
        except Exception as e:
            self.logger.error(f"Failed to delete data with id {data_id}: {e}")
            raise RuntimeError(f"Failed to delete data with id {data_id}") from e

    def update_data(self, data_id: str, new_data: Dict[str, Any], vectorizer_name: Optional[str] = None) -> None:
        if vectorizer_name:
            new_data["vector"] = self.vectorize(new_data, vectorizer_name)
        self.validate_data(new_data)
        try:
            self.table_manager.update(new_data, {"id": data_id})
            self.logger.info(f"Data updated with id {data_id}")
        except Exception as e:
            self.logger.error(f"Failed to update data with id {data_id}: {e}")
            raise RuntimeError(f"Failed to update data with id {data_id}") from e

    def search(self, query: str, similarity: bool = False, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            if similarity:
                return self.similarity_search(query, top_k)
            results = self.table_manager.search(query)
            self.logger.info(f"Search executed with query '{query}', found {len(results)} results")
            return results
        except Exception as e:
            self.logger.exception(f"Failed to execute search with query '{query}': {e}")
            raise RuntimeError(f"Failed to execute search with query '{query}'") from e

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
        try:
            result = self.client.execute_query(query, params=params, settings={"max_query_size": "10000000000000"})
            self.logger.info(f"Similarity search executed with embedding, top {top_k} results found")
            return [{"id": row[0], "title": row[1], "cosine_distance": row[2]} for row in result]
        except Exception as e:
            self.logger.error(f"Failed to execute similarity search: {e}")
            raise RuntimeError("Failed to execute similarity search") from e

    def validate_data(self, data: Dict[str, Any]) -> None:
        if "id" not in data:
            raise ValueError("Data must contain an 'id' field")

    def get_data(self, data_id: str) -> Optional[Dict[str, Any]]:
        query = f"SELECT * FROM {self.table_name} WHERE id = %(data_id)s"
        try:
            result = self.client.fetch_one(query, params={"data_id": data_id})
            if result:
                return dict(zip(self.table_manager.columns, result))
            return None
        except Exception as e:
            self.logger.error(f"Failed to get data with id {data_id}: {e}")
            raise RuntimeError(f"Failed to get data with id {data_id}") from e

    def set_data(self, data: Dict[str, Any], vectorizer_name: Optional[str] = None) -> None:
        if "id" in data:
            try:
                if self.get_data(data["id"]):
                    self.update_data(data["id"], data, vectorizer_name)
                else:
                    self.add_data(data, vectorizer_name)
            except Exception as e:
                self.logger.error(f"Failed to set data: {e}")
                raise RuntimeError("Failed to set data") from e
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
