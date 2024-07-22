"""RAG manager module for managing data in Clickhouse."""

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from clickhouserag.data_access.clickhouse_client import ClickhouseClient
from clickhouserag.data_access.clickhouse_table import ClickhouseTableManager
from clickhouserag.rag.base import RAGBase
from clickhouserag.vectorizers.base import VectorizerBase
from clickhouserag.vectorizers.manager import VectorizerManager


class RAGManager(RAGBase):
    """Manager class for RAG operations in Clickhouse."""

    def __init__(self, client: ClickhouseClient, table_name: str, table_schema: Optional[Dict[str, str]] = None, engine: str = "MergeTree", order_by: str = "id") -> None:
        """Initialize RAGManager.

        Args:
        ----
            client (ClickhouseClient): The Clickhouse client.
            table_name (str): The name of the table.
            table_schema (Optional[Dict[str, str]]): The schema of the table.
            engine (str): The engine to use for the table.
            order_by (str): The column to order the table by.

        """
        self.client = client
        self.table_manager = ClickhouseTableManager(client, table_name)
        self.table_name = table_name
        self.table_schema = table_schema
        self.engine = engine
        self.order_by = order_by
        self.vectorizer_manager = VectorizerManager()
        self.logger = logging.getLogger(__name__)

        if table_schema:
            self._initialize_table(table_schema, engine, order_by)

    def _initialize_table(self, table_schema: Dict[str, str], engine: str, order_by: str) -> None:
        """Initialize the table in Clickhouse based on the provided schema if it does not exist."""
        if not self._check_table_exists():
            self._create_table(table_schema, engine, order_by)
        else:
            self.logger.info(f"Table '{self.table_name}' already exists.")

    def _create_table(self, table_schema: Dict[str, str], engine: str, order_by: str) -> None:
        """Create a table in Clickhouse based on the provided schema."""
        try:
            fields = ", ".join([f"{name} {dtype}" for name, dtype in table_schema.items()])
            query = f"CREATE TABLE {self.table_name} ({fields}) ENGINE = {engine} ORDER BY {order_by}"
            self.client.execute_query(query)
            self.logger.info(f"Table '{self.table_name}' created with schema: {table_schema}, engine: {engine}, order by: {order_by}")
        except Exception as e:
            self.logger.error(f"Failed to create table '{self.table_name}': {e}")
            raise RuntimeError(f"Failed to create table '{self.table_name}'") from e

    def _check_table_exists(self) -> bool:
        """Check if the table exists in the database."""
        query = f"EXISTS TABLE {self.table_name}"
        try:
            result = self.client.execute_query(query)
            return result[0][0] == 1
        except Exception as e:
            self.logger.error(f"Failed to check if table '{self.table_name}' exists: {e}")
            raise RuntimeError(f"Failed to check if table '{self.table_name}' exists") from e

    def add_data(self, data: Dict[str, Any], vectorizer_name: Optional[str] = None) -> None:
        """Add data to the RAG."""
        if vectorizer_name:
            data["vector"] = self.vectorize(data, vectorizer_name)
        self.validate_data(data)
        try:
            self.table_manager.insert([data])
            self.logger.info(f"Data added with id {data.get('id')}")
        except Exception as e:
            self.logger.error(f"Failed to add data: {e}")
            raise RuntimeError("Failed to add data") from e

    def add_bulk_data(self, data_list: List[Dict[str, Any]], vectorizer_name: Optional[str] = None, vectorizer: VectorizerBase = None) -> None:
        """Add multiple data records to the RAG."""
        if vectorizer_name and vectorizer:
            raise ValueError("Only one of vectorizer_name and vectorizer should be provided.")
        elif not (vectorizer_name or vectorizer):
            raise ValueError("Either vectorizer_name or vectorizer should be provided.")
        elif vectorizer_name:
            for data in data_list:
                data["vector"] = self.vectorize(data, vectorizer_name)
        elif vectorizer:
            vectors = vectorizer.bulk_vectorize([data["title"] for data in data_list])
            for data, vector in zip(data_list, vectors):
                data["vector"] = vector

        for data in data_list:
            self.validate_data(data)

        try:
            self.table_manager.insert(data_list)
            self.logger.info(f"Bulk data added with {len(data_list)} records")
        except Exception as e:
            self.logger.error(f"Failed to add bulk data: {e}")
            raise RuntimeError("Failed to add bulk data") from e

    def delete_data(self, data_id: str) -> None:
        """Delete data from the RAG."""
        try:
            self.table_manager.delete({"id": data_id})
            self.logger.info(f"Data deleted with id {data_id}")
        except Exception as e:
            self.logger.error(f"Failed to delete data with id {data_id}: {e}")
            raise RuntimeError(f"Failed to delete data with id {data_id}") from e

    def update_data(self, data_id: str, new_data: Dict[str, Any], vectorizer_name: Optional[str] = None) -> None:
        """Update data in the RAG."""
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
        """Search the RAG."""
        if similarity:
            return self.similarity_search(query, top_k)
        else:
            try:
                results = self.table_manager.search(query)
                self.logger.info(f"Search executed with query '{query}', found {len(results)} results")
                return results
            except Exception as e:
                self.logger.error(f"Failed to execute search with query '{query}': {e}")
                raise RuntimeError(f"Failed to execute search with query '{query}'") from e

    def similarity_search(self, embedding: np.array, columns: List[str], top_k: Optional[int]) -> List[Dict[str, Any]]:
        """Search the RAG based on cosine similarity."""
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

    def compute_cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Compute the cosine similarity between two vectors."""
        dot_product = sum(p * q for p, q in zip(vector1, vector2))
        magnitude1 = sum(p ** 2 for p in vector1) ** 0.5
        magnitude2 = sum(q ** 2 for q in vector2) ** 0.5
        if magnitude1 * magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)

    def reset_database(self) -> None:
        """Reset the RAG database."""
        try:
            self.table_manager.reset_table()
            self.logger.info("RAG table reset")
        except Exception as e:
            self.logger.error(f"Failed to reset RAG table: {e}")
            raise RuntimeError("Failed to reset RAG table") from e

    def backup_database(self, path: str) -> None:
        """Backup the RAG database to a file."""
        try:
            data = self.table_manager.fetch_all()
            with open(path, "w") as file:
                json.dump(data, file)
            self.logger.info(f"Database backup created at {path}")
        except Exception as e:
            self.logger.error(f"Failed to backup database to {path}: {e}")
            raise RuntimeError(f"Failed to backup database to {path}") from e

    def restore_database(self, path: str, table_schema: Optional[Dict[str, str]] = None, engine: str = "MergeTree", order_by: str = "id") -> None:
        """Restore the RAG database from a file."""
        if table_schema:
            self._create_table(table_schema, engine, order_by)
        try:
            with open(path, "r") as file:
                data = json.load(file)
            self.table_manager.reset_table()
            self.table_manager.insert(data)
            self.logger.info(f"Database restored from {path}")
        except Exception as e:
            self.logger.error(f"Failed to restore database from {path}: {e}")
            raise RuntimeError(f"Failed to restore database from {path}") from e

    def save_to_file(self, path: str) -> None:
        """Save the RAG database to a file."""
        self.backup_database(path)

    def load_from_file(self, path: str, table_schema: Optional[Dict[str, str]] = None, engine: str = "MergeTree", order_by: str = "id") -> None:
        """Load the RAG database from a file."""
        self.restore_database(path, table_schema, engine, order_by)

    def validate_data(self, data: Dict[str, Any]) -> None:
        """Validate the data to be inserted into the RAG."""
        if "id" not in data:
            raise ValueError("Data must contain an 'id' field")
        # Additional data validation can be added here

    def get_data(self, data_id: str) -> Optional[Dict[str, Any]]:
        """Get data from the RAG by ID."""
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
        """Set data in the RAG."""
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

    def get_vectorizer(self, name: str) -> Optional[VectorizerBase]:
        """Get a vectorizer by name."""
        return self.vectorizer_manager.get_vectorizer(name)

    def add_vectorizer(self, name: str, vectorizer: VectorizerBase) -> None:
        """Add a vectorizer to the manager."""
        self.vectorizer_manager.add_vectorizer(name, vectorizer)

    def vectorize(self, data: Dict[str, Any], vectorizer_name: str) -> List[float]:
        """Vectorize data using the specified vectorizer."""
        vectorizer = self.get_vectorizer(vectorizer_name)
        if not vectorizer:
            raise ValueError(f"Vectorizer '{vectorizer_name}' not found")
        return vectorizer.vectorize(data["title"])

    def bulk_vectorize(self, data_list: List[Dict[str, Any]], vectorizer_name: str) -> List[List[float]]:
        """Vectorize a list of data using the specified vectorizer."""
        vectorizer = self.get_vectorizer(vectorizer_name)
        if not vectorizer:
            raise ValueError(f"Vectorizer '{vectorizer_name}' not found")
        return vectorizer.bulk_vectorize([data["title"] for data in data_list])
