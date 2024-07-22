from unittest.mock import MagicMock, patch

import numpy as np


def test_initialize_table(rag_manager):
    rag_manager._initialize_table = MagicMock()
    rag_manager._initialize_table(rag_manager.table_schema, rag_manager.engine, rag_manager.order_by)
    rag_manager._initialize_table.assert_called_once_with(rag_manager.table_schema, rag_manager.engine, rag_manager.order_by)

def test_add_data(rag_manager, sample_data):
    rag_manager.table_manager.insert = MagicMock()
    rag_manager.add_data(sample_data)
    rag_manager.table_manager.insert.assert_called_once_with([sample_data])

def test_add_bulk_data(rag_manager, sample_data):
    rag_manager.table_manager.insert = MagicMock()
    data_list = [sample_data, sample_data]
    with patch.object(rag_manager, "get_vectorizer", return_value=MagicMock(bulk_vectorize=lambda data: [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])):
        rag_manager.add_bulk_data(data_list, vectorizer_name="test_vectorizer")
    rag_manager.table_manager.insert.assert_called_once_with(data_list)

def test_delete_data(rag_manager):
    rag_manager.table_manager.delete = MagicMock()
    data_id = "1"
    rag_manager.delete_data(data_id)
    rag_manager.table_manager.delete.assert_called_once_with({"id": data_id})

def test_update_data(rag_manager, sample_data):
    rag_manager.table_manager.update = MagicMock()
    data_id = "1"
    new_data = {"id": "1", "title": "Updated Title", "vector": [0.4, 0.5, 0.6]}
    rag_manager.update_data(data_id, new_data)
    rag_manager.table_manager.update.assert_called_once_with(new_data, {"id": data_id})

def test_search(rag_manager):
    rag_manager.table_manager.search = MagicMock()
    query = "SELECT * FROM test_table"
    rag_manager.search(query)
    rag_manager.table_manager.search.assert_called_once_with(query)

def test_similarity_search(rag_manager, sample_data):
    embedding = np.array(sample_data["vector"])
    columns = ["id", "title", "cosine_distance"]
    top_k = 5
    rag_manager.client.execute_query = MagicMock(return_value=[("1", "Test Title", 0.9)])
    results = rag_manager.similarity_search(embedding, columns, top_k)
    rag_manager.client.execute_query.assert_called_once()
    assert len(results) == 1

def test_reset_database(rag_manager):
    rag_manager.table_manager.reset_table = MagicMock()
    rag_manager.reset_database()
    rag_manager.table_manager.reset_table.assert_called_once()

def test_backup_database(rag_manager):
    rag_manager.table_manager.fetch_all = MagicMock(return_value=[])
    path = "backup.json"
    with patch("logging.Logger.info"):
        rag_manager.backup_database(path)

