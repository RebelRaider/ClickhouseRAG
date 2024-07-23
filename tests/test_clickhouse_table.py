
def test_insert(table_manager, sample_data):
    table_manager.insert([sample_data])
    query = "INSERT INTO test_table VALUES"
    table_manager.client.execute_query.assert_called_once_with(query, [sample_data])

def test_update(table_manager, sample_data):
    values = {"title": "Updated Title"}
    conditions = {"id": "1"}
    table_manager.update(values, conditions)
    set_clause = "title = %(title)s"
    condition_clause = "id = %(id)s"
    query = f"ALTER TABLE test_table UPDATE {set_clause} WHERE {condition_clause}"
    table_manager.client.execute_query.assert_called_once_with(query, {**values, **conditions})

def test_delete(table_manager):
    conditions = {"id": "1"}
    table_manager.delete(conditions)
    condition_clause = "id = %(cond_id)s"
    query = f"DELETE FROM test_table WHERE {condition_clause}"
    params = {f"cond_{key}": value for key, value in conditions.items()}
    table_manager.client.execute_query.assert_called_once_with(query, params)

def test_search(table_manager):
    query = "SELECT * FROM test_table"
    table_manager.search(query)
    table_manager.client.fetch_all.assert_called_once_with(query, None)

def test_fetch_all(table_manager):
    table_manager.fetch_all()
    query = "SELECT * FROM test_table"
    table_manager.client.fetch_all.assert_called_once_with(query, None)

def test_reset_table(table_manager):
    table_manager.reset_table()
    query = "TRUNCATE TABLE test_table"
    table_manager.client.execute_query.assert_called_once_with(query, None)
