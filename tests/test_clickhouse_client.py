
def test_connect(clickhouse_client):
    clickhouse_client.connect()
    clickhouse_client.connect.assert_called_once()

def test_execute_query(clickhouse_client):
    query = "SELECT * FROM test"
    clickhouse_client.execute_query(query)
    clickhouse_client.execute_query.assert_called_once_with(query)  # No second argument

def test_fetch_one(clickhouse_client):
    query = "SELECT * FROM test WHERE id = 1"
    clickhouse_client.fetch_one(query)
    clickhouse_client.fetch_one.assert_called_once_with(query)  # No second argument

def test_fetch_all(clickhouse_client):
    query = "SELECT * FROM test"
    clickhouse_client.fetch_all(query)
    clickhouse_client.fetch_all.assert_called_once_with(query)  # No second argument

def test_close(clickhouse_client):
    clickhouse_client.close()
    clickhouse_client.close.assert_called_once()
