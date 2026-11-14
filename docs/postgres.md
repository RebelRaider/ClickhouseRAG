# PostgreSQL document delivery

Use PostgreSQL to record source document changes in the same transaction as
your application's writes. Workers deliver the latest committed revision of
each document to ClickHouse, where text, embeddings and relationship facts
become available to every configured retrieval mode. The add-on runs only
when an application enqueues and synchronizes changes.

```sh
pip install 'clickhouserag[postgres]'
```

Provision one outbox table per target ClickHouse collection and namespace.
The example uses three connections. The application owns the configuration,
transactions and connection lifecycle.

```python
from clickhouserag.postgres import PostgresOutbox

outbox = PostgresOutbox(table="product_docs_outbox")
with provisioner.transaction():
    outbox.initialize(provisioner)

# The collection is a configured RAG instance.
rag.initialize()
with application_db.transaction():
    save_product(application_db, product)
    outbox.enqueue(
        application_db,
        str(product.id),
        product.description,
        product.url,
        metadata={"team": "catalog"},
    )

# Run in a worker with a dedicated PostgreSQL connection.
delivered = outbox.sync(worker_db, rag, limit=32)
```

Call `enqueue_delete(connection, document_id)` in the application transaction
that removes a source document. The outbox retains its revision so that
recreating the document uses a greater revision. Keep all writes for the
same collection and namespace on one revision source.

## Delivery and operations

The outbox coalesces changes by document ID. A worker locks one pending row
with [`FOR UPDATE SKIP LOCKED`](https://www.postgresql.org/docs/current/sql-select.html#SQL-FOR-UPDATE-SHARE),
publishes that revision to ClickHouse and commits the acknowledgment.
A lost acknowledgment can cause the same revision to be retried. ClickHouse
accepts an already published identical revision without embedding it again.
Writers updating the same source row can wait while the worker holds its
lock during inference. Keep provider timeouts short enough for your workload.

Schedule `sync` from your worker. Exceptions stop the current call and roll
back that row's acknowledgment, while earlier successful rows remain
committed. Monitor the pending rows and surface failed operations to the
worker or operator. Intermediate edits can be coalesced. Preserve the outbox
and its revision history through backup and restore. If rebuilding ClickHouse,
re-index authoritative source documents or replay the outbox into a fresh
collection under a plan that preserves revision ownership.

PostgreSQL still commits independently of ClickHouse. After the source
transaction commits, retrieval catches up when a worker delivers the change.
Until then it can return the last published revision. Source rows and their
deletions must be enqueued by the application. A worker does not need access
to the application's other PostgreSQL tables.
