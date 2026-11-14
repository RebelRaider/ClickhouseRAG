# Operations

## Data layout and updates

Each table has one embedding model identity and dimension (1–4096).
Create a new table and re-embed when that identity changes. Database and table
names follow `[A-Za-z_][A-Za-z_0-9]{0,126}`. Store strings as UTF-8. Each batch
has a limit of 10,000 chunks. Tune the batch to your chunk and embedding sizes.

The current row is keyed by `(namespace, chunk_id)`. Use strictly increasing
positive UInt64 revisions for new payloads. A retry may repeat the same payload
and revision. Assign revisions from a single monotonic source per chunk. Two
different payloads at the same revision have undefined ordering. The API bounds
ingestion to a single validated batch per call. If a request fails after
transmission, repeat the identical batch and investigate persistent failures.
Coordinate multi-batch backfills in the caller. Physical duplicate rows can
remain until background merges.

`ReplacingMergeTree(revision)` and `SELECT … FINAL` resolve revisions as of each
query. Hybrid retrieval reads both legs in one statement with a shared table
snapshot. Filters on mutable metadata and text apply to the winning revision.
The store uses a schema manifest in the table comment, which is part of the
managed schema. Recreate the store object or call `validate_schema()` after
an authorized table replacement.

## Search and cost

- **Dense** uses native cosine similarity with Float32 storage. Larger scores
  are closer. It scans eligible vectors exactly. Positive and negative
  similarities are valid. Limit defaults to 10 and may be up to 1000.
- **Lexical** tokenizes Unicode with NFKC and case folding. It matches distinct
  letter or number terms and counts query terms present in each result. The
  token cap is 128. It uses a ClickHouse text index over the stored token array.
  Punctuation separates terms. There is no stemming, phrase match, or BM25.
- **Hybrid** combines ranks with reciprocal rank fusion and rank constant 60.
  Its score stays on the same scale when the lexical leg finds nothing.
  Candidates default to four times `limit` per leg, capped at 1000. Increasing
  them costs time and may improve recall. Ties use ascending chunk IDs.

Use selectively filtered namespaces and measure your workload before expanding
the collection. Measure p95 query latency and retrieval quality with your data.
Use `EXPLAIN indexes = 1` and `system.query_log` when tuning ClickHouse.
`FINAL` adds work where parts overlap. Review
[ClickHouse's merge guidance](https://clickhouse.com/resources/engineering/clickhouse-optimize-table-final)
before changing the merge strategy. Approximate search, graph expansion and
KAG orchestration are recorded in the [design](maintainer/design.md).

## Service permissions

The application owns the `clickhouse-connect` client. Use a secure endpoint
and certificate verification in hosted deployments. Use
`autogenerate_session_id=False` for shared HTTP clients and
`form_encode_query_params=True` so larger query parameter payloads go in the
request body. Set application appropriate connection and receive timeouts.

Run `initialize()` during provisioning with CREATE TABLE permission in the
target database. Give readers SELECT on the table, writers INSERT, and access
to the table metadata used by schema checks. The namespace limits each query.
Use separate ClickHouse grants and row policies for tenant authorization.
Manage retention and removal through ClickHouse administration.
Treat retrieved passages as untrusted content in generators and downstream
agents. Driver and server error or query logging follows your application and
ClickHouse configuration.

The 1.0 compatibility matrix covers self-managed 26.3 and 26.8 LTS. Verify
server settings, privileges and engine behavior for managed deployments.
