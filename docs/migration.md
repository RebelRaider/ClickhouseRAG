# Upgrade from 0.x

The package name stays `clickhouserag`. Version 1.0 moves to the official
ClickHouse Connect HTTP driver and a new table layout. Keep the old table while
reviewing the migration. The old `RAGManager`, table wrapper, vectorizer
registry and backup interface are removed in this major release.

| 0.x operation | 1.x operation |
| --- | --- |
| Configure the native TCP client | Create a `clickhouse_connect` HTTP client. Common ports are 8123 for HTTP and 8443 for HTTPS. |
| Define a table by passing a schema to the manager | Create a separate table with `ClickHouseStore.initialize()`. |
| Write records carrying vectors | Convert them to `Chunk` and `EmbeddedChunk` and call `insert()`. |
| Vectorize with a registered vectorizer | Implement `EmbeddingProvider` and call `Retriever.index()`. |
| Search a vector and get dictionary rows | Call `search_vector()` and read typed hits with sources. |
| Search via application SQL | Use the application-owned driver for that SQL. |
| Backup, restore or delete rows | Use ClickHouse administrative workflows. |

1. Export or stream the former IDs, text, source information, metadata and
   vectors from the old table. Inspect duplicate IDs, missing sources,
   vector dimensions and model identity.
2. Pick stable string IDs for documents and chunks. Use revision 1 for the
   initial import. Re-embed when dimensions, model identity or encoding
   configuration is unknown. Pass precomputed vectors only when they match
   the new store's embedding space.
3. Create a separate table and import bounded batches. Compare counts of
   logical chunk IDs and manually verify that sampled queries return the
   expected sources and metadata.
4. Switch application reads and writes to the new table after validation.
   Keep the former table during rollout and use your normal backups for
   rollback. Coordinate writes and replay any backlog if rolling back.

An alpha table predates the lexical token column. Import its passages into a
new table using the same procedure. Beta and RC use the 1.0 schema manifest.
