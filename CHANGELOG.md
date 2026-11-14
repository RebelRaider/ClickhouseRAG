# Changelog

<!-- version list -->

## v1.0.1 (2026-11-14)

### Bug Fixes

- **packaging**: Add project identity and useful PyPI metadata
  ([`35d2e87`](https://github.com/RebelRaider/ClickhouseRAG/commit/35d2e875aa6daf78682c33637505acaad413c6cf))

### Documentation

- Point the quickstart to the published wheel
  ([`0007aad`](https://github.com/RebelRaider/ClickhouseRAG/commit/0007aad53f7e2de12e37b6ed02b453378ada5abf))


## v1.0.0

- Stable Python API for database-backed dense, lexical and hybrid passage search.
- Includes a migration from the legacy `RAGManager` and old table schema.
- Verified CPython 3.11–3.15 and both ClickHouse 26.3 and 26.8 LTS lines.

## v1.0.0-rc.1

- Added a locked CI matrix, real ClickHouse services, an artifact smoke test
  and a GitHub Pages documentation build.
- Added an OIDC publishing workflow and migration and operations guidance.

## v1.0.0-beta.1

- Added Unicode lexical search and reciprocal-rank fusion on a single
  ClickHouse snapshot.
- Added the model-agnostic embedding provider protocol and retrieval wrapper.
- Added a reproducible synthetic retrieval benchmark.

## v1.0.0-alpha.1

- Replaced the 0.x managers with a typed ClickHouse store and versioned passage
  ingestion in a new schema.
- Added scoped exact cosine search and provenance in result records.
- Migrated packaging to a single PEP 621 configuration and uv.
