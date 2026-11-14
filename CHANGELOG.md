# Changelog

<!-- version list -->

## v1.3.0 (2026-11-14)

- Add Amazon Bedrock Converse and Titan V2 and Google Cloud publisher model
  adapters with application-owned cloud credentials.
- Combine scoped searches through `retrieve_many` or `ask_many`. For
  model-selected search terms and modes, configure a separate search planner
  and use `ask_planned`.
- Add a runnable example that uses Ollama for search planning and llama.cpp
  for embeddings and cited generation over ClickHouse.


## v1.3.0-rc.1 (2026-11-14)

### Documentation

- **models**: Verify a scoped pipeline across multiple local runtimes
  ([`ac569a4`](https://github.com/RebelRaider/ClickhouseRAG/commit/ac569a4dbe09f87d6b1a3c14465fe18d700a6b9e))

### Features

- **models**: Add Bedrock Runtime and Google Cloud publisher adapters
  ([`3d7acb7`](https://github.com/RebelRaider/ClickhouseRAG/commit/3d7acb7f18e06dc89edbadf0278cae712699b77c))

- **retrieval**: Compose several scoped searches into cited answers
  ([`026b033`](https://github.com/RebelRaider/ClickhouseRAG/commit/026b033a465269e2ec8eb33e26c8acf9fcdd166e))

- **retrieval**: Plan bounded searches with a separate model
  ([`0c3056a`](https://github.com/RebelRaider/ClickhouseRAG/commit/0c3056a50b7a733190b25741432f0433e0f15e35))


## v1.2.0 (2026-11-14)

- Compose independent models for embeddings, graph extraction, knowledge
  planning and cited generation. Includes native Ollama, Claude, Gemini,
  Cohere and Voyage plus OpenAI-compatible APIs and Azure OpenAI v1 auth.
- Add optional reranking, custom chunking and versioned index preparation.
- Use the PostgreSQL extra to deliver committed document changes into
  ClickHouse. Existing ClickHouse collections keep their schema and Ollama
  compatibility. See the [provider guide](https://rebelraider.github.io/ClickhouseRAG/providers/).


## v1.2.0-rc.1 (2026-11-14)

### Features

- **ingestion**: Support custom chunking and versioned index preparation
  ([`a1a3e24`](https://github.com/RebelRaider/ClickhouseRAG/commit/a1a3e24f8682e61c4d38f31f4d1377b51836d3c6))

- **ingestion**: Synchronize PostgreSQL document revisions to ClickHouse
  ([`f1e4500`](https://github.com/RebelRaider/ClickhouseRAG/commit/f1e45005ff0fcdd0e749754b6eda4a38286e9bca))

- **models**: Compose native and compatible inference providers
  ([`d0a1c30`](https://github.com/RebelRaider/ClickhouseRAG/commit/d0a1c30d4857bfd230e1d77b5caf39efe802fba3))

- **models**: Support Azure v1 keys and refreshing gateway credentials
  ([`40c7a27`](https://github.com/RebelRaider/ClickhouseRAG/commit/40c7a27fa82fb447d702e729e646f01b06df7760))

- **retrieval**: Rerank scoped candidates with native model providers
  ([`63299dc`](https://github.com/RebelRaider/ClickhouseRAG/commit/63299dc41637452e7daff77fa4fdc764e77dd8b1))


## v1.1.0 (2026-11-14)

- Add the document-oriented `RAG` API with atomic revision visibility and
  logical deletion in a managed ClickHouse table.
- Add cited relationship traversal and corpus overviews alongside hybrid text
  and vector search. Hybrid is the default.
- Add typed knowledge chains and intersections with complete path citations,
  an Ollama adapter, and retrieval, answer and citation evaluation.
- Keep the existing `ClickHouseStore` and `Retriever` APIs and tables usable.
  See the [framework walkthrough](https://rebelraider.github.io/ClickhouseRAG/framework/)
  and [measured benchmarks](https://rebelraider.github.io/ClickhouseRAG/benchmark/).

### Testing

- **graph**: Measure traversal cost under a shared hub
  ([`1d8d8e5`](https://github.com/RebelRaider/ClickhouseRAG/commit/1d8d8e53a2bf7536e81ab0bcada16acdb9ebb138))


## v1.1.0-rc.1 (2026-11-14)

Hybrid is the default for document retrieval and answers. Choose
`mode="mixed"` or `mode="graph"` for relationship expansion. The HotpotQA
bridge benchmark reports the evidence, answer and citation trade-offs.

### Bug Fixes

- **graph**: Distinguish typed nodes and bound evidence interleaving cost
  ([`8603636`](https://github.com/RebelRaider/ClickhouseRAG/commit/86036361e70d6bb09d45d34f3f6f5f553913fe3f))

- **knowledge**: Abstain on invalid model plans and validate extracted names
  ([`ae9ce3e`](https://github.com/RebelRaider/ClickhouseRAG/commit/ae9ce3eda84bb8fb8af3756cd96e21b2ce3b278b))

- **models**: Bound Ollama fact extraction per passage
  ([`4a5cdd6`](https://github.com/RebelRaider/ClickhouseRAG/commit/4a5cdd6ae1a6728b9cc6fcc5d762ebc20ac2edb2))

- **models**: Constrain citation IDs and planner fields in structured output
  ([`5300f55`](https://github.com/RebelRaider/ClickhouseRAG/commit/5300f55c1b6473edb4ac4885b0e8801dfbd0cce1))

### Documentation

- Link the framework walkthrough from the README
  ([`262a967`](https://github.com/RebelRaider/ClickhouseRAG/commit/262a9672f344f52130191bd8aed3696096ea6e50))

- Prepare framework adoption and release candidate metadata
  ([`508eb9c`](https://github.com/RebelRaider/ClickhouseRAG/commit/508eb9c0a86f272ec6e53a6f4eeda09323b360dc))

- Record the reproducible graph evaluation baseline
  ([`bcb5391`](https://github.com/RebelRaider/ClickhouseRAG/commit/bcb5391125e432481de8b8aa371b59436d2c4695))

- Specify the ClickHouse RAG framework and verification stages
  ([`13da92e`](https://github.com/RebelRaider/ClickhouseRAG/commit/13da92ebb1cb5dafd4829a6ab6b4a75e50b08b53))

### Features

- Add cited graph retrieval, typed knowledge reasoning and RAG evaluation
  ([`cfbd7cc`](https://github.com/RebelRaider/ClickhouseRAG/commit/cfbd7cc39f60ab91d04e4ac7eba5890898caf379))

- **documents**: Publish replacements and deletions with cited graph facts
  ([`a624e87`](https://github.com/RebelRaider/ClickhouseRAG/commit/a624e87eaf5e2de28d0c4a4829f080ea6fc184b1))

- **evaluation**: Add reproducible HotpotQA comparison harness
  ([`813838d`](https://github.com/RebelRaider/ClickhouseRAG/commit/813838d8a8a4890f014a1cef6a3cde92d4f64049))

- **evaluation**: Report quality and latency by query category
  ([`1e2b562`](https://github.com/RebelRaider/ClickhouseRAG/commit/1e2b56265d3c2545de6e3acd06325f7d11b98193))

- **graph**: Add scoped corpus summaries with diverse cited evidence
  ([`dda57ca`](https://github.com/RebelRaider/ClickhouseRAG/commit/dda57ca1cbe212d319d03b9af914b9f18412ed48))

- **knowledge**: Accept explicit plans for cited answers
  ([`d8009f2`](https://github.com/RebelRaider/ClickhouseRAG/commit/d8009f2fcda6b8c4fafd93e58c8edc43ca012362))

- **knowledge**: Generate typed intersections and require complete citations
  ([`7581975`](https://github.com/RebelRaider/ClickhouseRAG/commit/758197553e924a168f776a5d3edec10116f86b06))

- **knowledge**: Intersect typed paths with complete source evidence
  ([`c250563`](https://github.com/RebelRaider/ClickhouseRAG/commit/c250563decf16db0f5f6d36f966d8576afef10a6))

- **models**: Add local Ollama integration for the RAG pipeline
  ([`400dbac`](https://github.com/RebelRaider/ClickhouseRAG/commit/400dbacd8d8b6e6a11b13621c460d1a9c6066d13))

- **models**: Configure a RAG collection in one Ollama call
  ([`488898d`](https://github.com/RebelRaider/ClickhouseRAG/commit/488898d9edaf8f6d191c4945127f8f7016a2df2e))

### Performance Improvements

- **graph**: Rank visible entity candidates inside ClickHouse
  ([`5e36757`](https://github.com/RebelRaider/ClickhouseRAG/commit/5e367573dd6dcd2fc760c5edf5dc311d447120da))

### Refactoring

- **evaluation**: Name summary metrics in the report constructor
  ([`410cc5e`](https://github.com/RebelRaider/ClickhouseRAG/commit/410cc5eb15f3d6a2058eee26caf0d42dbc159c50))

### Testing

- **evaluation**: Exercise knowledge intersections and corpus overview
  ([`743a176`](https://github.com/RebelRaider/ClickhouseRAG/commit/743a1768145a4655d2600a7c1db1ea6ad5df11d9))


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
