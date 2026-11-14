# ClickHouse RAG 1.0 delivery

The project will provide a Python retrieval layer for teams already running ClickHouse.
It will ingest passages with embeddings, run filtered dense or hybrid search, and return
passages that can be cited or handed directly to any application or language model.

## Release sequence

| Milestone | Scope | Evidence before promotion |
| --- | --- | --- |
| Design | Audit 0.x, compare approaches, specify the public contract | Review the design and the migration boundary |
| 1.0.0-alpha.1 | Typed package, exact vector retrieval, batch ingestion, explicit schema lifecycle | Unit tests, installable wheel and real ClickHouse round trip |
| 1.0.0-beta.1 | Lexical and hybrid retrieval, provider extension points, benchmark and operations guidance | Same filters across all retrieval paths, repeatable retrieval evaluation, retrieval failure tests |
| 1.0.0-rc.1 | Full release pipeline, Pages, migration walkthrough, security and supply-chain checks | Green CI including CPython 3.11–3.15 and both target ClickHouse LTS lines, clean wheel and sdist installs, verified release rehearsal |
| 1.0.0 | Stable API, release artifacts and hosted documentation | Checks on the release commit, verify the published artifact and documented quickstart |
| Reuse | Separate Python library template and a playbook for OSS agents | Generate a fresh independent project and repeat its CI locally, verify template visibility if published |

Work progresses in separate branches with Conventional Commits. Prerelease labels
describe the version of the artifacts staged at each milestone. Only verified artifacts
may be published. The GitHub and PyPI states must be checked independently.

## Acceptance

- The packaged example runs from a clean environment against ClickHouse with no
  external embedding service.
- Exact retrieval has deterministic ordering. Dense, lexical and hybrid queries
  enforce the same namespace and metadata filters.
- The project documents consistency guarantees, limits and maintenance privileges.
- The stable interface and on-disk schema migration are explicit.
- CI exercises the supported Python range and real 26.3 and 26.8 LTS servers.
- Benchmarks report data, seed, conditions, quality and latency. Synthetic results
  are labelled as such. ANN promotion requires plan and recall evidence.


## Delivery record

- Design, alpha and beta are preserved in separate branches. The RC was
  merged through [PR #1](https://github.com/RebelRaider/ClickhouseRAG/pull/1)
  and [released on GitHub](https://github.com/RebelRaider/ClickhouseRAG/releases/tag/v1.0.0-rc.1).
- Stable 1.0.0 was promoted through
  [PR #3](https://github.com/RebelRaider/ClickhouseRAG/pull/3).
  See the [release run](https://github.com/RebelRaider/ClickhouseRAG/actions/runs/43591253502),
  [release files](https://github.com/RebelRaider/ClickhouseRAG/releases/tag/v1.0.0)
  and [hosted documentation](https://rebelraider.github.io/ClickhouseRAG/).
- Version 1.0.1 was published to [PyPI](https://pypi.org/project/clickhouserag/1.0.1/)
  using Trusted Publishing and to [GitHub Releases](https://github.com/RebelRaider/ClickhouseRAG/releases/tag/v1.0.1).

## ClickHouse RAG framework

The application-facing flow is to configure one ClickHouse client and model
provider, index documents, retrieve or ask, and inspect the supporting evidence.
Keep the existing 1.x passage APIs usable directly. New storage is provisioned
alongside existing tables. No separate graph or vector service is required.

| Stage | Deliverable | Verification |
| --- | --- | --- |
| Graph evidence | Same-namespace relationship indexing and bounded multi-hop retrieval with current passage citations | Both LTS servers, stale revisions, filtered intermediates, cycles, deterministic fanout |
| Framework and GraphRAG | Document chunking, extraction provider, automatic query seeds, mixed retrieval, context assembly and optional generation with validated citations | Installable end-to-end example, persisted state, repeat indexing, empty evidence and provider failure |
| Document lifecycle | Replace and delete a document atomically for readers | Boundary fault, shorter replacements, changed metadata, retry and concurrent generations on both LTS servers |
| Corpus overview | Global search across groups of scoped relationships | Separate workflows, disconnected evidence, filtered bridges and an explicit scan limit |
| KAG | Typed relation schema, validated structured plans, path intersections and evidence for each step | Multi-hop, invalid plans, ambiguity and unanswerable queries against real ClickHouse |
| Evaluation | Labelled dense, hybrid, graph and KAG comparison; ranking, answer/citation and latency metrics | Reproducible corpus and results by mode and query category, empty and failure cases |
| Stabilization | Migration and operations, quickstart, packaging, CI and design trade-offs | Lowest-friction installed examples, run across supported ClickHouse versions, artifact-level verification |

Sources and rationale are in the
[framework design](docs/maintainer/framework.md).

The framework was integrated in [PR #7](https://github.com/RebelRaider/ClickhouseRAG/pull/7)
and its wheel and sdist verified from the
[1.1.0 release candidate](https://github.com/RebelRaider/ClickhouseRAG/releases/tag/v1.1.0-rc.1).
The [benchmark](docs/benchmark.md) records the 12-case public comparison and
the synthetic shared-hub cost probe. Continue evaluating model planning,
graph candidate ranking and community quality on representative workloads.

## Composable model and retrieval integrations

Keep ClickHouse as the evidence store and retrieval engine. Compose independent
embedding, relation extraction, knowledge planning and answer stages. A model
change must not require an application rewrite. A change to the vector space
requires a fresh collection or an intentional backfill. Verify 1.1 workflows
alongside new provider combinations.

| Stage | Deliverable | Evidence |
| --- | --- | --- |
| Model adapters | OpenAI-compatible embeddings, Chat Completions and Responses, native Ollama, Claude, Gemini, Cohere and Voyage, plus a structured backend extension point | Request/response tests for every wire API and an independently composed ClickHouse + local model run |
| Pipeline composition | Support independent retrieval, reranking and generation choices, with bounded candidates and stable source IDs | Retrieval-only and full RAG runs, scoped evidence after reranking, ambiguous and failed provider responses |
| Additional local and managed runtimes | Exercise a local embedding server or in-process model and document capabilities across cloud gateways and engines | CPU or local end-to-end runs, supported Python matrix, optional dependencies stay optional |
| PostgreSQL add-on | Publish committed PostgreSQL document revisions into ClickHouse through an optional outbox | Source identity, restartable sync, replacements, deletion and concurrent workers tested on real databases |
| Delivery | Docs and tested artifacts for model roles and supported API capabilities | Clean installs, supported Python and ClickHouse CI, reproducible examples without hosted credentials |

Version 1.2.0 shipped the core [provider adapters](docs/providers.md), reranking,
custom chunking and optional [PostgreSQL delivery](docs/postgres.md). The
follow-up adds Bedrock, Google Cloud and bounded multi-query planning.
Hosted adapters use documented wire-contract fixtures in CI. The locally
verified examples use Ollama and llama.cpp over ClickHouse.
