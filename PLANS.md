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
- PyPI publishing uses the existing project and requires its owner to
  configure Trusted Publishing. The tag is already available on GitHub.
