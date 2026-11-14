# Releases

Use Conventional Commits. `fix:` bumps the patch version, `feat:` bumps the
minor version, and a breaking change footer or `!` bumps the major version.
Release preparation uses the locked Python Semantic Release tool. Changes to
the version and changelog pass through the same branch and CI as other changes.

```sh
git switch main
git pull --ff-only
uv run --locked --group release semantic-release version --print
uv run --locked --group release semantic-release version --no-commit --no-tag --skip-build
uv lock
```

Review the changed `pyproject.toml`, `uv.lock` and `CHANGELOG.md`, and use a
release branch and pull request to merge them. After merging and confirming CI
on the merge commit, create an annotated tag at that commit and push the single
tag. Tags use `vMAJOR.MINOR.PATCH` with an optional `-alpha.N`, `-beta.N` or
`-rc.N`. The tag must agree with the PEP 440 version in `pyproject.toml`.

For an RC, create `release/<major>.<minor>-rc` from the integrated feature
branch before running semantic-release. The branch's prerelease configuration
generates the RC version and changelog. Merge the reviewed candidate into main
before tagging. To promote it, generate the next stable version on main,
refresh the lockfile and review those changes through a release pull request.

```sh
git tag -a v1.0.0 -m "ClickHouse RAG 1.0.0" <verified-merge-sha>
git push origin refs/tags/v1.0.0
```

The release workflow verifies the tag, calls CI on the tagged commit and builds
wheel and sdist. The isolated publishing jobs use the uploaded artifacts.
Check the Actions run and the release downloads. For a Python prerelease smoke
test, install from the GitHub release assets.

For public PyPI updates, add a
[Trusted Publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
to `clickhouserag` for owner `RebelRaider`, repository `ClickhouseRAG`,
workflow `release.yml` and environment `pypi-publish`. Set
`PUBLISH_TO_PYPI=true` in repository Actions variables after configuring it.
PyPI receives stable releases only. After publishing, verify file digests and
the installed version from PyPI. On a failed upload, check which files PyPI has
received before choosing a rerun. Release tags and PyPI filenames should remain
immutable after publication.

For hosted documentation, select GitHub Actions as the Pages source and set
`PUBLISH_DOCS=true`. The Pages workflow publishes the strict build on main.
For release changes, use a green CI run, files from the tagged commit, and a
separate check of the Pages URL.
