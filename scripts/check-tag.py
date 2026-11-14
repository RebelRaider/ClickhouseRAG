"""Validate that a published tag matches the Python distribution version."""

import re
import sys
import tomllib
from pathlib import Path

from packaging.version import Version


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python scripts/check-tag.py vMAJOR.MINOR.PATCH[-alpha.N|-beta.N|-rc.N]"
        )
    tag = sys.argv[1]
    number = r"(?:0|[1-9][0-9]*)"
    if not re.fullmatch(rf"v{number}\.{number}\.{number}(?:-(?:alpha|beta|rc)\.[1-9][0-9]*)?", tag):
        raise SystemExit("invalid release tag")
    version = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    if Version(tag[1:]) != Version(version):
        raise SystemExit(f"tag {tag} does not match distribution version {version}")


if __name__ == "__main__":
    main()
