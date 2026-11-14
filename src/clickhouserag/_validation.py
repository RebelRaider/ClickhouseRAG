"""Validation at the database boundary."""

import math
import re
from collections.abc import Mapping, Sequence

from clickhouserag.models import Chunk

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]{0,126}")
_MAX_DIMENSIONS = 4096
_MAX_LIMIT = 1000
_MAX_BATCH = 10_000


def identifier(value: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError("database and table must be simple ClickHouse identifiers")
    return f"`{value}`"


def integer(value: int, name: str, maximum: int) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")


def nonblank(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    _unicode_string(value, name)


def _unicode_string(value: str, name: str) -> None:
    if "\x00" in value:
        raise ValueError(f"{name} must not contain NUL")
    try:
        value.encode()
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be valid Unicode") from exc


def string_map(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("metadata must map nonblank string keys to string values")
    result = dict(value)
    for key, item in result.items():
        nonblank(key, "metadata key")
        if not isinstance(item, str):
            raise ValueError("metadata values must be strings")
        # Empty values are supported and must differ from keys that are absent.
        _unicode_string(item, "metadata value")
    return result


def chunk_fields(chunk: Chunk) -> dict[str, str]:
    nonblank(chunk.chunk_id, "chunk_id")
    nonblank(chunk.document_id, "document_id")
    nonblank(chunk.text, "text")
    nonblank(chunk.source, "source")
    integer(chunk.revision, "revision", 2**64 - 1)
    return string_map(chunk.metadata)


def unit_vector(values: Sequence[float], dimensions: int) -> list[float]:
    if len(values) != dimensions or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"embedding must contain {dimensions} real numbers")
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("embedding must contain finite real numbers") from exc
    if any(isinstance(value, (bool, str, bytes, bytearray)) for value in values) or not all(
        math.isfinite(value) for value in vector
    ):
        raise ValueError("embedding must contain finite real numbers")
    scale = max(map(abs, vector))
    if scale == 0:
        raise ValueError("embedding must have nonzero length")
    scaled = [value / scale for value in vector]
    norm = math.sqrt(math.fsum(value * value for value in scaled))
    return [value / norm for value in scaled]
