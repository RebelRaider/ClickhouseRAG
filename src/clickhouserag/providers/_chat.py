"""Shared response parsing for structured text generation."""

from typing import Any

from clickhouserag._structured import ModelResponseError


def model_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelResponseError("Model did not return structured text")
    return value


def single_record(records: Any) -> dict[str, Any]:
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        raise ModelResponseError("Model did not return exactly one response choice")
    return records[0]


def text_blocks(blocks: Any, *, block_type: str = "text") -> str:
    if not isinstance(blocks, list) or any(not isinstance(block, dict) for block in blocks):
        raise ModelResponseError("Model returned invalid response blocks")
    return model_text(
        "".join(
            model_text(block.get("text")) for block in blocks if block.get("type") == block_type
        )
    )


def portable_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt internal task schemas to APIs that disallow array maxima."""
    result = dict(schema)
    maximum = result.pop("maxItems", None)
    if maximum is not None:
        description = result.get("description", "")
        result["description"] = f"{description} Return at most {maximum} items.".strip()
    kind = result.get("type")
    if isinstance(kind, list):
        result.pop("type")
        result["anyOf"] = [{"type": item} for item in kind]
    if "properties" in result:
        result["properties"] = {
            key: portable_schema(value) for key, value in result["properties"].items()
        }
    if "items" in result:
        result["items"] = portable_schema(result["items"])
    return result
