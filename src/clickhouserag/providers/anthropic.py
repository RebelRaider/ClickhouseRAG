"""Structured text with the Claude Messages API."""

import re
from typing import Any

from clickhouserag._structured import ModelResponseError
from clickhouserag._validation import integer, nonblank
from clickhouserag.providers._chat import portable_schema, text_blocks
from clickhouserag.providers._http import JsonEndpoint


class AnthropicMessages:
    """Generate schema-constrained JSON using a compatible Claude model."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = "https://api.anthropic.com/v1",
        api_version: str = "2026-10-21",
        max_tokens: int = 4096,
        timeout: int = 120,
    ) -> None:
        nonblank(model, "model")
        nonblank(api_key, "api_key")
        integer(max_tokens, "max_tokens", 2**31 - 1)
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", api_version):
            raise ValueError("api_version must have the form YYYY-MM-DD")
        self._endpoint = JsonEndpoint(
            base_url, timeout=timeout, key=api_key, headers={"anthropic-version": api_version}
        )
        self._model = model
        self._max_tokens = max_tokens

    def complete_json(self, instructions: str, input_text: str, schema: dict[str, Any]) -> str:
        """Return text from a completed Claude turn."""
        response = self._endpoint.post(
            "/messages",
            {
                "model": self._model,
                "max_tokens": self._max_tokens,
                "system": instructions,
                "messages": [{"role": "user", "content": input_text}],
                "output_config": {
                    "format": {"type": "json_schema", "schema": portable_schema(schema)}
                },
            },
        )
        if response.get("stop_reason") != "end_turn":
            raise ModelResponseError("Claude did not complete the requested JSON response")
        return text_blocks(response.get("content"))
