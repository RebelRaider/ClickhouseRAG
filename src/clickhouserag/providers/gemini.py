"""Gemini Developer API for embeddings and structured text."""

import re
from typing import Any

from clickhouserag._structured import ModelResponseError
from clickhouserag._validation import integer, nonblank
from clickhouserag.providers._chat import model_text, single_record
from clickhouserag.providers._embeddings import TextEmbeddings
from clickhouserag.providers._http import JsonEndpoint

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _resource(model: str) -> str:
    name = model.removeprefix("models/")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("model must be a Gemini model name")
    return f"models/{name}"


def _content(text: str) -> dict[str, list[dict[str, str]]]:
    return {"parts": [{"text": text}]}


def _endpoint(base_url: str, api_key: str, timeout: int) -> JsonEndpoint:
    nonblank(api_key, "api_key")
    return JsonEndpoint(
        base_url, timeout=timeout, key=api_key, key_header="x-goog-api-key", key_prefix=""
    )


class GeminiEmbeddings(TextEmbeddings):
    """Use Gemini text embeddings with optional asymmetric retrieval settings."""

    def __init__(
        self,
        model: str,
        dimensions: int,
        *,
        api_key: str,
        base_url: str = _BASE_URL,
        revision: str | None = None,
        use_task_type: bool = False,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 32,
        timeout: int = 120,
    ) -> None:
        self._resource = _resource(model)
        self._endpoint = _endpoint(base_url, api_key, timeout)
        super().__init__(
            "gemini",
            self._resource.removeprefix("models/"),
            dimensions,
            binding=(self._endpoint.url, use_task_type, query_prefix, document_prefix),
            revision=revision,
            batch_size=batch_size,
            max_batch=100,
        )
        self._use_task_type = use_task_type
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix

    def _request_vectors(self, texts: list[str], *, query: bool) -> list[Any]:
        prefix = self._query_prefix if query else self._document_prefix
        requests = [
            {
                "model": self._resource,
                "content": _content(prefix + text),
                "outputDimensionality": self.dimensions,
            }
            for text in texts
        ]
        if self._use_task_type:
            for request in requests:
                request["taskType"] = "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT"
        result = self._endpoint.post(
            f"/{self._resource}:batchEmbedContents", {"requests": requests}
        )
        rows = result.get("embeddings")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ModelResponseError("Gemini returned invalid embeddings")
        return [row.get("values") for row in rows]


class GeminiChat:
    """Use Gemini generateContent with native JSON Schema output."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = _BASE_URL,
        max_tokens: int = 4096,
        timeout: int = 120,
    ) -> None:
        integer(max_tokens, "max_tokens", 2**31 - 1)
        self._resource = _resource(model)
        self._endpoint = _endpoint(base_url, api_key, timeout)
        self._max_tokens = max_tokens

    def complete_json(self, instructions: str, input_text: str, schema: dict[str, Any]) -> str:
        """Return output text from a completed Gemini candidate."""
        result = self._endpoint.post(
            f"/{self._resource}:generateContent",
            {
                "systemInstruction": _content(instructions),
                "contents": [{"role": "user", **_content(input_text)}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseJsonSchema": schema,
                    "maxOutputTokens": self._max_tokens,
                },
            },
        )
        candidate = single_record(result.get("candidates"))
        content = candidate.get("content")
        if candidate.get("finishReason") != "STOP" or not isinstance(content, dict):
            raise ModelResponseError("Gemini did not complete the requested JSON response")
        parts = content.get("parts")
        if not isinstance(parts, list) or any(not isinstance(part, dict) for part in parts):
            raise ModelResponseError("Gemini returned invalid content parts")
        return model_text(
            "".join(model_text(part.get("text")) for part in parts if not part.get("thought"))
        )
