"""Gemini model adapters for Google Cloud publisher endpoints."""

import re
from typing import Any

from clickhouserag._structured import ModelResponseError
from clickhouserag._validation import integer
from clickhouserag.providers._embeddings import TextEmbeddings
from clickhouserag.providers._http import Credential, JsonEndpoint
from clickhouserag.providers.gemini import _content, _generated_text, _resource


def _publisher(project: str, location: str, access_token: Credential, timeout: int) -> JsonEndpoint:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", project) or not re.fullmatch(
        r"[a-z0-9][a-z0-9-]*", location
    ):
        raise ValueError("project and location must be Google Cloud identifiers")
    host = (
        "aiplatform.googleapis.com"
        if location == "global"
        else f"{location}-aiplatform.googleapis.com"
    )
    return JsonEndpoint(
        f"https://{host}/v1/projects/{project}/locations/{location}/publishers/google",
        timeout=timeout,
        key=access_token,
    )


class VertexGeminiChat:
    """Use Google Cloud Gemini generateContent authenticated with an OAuth token."""

    def __init__(
        self,
        model: str,
        *,
        project: str,
        location: str,
        access_token: Credential,
        max_tokens: int = 4096,
        timeout: int = 120,
    ) -> None:
        integer(max_tokens, "max_tokens", 2**31 - 1)
        self._resource = _resource(model)
        self._endpoint = _publisher(project, location, access_token, timeout)
        self._max_tokens = max_tokens

    def complete_json(self, instructions: str, input_text: str, schema: dict[str, Any]) -> str:
        """Generate one structured response with a Cloud publisher model."""
        return _generated_text(
            self._endpoint.post(
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
        )


class VertexTextEmbeddings(TextEmbeddings):
    """Use an embedding publisher model with the Google Cloud predict API."""

    def __init__(
        self,
        model: str,
        dimensions: int,
        *,
        project: str,
        location: str,
        access_token: Credential,
        revision: str | None = None,
        batch_size: int = 1,
        timeout: int = 120,
    ) -> None:
        self._resource = _resource(model)
        self._endpoint = _publisher(project, location, access_token, timeout)
        super().__init__(
            "vertex",
            self._resource.removeprefix("models/"),
            dimensions,
            binding=(self._endpoint.url,),
            revision=revision,
            batch_size=batch_size,
            max_batch=5,
        )

    def _request_vectors(self, texts: list[str], *, query: bool) -> list[Any]:
        task = "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT"
        result = self._endpoint.post(
            f"/{self._resource}:predict",
            {
                "instances": [{"content": text, "task_type": task} for text in texts],
                "parameters": {"outputDimensionality": self.dimensions, "autoTruncate": False},
            },
        )
        rows = result.get("predictions")
        if not isinstance(rows, list):
            raise ModelResponseError("Google Cloud returned invalid embedding predictions")
        vectors = []
        for row in rows:
            embedding = row.get("embeddings") if isinstance(row, dict) else None
            if not isinstance(embedding, dict):
                raise ModelResponseError("Google Cloud returned an invalid embedding prediction")
            statistics = embedding.get("statistics")
            if isinstance(statistics, dict) and statistics.get("truncated"):
                raise ModelResponseError("Google Cloud truncated an embedding input")
            vectors.append(embedding.get("values"))
        return vectors
