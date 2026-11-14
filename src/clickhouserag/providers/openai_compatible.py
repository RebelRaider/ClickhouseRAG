"""Embeddings and structured chat for servers with OpenAI-compatible APIs."""

import json
from typing import Any, Literal

from clickhouserag._structured import ModelResponseError
from clickhouserag._validation import integer, nonblank
from clickhouserag.providers._chat import model_text, single_record, text_blocks
from clickhouserag.providers._embeddings import TextEmbeddings, indexed_vectors
from clickhouserag.providers._http import JsonEndpoint


class OpenAICompatibleEmbeddings(TextEmbeddings):
    """Use the /embeddings protocol on a service such as vLLM or TEI."""

    def __init__(
        self,
        model: str,
        dimensions: int,
        *,
        base_url: str,
        api_key: str | None = None,
        revision: str | None = None,
        request_dimensions: bool = False,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 64,
        timeout: int = 120,
    ) -> None:
        self._endpoint = JsonEndpoint(base_url, timeout=timeout, key=api_key)
        super().__init__(
            "openai-compatible",
            model,
            dimensions,
            binding=(self._endpoint.url, request_dimensions, query_prefix, document_prefix),
            revision=revision,
            batch_size=batch_size,
            max_batch=2048,
        )
        self._request_dimensions = request_dimensions
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix

    def _request_vectors(self, texts: list[str], *, query: bool) -> list[Any]:
        prefix = self._query_prefix if query else self._document_prefix
        body = {
            "model": self.model,
            "input": [prefix + text for text in texts],
            "encoding_format": "float",
        }
        if self._request_dimensions:
            body["dimensions"] = self.dimensions
        result = self._endpoint.post("/embeddings", body)
        return indexed_vectors(result.get("data"), len(texts))


class OpenAICompatibleChat:
    """Use Chat Completions with JSON Schema or explicit JSON mode."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None = None,
        json_mode: Literal["schema", "object"] = "schema",
        max_tokens: int = 4096,
        token_parameter: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens",
        timeout: int = 120,
    ) -> None:
        nonblank(model, "model")
        integer(max_tokens, "max_tokens", 2**31 - 1)
        if json_mode not in ("schema", "object"):
            raise ValueError("json_mode must be schema or object")
        if token_parameter not in ("max_completion_tokens", "max_tokens"):
            raise ValueError("unsupported completion token parameter")
        self._endpoint = JsonEndpoint(base_url, timeout=timeout, key=api_key)
        self._model = model
        self._json_mode = json_mode
        self._token_parameter = token_parameter
        self._max_tokens = max_tokens

    def complete_json(self, instructions: str, input_text: str, schema: dict[str, Any]) -> str:
        """Return the JSON text of one completed, non-refusal message."""
        if self._json_mode == "object":
            response_format = {"type": "json_object"}
            instructions += f"\nReturn a JSON object matching this schema: {json.dumps(schema)}"
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "rag_result", "strict": True, "schema": schema},
            }
        response = self._endpoint.post(
            "/chat/completions",
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": input_text},
                ],
                "response_format": response_format,
                self._token_parameter: self._max_tokens,
                "stream": False,
            },
        )
        choice = single_record(response.get("choices"))
        message = choice.get("message")
        if (
            choice.get("finish_reason") != "stop"
            or not isinstance(message, dict)
            or message.get("refusal")
        ):
            raise ModelResponseError("Model did not complete the requested JSON response")
        return model_text(message.get("content"))


class OpenAICompatibleResponses:
    """Use a Responses endpoint with a JSON Schema text format and store disabled."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None = None,
        max_tokens: int = 4096,
        timeout: int = 120,
    ) -> None:
        nonblank(model, "model")
        integer(max_tokens, "max_tokens", 2**31 - 1)
        self._endpoint = JsonEndpoint(base_url, timeout=timeout, key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def complete_json(self, instructions: str, input_text: str, schema: dict[str, Any]) -> str:
        """Read completed assistant text, skipping non-message output items."""
        response = self._endpoint.post(
            "/responses",
            {
                "model": self._model,
                "instructions": instructions,
                "input": input_text,
                "max_output_tokens": self._max_tokens,
                "store": False,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "rag_result",
                        "strict": True,
                        "schema": schema,
                    }
                },
            },
        )
        output = response.get("output")
        if response.get("status") != "completed" or not isinstance(output, list):
            raise ModelResponseError("Model did not complete the requested JSON response")
        message = single_record(
            [item for item in output if isinstance(item, dict) and item.get("type") == "message"]
        )
        blocks = message.get("content")
        if (
            message.get("role") != "assistant"
            or message.get("status") != "completed"
            or not isinstance(blocks, list)
            or any(
                not isinstance(block, dict) or block.get("type") == "refusal" for block in blocks
            )
        ):
            raise ModelResponseError("Model did not return a completed assistant message")
        return text_blocks(blocks, block_type="output_text")
