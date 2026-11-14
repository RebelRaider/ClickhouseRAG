"""Structured generation and Titan V2 embeddings through Amazon Bedrock Runtime."""

import json
from collections.abc import Callable
from typing import Any

from clickhouserag._structured import ModelResponseError, _json_object
from clickhouserag._validation import integer, nonblank
from clickhouserag.providers._chat import model_text, portable_schema
from clickhouserag.providers._embeddings import TextEmbeddings

_MAX_BODY = 1024 * 1024


class BedrockConverse:
    """Use the converse method of an application-authenticated Bedrock Runtime client."""

    def __init__(self, model: str, *, converse: Callable[..., Any], max_tokens: int = 4096) -> None:
        nonblank(model, "model")
        integer(max_tokens, "max_tokens", 2**31 - 1)
        self._model = model
        self._converse = converse
        self._max_tokens = max_tokens

    def complete_json(self, instructions: str, input_text: str, schema: dict[str, Any]) -> str:
        """Read completed assistant output from a schema-capable Bedrock model."""
        response = self._converse(
            modelId=self._model,
            system=[{"text": instructions}],
            messages=[{"role": "user", "content": [{"text": input_text}]}],
            inferenceConfig={"maxTokens": self._max_tokens},
            outputConfig={
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {
                            "name": "rag_result",
                            "schema": json.dumps(portable_schema(schema)),
                        }
                    },
                }
            },
        )
        output = response.get("output")
        message = output.get("message") if isinstance(output, dict) else None
        if (
            response.get("stopReason") != "end_turn"
            or not isinstance(message, dict)
            or message.get("role") != "assistant"
        ):
            raise ModelResponseError("Bedrock did not complete the requested JSON response")
        blocks = message.get("content")
        if not isinstance(blocks, list) or any(not isinstance(block, dict) for block in blocks):
            raise ModelResponseError("Bedrock returned invalid message content")
        return model_text("".join(model_text(block["text"]) for block in blocks if "text" in block))


class BedrockTitanEmbeddings(TextEmbeddings):
    """Use Titan Text Embeddings V2 via the Bedrock Runtime invoke_model method."""

    def __init__(
        self,
        dimensions: int,
        *,
        invoke_model: Callable[..., Any],
        deployment: str,
        model: str = "amazon.titan-embed-text-v2:0",
        revision: str | None = None,
    ) -> None:
        nonblank(deployment, "deployment")
        if dimensions not in (256, 512, 1024):
            raise ValueError("Titan V2 embedding dimensions must be 256, 512, or 1024")
        super().__init__(
            "bedrock-titan",
            model,
            dimensions,
            binding=(deployment,),
            revision=revision,
            batch_size=1,
            max_batch=1,
        )
        self._invoke_model = invoke_model

    def _request_vectors(self, texts: list[str], *, query: bool) -> list[Any]:
        result = self._invoke_model(
            modelId=self.model,
            body=json.dumps(
                {"inputText": texts[0], "dimensions": self.dimensions, "normalize": True}
            ),
            contentType="application/json",
            accept="application/json",
        )
        body = result.get("body")
        if body is None:
            raise ModelResponseError("Bedrock did not return an embedding body")
        try:
            payload = body.read(_MAX_BODY + 1)
        finally:
            body.close()
        if len(payload) > _MAX_BODY:
            raise ModelResponseError("Bedrock embedding response exceeded the size limit")
        return [_json_object(payload).get("embedding")]
