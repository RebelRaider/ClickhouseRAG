import json
from io import BytesIO
from unittest.mock import Mock, patch

import pytest

from clickhouserag import Chunk, Context
from clickhouserag.providers import (
    BedrockConverse,
    BedrockTitanEmbeddings,
    ModelResponseError,
    StructuredLLM,
    VertexGeminiChat,
    VertexTextEmbeddings,
)

_CONTEXT = Context((Chunk("evidence", "doc", "Ledger writes Vault.", "runbook"),))


def test_bedrock_converse_and_titan_v2() -> None:
    value = '{"answer":"Vault","citations":["1"]}'
    converse = Mock(
        return_value={
            "stopReason": "end_turn",
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"reasoningContent": {}}, {"text": value}],
                }
            },
        }
    )
    provider = StructuredLLM(BedrockConverse("foundation-model", converse=converse))
    assert provider.generate("Where does Ledger write?", _CONTEXT).citation_ids == ["evidence"]
    request = converse.call_args.kwargs
    assert request["modelId"] == "foundation-model"
    schema = json.loads(request["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"])
    assert schema["properties"]["answer"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
    assert "maxItems" not in schema["properties"]["citations"]
    reply = converse.return_value
    converse.return_value = {**reply, "stopReason": "max_tokens"}
    with pytest.raises(ModelResponseError, match="did not complete"):
        provider.generate("question", _CONTEXT)

    body = BytesIO(json.dumps({"embedding": [1] + [0] * 255}).encode())
    invoke = Mock(return_value={"body": body})
    encoder = BedrockTitanEmbeddings(256, invoke_model=invoke, deployment="aws/us-west-2")
    assert len(encoder.embed_query("Vault")) == 256
    assert body.closed
    request = invoke.call_args.kwargs
    assert request["modelId"] == "amazon.titan-embed-text-v2:0"
    assert json.loads(request["body"]) == {
        "inputText": "Vault",
        "dimensions": 256,
        "normalize": True,
    }


def test_vertex_generation_and_prediction_payloads() -> None:
    chat = VertexGeminiChat(
        "chat", project="example-project", location="global", access_token=lambda: "fixture"
    )
    encoder = VertexTextEmbeddings(
        "gemini-embedding-001",
        2,
        project="example-project",
        location="us-central1",
        access_token="fixture",
    )
    text = '{"answer":"Vault","citations":["1"]}'
    reply = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": text}]}}]}
    with patch("clickhouserag.providers._http.http.client.HTTPSConnection") as factory:
        conn = factory.return_value
        response = conn.getresponse.return_value
        response.status = 200
        response.read.side_effect = [
            json.dumps(reply).encode(),
            b'{"predictions":[{"embeddings":{"values":[0,1],"statistics":{"truncated":false}}}]}',
            b'{"predictions":[{"embeddings":{"values":[1,0],"statistics":{"truncated":true}}}]}',
        ]
        assert StructuredLLM(chat).generate("question", _CONTEXT).text == "Vault"
        assert encoder.embed_documents(["text"]) == [[0, 1]]
        with pytest.raises(ModelResponseError, match="truncated"):
            encoder.embed_query("query")
        assert [call.args[0] for call in factory.call_args_list] == [
            "aiplatform.googleapis.com",
            "us-central1-aiplatform.googleapis.com",
            "us-central1-aiplatform.googleapis.com",
        ]
        request, doc, query = [call.args for call in conn.request.call_args_list]
        assert (
            request[1] == "/v1/projects/example-project/locations/global/publishers/google"
            "/models/chat:generateContent"
        )
        assert request[3]["Authorization"] == "Bearer fixture"
        assert doc[1].endswith("/models/gemini-embedding-001:predict")
        assert json.loads(doc[2])["instances"] == [
            {"content": "text", "task_type": "RETRIEVAL_DOCUMENT"}
        ]
        assert json.loads(query[2])["instances"] == [
            {"content": "query", "task_type": "RETRIEVAL_QUERY"}
        ]
