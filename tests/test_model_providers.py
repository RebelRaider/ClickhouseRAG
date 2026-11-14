import json
from typing import Any
from unittest.mock import patch

import pytest

from clickhouserag import Chunk, Context, KnowledgeQuery, Predicate
from clickhouserag.providers import (
    AnthropicMessages,
    CohereChat,
    CohereEmbeddings,
    GeminiChat,
    GeminiEmbeddings,
    ModelHTTPError,
    ModelResponseError,
    OpenAICompatibleChat,
    OpenAICompatibleEmbeddings,
    OpenAICompatibleResponses,
    StructuredLLM,
    VoyageEmbeddings,
)

_BASE_URL = "https://model.example/v1"
_SCHEMA = (Predicate("calls", "service", "service"),)
_PASSAGE = Chunk("s1", "document", "Gateway calls Ledger.", "runbook")
_CONTEXT = Context((_PASSAGE,))


def _indexed(*vectors: list[float]) -> dict:
    return {
        "data": [{"index": i, "embedding": row} for i, row in reversed(list(enumerate(vectors)))]
    }


def test_openai_compatible_batches_restore_order_and_keep_configuration_in_model_id() -> None:
    model = OpenAICompatibleEmbeddings(
        "embed",
        2,
        base_url=_BASE_URL,
        query_prefix="q: ",
        document_prefix="d: ",
        request_dimensions=True,
        batch_size=2,
        revision="v1",
    )
    with patch.object(
        model._endpoint,
        "post",
        side_effect=[_indexed([1, 0], [0, 1]), _indexed([1, 1]), _indexed([0, 1])],
    ) as post:
        assert model.embed_documents(["a", "b", "c"]) == [[1, 0], [0, 1], [1, 1]]
        assert model.embed_query("terms") == [0, 1]
        assert [call.args[1]["input"] for call in post.call_args_list] == [
            ["d: a", "d: b"],
            ["d: c"],
            ["q: terms"],
        ]
        assert all(call.args[1]["dimensions"] == 2 for call in post.call_args_list)
    assert model.model_id != OpenAICompatibleEmbeddings("embed", 2, base_url=_BASE_URL).model_id


@pytest.mark.parametrize("vector", [[0, 0], [float("inf"), 1], [True, 1], [1]])
def test_invalid_embedding_rows_fail_before_use(vector: list) -> None:
    model = OpenAICompatibleEmbeddings("embed", 2, base_url=_BASE_URL)
    with (
        patch.object(model._endpoint, "post", return_value=_indexed(vector)) as post,
        pytest.raises(ModelResponseError, match="embedding"),
    ):
        model.embed_query("question")
    assert "dimensions" not in post.call_args.args[1]


def test_duplicate_index_does_not_silently_reorder_passages() -> None:
    model = OpenAICompatibleEmbeddings("embed", 2, base_url=_BASE_URL)
    indexed = {"data": [{"index": 0, "embedding": [1, 2]}] * 2}
    with (
        patch.object(model._endpoint, "post", return_value=indexed),
        pytest.raises(ModelResponseError, match="repeated embedding index"),
    ):
        model.embed_documents(["one", "two"])


def test_native_embedding_dialects_distinguish_query_and_document() -> None:
    key = "fixture"
    gemini = GeminiEmbeddings("models/embed", 2, api_key=key, use_task_type=True)
    cohere = CohereEmbeddings("embed", 2, api_key=key, request_dimensions=True)
    voyage = VoyageEmbeddings("embed", 2, api_key=key, request_dimensions=True)
    with patch.object(
        gemini._endpoint, "post", return_value={"embeddings": [{"values": [0, 1]}]}
    ) as post:
        assert gemini.embed_documents(["source"]) == [[0, 1]]
        gemini.embed_query("question")
        assert post.call_args.args[0] == "/models/embed:batchEmbedContents"
        docs, query = (call.args[1]["requests"][0] for call in post.call_args_list)
        assert docs["taskType"] == "RETRIEVAL_DOCUMENT" and query["taskType"] == "RETRIEVAL_QUERY"
        assert query["outputDimensionality"] == 2 and query["model"] == "models/embed"
    with patch.object(
        cohere._endpoint, "post", return_value={"embeddings": {"float": [[1, 2]]}}
    ) as post:
        assert cohere.embed_documents(["source"]) == [[1, 2]]
        cohere.embed_query("question")
        docs, query = (call.args[1] for call in post.call_args_list)
        assert docs["input_type"] == "search_document" and query["input_type"] == "search_query"
        assert query["truncate"] == "NONE" and query["output_dimension"] == 2
    with patch.object(voyage._endpoint, "post", return_value=_indexed([1, 2])) as post:
        assert voyage.embed_documents(["source"]) == [[1, 2]]
        voyage.embed_query("question")
        docs, query = (call.args[1] for call in post.call_args_list)
        assert docs["input_type"] == "document" and query["input_type"] == "query"
        assert query["truncation"] is False and query["output_dimension"] == 2


def _reply(api: str, output: dict[str, Any]) -> dict:
    value = json.dumps(output)
    text = [{"type": "text", "text": value}]
    return {
        "chat": {"choices": [{"finish_reason": "stop", "message": {"content": value}}]},
        "responses": {
            "status": "completed",
            "output": [
                {"type": "reasoning"},
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": value}],
                },
            ],
        },
        "anthropic": {"stop_reason": "end_turn", "content": text},
        "gemini": {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "parts": [
                            {"text": "details", "thought": True},
                            {"text": value},
                        ]
                    },
                }
            ]
        },
        "cohere": {"finish_reason": "COMPLETE", "message": {"content": text}},
    }[api]


@pytest.mark.parametrize("api", ["chat", "responses", "anthropic", "gemini", "cohere"])
def test_each_chat_dialect_runs_all_three_shared_model_tasks(api: str) -> None:
    chat = {
        "chat": OpenAICompatibleChat("chat", base_url=_BASE_URL),
        "responses": OpenAICompatibleResponses("chat", base_url=_BASE_URL),
        "anthropic": AnthropicMessages("chat", api_key="fixture"),
        "gemini": GeminiChat("chat", api_key="fixture"),
        "cohere": CohereChat("chat", api_key="fixture"),
    }[api]
    model = StructuredLLM(chat, schema=_SCHEMA)
    outputs = [
        {"relations": [{"subject": "Gateway", "predicate": "calls", "object": "Ledger"}]},
        {"paths": [{"entity": "Gateway", "predicates": ["calls"]}]},
        {"answer": "Ledger", "citations": ["1"]},
    ]
    with patch.object(
        chat._endpoint, "post", side_effect=[_reply(api, output) for output in outputs]
    ) as post:
        assert model.extract(_PASSAGE)[0].object_type == "service"
        assert model.plan("What does Gateway call?", _SCHEMA) == KnowledgeQuery(
            "Gateway", "service", ("calls",)
        )
        assert model.generate("What does Gateway call?", _CONTEXT).citation_ids == ["s1"]
    path, body = post.call_args.args
    if api == "chat":
        assert (
            path == "/chat/completions" and body["response_format"]["json_schema"]["strict"] is True
        )
    elif api == "responses":
        assert path == "/responses" and body["store"] is False
    elif api == "anthropic":
        assert path == "/messages" and '"maxItems"' not in json.dumps(body["output_config"])
    elif api == "gemini":
        assert path == "/models/chat:generateContent"
        assert "responseJsonSchema" in body["generationConfig"]
    elif api == "cohere":
        assert path == "/chat" and '"maxItems"' not in json.dumps(body["response_format"])


def test_non_schema_json_mode_still_checks_citations() -> None:
    chat = OpenAICompatibleChat("chat", base_url=_BASE_URL, json_mode="object")
    model = StructuredLLM(chat)
    with patch.object(
        chat._endpoint,
        "post",
        return_value=_reply("chat", {"answer": "made up", "citations": ["2"]}),
    ) as post:
        with pytest.raises(ModelResponseError, match="unknown evidence"):
            model.generate("question", _CONTEXT)
        body = post.call_args.args[1]
        assert body["response_format"] == {"type": "json_object"}
        assert '"citations"' in body["messages"][0]["content"]


def test_http_response_error_has_status_but_no_payload_and_redirect_is_not_followed() -> None:
    chat = GeminiChat("chat", base_url=_BASE_URL, api_key="fixture")
    with patch("clickhouserag.providers._http.http.client.HTTPSConnection") as factory:
        connection = factory.return_value
        response = connection.getresponse.return_value
        response.status = 429
        response.getheader.return_value = "2"
        with pytest.raises(ModelHTTPError) as error:
            chat.complete_json("instructions", "input", {})
        assert error.value.status == 429 and error.value.retry_after == "2"
        assert connection.request.call_args.args[1] == "/v1/models/chat:generateContent"
        assert connection.request.call_args.args[3]["x-goog-api-key"] == "fixture"
        response.read.assert_not_called()
        response.status = 307
        with pytest.raises(ModelHTTPError, match="307"):
            chat.complete_json("instructions", "input", {})
        assert factory.call_count == 2
        response.read.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "http://model.example/v1",
        "https://user@model.example/v1",
        "https://model.example/v1?key=secret",
    ],
)
def test_authenticated_cloud_url_must_not_leak_keys(url: str) -> None:
    with pytest.raises(ValueError):
        OpenAICompatibleChat("chat", base_url=url, api_key="fixture")


def test_refusal_or_truncation_never_becomes_cited_answer() -> None:
    chat = OpenAICompatibleResponses("chat", base_url=_BASE_URL)
    payload = _reply("responses", {"answer": "answer", "citations": ["1"]})
    incomplete = {**payload, "status": "incomplete"}
    payload["output"][1]["content"] = [{"type": "refusal", "refusal": "declined"}]
    with patch.object(chat._endpoint, "post", side_effect=[incomplete, payload]):
        for _ in range(2):
            with pytest.raises(ModelResponseError, match="complete"):
                StructuredLLM(chat).generate("question", _CONTEXT)


def test_azure_v1_key_header_and_bearer_token_refresh() -> None:
    url = "https://resource.openai.azure.com/openai/v1/"
    refreshed = iter(["first-token", "second-token"])
    key_client = OpenAICompatibleEmbeddings(
        "deployment", 2, base_url=url, api_key="fixture", auth="api-key"
    )
    token_client = OpenAICompatibleEmbeddings(
        "deployment", 2, base_url=url, api_key=lambda: next(refreshed)
    )
    assert key_client.model_id == token_client.model_id
    with patch("clickhouserag.providers._http.http.client.HTTPSConnection") as factory:
        connection = factory.return_value
        response = connection.getresponse.return_value
        response.status = 200
        response.read.return_value = json.dumps(_indexed([1, 2])).encode()
        for client in (key_client, token_client, token_client):
            assert client.embed_query("query") == [1, 2]
        assert all(
            call.args[1] == "/openai/v1/embeddings" for call in connection.request.call_args_list
        )
        fixed, first, second = (call.args[3] for call in connection.request.call_args_list)
        assert fixed["api-key"] == "fixture" and "Authorization" not in fixed
        assert first["Authorization"] == "Bearer first-token"
        assert second["Authorization"] == "Bearer second-token"


def test_structured_search_plans_validate_the_wire_output() -> None:
    chat = OpenAICompatibleResponses("chat", base_url=_BASE_URL)
    model = StructuredLLM(chat)
    results = [
        {"searches": [{"text": "ERR-42", "mode": "lexical"}]},
        {"searches": [{"text": "rewrite", "mode": "graph"}]},
        {"searches": []},
    ]
    with patch.object(
        chat._endpoint, "post", side_effect=[_reply("responses", item) for item in results]
    ):
        (first,) = model.plan_search("Recover from ERR-42", ("lexical", "hybrid"))
        assert first.text == "ERR-42" and first.mode == "lexical" and first.seeds is None
        with pytest.raises(ModelResponseError, match="unavailable"):
            model.plan_search("question", ("lexical", "hybrid"))
        assert model.plan_search("question", ("lexical", "hybrid")) == ()
