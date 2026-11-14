import json
from unittest.mock import patch

import pytest

from clickhouserag import Chunk, Context, KnowledgeQuery, Predicate
from clickhouserag.ollama import ModelResponseError, Ollama


def _chat(payload: dict) -> bytes:
    return json.dumps({"done": True, "message": {"content": json.dumps(payload)}}).encode()


def test_provider_implements_structured_requests_and_citations() -> None:
    schema = (Predicate("calls", "service", "service"),)
    adapter = Ollama(embedding_model="embed", chat_model="chat", dimensions=2, schema=schema)
    passage = Chunk("first", "doc", "Gateway calls Ledger.", "source")
    with patch("clickhouserag.ollama.http.client.HTTPConnection") as factory:
        connection = factory.return_value
        response = connection.getresponse.return_value
        response.status = 200
        response.read.side_effect = [
            b'{"embeddings":[[1,0]]}',
            _chat(
                {"relations": [{"subject": "Gateway", "predicate": "calls", "object": "Ledger"}]}
            ),
            _chat({"entity": "Gateway", "entity_type": "service", "predicates": ["calls"]}),
            _chat({"answer": "Ledger", "citations": ["1"]}),
        ]
        assert adapter.embed_query("query") == [1, 0]
        relations = adapter.extract(passage)
        assert [(fact.subject_type, fact.object_type) for fact in relations] == [
            ("service", "service")
        ]
        assert adapter.plan("Who does Gateway call?", schema) == KnowledgeQuery(
            "Gateway", "service", ("calls",)
        )
        answer = adapter.generate("Who does it call?", Context((passage,)))
        assert answer.text == "Ledger"
        assert answer.citation_ids == ["first"]
        methods = [(call.args[0], call.args[1]) for call in connection.request.call_args_list]
        assert methods == [("POST", "/api/embed"), *(("POST", "/api/chat"),) * 3]
        request = json.loads(connection.request.call_args.args[2])
        assert request["format"]["required"] == ["answer", "citations"]
        assert json.loads(request["messages"][1]["content"])["evidence"] == [
            {"id": "1", "text": passage.text}
        ]


def test_provider_rejects_unexpected_citations_and_http_failures() -> None:
    adapter = Ollama(embedding_model="embed", chat_model="chat", dimensions=2)
    context = Context((Chunk("first", "doc", "Evidence", "source"),))
    with patch("clickhouserag.ollama.http.client.HTTPConnection") as factory:
        response = factory.return_value.getresponse.return_value
        response.status = 200
        response.read.return_value = _chat({"answer": "unknown", "citations": ["88"]})
        with pytest.raises(ModelResponseError, match="unknown evidence"):
            adapter.generate("question", context)
        response.status = 503
        with pytest.raises(ModelResponseError, match="503"):
            adapter.embed_query("question")


def test_provider_rejects_embedding_count_mismatch() -> None:
    adapter = Ollama(embedding_model="embed", chat_model="chat", dimensions=2)
    with (
        patch.object(adapter, "_request", return_value={"embeddings": []}),
        pytest.raises(ModelResponseError, match="number of embeddings"),
    ):
        adapter.embed_query("question")


@pytest.mark.parametrize("url", ["file:///tmp", "http://user:pw@localhost", "http://localhost/api"])
def test_provider_requires_an_http_origin(url: str) -> None:
    with pytest.raises(ValueError, match="HTTP"):
        Ollama(embedding_model="embed", chat_model="chat", dimensions=2, url=url)
