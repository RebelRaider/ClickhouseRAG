import json
from unittest.mock import patch

import pytest

from clickhouserag import Chunk
from clickhouserag.providers import CohereReranker, ModelResponseError, TEIReranker, VoyageReranker

_PASSAGES = (Chunk("a", "p", "First", "ref:a"), Chunk("b", "p", "Second", "ref:b"))


def test_reranker_dialects_and_tie_breaking() -> None:
    cohere = CohereReranker("rerank", base_url="https://endpoint.example/v2")
    voyage = VoyageReranker("rerank", api_key="fixture")
    tei = TEIReranker("http://localhost:8080")
    scores = [{"index": 1, "relevance_score": 0.5}, {"index": 0, "relevance_score": 0.5}]
    with patch.object(cohere._endpoint, "post", return_value={"results": scores}) as post:
        assert cohere.rank("query", _PASSAGES) == [0, 1]
        assert post.call_args.args[1]["top_n"] == 2
    with patch.object(voyage._endpoint, "post", return_value={"data": scores}) as post:
        assert voyage.rank("query", _PASSAGES) == [0, 1]
        assert post.call_args.args[1]["truncation"] is False
    with patch("clickhouserag.providers._http.http.client.HTTPConnection") as factory:
        connection = factory.return_value
        response = connection.getresponse.return_value
        response.status = 200
        response.read.return_value = b'[{"index":0,"score":0.2},{"index":1,"score":0.8}]'
        assert tei.rank("query", _PASSAGES) == [1, 0]
        _, path, payload, _ = connection.request.call_args.args
        assert path == "/rerank" and json.loads(payload)["texts"] == ["First", "Second"]


def test_duplicate_reranker_indices_are_rejected() -> None:
    model = CohereReranker("rerank")
    result = {"results": [{"index": 0, "relevance_score": 0.5}] * 2}
    with (
        patch.object(model._endpoint, "post", return_value=result),
        pytest.raises(ModelResponseError, match="repeated index"),
    ):
        model.rank("query", _PASSAGES)
