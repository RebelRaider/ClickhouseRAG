from collections.abc import Sequence
from unittest.mock import Mock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from clickhouserag import Chunk, ClickHouseStore, Retriever, SearchHit
from clickhouserag._lexical import terms
from clickhouserag._ranking import reciprocal_rank_fusion


class ToyProvider:
    model_id = "toy"

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[1, 0]] * len(texts)

    def embed_query(self, text: str) -> Sequence[float]:
        return [1, 0]


def test_provider_routes_passages_and_query() -> None:
    store = Mock(spec=ClickHouseStore, embedding_model="toy")
    store.insert.return_value = 1
    retriever = Retriever(store, ToyProvider())
    chunk = Chunk("1", "doc", "Text", "source")
    assert retriever.index([chunk], namespace="collection") == 1
    inserted = next(store.insert.call_args.args[0])
    assert inserted.chunk == chunk
    assert inserted.embedding == [1, 0]
    assert store.insert.call_args.kwargs == {"namespace": "collection"}

    retriever.search("query", namespace="collection", metadata={"lang": "ru"}, limit=4)
    store.search_hybrid.assert_called_once_with(
        "query", [1, 0], namespace="collection", metadata={"lang": "ru"}, limit=4
    )


def test_provider_rejects_model_mismatch() -> None:
    store = Mock(spec=ClickHouseStore, embedding_model="other")
    with pytest.raises(ValueError, match="model_id"):
        Retriever(store, ToyProvider())


def test_wrong_vector_count_leaves_table_unmodified() -> None:
    store = Mock(spec=ClickHouseStore, embedding_model="toy")
    provider = Mock(spec=ToyProvider)
    provider.model_id = "toy"
    provider.embed_documents.return_value = []
    retriever = Retriever(store, provider)
    with pytest.raises(ValueError, match="number"):
        retriever.index([Chunk("1", "doc", "Text", "source")])
    store.insert.assert_not_called()


def test_unicode_normalization() -> None:
    assert terms("Straße STRASSE; \uff21BC Альфа; RFC_9110") == [
        "strasse",
        "abc",
        "альфа",
        "rfc",
        "9110",
    ]


@given(st.text())
def test_tokenization_is_stable_on_normalized_tokens(text: str) -> None:
    output = terms(text)
    assert len(output) == len(set(output))
    assert terms(" ".join(output)) == output


def test_fusion_uses_ranks_and_deterministic_ties() -> None:
    hits = [SearchHit(Chunk(key, "doc", key, "source"), 999.0) for key in "abc"]
    a, b, c = hits
    fused = reciprocal_rank_fusion([a, c], [b, c], 3)
    assert [hit.chunk.chunk_id for hit in fused] == ["c", "a", "b"]
    assert fused[0].score == pytest.approx(2 / 62)
