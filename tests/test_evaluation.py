from math import log2

import pytest

from clickhouserag import Answer, Chunk, Context, EvalCase, evaluate


def test_evaluation_reports_ranking_citations_and_abstention() -> None:
    relevant = Chunk("hit", "doc", "Evidence", "source")
    irrelevant = Chunk("miss", "doc", "Noise", "noise")
    cases = [
        EvalCase("1", "question", frozenset({"hit", "other"}), ("Vault",), "multi-hop"),
        EvalCase("2", "absent", frozenset(), category="unanswerable"),
    ]

    def run(case: EvalCase) -> Answer:
        if case.case_id == "1":
            return Answer(
                " VAULT ", (relevant, irrelevant), Context((irrelevant, relevant, relevant))
            )
        return Answer(None, (), Context(()))

    result = evaluate(cases, run, k=2)
    assert result.recall == 0.5
    assert result.mrr == 0.5
    assert result.ndcg == pytest.approx((1 / log2(3)) / (1 + 1 / log2(3)))
    assert result.exact_match == 1
    assert result.citation_precision == 0.5
    assert result.citation_recall == 0.5
    assert result.abstention_accuracy == 1
    assert result.cases[0].retrieved == ("miss", "hit")
    assert result.cases[1].recall is None
    assert result.latency_p95_ms >= 0
    assert result.by_category["multi-hop"].recall == 0.5
    assert result.by_category["unanswerable"].recall is None
    assert result.by_category["unanswerable"].abstention_accuracy == 1


def test_evaluation_invalid_cases_rejected_before_run() -> None:
    case = EvalCase("same", "question", frozenset())

    def run(_: EvalCase) -> Context:
        pytest.fail("invalid input must not execute the pipeline")

    with pytest.raises(ValueError, match="unique"):
        evaluate([case, case], run)
    with pytest.raises(ValueError, match="at least one"):
        evaluate([], run)
