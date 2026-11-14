"""Deterministic retrieval and citation evaluation with per-case timings."""

from collections.abc import Callable, Iterable, Sequence, Set
from dataclasses import dataclass
from math import ceil, log2
from statistics import mean
from time import perf_counter

from clickhouserag._validation import _MAX_LIMIT, integer, nonblank
from clickhouserag.rag import Answer, Context


@dataclass(frozen=True, slots=True)
class EvalCase:
    """Labelled query with relevant passage IDs and optional accepted exact answers."""

    case_id: str
    query: str
    relevant: frozenset[str]
    answers: tuple[str, ...] = ()
    category: str = "default"


@dataclass(frozen=True, slots=True)
class CaseScore:
    """Metrics and timing for one evaluation case."""

    case_id: str
    category: str
    retrieved: tuple[str, ...]
    recall: float | None
    reciprocal_rank: float | None
    ndcg: float | None
    exact_match: bool | None
    citation_precision: float | None
    citation_recall: float | None
    abstained: bool | None
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Macro metrics over applicable cases, keeping measurements per query."""

    k: int
    cases: tuple[CaseScore, ...]
    recall: float | None
    mrr: float | None
    ndcg: float | None
    exact_match: float | None
    citation_precision: float | None
    citation_recall: float | None
    abstention_accuracy: float | None
    latency_p95_ms: float


def evaluate(
    cases: Iterable[EvalCase],
    run: Callable[[EvalCase], Context | Answer],
    *,
    k: int = 10,
) -> Evaluation:
    """Run each labelled case once. Empty qrels label queries with no known answer."""
    integer(k, "k", _MAX_LIMIT)
    items = tuple(cases)
    if not items:
        raise ValueError("evaluation requires at least one case")
    if len({case.case_id for case in items}) != len(items):
        raise ValueError("evaluation case IDs must be unique")
    for case in items:
        nonblank(case.case_id, "case ID")
        nonblank(case.query, "query")
        if case.answers and not case.relevant:
            raise ValueError("an answerable case must have relevant passage IDs")
    scores = tuple(_run_case(case, run, k) for case in items)
    latencies = sorted(score.elapsed_ms for score in scores)
    abstentions = [
        score.abstained == (not case.relevant)
        for case, score in zip(items, scores, strict=True)
        if score.abstained is not None
    ]
    return Evaluation(
        k,
        scores,
        _macro(score.recall for score in scores),
        _macro(score.reciprocal_rank for score in scores),
        _macro(score.ndcg for score in scores),
        _macro(score.exact_match for score in scores),
        _macro(score.citation_precision for score in scores),
        _macro(score.citation_recall for score in scores),
        mean(abstentions) if abstentions else None,
        latencies[ceil(len(latencies) * 0.95) - 1],
    )


def _run_case(case: EvalCase, run: Callable[[EvalCase], Context | Answer], k: int) -> CaseScore:
    start = perf_counter()
    output = run(case)
    elapsed_ms = (perf_counter() - start) * 1000
    context = output.context if isinstance(output, Answer) else output
    ids = tuple(dict.fromkeys(chunk.chunk_id for chunk in context.passages))[:k]
    recall, rr, ndcg = _ranking(ids, case.relevant, k)
    exact_match = precision = citation_recall = abstained = None
    if isinstance(output, Answer):
        abstained = output.text is None
        if case.answers:
            exact_match = output.text is not None and _normalize(output.text) in {
                _normalize(answer) for answer in case.answers
            }
        citations = {chunk.chunk_id for chunk in output.citations}
        true_citations = len(citations & case.relevant)
        if citations or case.relevant:
            precision = true_citations / len(citations) if citations else 0.0
        if case.relevant:
            citation_recall = true_citations / len(case.relevant)
    return CaseScore(
        case.case_id,
        case.category,
        ids,
        recall,
        rr,
        ndcg,
        exact_match,
        precision,
        citation_recall,
        abstained,
        elapsed_ms,
    )


def _ranking(
    ids: Sequence[str], relevant: Set[str], k: int
) -> tuple[float | None, float | None, float | None]:
    if not relevant:
        return None, None, None
    ranks = [index for index, key in enumerate(ids, start=1) if key in relevant]
    dcg = sum(1 / log2(rank + 1) for rank in ranks)
    ideal = sum(1 / log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return len(ranks) / len(relevant), 1 / ranks[0] if ranks else 0.0, dcg / ideal


def _macro(scores: Iterable[float | bool | None]) -> float | None:
    values = [float(score) for score in scores if score is not None]
    return mean(values) if values else None


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())
