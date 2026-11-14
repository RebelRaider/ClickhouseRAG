"""Deterministic retrieval and citation evaluation with per-case timings."""

from collections import defaultdict
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
class Metrics:
    """Macro metrics over evaluated cases for one retrieval or answer run."""

    recall: float | None
    mrr: float | None
    ndcg: float | None
    exact_match: float | None
    citation_precision: float | None
    citation_recall: float | None
    abstention_accuracy: float | None
    latency_p95_ms: float


@dataclass(frozen=True, slots=True)
class Evaluation(Metrics):
    """Overall and category-level metrics, with individual case measurements."""

    k: int
    cases: tuple[CaseScore, ...]
    by_category: dict[str, Metrics]


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
        nonblank(case.category, "category")
        nonblank(case.query, "query")
        if case.answers and not case.relevant:
            raise ValueError("an answerable case must have relevant passage IDs")
    scores = tuple(_run_case(case, run, k) for case in items)
    pairs = list(zip(items, scores, strict=True))
    grouped: defaultdict[str, list[tuple[EvalCase, CaseScore]]] = defaultdict(list)
    for item, score in pairs:
        grouped[item.category].append((item, score))
    aggregate = _aggregate(pairs)
    return Evaluation(
        aggregate.recall,
        aggregate.mrr,
        aggregate.ndcg,
        aggregate.exact_match,
        aggregate.citation_precision,
        aggregate.citation_recall,
        aggregate.abstention_accuracy,
        aggregate.latency_p95_ms,
        k,
        scores,
        {name: _aggregate(group) for name, group in sorted(grouped.items())},
    )


def _aggregate(pairs: list[tuple[EvalCase, CaseScore]]) -> Metrics:
    latencies = sorted(score.elapsed_ms for _, score in pairs)
    abstentions = [
        score.abstained == (not case.relevant)
        for case, score in pairs
        if score.abstained is not None
    ]
    return Metrics(
        _macro(score.recall for _, score in pairs),
        _macro(score.reciprocal_rank for _, score in pairs),
        _macro(score.ndcg for _, score in pairs),
        _macro(score.exact_match for _, score in pairs),
        _macro(score.citation_precision for _, score in pairs),
        _macro(score.citation_recall for _, score in pairs),
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
