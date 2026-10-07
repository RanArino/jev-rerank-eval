"""Ranking metrics over quote-based gold labels. Standard library only.

A gold label is an exact quote in one document with a grade (3 answers, 2 supports,
1 on topic only). A candidate chunk covers a gold passage when the chunks being scored
hold at least half of the passage's characters, because a sentence may straddle two
chunks. A candidate's grade is the highest grade among passages it covers on its own.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

COVERAGE = 0.5
RELEVANT_GRADE = 2
NDCG_K = 10


@dataclass(frozen=True)
class Span:
    document_id: str
    start: int
    end: int


@dataclass(frozen=True)
class Gold:
    document_id: str
    start: int
    end: int
    grade: int


def resolve_gold(query: dict, documents: dict[str, str]) -> tuple[Gold, ...]:
    """Turn each quote into its character span; a quote must occur exactly once."""
    gold = []
    for label in query["gold"]:
        text = documents[label["document_id"]]
        start = text.find(label["quote"])
        if start < 0 or text.find(label["quote"], start + 1) >= 0:
            raise ValueError(f"{query['id']}: a quote must occur exactly once")
        gold.append(Gold(label["document_id"], start, start + len(label["quote"]), label["grade"]))
    return tuple(gold)


def covered(gold: Gold, spans: Iterable[Span]) -> bool:
    inside = set()
    for span in spans:
        if span.document_id == gold.document_id:
            inside.update(range(max(span.start, gold.start), min(span.end, gold.end)))
    return len(inside) >= COVERAGE * (gold.end - gold.start)


def grade(span: Span, gold: Sequence[Gold]) -> int:
    return max((g.grade for g in gold if covered(g, [span])), default=0)


def dcg(grades: Sequence[int]) -> float:
    return sum((2**g - 1) / math.log2(rank + 1) for rank, g in enumerate(grades, 1))


def ndcg(ranked: Sequence[int], pool: Sequence[int], k: int = NDCG_K) -> float | None:
    """IDCG comes from the whole labelled pool, so a shorter list is judged fairly."""
    ideal = dcg(sorted(pool, reverse=True)[:k])
    return None if ideal == 0 else dcg(ranked[:k]) / ideal


def pooled_relevant(pool: Sequence[Span], gold: Sequence[Gold]) -> tuple[Gold, ...]:
    """The fixed recall denominator: grade >= 2 passages the pool can reach."""
    return tuple(g for g in gold if g.grade >= RELEVANT_GRADE and covered(g, pool))


def recall(spans: Sequence[Span], relevant: Sequence[Gold]) -> float | None:
    if not relevant:
        return None
    return sum(covered(g, spans) for g in relevant) / len(relevant)


def paired_bootstrap(
    deltas: Sequence[float], resamples: int = 10_000, seed: int = 20261006
) -> tuple[float, float] | None:
    """A 95% interval for the mean per-query difference, resampling queries in pairs."""
    if not deltas:
        return None
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[min(resamples - 1, int(0.975 * resamples))]
