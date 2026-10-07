"""Score a reranker's answers against the original vector order on the frozen pools.

    python -m rerank_eval.evaluate results/jev-1.13.0.json

Both orders run the same context rules, so the only difference is the order:

- original: the frozen cosine order;
- reranked: usefulness probability, highest first (a Boolean answer counts as 1 or 0),
  ties broken by the original rank;
- context seeds: walk the order, skip duplicate bodies and anything under the relevance
  floor, and take at most 12 chunks, at most 4 per document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rerank_eval.metrics import (
    Gold,
    Span,
    grade,
    ndcg,
    paired_bootstrap,
    pooled_relevant,
    recall,
    resolve_gold,
)

ROOT = Path(__file__).resolve().parents[2]
MAX_CHUNKS = 12
MAX_PER_DOCUMENT = 4
TOP_K_PREFIXES = (20, 40, 60, 100, 150)
FLOORS = tuple(round(step * 0.05, 2) for step in range(21))
GATES = {"mean_improvement": 0.03, "category_regression": 0.02}


def load(root: Path = ROOT) -> tuple[dict, list[dict], dict]:
    corpus = json.loads((root / "data/corpus.json").read_text("utf-8"))
    queries = json.loads((root / "data/queries.json").read_text("utf-8"))["queries"]
    pools = json.loads((root / "data/pools.json").read_text("utf-8"))
    return corpus, queries, pools


def score_of(answer: Mapping[str, Any] | None) -> float | None:
    if answer is None:
        return None
    if answer.get("probability") is not None:
        return float(answer["probability"])
    return 1.0 if answer.get("value") else 0.0


def reranked(refs: Sequence[str], answers: Mapping[str, Any]) -> list[str]:
    """Answered candidates by score, then unanswered ones, each in original order."""

    def key(item: tuple[int, str]) -> tuple[int, float, int]:
        rank, ref = item
        value = score_of(answers.get(ref))
        return (1, 0.0, rank) if value is None else (0, -value, rank)

    return [ref for _, ref in sorted(enumerate(refs), key=key)]


def seeds(
    order: Sequence[str],
    chunks: Mapping[str, dict],
    answers: Mapping[str, Any] | None = None,
    floor: float = 0.0,
) -> list[str]:
    taken: list[str] = []
    per_document: dict[str, int] = {}
    bodies: set[str] = set()
    for ref in order:
        chunk = chunks[ref]
        value = score_of(answers.get(ref)) if answers is not None else None
        if answers is not None and value is not None and floor > 0 and value < floor:
            continue
        body = hashlib.sha256(chunk["body"].encode("utf-8")).hexdigest()
        if body in bodies:
            continue
        if len(taken) >= MAX_CHUNKS or per_document.get(chunk["document_id"], 0) >= MAX_PER_DOCUMENT:
            continue
        taken.append(ref)
        bodies.add(body)
        per_document[chunk["document_id"]] = per_document.get(chunk["document_id"], 0) + 1
    return taken


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def evaluate(
    results: Mapping[str, Any], root: Path = ROOT, top_k: int | None = None
) -> dict[str, Any]:
    """Score a result file; ``top_k`` cuts every query to that original-order prefix, so
    implementations measured on different TopK compare on the same candidates."""
    corpus, queries, pools = load(root)
    documents = {d["id"]: d["text"] for d in corpus["documents"]}
    chunks = {c["ref"]: c for c in pools["chunks"]}
    answers_by_query = results["answers"]

    def span(ref: str) -> Span:
        c = chunks[ref]
        return Span(c["document_id"], c["start"], c["end"])

    per_query: dict[str, dict[str, Any]] = {}
    prefix_rows: dict[int, dict[str, list[float]]] = {
        k: {"recall": [], "ndcg": []} for k in TOP_K_PREFIXES
    }
    floor_rows: dict[float, dict[str, list[float]]] = {
        f: {"recall": [], "precision": []} for f in FLOORS
    }
    for query in queries:
        gold: tuple[Gold, ...] = resolve_gold(query, documents)
        answers = answers_by_query.get(query["id"])
        if answers is None:
            continue
        size = len(answers) if top_k is None else top_k
        if size > len(answers):
            raise ValueError(f"{query['id']}: the results cover only {len(answers)} candidates")
        refs = [ref for ref, _ in pools["pools"][query["id"]]][:size]
        answers = {ref: answers[ref] for ref in refs if ref in answers}
        pool_grades = [grade(span(r), gold) for r in refs]
        relevant = pooled_relevant([span(r) for r in refs], gold)
        row: dict[str, Any] = {"category": query["category"]}
        for name, order in (("original", refs), ("reranked", reranked(refs, answers))):
            chosen = seeds(order, chunks, answers if name == "reranked" else None)
            row[name] = {
                "ndcg@10": ndcg([grade(span(r), gold) for r in order], pool_grades),
                "recall@k": recall([span(r) for r in order[:MAX_CHUNKS]], relevant),
                "seed_recall": recall([span(r) for r in chosen], relevant),
            }
        per_query[query["id"]] = row
        for k in TOP_K_PREFIXES:
            if k > len(refs):
                continue
            head = refs[:k]
            value = recall([span(r) for r in head], relevant)
            if value is not None:
                prefix_rows[k]["recall"].append(value)
            score = ndcg(
                [grade(span(r), gold) for r in reranked(head, answers)], pool_grades
            )
            if score is not None:
                prefix_rows[k]["ndcg"].append(score)
        for floor in FLOORS:
            chosen = seeds(reranked(refs, answers), chunks, answers, floor)
            value = recall([span(r) for r in chosen], relevant)
            if value is not None:
                floor_rows[floor]["recall"].append(value)
            if chosen:
                floor_rows[floor]["precision"].append(
                    sum(grade(span(r), gold) >= 2 for r in chosen) / len(chosen)
                )

    scored = {q: r for q, r in per_query.items() if r["original"]["ndcg@10"] is not None}

    def summary(ids: Sequence[str]) -> dict[str, Any]:
        out: dict[str, Any] = {"queries": len(ids)}
        for name in ("original", "reranked"):
            out[name] = {
                metric: _mean(
                    [scored[q][name][metric] for q in ids if scored[q][name][metric] is not None]
                )
                for metric in ("ndcg@10", "recall@k", "seed_recall")
            }
        out["delta_ndcg@10"] = _mean(
            [scored[q]["reranked"]["ndcg@10"] - scored[q]["original"]["ndcg@10"] for q in ids]
        )
        return out

    overall = summary(sorted(scored))
    categories = {
        c: summary(sorted(q for q in scored if scored[q]["category"] == c))
        for c in sorted({r["category"] for r in scored.values()})
    }
    deltas = [
        scored[q]["reranked"]["ndcg@10"] - scored[q]["original"]["ndcg@10"] for q in sorted(scored)
    ]
    interval = paired_bootstrap(deltas)
    return {
        "model": results.get("model"),
        "top_k": top_k,
        "overall": overall,
        "bootstrap_95": interval,
        "categories": categories,
        "gates": {
            "mean_improvement": overall["delta_ndcg@10"] >= GATES["mean_improvement"],
            "lower_bound_above_zero": interval is not None and interval[0] > 0,
            "category_regression": all(
                c["delta_ndcg@10"] >= -GATES["category_regression"] for c in categories.values()
            ),
            # Overall and in every category (the gate approved before the data was seen).
            "recall_not_lower": all(
                group["reranked"]["recall@k"] >= group["original"]["recall@k"]
                for group in (overall, *categories.values())
                if group["original"]["recall@k"] is not None
            ),
        },
        "top_k_prefixes": {
            str(k): {"pooled_recall": _mean(v["recall"]), "reranked_ndcg@10": _mean(v["ndcg"])}
            for k, v in prefix_rows.items()
            if v["ndcg"]
        },
        "relevance_floor": {
            f"{f:.2f}": {
                "seed_recall": _mean(v["recall"]),
                "seed_precision": _mean(v["precision"]),
            }
            for f, v in floor_rows.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--top-k", type=int, help="score only this original-order prefix")
    args = parser.parse_args()
    report = evaluate(json.loads(args.results.read_text("utf-8")), top_k=args.top_k)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
