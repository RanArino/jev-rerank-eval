import json
from pathlib import Path

import pytest

from rerank_eval.evaluate import evaluate, reranked, seeds
from rerank_eval.metrics import Gold, Span, covered, dcg, ndcg, paired_bootstrap, recall

ROOT = Path(__file__).resolve().parents[1]


def test_a_chunk_covers_a_passage_when_it_holds_at_least_half_of_it() -> None:
    gold = Gold("d", 100, 200, 3)
    assert covered(gold, [Span("d", 150, 400)])
    assert not covered(gold, [Span("d", 151, 400)])
    # Two overlapping chunks are counted once, not twice.
    assert not covered(gold, [Span("d", 100, 130), Span("d", 110, 140)])
    assert not covered(gold, [Span("other", 0, 1000)])


def test_ndcg_uses_the_whole_pool_for_its_ideal() -> None:
    assert dcg([3, 0]) == 7.0
    assert ndcg([3, 2], [3, 2]) == 1.0
    # The best item is in the pool but ranked below the cut: the ideal still counts it.
    assert ndcg([0], [0, 3], k=1) == 0.0
    assert ndcg([0, 1], [0, 1, 0]) == pytest.approx(dcg([0, 1]) / dcg([1]))
    assert ndcg([0, 0], [0, 0]) is None


def test_recall_without_relevant_passages_is_undefined_not_perfect() -> None:
    assert recall([Span("d", 0, 10)], []) is None


def test_reranking_breaks_ties_by_the_original_rank() -> None:
    answers = {"a": {"probability": 0.5}, "b": {"probability": 0.9}, "c": {"probability": 0.5}}
    assert reranked(["a", "b", "c", "d"], answers) == ["b", "a", "c", "d"]
    assert reranked(["a", "b"], {"a": {"value": False}, "b": {"value": True}}) == ["b", "a"]


def test_seeds_cap_per_document_and_skip_duplicates_and_low_scores() -> None:
    chunks = {f"x{i}": {"document_id": "x", "body": f"b{i}"} for i in range(6)}
    chunks["dup"] = {"document_id": "y", "body": "b0"}
    order = ["x0", "dup", "x1", "x2", "x3", "x4"]
    assert seeds(order, chunks) == ["x0", "x1", "x2", "x3"]
    answers = {ref: {"probability": 0.1 if ref == "x1" else 0.9} for ref in order}
    assert seeds(order, chunks, answers, floor=0.25) == ["x0", "x2", "x3", "x4"]


def test_paired_bootstrap_is_deterministic() -> None:
    assert paired_bootstrap([0.1, 0.2, 0.3]) == paired_bootstrap([0.1, 0.2, 0.3])
    assert paired_bootstrap([]) is None


def test_the_published_jev_results_reproduce() -> None:
    report = evaluate(json.loads((ROOT / "results/jev-1.13.0.json").read_text("utf-8")))
    overall = report["overall"]
    assert overall["queries"] == 56
    assert overall["original"]["ndcg@10"] == pytest.approx(0.752904, abs=1e-6)
    assert overall["reranked"]["ndcg@10"] == pytest.approx(0.911933, abs=1e-6)
    assert report["bootstrap_95"] == pytest.approx((0.090248, 0.234053), abs=1e-6)
    assert all(report["gates"].values())


def test_the_published_llm_results_reproduce() -> None:
    results = json.loads((ROOT / "results/gpt-6-luna-medium.json").read_text("utf-8"))
    report = evaluate(results)
    assert report["overall"]["queries"] == 56
    assert round(report["overall"]["delta_ndcg@10"], 3) == 0.116
    low, high = report["bootstrap_95"]
    assert (round(low, 3), round(high, 3)) == (0.065, 0.172)
    # Ranked recall falls in the ambiguous/no-answer category, so that gate fails.
    assert report["gates"]["recall_not_lower"] is False
    assert report["gates"]["mean_improvement"] and report["gates"]["lower_bound_above_zero"]


def test_a_top_k_prefix_compares_implementations_on_the_same_candidates() -> None:
    jev = json.loads((ROOT / "results/jev-1.13.0.json").read_text("utf-8"))
    report = evaluate(jev, top_k=60)
    assert report["top_k"] == 60
    assert round(report["overall"]["reranked"]["ndcg@10"], 3) == 0.919
    llm = json.loads((ROOT / "results/gpt-6-luna-medium.json").read_text("utf-8"))
    # A result measured on 60 candidates cannot be scored as if it had 150.
    with pytest.raises(ValueError, match="cover only"):
        evaluate(llm, top_k=150)


def test_the_published_clef_results_reproduce() -> None:
    results = json.loads((ROOT / "results/clef.json").read_text("utf-8"))
    report = evaluate(results)
    assert round(report["overall"]["delta_ndcg@10"], 3) == 0.098
    low, high = report["bootstrap_95"]
    assert (round(low, 3), round(high, 3)) == (0.022, 0.172)
    assert report["gates"]["recall_not_lower"] is False
    assert report["gates"]["category_regression"] is True
