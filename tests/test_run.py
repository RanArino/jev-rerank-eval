import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from rerank_eval import run


def _args(tmp_path: Path, **extra: Any) -> argparse.Namespace:
    values = {
        "implementation": "jev",
        "model": "jev-1.13.0",
        "effort": "low",
        "top_k": 150,
        "timeout": 5.0,
        "max_calls": 3,
        "max_usd": 1.0,
        "output": tmp_path / "out.json",
    }
    return argparse.Namespace(**{**values, **extra})


def _fake(monkeypatch: pytest.MonkeyPatch, fail: bool = False) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    async def ask(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
        sent.append(state)
        if fail:
            raise TimeoutError
        return {"probability": 0.7}, {"input_tokens": 400, "output_tokens": 0}

    async def close() -> None:
        return None

    monkeypatch.setattr(run, "jev_asker", lambda model, timeout: (ask, close))
    return sent


def test_a_run_stops_at_its_limit_and_resumes_without_resending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = _fake(monkeypatch)
    first = asyncio.run(run.run(_args(tmp_path)))
    assert len(sent) == 3 and first["usage"]["calls"] == 3
    # Production sends an empty title for the user's own chunks.
    assert {s["candidate"]["title"] for s in sent} == {""}

    second = asyncio.run(run.run(_args(tmp_path, max_calls=5)))
    assert len(sent) == 5 and second["usage"]["calls"] == 5
    lines = (tmp_path / "out.jsonl").read_text().splitlines()
    assert len({(json.loads(x)["query_id"], json.loads(x)["ref"]) for x in lines}) == 5


def test_a_failed_call_is_kept_and_charged_its_worst_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake(monkeypatch, fail=True)
    report = asyncio.run(run.run(_args(tmp_path, max_calls=1)))
    (record,) = [json.loads(x) for x in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert record["answer"] is None and record["error"] == "TimeoutError"
    assert report["usage"]["usd"] > 0 and report["answers"] == {}


def test_the_dollar_limit_stops_before_a_call_could_cross_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = _fake(monkeypatch)
    asyncio.run(run.run(_args(tmp_path, max_usd=0.0)))
    assert sent == []
