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


def test_clef_uses_the_workers_ai_api_and_reads_its_noul_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: dict[str, Any] = {}

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "success": True,
                    "result": {
                        "answers": {"useful": {"noul": 0.7}},
                        "usage": {"input_tokens": 400, "output_tokens": 0},
                    },
                }
            ).encode()

    def urlopen(request: Any, timeout: float) -> Response:
        sent.update(
            url=request.full_url,
            headers=dict(request.header_items()),
            body=json.loads(request.data),
            timeout=timeout,
        )
        return Response()

    monkeypatch.setattr(run, "urlopen", urlopen)
    ask, close = run.clef_asker("clef", 5.0, "account-id", "api-token")
    answer, usage = asyncio.run(ask({"candidate": "example"}))
    asyncio.run(close())

    assert answer == {"probability": 0.7}
    assert usage == {"input_tokens": 400, "output_tokens": 0}
    assert sent["url"].endswith("/accounts/account-id/ai/run/@cf/cloudflare/clef")
    assert sent["headers"]["Authorization"] == "Bearer api-token"
    assert sent["body"]["questions"]["useful"]["type"] == "noul"


def test_clef_requires_its_cloudflare_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="CLOUDFLARE_ACCOUNT_ID"):
        asyncio.run(run.run(_args(tmp_path, implementation="clef", model="clef")))
