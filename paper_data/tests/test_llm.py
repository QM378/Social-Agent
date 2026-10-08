import json
import httpx
import pytest
from pydantic import BaseModel

from social_agent.llm import LLMConfig, MockClient, CallBudget, OpenAICompatClient, _parse, schema_prompt_block


class Out(BaseModel):
    a: str
    b: int


def _client(handler, **kw):
    cfg = LLMConfig(max_backoff_retries=2, **kw)
    c = OpenAICompatClient(cfg, budget=CallBudget(kw.pop("max_calls", None)))
    c._http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=cfg.base_url)
    return c


def _ok(content, usage=None, finish="stop"):
    body = {"choices": [{"message": {"content": content}, "finish_reason": finish}]}
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


def test_parse_valid_fenced_and_invalid():
    assert _parse('{"a":"x","b":1}', Out)[0].b == 1
    assert _parse('```json\n{"a":"x","b":2}\n```', Out)[0].b == 2
    assert _parse('{"a":"x"}', Out)[0] is None
    assert _parse('{"a":"x", "b": ', Out)[0] is None


@pytest.mark.asyncio
async def test_500_then_success_counts_two_requests_one_retry(monkeypatch):
    import asyncio
    monkeypatch.setattr(asyncio, "sleep", lambda *_: asyncio.sleep(0) if False else _noop())
    n = {"i": 0}

    def h(req):
        n["i"] += 1
        return httpx.Response(500, text="boom") if n["i"] == 1 else _ok('{"a":"x","b":1}', {"prompt_tokens": 10, "completion_tokens": 5})
    c = _client(h)
    r = await c.structured("s", "u", Out)
    assert r.ok and r.calls == 2 and r.network_retries == 1 and r.json_repairs == 0
    assert c.budget.calls == 2


async def _noop():
    return None


@pytest.mark.asyncio
async def test_budget_zero_sends_nothing():
    def h(req):
        raise AssertionError("must not be called")
    c = _client(h)
    c.budget = CallBudget(max_calls=0)
    r = await c.structured("s", "u", Out)
    assert not r.ok and r.calls == 0 and "budget_exceeded" in r.error and c.budget.calls == 0


@pytest.mark.asyncio
async def test_tokens_accumulate_across_repair():
    n = {"i": 0}

    def h(req):
        n["i"] += 1
        if n["i"] == 1:
            return _ok('{"a":"x"}', {"prompt_tokens": 10, "completion_tokens": 3})
        return _ok('{"a":"x","b":1}', {"prompt_tokens": 20, "completion_tokens": 4})
    r = await _client(h).structured("s", "u", Out)
    assert r.ok and r.calls == 2 and r.json_repairs == 1
    assert r.prompt_tokens == 30 and r.completion_tokens == 7 and not r.tokens_incomplete
    assert [a.status for a in r.attempts] == ["invalid_output", "ok"]


@pytest.mark.asyncio
async def test_missing_usage_marks_incomplete_but_keeps_subtotal():
    n = {"i": 0}

    def h(req):
        n["i"] += 1
        return _ok('{"a":"x"}') if n["i"] == 1 else _ok('{"a":"x","b":1}', {"prompt_tokens": 20, "completion_tokens": 4})
    r = await _client(h).structured("s", "u", Out)
    assert r.ok and r.tokens_incomplete and r.prompt_tokens == 20


@pytest.mark.asyncio
async def test_schema_unsupported_downgrades_once_and_prompts_schema():
    seen = []

    def h(req):
        body = json.loads(req.content)
        seen.append(body)
        if body["response_format"]["type"] == "json_schema":
            return httpx.Response(400, text='{"error": "response_format json_schema is not supported"}')
        return _ok('{"a":"x","b":1}', {"prompt_tokens": 1, "completion_tokens": 1})
    c = _client(h)
    r = await c.structured("s", "u", Out)
    assert r.ok and r.schema_downgrades == 1 and r.calls == 2 and r.network_retries == 0
    assert seen[1]["response_format"]["type"] == "json_object"
    assert schema_prompt_block(Out).splitlines()[0] in seen[1]["messages"][0]["content"]
    r2 = await c.structured("s", "u", Out)          # no second downgrade attempt
    assert r2.calls == 1 and seen[2]["response_format"]["type"] == "json_object"


@pytest.mark.asyncio
async def test_other_400_and_404_are_not_downgraded_or_retried():
    def h400(req):
        return httpx.Response(400, text='{"error":"maximum context length exceeded"}')
    r = await _client(h400).structured("s", "u", Out)
    assert not r.ok and r.calls == 1 and r.schema_downgrades == 0 and "context length" in r.error

    def h404(req):
        return httpx.Response(404, text='{"error":"model not found"}')
    r = await _client(h404).structured("s", "u", Out)
    assert not r.ok and r.calls == 1 and r.network_retries == 0 and "http_404" in r.error


@pytest.mark.asyncio
async def test_truncation_is_reported_not_parsed():
    r = await _client(lambda req: _ok('{"a":"x","b":', finish="length")).structured("s", "u", Out)
    assert not r.ok and r.truncated and r.error.startswith("truncated") and r.calls == 1


@pytest.mark.asyncio
async def test_mock_tags_backend_and_counts_budget():
    c = MockClient(LLMConfig(), responder=lambda s, u, sc: {"a": "ok", "b": 3}, budget=CallBudget(max_calls=2))
    r1 = await c.structured("s", "u", Out)
    assert r1.ok and r1.backend == "mock" and r1.parsed.b == 3 and r1.calls == 1
    await c.structured("s", "u", Out)
    r3 = await c.structured("s", "u", Out)
    assert not r3.ok and "budget_exceeded" in r3.error and r3.calls == 0


@pytest.mark.asyncio
async def test_no_retry_budget_reports_zero_retries():
    r = await _client_kw(lambda req: httpx.Response(500, text="boom"), max_backoff_retries=0).structured("s", "u", Out)
    assert not r.ok and r.calls == 1 and r.network_retries == 0 and r.attempts[0].reason == "initial"


def _client_kw(handler, **kw):
    cfg = LLMConfig(**kw)
    c = OpenAICompatClient(cfg, budget=CallBudget())
    c._http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=cfg.base_url)
    return c


@pytest.mark.asyncio
async def test_no_repair_budget_reports_zero_repairs():
    r = await _client_kw(lambda req: _ok('{"a":"x"}'), max_repairs=0, max_backoff_retries=0).structured("s", "u", Out)
    assert not r.ok and r.calls == 1 and r.json_repairs == 0 and r.error.startswith("schema_error")


@pytest.mark.asyncio
async def test_invalid_schema_400_is_not_a_downgrade():
    r = await _client(lambda req: httpx.Response(400, text='{"error":"Invalid json_schema: required must include every property"}')).structured("s", "u", Out)
    assert not r.ok and r.calls == 1 and r.schema_downgrades == 0 and "Invalid json_schema" in r.error


@pytest.mark.asyncio
async def test_200_non_json_body_is_a_recorded_failure():
    r = await _client(lambda req: httpx.Response(200, text="<html>gateway</html>")).structured("s", "u", Out)
    assert not r.ok and r.calls == 1 and r.attempts[0].status == "bad_response" and r.error.startswith("bad_response")


@pytest.mark.asyncio
async def test_attempt_reasons_and_latency_split(monkeypatch):
    import asyncio as _a
    waited = {"s": 0.0}

    async def fake_sleep(t):
        waited["s"] += t
    monkeypatch.setattr(_a, "sleep", fake_sleep)
    n = {"i": 0}

    def h(req):
        n["i"] += 1
        if n["i"] == 1:
            return httpx.Response(503, text="busy")
        if n["i"] == 2:
            return _ok('{"a":"x"}', {"prompt_tokens": 1, "completion_tokens": 1})
        return _ok('{"a":"x","b":1}', {"prompt_tokens": 1, "completion_tokens": 1})
    r = await _client(h).structured("s", "u", Out)
    assert r.ok and [a.reason for a in r.attempts] == ["initial", "network_retry", "json_repair"]
    assert r.network_retries == 1 and r.json_repairs == 1 and r.calls == 3
    assert r.http_latency_s <= r.wall_s and waited["s"] > 0


def test_schema_prompt_block_contains_full_schema():
    from typing import Literal
    from pydantic import Field as F

    class Inner(BaseModel):
        k: Literal["x", "y"]

    class Outer(BaseModel):
        items: list[Inner]
        n: int = F(ge=0, le=3)
    blk = schema_prompt_block(Outer)
    assert '"$defs"' in blk and '"Inner"' in blk and '"enum"' in blk and '"maximum": 3' in blk and '"items"' in blk


@pytest.mark.asyncio
async def test_attempt_records_exact_messages_and_response_without_auth():
    n = {"i": 0}

    def h(req):
        n["i"] += 1
        return _ok('{"a":"x"}', {"prompt_tokens": 1, "completion_tokens": 1}) if n["i"] == 1 else _ok('{"a":"x","b":1}', {"prompt_tokens": 1, "completion_tokens": 1})
    r = await _client(h).structured("SYS", "USER", Out)
    a0, a1 = r.attempts
    assert a0.request_messages[0]["content"] == "SYS" and a0.response_content == '{"a":"x"}'
    assert len(a1.request_messages) == 4 and "invalid" in a1.request_messages[-1]["content"].lower()   # repair turn differs
    assert "api_key" not in json.dumps(a1.request_params) and "Authorization" not in json.dumps(a1.request_params)
