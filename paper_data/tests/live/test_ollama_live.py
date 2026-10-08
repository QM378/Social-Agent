"""Live checks against a running Ollama endpoint. Skipped unless SOCIAL_AGENT_LIVE_OLLAMA=1.
In CI/container we serve a tiny RANDOM model (garbage content): these tests verify the HTTP contract only."""
import asyncio
import os

import pytest
from pydantic import BaseModel
from typing import Literal

from social_agent.llm import LLMConfig, OpenAICompatClient

pytestmark = pytest.mark.skipif(os.environ.get("SOCIAL_AGENT_LIVE_OLLAMA") != "1", reason="needs a live Ollama")
BASE = os.environ.get("SOCIAL_AGENT_OLLAMA_URL", "http://127.0.0.1:11434/v1")
MODEL = os.environ.get("SOCIAL_AGENT_OLLAMA_MODEL", "tiny-8k")


class Small(BaseModel):
    pace: Literal["slow", "fast", "unknown"]
    n: int


async def _with(cfg, fn):
    c = OpenAICompatClient(cfg)
    try:
        return await fn(c)
    finally:
        await c.aclose()


def test_models_listed_and_schema_accepted():
    async def body(c):
        models = await c.list_models()
        assert any(m.startswith(MODEL) for m in models)
        r = await c.structured("Return JSON.", "hi", Small)
        assert r.calls == 1 and r.schema_downgrades == 0 and r.attempts[0].http_status == 200
        assert not r.tokens_incomplete and r.prompt_tokens and r.completion_tokens
        assert c.schema_mode_supported is True
        if r.ok:
            assert r.parsed.pace in ("slow", "fast", "unknown")
        else:
            assert r.truncated or r.error
    asyncio.run(_with(LLMConfig(base_url=BASE, model=MODEL, max_tokens=64, reasoning_effort=None), body))


def test_reasoning_effort_on_non_thinking_model_is_a_plain_400_not_a_downgrade():
    async def body(c):
        r = await c.structured("Return JSON.", "hi", Small)
        if not r.ok and r.attempts and r.attempts[0].http_status == 400:
            assert r.schema_downgrades == 0 and r.calls == 1 and "thinking" in r.error
        else:
            pytest.skip("model supports thinking; nothing to check here")
    asyncio.run(_with(LLMConfig(base_url=BASE, model=MODEL, max_tokens=64, reasoning_effort="low"), body))


def test_unknown_model_is_404_or_400_and_not_retried():
    async def body(c):
        r = await c.structured("x", "y", Small)
        assert not r.ok and r.calls == 1 and r.network_retries == 0 and r.attempts[0].http_status in (400, 404)
    asyncio.run(_with(LLMConfig(base_url=BASE, model="does-not-exist-xyz", max_tokens=16, reasoning_effort=None), body))
