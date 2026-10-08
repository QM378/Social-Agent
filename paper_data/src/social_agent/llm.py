"""Async client for OpenAI-compatible endpoints (Ollama), with structured JSON output, retries and a mock for tests.

Accounting:
- Every real HTTP request is one `Attempt`; calls / retries / latency / usage are summed from attempts.
- A request blocked by the budget is NOT an attempt (calls stay 0 for it).
- Three independent counters: network_retries (429/5xx/transport), json_repairs, schema_downgrades.
- Usage is accumulated across attempts; if any attempt lacks usage, `tokens_incomplete=True`
  and the reported totals are a known subtotal, not an exact total.
- Schema downgrade happens at most once, only when the backend says structured output / response_format
  is unsupported; other 400s are reported as-is. After downgrade the schema and enums are put in the prompt.
- Non-retryable HTTP (400/401/403/404/422) is never retried. Every failure has a non-empty `error`.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
_FEATURE_WORDS = ("response_format", "json_schema", "structured output", "structured_output", "guided decoding", "guided_json")
_UNSUPPORTED_WORDS = ("not supported", "unsupported", "does not support", "doesn't support", "not enabled", "is not available",
                      "unrecognized request argument", "unknown parameter", "extra inputs are not permitted")


@dataclass
class LLMConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "openai/gpt-oss-20b"
    api_key: str = "EMPTY"
    timeout_s: float = 120.0
    temperature: float = 0.0
    seed: Optional[int] = 0
    max_tokens: int = 2048
    max_json_tokens: int = 512
    reasoning_effort: Optional[str] = None
    json_schema_mode: bool = True
    max_repairs: int = 1
    max_backoff_retries: int = 3
    concurrency: int = 1
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class Attempt:
    index: int
    reason: str                     # initial | network_retry | json_repair | schema_downgrade
    status: str                     # ok | invalid_output | truncated | http_error | transport_error | bad_response
    http_status: Optional[int]
    latency_s: float
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    usage_missing: bool
    schema_mode: str                # json_schema | json_object_prompted
    downgraded_here: bool = False
    error: Optional[str] = None
    request_messages: Optional[list] = None      # exact messages sent in THIS attempt (repairs/downgrades change them)
    request_params: Optional[dict] = None        # body minus messages; never includes auth
    response_content: Optional[str] = None       # assistant content of this attempt (or error body)


@dataclass
class LLMResult:
    ok: bool
    parsed: Optional[BaseModel]
    raw_text: str
    reasoning_text: Optional[str]
    backend: str
    model: str
    attempts: list[Attempt] = field(default_factory=list)
    error: Optional[str] = None
    request_id: Optional[str] = None
    truncated: bool = False
    wall_s: float = 0.0             # task total including backoff waits (what a user actually waits)

    # ---- derived accounting: counts are of attempts actually SENT for that reason ----
    @property
    def calls(self) -> int:
        return len(self.attempts)

    @property
    def network_retries(self) -> int:
        return sum(1 for a in self.attempts if a.reason == "network_retry")

    @property
    def json_repairs(self) -> int:
        return sum(1 for a in self.attempts if a.reason == "json_repair")

    @property
    def schema_downgrades(self) -> int:
        return sum(1 for a in self.attempts if a.reason == "schema_downgrade")

    @property
    def http_latency_s(self) -> float:
        """Sum of HTTP request durations only (excludes backoff waits)."""
        return sum(a.latency_s for a in self.attempts)

    @property
    def latency_s(self) -> float:
        return self.wall_s

    @property
    def tokens_incomplete(self) -> bool:
        return any(a.usage_missing for a in self.attempts)

    @property
    def prompt_tokens(self) -> Optional[int]:
        vals = [a.prompt_tokens for a in self.attempts if a.prompt_tokens is not None]
        return sum(vals) if vals else None

    @property
    def completion_tokens(self) -> Optional[int]:
        vals = [a.completion_tokens for a in self.attempts if a.completion_tokens is not None]
        return sum(vals) if vals else None

    def to_record(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "error": self.error, "request_id": self.request_id, "backend": self.backend,
            "model": self.model, "truncated": self.truncated, "calls": self.calls,
            "network_retries": self.network_retries, "json_repairs": self.json_repairs,
            "schema_downgrades": self.schema_downgrades, "http_latency_s": round(self.http_latency_s, 4),
            "wall_s": round(self.wall_s, 4),
            "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
            "tokens_incomplete": self.tokens_incomplete,
            "attempts": [a.__dict__ for a in self.attempts],
            "raw_text": self.raw_text, "reasoning_text": self.reasoning_text,
            "parsed": self.parsed.model_dump() if self.parsed else None,
        }


class BudgetExceeded(RuntimeError):
    pass


class CallBudget:
    """Counts real requests actually sent (including retries and repairs). Blocked requests are not counted."""

    def __init__(self, max_calls: Optional[int] = None):
        self.max_calls = max_calls
        self.calls = 0
        self._lock = asyncio.Lock()

    async def take(self) -> None:
        async with self._lock:
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise BudgetExceeded(f"max_calls={self.max_calls} reached")
            self.calls += 1


class BaseClient:
    backend: str = "base"

    def __init__(self, cfg: LLMConfig, budget: Optional[CallBudget] = None):
        self.cfg = cfg
        self.budget = budget or CallBudget()
        self._sem = asyncio.Semaphore(max(1, cfg.concurrency))

    async def structured(self, system: str, user: str, schema: Type[T], *, request_id: Optional[str] = None) -> LLMResult:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


def schema_prompt_block(schema: Type[BaseModel]) -> str:
    """Full JSON Schema (with $defs, array items, enums, constraints) for prompting when json_schema mode is unavailable."""
    js = schema.model_json_schema()
    return ("Output exactly one JSON object that validates against this JSON Schema. No extra fields, no prose, "
            "no markdown fences.\n" + json.dumps(js, indent=1))


class OpenAICompatClient(BaseClient):
    backend = "openai_compat"

    def __init__(self, cfg: LLMConfig, budget: Optional[CallBudget] = None):
        super().__init__(cfg, budget)
        self._http = httpx.AsyncClient(base_url=cfg.base_url, timeout=cfg.timeout_s, trust_env=False,
                                       headers={"Authorization": f"Bearer {cfg.api_key}"})
        self.schema_mode_supported: Optional[bool] = None if cfg.json_schema_mode else False

    async def aclose(self) -> None:
        await self._http.aclose()

    async def list_models(self) -> list[str]:
        r = await self._http.get("/models")
        r.raise_for_status()
        return [m.get("id", "") for m in r.json().get("data", [])]

    def _body(self, messages: list[dict[str, str]], schema: Type[BaseModel], use_schema: bool) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.cfg.model, "messages": messages,
                                "temperature": self.cfg.temperature, "max_tokens": self.cfg.max_tokens}
        if self.cfg.seed is not None:
            body["seed"] = self.cfg.seed
        if use_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "schema": schema.model_json_schema(), "strict": True}}
        else:
            body["response_format"] = {"type": "json_object"}
        if self.cfg.reasoning_effort:
            body["reasoning_effort"] = self.cfg.reasoning_effort
        body.update(self.cfg.extra_body)
        return body

    @staticmethod
    def _is_schema_unsupported(resp: httpx.Response) -> bool:
        """True only for an explicit 'feature not supported' error about structured output; an invalid schema is not that."""
        if resp.status_code != 400:
            return False
        txt = resp.text.lower()
        if "invalid" in txt and "json_schema" in txt and not any(w in txt for w in _UNSUPPORTED_WORDS):
            return False
        return any(f in txt for f in _FEATURE_WORDS) and any(w in txt for w in _UNSUPPORTED_WORDS)

    async def structured(self, system: str, user: str, schema: Type[T], *, request_id: Optional[str] = None) -> LLMResult:
        async with self._sem:
            return await self._structured(system, user, schema, request_id)

    async def _structured(self, system: str, user: str, schema: Type[T], request_id: Optional[str]) -> LLMResult:
        res = LLMResult(False, None, "", None, self.backend, self.cfg.model, request_id=request_id)
        t_task = time.perf_counter()
        use_schema = self.schema_mode_supported is not False
        sys_text = system if use_schema else system + "\n\n" + schema_prompt_block(schema)
        messages = [{"role": "system", "content": sys_text}, {"role": "user", "content": user}]
        repairs_left = self.cfg.max_repairs
        net_retries_left = self.cfg.max_backoff_retries
        downgrade_left = 1 if self.schema_mode_supported is None else 0
        delay = 1.0
        reason = "initial"

        def finish(err: Optional[str]) -> LLMResult:
            res.error = err
            res.wall_s = time.perf_counter() - t_task
            return res

        while True:
            try:
                await self.budget.take()
            except BudgetExceeded as e:
                return finish(f"budget_exceeded:{e}")
            mode = "json_schema" if use_schema else "json_object_prompted"
            t0 = time.perf_counter()
            att = Attempt(len(res.attempts), reason, "ok", None, 0.0, None, None, True, mode)
            body = self._body(messages, schema, use_schema)
            att.request_messages = [dict(m) for m in messages]
            att.request_params = {k: v for k, v in body.items() if k != "messages"}
            try:
                r = await self._http.post("/chat/completions", json=body)
            except httpx.TransportError as e:
                att.status, att.error, att.latency_s = "transport_error", f"{type(e).__name__}:{e}", time.perf_counter() - t0
                res.attempts.append(att)
                if net_retries_left > 0:
                    net_retries_left -= 1; reason = "network_retry"
                    await asyncio.sleep(delay + random.random() * 0.5); delay = min(delay * 2, 30.0)
                    continue
                return finish(att.error)
            att.latency_s = time.perf_counter() - t0
            att.http_status = r.status_code

            if r.status_code != 200:
                att.status = "http_error"
                att.error = f"http_{r.status_code}:{r.text[:300]}"
                att.response_content = r.text[:2000]
                res.attempts.append(att)
                if use_schema and downgrade_left > 0 and self._is_schema_unsupported(r):
                    downgrade_left -= 1
                    att.downgraded_here = True
                    self.schema_mode_supported = False
                    use_schema = False
                    messages[0] = {"role": "system", "content": system + "\n\n" + schema_prompt_block(schema)}
                    reason = "schema_downgrade"
                    continue
                if r.status_code in _RETRYABLE_STATUS and net_retries_left > 0:
                    net_retries_left -= 1; reason = "network_retry"
                    await asyncio.sleep(delay + random.random() * 0.5); delay = min(delay * 2, 30.0)
                    continue
                return finish(att.error)

            try:
                data = r.json()
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message", {})
            except (ValueError, AttributeError, TypeError) as e:
                att.status, att.error = "bad_response", f"bad_response:body is not a chat completion JSON ({type(e).__name__})"
                res.attempts.append(att)
                return finish(att.error)
            if use_schema and self.schema_mode_supported is None:
                self.schema_mode_supported = True
            res.raw_text = msg.get("content") or ""
            att.response_content = res.raw_text
            res.reasoning_text = msg.get("reasoning_content") or msg.get("reasoning")
            usage = data.get("usage") or {}
            att.prompt_tokens, att.completion_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
            att.usage_missing = att.prompt_tokens is None or att.completion_tokens is None
            if choice.get("finish_reason") == "length":
                att.status, att.error = "truncated", "truncated:finish_reason=length"
                res.attempts.append(att)
                res.truncated = True
                return finish(att.error)
            parsed, perr = _parse(res.raw_text, schema)
            if parsed is not None:
                res.attempts.append(att)
                res.ok, res.parsed = True, parsed
                return finish(None)
            att.status, att.error = "invalid_output", perr
            res.attempts.append(att)
            if repairs_left > 0:
                repairs_left -= 1; reason = "json_repair"
                messages = messages + [
                    {"role": "assistant", "content": res.raw_text},
                    {"role": "user", "content": f"Your previous output was invalid ({perr}). "
                                                f"Return only a JSON object matching the schema.\n\n{schema_prompt_block(schema)}"}]
                continue
            return finish(perr)


def _parse(text: str, schema: Type[T]) -> tuple[Optional[T], Optional[str]]:
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.startswith("json"):
            s = s[4:]
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end <= start:
            return None, f"json_error:{e.msg}"
        try:
            obj = json.loads(s[start:end + 1])
        except json.JSONDecodeError as e2:
            return None, f"json_error:{e2.msg}"
    try:
        return schema.model_validate(obj), None
    except ValidationError as e:
        e0 = e.errors()[0]
        return None, f"schema_error:{'.'.join(str(x) for x in e0.get('loc', ()))}:{e0.get('msg', '')}"


class MockClient(BaseClient):
    """Deterministic mock. Every result carries backend='mock'; one Attempt per accepted request."""
    backend = "mock"

    def __init__(self, cfg: LLMConfig, responder=None, budget: Optional[CallBudget] = None):
        super().__init__(cfg, budget)
        self._responder = responder

    async def structured(self, system: str, user: str, schema: Type[T], *, request_id: Optional[str] = None) -> LLMResult:
        res = LLMResult(False, None, "", None, self.backend, "mock", request_id=request_id)
        try:
            await self.budget.take()
        except BudgetExceeded as e:
            res.error = f"budget_exceeded:{e}"
            return res
        out = self._responder(system, user, schema) if self._responder else {}
        raw = out if isinstance(out, str) else json.dumps(out)
        parsed, err = _parse(raw, schema)
        res.attempts.append(Attempt(0, "initial", "ok" if parsed else "invalid_output", 200, 0.0,
                                    len(user.split()), len(raw.split()), False, "mock", error=err))
        res.raw_text, res.parsed, res.ok, res.error = raw, parsed, parsed is not None, err
        return res


def make_client(cfg: LLMConfig, *, mock: bool = False, budget: Optional[CallBudget] = None, responder=None) -> BaseClient:
    return MockClient(cfg, responder=responder, budget=budget) if mock else OpenAICompatClient(cfg, budget=budget)


def load_llm_config(path) -> LLMConfig:
    """Read a matcher config (configs/matcher_*.yaml): its 'llm' block becomes an LLMConfig (one request at a time)."""
    import yaml
    d = yaml.safe_load(open(path, encoding="utf-8"))
    return LLMConfig(**{**d["llm"], "concurrency": 1})
