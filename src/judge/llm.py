"""One async client layer (httpx) for both providers.

Every LLM call in the repo goes through LLMClient.complete(): semaphore-bounded concurrency,
exponential backoff with jitter that honours Retry-After on 429/5xx, a disk cache with a nonce,
a CallLog row appended to JSONL for every call (cached or not), and per-run token/call counters.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from judge.cache import DiskCache, make_key, prompt_hash
from judge.config import Settings

RETRYABLE = {429, 500, 502, 503, 504}
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
GROQ_BASE = "https://api.groq.com/openai/v1"


class LLMError(RuntimeError):
    pass


@dataclass
class CallLog:
    run_id: str
    role: str  # judge | generator
    model: str
    case_id: str
    prompt_hash: str
    prompt: str
    raw_response: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    cached: bool
    nonce: str
    reasoning: str = ""
    system: str = ""
    temperature: float = 0.0
    ts: float = field(default_factory=time.time)


@dataclass
class Counters:
    calls: int = 0
    network_calls: int = 0
    cached_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


def parse_retry_after(headers: httpx.Headers) -> float | None:
    v = headers.get("retry-after")
    if v is None:
        return None
    try:
        return max(0.0, float(v))
    except ValueError:
        return None


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        role: str,
        *,
        run_id: str | None = None,
        log_name: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep=asyncio.sleep,
    ):
        if role not in ("judge", "generator"):
            raise ValueError(role)
        settings.enforce_family_guard()
        self.s = settings
        self.role = role
        if role == "judge":
            self.provider, self.model = settings.judge_provider, settings.judge_model
            self.default_temp, conc = settings.judge_temperature, settings.judge_concurrency
        else:
            self.provider, self.model = settings.gen_provider, settings.gen_model
            self.default_temp, conc = settings.gen_temperature, settings.gen_concurrency
        self.api_key = settings.api_key_for(self.provider)
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.cache = DiskCache(settings.cache_dir)
        log_dir = Path(settings.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / (log_name or f"calls_{role}_{self.run_id}.jsonl")
        self.counters = Counters()
        self._sem = asyncio.Semaphore(conc)
        self._sleep = sleep
        self._http = httpx.AsyncClient(timeout=settings.request_timeout_seconds, transport=transport)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    # ---- request building / response parsing per provider ----
    def _params(self, temperature: float, max_tokens: int | None) -> dict[str, Any]:
        p: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
        if self.provider == "groq":
            p["reasoning_effort"] = self.s.judge_reasoning_effort
        return p

    def _build(self, prompt: str, system: str | None, params: dict[str, Any]):
        if self.provider == "gemini":
            body: dict[str, Any] = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": params["temperature"]},
            }
            if params["max_tokens"]:
                body["generationConfig"]["maxOutputTokens"] = params["max_tokens"]
            if system:
                body["systemInstruction"] = {"parts": [{"text": system}]}
            return f"{GEMINI_BASE}/models/{self.model}:generateContent", {"x-goog-api-key": self.api_key}, body
        if self.provider == "groq":
            msgs = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": prompt}
            ]
            body = {"model": self.model, "messages": msgs, "temperature": params["temperature"]}
            if params["max_tokens"]:
                body["max_completion_tokens"] = params["max_tokens"]
            if "gpt-oss" in self.model:
                body["reasoning_effort"] = params["reasoning_effort"]
            return f"{GROQ_BASE}/chat/completions", {"Authorization": f"Bearer {self.api_key}"}, body
        raise ValueError(f"unknown provider {self.provider!r}")

    def _parse(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.provider == "gemini":
            cands = data.get("candidates") or []
            parts = (cands[0].get("content", {}).get("parts", []) if cands else [])
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            u = data.get("usageMetadata", {})
            return {
                "text": text,
                "reasoning": "",
                "tokens_in": u.get("promptTokenCount", 0) or 0,
                "tokens_out": (u.get("candidatesTokenCount", 0) or 0) + (u.get("thoughtsTokenCount", 0) or 0),
            }
        msg = (data.get("choices") or [{}])[0].get("message", {})
        u = data.get("usage", {})
        return {
            "text": msg.get("content") or "",
            "reasoning": msg.get("reasoning") or "",
            "tokens_in": u.get("prompt_tokens", 0) or 0,
            "tokens_out": u.get("completion_tokens", 0) or 0,
        }

    async def _post_with_backoff(self, url: str, headers: dict, body: dict) -> dict[str, Any]:
        s = self.s
        for attempt in range(s.max_retries):
            try:
                r = await self._http.post(url, headers=headers, json=body)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt == s.max_retries - 1:
                    raise LLMError(f"transport error after {s.max_retries} attempts: {exc!r}") from exc
                await self._sleep(self._delay(attempt, None))
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code in RETRYABLE and attempt < s.max_retries - 1:
                await self._sleep(self._delay(attempt, parse_retry_after(r.headers)))
                continue
            raise LLMError(f"{self.provider} HTTP {r.status_code}: {r.text[:300]}")
        raise LLMError("unreachable")

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        s = self.s
        backoff = random.uniform(0, min(s.backoff_max_seconds, s.backoff_base_seconds * 2**attempt))
        if retry_after is not None:
            # honour the server's floor, plus a little jitter so concurrent callers don't re-sync
            return retry_after + random.uniform(0, 0.5)
        return backoff

    # ---- public API ----
    async def complete(
        self,
        prompt: str,
        *,
        case_id: str = "",
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        nonce: str = "",
    ) -> dict[str, Any]:
        temp = self.default_temp if temperature is None else temperature
        params = self._params(temp, max_tokens)
        key = make_key(self.model, params, f"{system or ''}\x00{prompt}", nonce)
        t0 = time.perf_counter()
        self.counters.calls += 1

        cached = self.cache.get(key)
        if cached is not None:
            self.counters.cached_calls += 1
            result = {**cached, "cached": True}
        else:
            url, headers, body = self._build(prompt, system, params)
            async with self._sem:
                data = await self._post_with_backoff(url, headers, body)
            self.counters.network_calls += 1
            result = self._parse(data)
            self.cache.set(key, result)
            result = {**result, "cached": result.get("cached", False)}

        self.counters.tokens_in += result["tokens_in"]
        self.counters.tokens_out += result["tokens_out"]
        self._append_log(
            CallLog(
                run_id=self.run_id, role=self.role, model=self.model, case_id=case_id,
                prompt_hash=prompt_hash(prompt), prompt=prompt, raw_response=result["text"],
                tokens_in=result["tokens_in"], tokens_out=result["tokens_out"],
                latency_ms=int((time.perf_counter() - t0) * 1000), cached=result["cached"],
                nonce=nonce, reasoning=result.get("reasoning", ""), system=system or "",
                temperature=temp,
            )
        )
        return result

    def _append_log(self, row: CallLog) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


# ---- model-listing helpers used by the smoke test ----
async def list_groq_models(settings: Settings, transport=None) -> list[str]:
    async with httpx.AsyncClient(timeout=30, transport=transport) as c:
        r = await c.get(f"{GROQ_BASE}/models", headers={"Authorization": f"Bearer {settings.groq_api_key}"})
    if r.status_code != 200:
        raise LLMError(f"Groq /models HTTP {r.status_code}: {r.text[:200]}")
    return [m["id"] for m in r.json().get("data", [])]


async def gemini_model_exists(settings: Settings, model: str, transport=None) -> bool:
    async with httpx.AsyncClient(timeout=30, transport=transport) as c:
        r = await c.get(f"{GEMINI_BASE}/models/{model}", headers={"x-goog-api-key": settings.gemini_api_key})
    if r.status_code == 404:
        return False
    if r.status_code != 200:
        raise LLMError(f"Gemini model lookup HTTP {r.status_code}: {r.text[:200]}")
    return True
