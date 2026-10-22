import httpx
import pytest

from judge.config import FamilyGuardError, Settings, family_of
from judge.llm import LLMClient, LLMError


def make_settings(tmp_path, **over) -> Settings:
    base = dict(
        _env_file=None,
        gemini_api_key="g-test", groq_api_key="q-test",
        gen_provider="gemini", gen_model="gemini-3.5-flash-lite", gen_temperature=0.0,
        judge_provider="groq", judge_model="openai/gpt-oss-120b", judge_temperature=0.0,
        cache_dir=str(tmp_path / "cache"), log_dir=str(tmp_path / "logs"),
        backoff_base_seconds=0.01, backoff_max_seconds=0.02, max_retries=4,
    )
    base.update(over)
    return Settings(**base)


def groq_ok(text="pong"):
    return {
        "choices": [{"message": {"content": text, "reasoning": "thinking"}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }


class Counting:
    def __init__(self, responses):
        self.responses, self.n = list(responses), 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.n += 1
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return r


async def test_second_call_is_cached_and_logged(tmp_path):
    h = Counting([httpx.Response(200, json=groq_ok())])
    async with LLMClient(make_settings(tmp_path), "judge", run_id="t", transport=httpx.MockTransport(h)) as c:
        a = await c.complete("hi", case_id="c1")
        b = await c.complete("hi", case_id="c1")
        assert (a["cached"], b["cached"]) == (False, True)
        assert h.n == 1
        assert c.counters.calls == 2 and c.counters.network_calls == 1 and c.counters.cached_calls == 1
        rows = c.log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(rows) == 2  # cached calls are logged too
        assert '"prompt_hash"' in rows[0] and '"raw_response": "pong"' in rows[0]


async def test_nonce_bypasses_cache(tmp_path):
    h = Counting([httpx.Response(200, json=groq_ok())])
    async with LLMClient(make_settings(tmp_path), "judge", transport=httpx.MockTransport(h)) as c:
        await c.complete("hi")
        await c.complete("hi", nonce="retest-1")
        await c.complete("hi", nonce="retest-1")  # same nonce -> hit
        await c.complete("hi", nonce="retest-2")
        assert h.n == 3


async def test_backoff_honours_retry_after(tmp_path):
    h = Counting([httpx.Response(429, headers={"retry-after": "7"}), httpx.Response(200, json=groq_ok())])
    sleeps: list[float] = []

    async def fake_sleep(d):
        sleeps.append(d)

    async with LLMClient(make_settings(tmp_path), "judge", transport=httpx.MockTransport(h), sleep=fake_sleep) as c:
        r = await c.complete("hi")
    assert r["text"] == "pong" and h.n == 2
    assert 7.0 <= sleeps[0] <= 7.5


async def test_non_retryable_raises_immediately(tmp_path):
    h = Counting([httpx.Response(401, text="bad key")])
    async with LLMClient(make_settings(tmp_path), "judge", transport=httpx.MockTransport(h)) as c:
        with pytest.raises(LLMError):
            await c.complete("hi")
    assert h.n == 1


async def test_gives_up_after_max_retries(tmp_path):
    h = Counting([httpx.Response(503)])

    async def nosleep(_):
        pass

    async with LLMClient(make_settings(tmp_path), "judge", transport=httpx.MockTransport(h), sleep=nosleep) as c:
        with pytest.raises(LLMError):
            await c.complete("hi")
    assert h.n == 4


async def test_gemini_parsing(tmp_path):
    body = {
        "candidates": [{"content": {"parts": [{"text": "pong"}]}}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "thoughtsTokenCount": 4},
    }
    h = Counting([httpx.Response(200, json=body)])
    async with LLMClient(make_settings(tmp_path), "generator", transport=httpx.MockTransport(h)) as c:
        r = await c.complete("hi")
    assert (r["text"], r["tokens_in"], r["tokens_out"]) == ("pong", 5, 6)


def test_family_guard_blocks_same_family(tmp_path):
    s = make_settings(tmp_path, judge_provider="gemini", judge_model="gemini-3.8-flash")
    with pytest.raises(FamilyGuardError):
        s.enforce_family_guard()
    with pytest.raises(FamilyGuardError):
        LLMClient(s, "judge")


def test_family_guard_override_warns(tmp_path, caplog):
    s = make_settings(tmp_path, judge_provider="gemini", judge_model="gemini-3.8-flash", allow_same_family=True)
    with caplog.at_level("WARNING"):
        assert s.enforce_family_guard() == ("google", "google")
    assert "ALLOW_SAME_FAMILY" in caplog.text


def test_family_guard_allows_different_family(tmp_path):
    assert make_settings(tmp_path).enforce_family_guard() == ("google", "openai")


def test_unknown_family_refused(tmp_path):
    with pytest.raises(FamilyGuardError):
        make_settings(tmp_path, judge_model="acme/mystery-9b").enforce_family_guard()


def test_family_of():
    assert family_of("openai/gpt-oss-120b") == "openai"
    assert family_of("meta-llama/llama-4") == "meta"


def test_gen_and_judge_settings_are_independent(tmp_path):
    s = make_settings(tmp_path, gen_temperature=0.7, judge_temperature=0.0, judge_concurrency=3, gen_concurrency=1)
    assert (s.gen_temperature, s.judge_temperature, s.gen_concurrency, s.judge_concurrency) == (0.7, 0.0, 1, 3)
