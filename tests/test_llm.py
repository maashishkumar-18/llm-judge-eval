import httpx
import pytest

from judge.config import FamilyGuardError, Settings, family_of
from judge.llm import LLMClient, LLMError


def make_settings(tmp_path, **over) -> Settings:
    base = dict(
        _env_file=None,
        gemini_api_key="g-test", groq_api_key="q-test",
        gen_provider="groq", gen_model="openai/gpt-oss-20b", gen_temperature=0.0,
        judge_provider="gemini", judge_model="gemini-3.5-flash-lite", judge_temperature=0.0,
        judge_rpm_limit=1000, judge_daily_cap=450, ledger_path=str(tmp_path / "ledger.json"),
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


def gemini_ok(text="pong", p=5, c=1, th=4):
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {"promptTokenCount": p, "candidatesTokenCount": c, "thoughtsTokenCount": th},
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
    async with LLMClient(make_settings(tmp_path), "generator", run_id="t", transport=httpx.MockTransport(h)) as c:
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
    async with LLMClient(make_settings(tmp_path), "generator", transport=httpx.MockTransport(h)) as c:
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

    async with LLMClient(make_settings(tmp_path), "generator", transport=httpx.MockTransport(h), sleep=fake_sleep) as c:
        r = await c.complete("hi")
    assert r["text"] == "pong" and h.n == 2
    assert 7.0 <= sleeps[0] <= 7.5


async def test_non_retryable_raises_immediately(tmp_path):
    h = Counting([httpx.Response(401, text="bad key")])
    async with LLMClient(make_settings(tmp_path), "generator", transport=httpx.MockTransport(h)) as c:
        with pytest.raises(LLMError):
            await c.complete("hi")
    assert h.n == 1


async def test_gives_up_after_max_retries(tmp_path):
    h = Counting([httpx.Response(503)])

    async def nosleep(_):
        pass

    async with LLMClient(make_settings(tmp_path), "generator", transport=httpx.MockTransport(h), sleep=nosleep) as c:
        with pytest.raises(LLMError):
            await c.complete("hi")
    assert h.n == 4


async def test_gemini_judge_parsing(tmp_path):
    h = Counting([httpx.Response(200, json=gemini_ok())])
    async with LLMClient(make_settings(tmp_path), "judge", transport=httpx.MockTransport(h)) as c:
        r = await c.complete("hi")
    assert (r["text"], r["tokens_in"], r["tokens_out"]) == ("pong", 5, 5)  # thoughts counted in tokens_out


async def test_generator_strips_reasoning_and_requests_low_effort(tmp_path):
    seen = {}

    def handler(request):
        seen.update(__import__("json").loads(request.content))
        return httpx.Response(200, json=groq_ok("final answer"))

    async with LLMClient(make_settings(tmp_path), "generator", transport=httpx.MockTransport(handler)) as c:
        r = await c.complete("hi", case_id="c1")
        row = c.log_path.read_text(encoding="utf-8")
    assert seen["reasoning_effort"] == "low" and seen["include_reasoning"] is False
    assert r["text"] == "final answer" and r["reasoning"] == ""
    assert "thinking" not in row  # reasoning never reaches the log or the cache


async def test_generator_keeps_reasoning_when_strip_disabled(tmp_path):
    h = Counting([httpx.Response(200, json=groq_ok())])
    s = make_settings(tmp_path, gen_strip_reasoning=False)
    async with LLMClient(s, "generator", transport=httpx.MockTransport(h)) as c:
        r = await c.complete("hi")
    assert r["reasoning"] == "thinking"


async def test_judge_json_mode_off_by_default_and_on_by_flag(tmp_path):
    bodies = []

    def handler(request):
        bodies.append(__import__("json").loads(request.content))
        return httpx.Response(200, json=gemini_ok("{}"))

    schema = {"type": "object"}
    async with LLMClient(make_settings(tmp_path), "judge", transport=httpx.MockTransport(handler)) as c:
        await c.complete("p", response_schema=schema)  # schema ignored while flag is off
    assert "responseMimeType" not in bodies[0]["generationConfig"]
    assert "responseJsonSchema" not in bodies[0]["generationConfig"]
    assert make_settings(tmp_path).judge_json_mode is False

    s_on = make_settings(tmp_path, judge_json_mode=True, cache_dir=str(tmp_path / "cache2"))
    async with LLMClient(s_on, "judge", transport=httpx.MockTransport(handler)) as c:
        await c.complete("p", response_schema=schema)
    gc = bodies[1]["generationConfig"]
    assert gc["responseMimeType"] == "application/json" and gc["responseJsonSchema"] == schema


async def test_json_mode_flag_never_applies_to_generator(tmp_path):
    h = Counting([httpx.Response(200, json=groq_ok())])
    s = make_settings(tmp_path, judge_json_mode=True)
    async with LLMClient(s, "generator", transport=httpx.MockTransport(h)) as c:
        assert c.json_mode is False


async def test_token_ledger_counts_cached_and_network_calls(tmp_path):
    h = Counting([httpx.Response(200, json=gemini_ok(p=10, c=2, th=3))])
    async with LLMClient(make_settings(tmp_path), "judge", transport=httpx.MockTransport(h)) as c:
        await c.complete("a")
        await c.complete("a")  # cached: still counted in the ledger, flagged cached
        await c.complete("b")
        k = c.counters
        assert (k.calls, k.network_calls, k.cached_calls) == (3, 2, 1)
        assert (k.tokens_in, k.tokens_out) == (30, 15)
        rows = [__import__("json").loads(l) for l in c.log_path.read_text(encoding="utf-8").splitlines()]
        assert sum(r["tokens_in"] for r in rows) == k.tokens_in
        assert sum(r["tokens_out"] for r in rows) == k.tokens_out
        assert [r["cached"] for r in rows] == [False, True, False]


def test_family_guard_blocks_same_family(tmp_path):
    s = make_settings(tmp_path, judge_provider="groq", judge_model="openai/gpt-oss-20b")
    with pytest.raises(FamilyGuardError):
        s.enforce_family_guard()
    with pytest.raises(FamilyGuardError):
        LLMClient(s, "judge")


def test_family_guard_override_warns(tmp_path, caplog):
    s = make_settings(tmp_path, judge_provider="groq", judge_model="openai/gpt-oss-20b", allow_same_family=True)
    with caplog.at_level("WARNING"):
        assert s.enforce_family_guard() == ("openai", "openai")
    assert "ALLOW_SAME_FAMILY" in caplog.text


def test_family_guard_allows_different_family(tmp_path):
    assert make_settings(tmp_path).enforce_family_guard() == ("openai", "google")


def test_unknown_family_refused(tmp_path):
    with pytest.raises(FamilyGuardError):
        make_settings(tmp_path, judge_provider="groq", judge_model="acme/mystery-9b").enforce_family_guard()


def test_family_of():
    assert family_of("openai/gpt-oss-120b") == "openai"
    assert family_of("meta-llama/llama-4") == "meta"
    assert family_of("gemini-3.5-flash-lite") == "google"
    assert family_of("gemma-4-31b-it") == "google"  # Gemma is Google family


def test_gen_and_judge_settings_are_independent(tmp_path):
    s = make_settings(tmp_path, gen_temperature=0.7, judge_temperature=0.0, judge_concurrency=3, gen_concurrency=1)
    assert (s.gen_temperature, s.judge_temperature, s.gen_concurrency, s.judge_concurrency) == (0.7, 0.0, 1, 3)


def test_swapped_role_defaults(tmp_path):
    s = Settings(_env_file=None, cache_dir=str(tmp_path), log_dir=str(tmp_path))
    assert (s.gen_provider, s.gen_model, s.gen_reasoning_effort, s.gen_strip_reasoning) == (
        "groq", "openai/gpt-oss-20b", "low", True)
    assert (s.judge_provider, s.judge_model, s.judge_json_mode) == ("gemini", "gemini-3.5-flash-lite", False)
    assert (s.judge_rpm_limit, s.judge_daily_cap, s.judge_concurrency) == (12, 450, 2)
    assert s.enforce_family_guard() == ("openai", "google")


def test_missing_key_names_the_env_variable(tmp_path):
    from judge.config import MissingKeyError

    s = make_settings(tmp_path, gemini_api_key="")
    with pytest.raises(MissingKeyError, match="GEMINI_API_KEY"):
        s.api_key_for("gemini")
    with pytest.raises(MissingKeyError, match="GEMINI_API_KEY"):
        LLMClient(s, "judge")
    assert s.api_key_for("groq") == "q-test"


def test_cli_reports_a_missing_key_without_a_traceback(tmp_path, monkeypatch, capsys):
    from judge import cli
    from judge.config import MissingKeyError

    def boom(_args):
        raise MissingKeyError("no API key configured for provider 'groq': set GROQ_API_KEY in .env")

    monkeypatch.setattr(cli, "cmd_budget", boom)
    with pytest.raises(SystemExit) as e:
        cli.main(["budget"])
    assert e.value.code == 2
    assert "GROQ_API_KEY" in capsys.readouterr().err
