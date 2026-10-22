import json

import httpx
import pytest

from judge.budget import BudgetExceeded, load_budget
from judge.checkpoint import run_checkpointed
from judge.config import Settings
from judge.ledger import DailyCapReached, ExperimentCapReached, JudgeLedger, RateLimiter
from judge.llm import LLMClient


def settings(tmp_path, **over) -> Settings:
    base = dict(
        _env_file=None, gemini_api_key="g", groq_api_key="q",
        cache_dir=str(tmp_path / "cache"), log_dir=str(tmp_path / "logs"),
        ledger_path=str(tmp_path / "ledger.json"), judge_rpm_limit=1000,
        backoff_base_seconds=0.01, backoff_max_seconds=0.02,
    )
    base.update(over)
    return Settings(**base)


def gemini_ok():
    return httpx.Response(200, json={
        "candidates": [{"content": {"parts": [{"text": "{}"}]}}],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
    })


# ---- ledger ----
def test_ledger_persists_and_caps(tmp_path):
    day = ["2026-10-23"]
    L = JudgeLedger(tmp_path / "l.json", 3, today=lambda: day[0])
    for _ in range(3):
        L.reserve()
    assert L.used_today() == 3
    with pytest.raises(DailyCapReached):
        L.reserve()
    # persistence: a brand-new object over the same file still sees the usage
    assert JudgeLedger(tmp_path / "l.json", 3, today=lambda: day[0]).used_today() == 3
    # midnight Pacific rollover resets the daily count
    day[0] = "2026-10-24"
    assert L.used_today() == 0
    L.reserve()
    assert L.used_today() == 1


def test_ledger_experiment_cap(tmp_path):
    L = JudgeLedger(tmp_path / "l.json", 100, {"probes": 2})
    L.reserve("probes")
    L.reserve("probes")
    with pytest.raises(ExperimentCapReached):
        L.reserve("probes")
    assert L.used_today() == 2  # the refused call was not counted
    L.reserve("other")  # experiments without a cap are only bound by the daily cap


async def test_client_counts_only_noncached_judge_calls_and_stops_at_cap(tmp_path):
    n = {"http": 0}

    def handler(request):
        n["http"] += 1
        return gemini_ok()

    s = settings(tmp_path, judge_daily_cap=2)
    async with LLMClient(s, "judge", transport=httpx.MockTransport(handler)) as c:
        await c.complete("a")
        await c.complete("a")  # cached -> not counted
        await c.complete("a")  # cached -> not counted
        assert c.ledger.used_today() == 1
        await c.complete("b")
        assert c.ledger.used_today() == 2
        with pytest.raises(DailyCapReached):
            await c.complete("c")  # at the cap: refused BEFORE any request
        assert n["http"] == 2
        r = await c.complete("a")  # cache still serves at the cap
        assert r["cached"] is True
        # nonce bypass is a real call, so it is refused too
        with pytest.raises(DailyCapReached):
            await c.complete("a", nonce="retest-1")
        assert c.ledger.used_today() == 2


async def test_cap_survives_new_client_instance(tmp_path):
    s = settings(tmp_path, judge_daily_cap=1)
    async with LLMClient(s, "judge", transport=httpx.MockTransport(lambda r: gemini_ok())) as c:
        await c.complete("a")
    async with LLMClient(s, "judge", transport=httpx.MockTransport(lambda r: gemini_ok())) as c2:
        with pytest.raises(DailyCapReached):
            await c2.complete("new prompt")


async def test_generator_calls_never_touch_the_ledger(tmp_path):
    s = settings(tmp_path, judge_daily_cap=1)
    body = {"choices": [{"message": {"content": "x"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    async with LLMClient(s, "generator", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))) as c:
        for i in range(3):
            await c.complete(f"p{i}")
        assert c.ledger is None


async def test_experiment_cap_enforced_by_client_and_unknown_experiment_rejected(tmp_path):
    s = settings(tmp_path)
    with pytest.raises(KeyError):
        LLMClient(s, "judge", experiment="typo_experiment")
    cap = load_budget().get("clustering_naive").cap
    async with LLMClient(s, "judge", experiment="clustering_naive", transport=httpx.MockTransport(lambda r: gemini_ok())) as c:
        for i in range(cap):
            await c.complete(f"p{i}")
        with pytest.raises(ExperimentCapReached):
            await c.complete("one too many")


async def test_judge_default_max_tokens_is_generous(tmp_path):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return gemini_ok()

    async with LLMClient(settings(tmp_path), "judge", transport=httpx.MockTransport(handler)) as c:
        await c.complete("a")
    assert seen["generationConfig"]["maxOutputTokens"] == 8192


# ---- limiter (fake clock) ----
async def test_rate_limiter_blocks_the_13th_request_until_window_slides():
    now = [0.0]
    slept: list[float] = []

    async def fake_sleep(d):
        slept.append(d)
        now[0] += d

    L = RateLimiter(12, clock=lambda: now[0], sleep=fake_sleep)
    for _ in range(12):
        await L.acquire()
        now[0] += 1.0  # 12 requests over 12 s
    assert slept == []
    await L.acquire()  # 13th: oldest stamp is t=0, window ends t=60
    assert slept and abs(sum(slept) - (60.0 - 12.0)) < 1e-9
    assert now[0] == 60.0


# ---- budget ----
def test_budget_totals_match_the_lean_plan():
    b = load_budget()
    planned = {n: e.planned for n, e in b.experiments.items()}
    assert planned == {
        "pointwise_mitigated": 40, "pairwise_mitigated": 40, "pairwise_naive": 24, "controls": 12,
        "probes": 54, "sensitivity": 24, "clustering_naive": 12, "self_preference": 12, "test_retest": 20, "kappa_adversarial": 0,
    }
    assert b.total_planned == 238
    assert b.validate() == []  # every plan fits inside its own cap


def test_budget_refuses_run_over_cap():
    b = load_budget()
    b.check_run("pointwise_mitigated", 40)
    b.check_run("pointwise_mitigated", 6, spent=40)  # exactly at cap 46
    with pytest.raises(BudgetExceeded):
        b.check_run("pointwise_mitigated", 7, spent=40)
    with pytest.raises(BudgetExceeded):
        b.check_run("kappa_adversarial", 1)  # reuses verdicts: zero calls allowed


# ---- checkpoint/resume ----
async def test_run_stops_at_cap_then_resumes_from_checkpoint(tmp_path):
    L = JudgeLedger(tmp_path / "l.json", 3)
    calls: list[str] = []

    async def fn(key):
        L.reserve()
        calls.append(key)
        return {"v": key.upper()}

    keys = [f"c{i}" for i in range(5)]
    ck = tmp_path / "ck.jsonl"
    st = await run_checkpointed(keys, fn, ck, concurrency=1)
    assert st.ran == 3 and not st.complete and "daily judge cap" in st.stopped_reason
    assert st.remaining == ["c3", "c4"]

    L.daily_cap = 10  # "next day"
    st2 = await run_checkpointed(keys, fn, ck, concurrency=1)
    assert st2.skipped == 3 and st2.ran == 2 and st2.complete
    assert calls == keys  # nothing was redone
    assert st2.results["c4"] == {"v": "C4"}
