import json

import httpx
import pytest

from judge.config import Settings
from judge.llm import LLMClient
from judge.pointwise import PointwiseResult, judge_case, judge_suite, replay_suite, summarize_pointwise
from judge.replay import ReplayClient, ReplayMiss, read_log
from judge.schemas import Case, load_rubric

from .corpus import GOOD_JSON, crit, good

ANSWER = "The fine is $0.25 x 10 = $2.50."
CASE = Case(
    id="t1", input="How much is a 10-day fine?", system_prompt="Answer in one sentence.",
    model_output=ANSWER, expected_output="$2.50",
)
FREE = Case(id="t2", input="Summarise.", system_prompt="Two bullets.", model_output="- one\n- two")


def settings(tmp_path, **over) -> Settings:
    base = dict(
        _env_file=None, gemini_api_key="g", groq_api_key="q", judge_rpm_limit=1000,
        cache_dir=str(tmp_path / "cache"), log_dir=str(tmp_path / "logs"), ledger_path=str(tmp_path / "ledger.json"),
        backoff_base_seconds=0.001, backoff_max_seconds=0.002, max_retries=2,
    )
    base.update(over)
    return Settings(**base)


def gemini(text: str, tin=100, tout=50):
    return httpx.Response(200, json={
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {"promptTokenCount": tin, "candidatesTokenCount": tout},
    })


class Script:
    """MockTransport handler: replies from a queue (str -> Gemini text, Response -> as is) and records bodies."""

    def __init__(self, *replies):
        self.replies, self.bodies = list(replies), []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        r = self.replies.pop(0)
        return r if isinstance(r, httpx.Response) else gemini(r)

    def prompt(self, i: int) -> str:
        return self.bodies[i]["contents"][0]["parts"][0]["text"]


def client(tmp_path, script, log_name="j.jsonl", **over):
    async def nosleep(_):
        return None

    return LLMClient(
        settings(tmp_path, **over), "judge", log_name=log_name, transport=httpx.MockTransport(script), sleep=nosleep
    )


async def test_clean_reply_is_strict_and_overall_is_recomputed_in_code(tmp_path):
    # the judge claims overall 1.0 and a wrong pass flag; both are ignored. 5/4/3 -> .4*5+.35*4+.25*3 = 4.15
    reply = json.dumps({**good(5, 4, 3, evidence="$2.50"), "overall": 1.0, "pass": False})
    script = Script(reply)
    async with client(tmp_path, script) as c:
        r = await judge_case(c, CASE, label="A")
    assert (r.parse_status, r.error_kind, r.repairs) == ("strict", "", [])
    assert r.scores == {"faithfulness": 5, "instruction_following": 4, "completeness": 3}
    assert r.overall == pytest.approx(4.15) and r.passed is True
    assert r.mode == "reference" and r.template_id == "pointwise_ref.v1" and len(script.bodies) == 1
    assert r.unsupported_high_scores == []  # "$2.50" is a verbatim span of the answer


async def test_reference_free_case_uses_the_free_template(tmp_path):
    async with client(tmp_path, Script(GOOD_JSON)) as c:
        r = await judge_case(c, FREE)
    assert r.mode == "free" and r.template_id == "pointwise_free.v1"


async def test_gpt_oss_style_reasoning_before_the_json_is_extracted(tmp_path):
    reply = "We need to grade. The reference is $2.50 and the answer matches {evidence}. Now output.\n" + GOOD_JSON
    async with client(tmp_path, Script(reply)) as c:
        r = await judge_case(c, CASE)
    assert r.parse_status == "extract" and len(r.calls) == 1


async def test_repairable_reply_is_status_repair_with_the_stage_recorded(tmp_path):
    reply = GOOD_JSON.replace('"score": 3\n  }', '"score": 3,\n  }')
    async with client(tmp_path, Script(reply)) as c:
        r = await judge_case(c, CASE)
    assert r.parse_status == "repair" and r.repairs == ["comments_trailing_commas"]


async def test_bad_score_triggers_one_reask_that_quotes_the_error(tmp_path):
    script = Script(json.dumps(good(9, 4, 3)), GOOD_JSON)
    async with client(tmp_path, script) as c:
        r = await judge_case(c, CASE, label="A")
    assert r.parse_status == "reask" and r.scores["faithfulness"] == 5
    assert [x.tag for x in r.calls] == ["t1|A", "t1|A|reask"]
    assert "faithfulness.score" in script.prompt(1) and "# Correction needed" in script.prompt(1)
    assert script.prompt(1).startswith(script.prompt(0))  # the re-ask restates the whole original task
    rows = read_log(tmp_path / "logs" / "j.jsonl")
    assert [x["case_id"] for x in rows] == ["t1|A", "t1|A|reask"]  # the re-ask is logged like any judge call


async def test_unparseable_after_reask_is_invalid_counted_and_kept(tmp_path):
    script = Script("I cannot do this.", "Still no.", GOOD_JSON)
    async with client(tmp_path, script) as c:
        bad = await judge_case(c, CASE)
        ok = await judge_case(c, FREE)
    assert bad.parse_status == "invalid" and bad.error_kind == "parse" and bad.scores is None
    assert len(bad.calls) == 2 and "after re-ask" in bad.error  # exactly one re-ask, no loop
    s = summarize_pointwise([bad, ok])
    assert s["n"] == 2 and s["n_invalid"] == 1 and s["n_scored"] == 1 and s["parse_status"]["invalid"] == 1
    assert s["pass_rate_scored"] == 1.0 and s["pass_rate_invalid_as_fail"] == 0.5  # INVALID is never silently dropped
    assert s["reasks"] == 1 and s["judge_calls"] == 3


async def test_exhausted_judge_call_is_an_invalid_call_error_not_a_crash(tmp_path):
    script = Script(httpx.Response(503, text="busy"), httpx.Response(503, text="busy"))
    async with client(tmp_path, script) as c:
        r = await judge_case(c, CASE)
    assert r.parse_status == "invalid" and r.error_kind == "call" and "503" in r.error


async def test_high_score_without_a_verbatim_quote_is_flagged(tmp_path):
    reply = json.dumps({
        "faithfulness": crit(5, evidence="The fine is two dollars fifty"),  # paraphrase, not in the answer
        "instruction_following": crit(4, evidence=ANSWER),
        "completeness": crit(2, evidence="NONE"),  # a low score may have no quote
    })
    async with client(tmp_path, Script(reply)) as c:
        r = await judge_case(c, CASE)
    assert r.unsupported_high_scores == ["faithfulness"]


def test_summary_hand_computed():
    rb = load_rubric()

    def res(cid, f, i, c, status="strict"):
        sc = {"faithfulness": f, "instruction_following": i, "completeness": c}
        return PointwiseResult(cid, "A", "free", "t", "s", "v", status, scores=sc, overall=rb.overall(sc), passed=rb.passes(sc))

    rs = [res("a", 5, 5, 5), res("b", 4, 3, 3), res("c", 2, 3, 3, "repair"),
          PointwiseResult("d", "A", "free", "t", "s", "v", "invalid", error_kind="parse")]
    s = summarize_pointwise(rs)
    # overall: a 5.0, b .4*4+.35*3+.25*3 = 3.4 (fails 3.5), c .8+1.05+.75 = 2.6 ; mean (5+3.4+2.6)/3
    assert s["overall"]["mean"] == pytest.approx(11.0 / 3, abs=1e-3)
    assert s["pass_rate_scored"] == pytest.approx(1 / 3, abs=1e-3) and s["pass_rate_invalid_as_fail"] == 0.25
    assert s["parse_status"] == {"strict": 2, "extract": 0, "repair": 1, "reask": 0, "invalid": 1}
    assert s["criteria_mean"]["faithfulness"] == pytest.approx(11 / 3, abs=1e-3)


# ---- suite run, checkpoint, replay
async def run_suite(tmp_path, script, cases, **kw):
    async with client(tmp_path, script, log_name="pointwise_T.jsonl") as c:
        return await judge_suite(
            c, cases, "T", out_path=tmp_path / "results" / "pointwise_T.json",
            checkpoint_path=tmp_path / "cp.jsonl", cases_file="unused.json", **kw,
        )


async def test_suite_replay_reproduces_every_verdict_with_zero_api_calls(tmp_path):
    cases = [CASE, FREE, Case(id="t3", input="q3", model_output="a"), Case(id="t4", input="q4", model_output="a")]
    script = Script(
        GOOD_JSON,                                                    # t1 strict
        "Reasoning first. " + json.dumps(good(3, 3, 3)),              # t2 extract
        json.dumps(good(9, 9, 9)), json.dumps(good(2, 2, 2)),         # t3 reask
        "nope", "nope again",                                         # t4 invalid
    )
    doc, remaining = await run_suite(tmp_path, script, cases, concurrency=1)
    assert remaining == []
    assert [r["parse_status"] for r in doc["results"]] == ["strict", "extract", "reask", "invalid"]
    assert doc["summary"]["n_invalid"] == 1 and doc["summary"]["judge_calls"] == 6
    saved = json.loads((tmp_path / "results" / "pointwise_T.json").read_text(encoding="utf-8"))
    assert saved["summary"] == doc["summary"]

    replay = ReplayClient(tmp_path / "logs" / "pointwise_T.jsonl")  # no http client, no keys, nothing to call
    assert not hasattr(replay, "_http")
    rows = await replay_suite(saved, cases, replay)
    assert all(r["same"] for r in rows), rows
    assert [r["replayed_status"] for r in rows] == ["strict", "extract", "reask", "invalid"]
    assert replay.served == 6 and len(script.replies) == 0  # every logged response, both re-asks included


async def test_replay_detects_a_log_that_no_longer_matches(tmp_path):
    doc, _ = await run_suite(tmp_path, Script(GOOD_JSON), [CASE])
    log = tmp_path / "logs" / "pointwise_T.jsonl"
    changed = CASE.model_copy(update={"model_output": "a different answer"})  # prompt differs -> no logged response
    rows = await replay_suite(doc, [changed], ReplayClient(log))
    assert rows[0]["same"] is False and rows[0]["replayed_status"] == "no-log"
    tampered = json.loads(json.dumps(doc))  # a verdict edited after the fact is reported as a diff
    tampered["results"][0]["scores"]["faithfulness"] = 1
    rows = await replay_suite(tampered, [CASE], ReplayClient(log))
    assert rows[0]["same"] is False and "scores" in rows[0]["diffs"]


async def test_replay_client_raises_on_unknown_request(tmp_path):
    (tmp_path / "log.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ReplayMiss):
        await ReplayClient(tmp_path / "log.jsonl").complete("never logged", case_id="x")


async def test_call_failures_are_retried_on_resume_but_parse_failures_are_kept(tmp_path):
    cases = [CASE, FREE]
    first = Script(httpx.Response(503, text="busy"), httpx.Response(503, text="busy"), "garbage", "garbage")
    doc, _ = await run_suite(tmp_path, first, cases, concurrency=1)
    assert {r["case_id"]: r["error_kind"] for r in doc["results"]} == {"t1": "call", "t2": "parse"}
    second = Script(GOOD_JSON)  # only t1 is retried; t2's INVALID is a settled result
    doc, _ = await run_suite(tmp_path, second, cases, concurrency=1)
    kinds = {r["case_id"]: (r["parse_status"], r["error_kind"]) for r in doc["results"]}
    assert kinds == {"t1": ("strict", ""), "t2": ("invalid", "parse")}
    assert len(second.bodies) == 1


async def test_second_run_is_served_from_cache_with_no_new_calls(tmp_path):
    await run_suite(tmp_path, Script(GOOD_JSON), [CASE])
    again = Script()  # would IndexError on any request
    doc, _ = await run_suite(tmp_path, again, [CASE], fresh=True)  # fresh: checkpoint dropped, cache still serves
    assert again.bodies == [] and doc["summary"]["judge_calls_network"] == 0 and doc["summary"]["judge_calls"] == 1
