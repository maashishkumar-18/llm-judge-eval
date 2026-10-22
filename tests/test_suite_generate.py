import json

import httpx
import yaml

from judge.checks import complies, run_checks, validate_spec
from judge.generate import generate_config
from judge.llm import LLMClient
from judge.suite import CANDIDATES_PATH, SUITE_PATH, GenConfig, load_candidates, load_config, summarize
from tests.test_llm import groq_ok, make_settings


def test_candidate_pool_shape():
    rows = load_candidates(CANDIDATES_PATH, include_dropped=True)  # also validates every check spec
    assert summarize(rows) == {"cases": 44, "reference_based": 31, "reference_free": 13}
    assert all(r["system_prompt"].strip() for r in rows)
    text = " ".join(r["input"] for r in rows)
    assert "Northwind" not in text and "30 days of delivery" not in text  # anchors must not leak into the suite


def test_checks_hand_computed():
    assert not complies("One. Two. Three.", [{"type": "max_sentences", "n": 2}])
    assert complies("The fine is 2.50 (10 days). It can be renewed twice.", [{"type": "max_sentences", "n": 2}])  # 2.50 is not a stop
    assert complies("- a\n- b\n- c", [{"type": "exact_bullets", "n": 3}])
    assert not complies("- a\n- b\nc", [{"type": "exact_bullets", "n": 3}])
    assert complies('{"name": "x", "date": "d", "amount": 1}', [{"type": "json_keys", "keys": ["name", "date", "amount"]}])
    assert not complies('{"name": "x"}', [{"type": "json_keys", "keys": ["name", "date"]}])
    assert not complies("```json\n{}\n```", [{"type": "json_keys", "keys": []}])  # a fence is not bare JSON
    assert complies("billing", [{"type": "one_of", "options": ["billing", "other"]}])
    assert not complies("Billing.", [{"type": "one_of", "options": ["billing", "other"]}])
    assert not complies("a very good day", [{"type": "must_not_include", "words": ["very"]}])
    assert complies("a 350 ml bottle", [{"type": "must_include", "words": ["350 ml"]}])  # narrow no-break space
    assert complies("every day", [{"type": "must_not_include", "words": ["very"]}])  # whole-word match only
    assert [r.passed for r in run_checks("hello world", [{"type": "min_words", "n": 2}, {"type": "max_words", "n": 1}])] == [True, False]


def test_bad_check_spec_rejected():
    for bad in ({"type": "nope"}, {"type": "max_words"}, {"type": "max_words", "n": 3, "extra": 1}):
        try:
            validate_spec(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


async def test_generate_prompt_assembly_resume_and_files(tmp_path):
    rows = [
        {"id": "x1", "system_prompt": "Reply in one word.", "input": "Say hi", "expected_output": "hi",
         "meta": {"kind": "format", "checks": [{"type": "max_words", "n": 1}]}},
        {"id": "x2", "system_prompt": "Reply in one word.", "input": "Say bye",
         "meta": {"kind": "format", "checks": [{"type": "max_words", "n": 1}]}},
    ]
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=groq_ok("two words"))

    cfg = GenConfig(name="T", prompt_version="v9", preamble="PREAMBLE")
    s = make_settings(tmp_path)
    async with LLMClient(s, "generator", run_id="t", transport=httpx.MockTransport(handler)) as c:
        cases, st = await generate_config(s, rows, cfg, tmp_path / "out", client=c)
        assert (st.ran, st.skipped) == (2, 0)
        sys_msg = seen[0]["messages"][0]
        assert sys_msg["role"] == "system" and sys_msg["content"] == "PREAMBLE\n\nReply in one word."
        n = len(seen)
        _, st2 = await generate_config(s, rows, cfg, tmp_path / "out", client=c)
        assert (st2.ran, st2.skipped) == (0, 2) and len(seen) == n  # resumed, no new calls
    saved = json.loads((tmp_path / "out" / "T.json").read_text(encoding="utf-8"))
    assert [d["id"] for d in saved] == ["x1", "x2"] and saved[0]["expected_output"] == "hi" and saved[1]["expected_output"] is None
    assert saved[0]["model_output"] == "two words"  # reasoning field dropped, content only
    assert saved[0]["system_prompt"] == "Reply in one word."  # the judge sees the case prompt, not the preamble
    assert saved[0]["meta"]["config"] == "T" and saved[0]["meta"]["constraint_results"][0]["passed"] is False


def test_locked_suite_matches_curation():
    locked = load_candidates(SUITE_PATH)
    pool = load_candidates(CANDIDATES_PATH)  # drops meta.drop cases
    assert [r["id"] for r in locked] == [r["id"] for r in pool]
    assert summarize(locked) == {"cases": 20, "reference_based": 13, "reference_free": 7}
    probes = [r["id"] for r in locked if r["meta"].get("probe_base")]
    assert probes == ["c01", "c09", "c16", "c19", "c24", "c44"]
    assert {"c03", "c05", "c26", "c27"}.isdisjoint(r["id"] for r in locked)  # swapped out for c41-c44
    assert [r["id"] for r in locked if r["meta"].get("temptation")] == ["c41", "c42", "c43", "c44"]
    assert all(r.get("expected_output") for r in locked if r["id"] in probes)


def test_exact_match_hand_computed():
    from judge.checks import exact_match

    assert exact_match("Billing.", "billing")
    assert exact_match("a b", "A  B")  # narrow no-break space, case, whitespace
    assert exact_match('{"a":1, "b":2}', '{"b": 2, "a": 1}')  # JSON compared parsed
    assert not exact_match("The fine is 2.50.", "The fine is 2.50. Renew twice.")
    assert not exact_match("400", "400 euros")


def test_config_c_uses_config_a_prompt():
    a, c = load_config("A"), load_config("C")
    assert c.preamble == a.preamble and c.prompt_version == a.prompt_version
    assert c.gen_model == "qwen/qwen3.8-27b" and a.gen_model is None


async def test_qwen_sends_reasoning_params_and_strips_think(tmp_path):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "<think>hmm</think>\n11:05", "reasoning": "x"}}],
                                         "usage": {"prompt_tokens": 5, "completion_tokens": 9}})

    s = make_settings(tmp_path, gen_model="qwen/qwen3.8-27b")
    async with LLMClient(s, "generator", run_id="q", transport=httpx.MockTransport(handler)) as c:
        r = await c.complete("t", system="s", max_tokens=600)
    assert r["text"] == "11:05" and r["reasoning"] == ""
    assert seen[0]["reasoning_effort"] == "low" and seen[0]["reasoning_format"] == "hidden"
    assert "include_reasoning" not in seen[0] and seen[0]["max_completion_tokens"] == 600
