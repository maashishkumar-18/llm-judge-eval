import asyncio
import json
from pathlib import Path

import pytest

from judge.config import FamilyGuardError
from judge.llm import LLMClient
from judge.pairwise import TIE, OrderResult, PairResult, judge_pairs
from judge.probes import BASES, build_wrong
from judge.schemas import Case
from judge.selfpref import (
    FACTS,
    GEMINI,
    GPT,
    clopper_pearson,
    correct_by_reference,
    gemini_generator_settings,
    normalize,
    select_pairs,
    summarize,
)
from tests.test_llm import make_settings
from tests.test_pairwise import FakeJudge

SUITE = [Case(**c) for c in json.loads(Path("data/outputs/A.json").read_text(encoding="utf-8"))]
REF = [c for c in SUITE if c.reference_based]
BY_ID = {c.id: c for c in SUITE}


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- family guard (the structural mitigation)

def test_guard_passes_for_the_project_pair_and_the_selfpref_generator(tmp_path):
    assert make_settings(tmp_path).enforce_family_guard() == ("openai", "google")  # gpt-oss generates, Gemini judges
    gs = gemini_generator_settings(make_settings(tmp_path))
    assert (gs.gen_provider, gs.gen_model) == ("gemini", "gemini-3.5-flash-lite")
    assert gs.enforce_family_guard() == ("google", "openai")  # no ALLOW_SAME_FAMILY needed
    assert not gs.allow_same_family


def test_guard_refuses_a_gemini_or_gemma_generator_with_the_gemini_judge(tmp_path):
    for model in ("gemini-3.5-flash-lite", "gemma-4-31b-it"):
        s = make_settings(tmp_path, gen_provider="gemini", gen_model=model)
        with pytest.raises(FamilyGuardError, match="self-enhancement"):
            s.enforce_family_guard()
        with pytest.raises(FamilyGuardError):
            LLMClient(s, "judge")  # the guard runs at every client construction


def test_guard_refuses_an_openai_judge_and_unknown_families(tmp_path):
    with pytest.raises(FamilyGuardError):
        make_settings(tmp_path, judge_provider="groq", judge_model="openai/gpt-oss-120b").enforce_family_guard()
    with pytest.raises(FamilyGuardError, match="cannot determine"):
        make_settings(tmp_path, judge_model="acme/mystery-9b").enforce_family_guard()


def test_swapped_generator_client_constructs_and_only_generates(tmp_path):
    gs = gemini_generator_settings(make_settings(tmp_path))

    async def go():
        async with LLMClient(gs, "generator", log_name="x.jsonl") as c:
            return c.provider, c.model, c.ledger, c.limiter

    provider, model, ledger, limiter = run(go())
    assert (provider, model) == ("gemini", "gemini-3.5-flash-lite")
    assert ledger is None and limiter is None  # generator side: not in the judge ledger


# ---------------------------------------------------------------- Clopper-Pearson (hand-computed)

def test_clopper_pearson_known_values():
    assert clopper_pearson(6, 6) == (pytest.approx(0.025 ** (1 / 6), abs=1e-4), 1.0)  # 0.5407
    assert clopper_pearson(0, 6) == (0.0, pytest.approx(1 - 0.025 ** (1 / 6), abs=1e-4))  # 0.4593
    assert clopper_pearson(3, 6) == (pytest.approx(0.1181, abs=1e-4), pytest.approx(0.8819, abs=1e-4))
    # k=2, n=3: lower solves 3p^2 - 2p^3 = 0.025, upper is 0.975^(1/3)
    lo, hi = clopper_pearson(2, 3)
    assert lo == pytest.approx(0.0943, abs=1e-4) and hi == pytest.approx(0.975 ** (1 / 3), abs=1e-4)
    assert clopper_pearson(1, 1) == (pytest.approx(0.025, abs=1e-4), 1.0)
    assert clopper_pearson(0, 0) is None
    with pytest.raises(ValueError):
        clopper_pearson(4, 3)


# ---------------------------------------------------------------- deterministic correctness

def test_gpt_oss_answers_known_wrong_are_rejected_and_the_rest_accepted():
    ok = {c.id: correct_by_reference(c, c.model_output) for c in REF}
    assert ok["c09"] is False and ok["c16"] is False  # A said "yes" / "not stated"
    assert [i for i, v in ok.items() if not v] == ["c09", "c16"]


def test_reference_text_passes_its_own_fact_check():
    for c in REF:
        assert correct_by_reference(c, c.expected_output) or c.id in {"c43"}, c.id  # reference may break the case's own limit
    assert correct_by_reference(BY_ID["c43"], BY_ID["c43"].model_output)


@pytest.mark.parametrize("cid", BASES)
def test_the_j6_confidently_wrong_answers_fail_the_check(cid):
    assert not correct_by_reference(BY_ID[cid], build_wrong(cid))


def test_constraint_violation_makes_a_right_answer_incorrect():
    c = BY_ID["c24"]  # two sentences max, no "very"
    assert correct_by_reference(c, "The beam is cracked: a cracked support beam. It reopens in August.")
    assert not correct_by_reference(c, "A cracked support beam closed it. It is very likely to reopen in August.")
    assert not correct_by_reference(c, "A cracked support beam. Reopening in August. Thanks. Bye.")
    assert not correct_by_reference(c, "")


def test_every_reference_case_has_a_fact_check_and_normalize_straightens_quotes():
    assert {c.id for c in REF} <= set(FACTS)
    assert normalize("Doesn’t  say so") == "doesn't say so"


# ---------------------------------------------------------------- selection rule

def test_select_pairs_audit_and_first_n():
    cs = [BY_ID[i] for i in ("c19", "c24", "c01", "c09", "c44")]
    gem = {
        "c19": "The late registration fee today is 45.",  # identical to gpt-oss -> dropped
        "c24": "Inspectors found a cracked support beam; the bridge should reopen in August.",
        "c01": "Fine: 10 days at 0.25 is 2.50. Renewals: up to 2 times.",
        "c09": "No, walk-ins stop at 12:30.",  # correct, but the gpt-oss answer is wrong
        "c44": "There are 124 seats right now.",  # wrong
    }
    sel, audit = select_pairs(cs, gem, n=1)
    out = {a["case_id"]: a["outcome"] for a in audit}
    assert sel == ["c24"]
    assert out == {
        "c19": "identical text", "c24": "selected", "c01": "eligible but beyond the first 1",
        "c09": "gpt-oss answer incorrect", "c44": "gemini answer incorrect",
    }
    assert select_pairs(cs, gem, n=6)[0] == ["c24", "c01"]
    assert select_pairs(cs, {}, n=6)[0] == []  # no Gemini answers -> nothing selected


# ---------------------------------------------------------------- summary (hand-computed)

def pair(cid, verdict, flipped=False):
    mk = lambda f, s, w: OrderResult(f, s, "mitigated", "t", "s", "strict", position_winner=None if w is None else TIE if w == TIE else "first" if w == f else "second", winner=w)  # noqa: E731
    w1 = {GEMINI: GEMINI, GPT: GPT, TIE: TIE}[verdict]
    return PairResult(cid, "mitigated", GPT, GEMINI, False, [mk(GPT, GEMINI, w1), mk(GEMINI, GPT, w1)], verdict, "both_orders", flipped, False, verdict)


def test_summary_hand_computed():
    ps = [pair("a", GEMINI), pair("b", GEMINI), pair("c", GPT), pair("d", TIE), pair("e", TIE), pair("f", TIE)]
    s = summarize(ps)
    assert (s["gemini_wins"], s["gpt_oss_wins"], s["ties"]) == (2, 1, 3)
    assert s["own_family_win_rate"] == pytest.approx((2 + 1.5) / 6, abs=1e-4)  # 0.5833
    assert s["n_decisive"] == 3 and s["decisive_gemini_share"] == pytest.approx(2 / 3, abs=1e-4)
    lo, hi = s["ci95_clopper_pearson_decisive"]
    assert lo == pytest.approx(0.0943, abs=1e-4) and hi == pytest.approx(0.9916, abs=1e-4)
    assert s["single_order_judgements"] == {"n": 12, "gemini": 4, "gpt_oss": 2, "tie": 6}


def test_summary_all_ties_has_no_interval_and_half_win_rate():
    s = summarize([pair("a", TIE), pair("b", TIE)])
    assert s["own_family_win_rate"] == 0.5 and s["n_decisive"] == 0 and s["ci95_clopper_pearson_decisive"] is None


# ---------------------------------------------------------------- end to end with a fake judge

def _cases():
    base = BY_ID["c19"]
    x = base.model_copy(update={"model_output": "GPT-ish answer 45"})
    y = base.model_copy(update={"model_output": "GEM-ish answer 45"})
    return [x], [y]


def test_order_swap_removes_a_position_only_judge(tmp_path):
    xs, ys = _cases()
    fake = FakeJudge(lambda a, b: "A")  # always the first-listed answer
    got, left = run(judge_pairs(fake, xs, ys, ["c19"], mode="mitigated", checkpoint_path=tmp_path / "cp.jsonl", x_label=GPT, y_label=GEMINI))
    assert not left and got[0].verdict == TIE and got[0].flipped and got[0].hard_flip
    s = summarize(got)
    assert s["own_family_win_rate"] == 0.5 and s["ci95_clopper_pearson_decisive"] is None


def test_a_judge_that_prefers_gemini_text_is_detected_in_both_orders(tmp_path):
    xs, ys = _cases()
    fake = FakeJudge(lambda a, b: "A" if "GEM" in a else "B" if "GEM" in b else "TIE")
    got, _ = run(judge_pairs(fake, xs, ys, ["c19"], mode="mitigated", checkpoint_path=tmp_path / "cp.jsonl", x_label=GPT, y_label=GEMINI))
    assert got[0].verdict == GEMINI and not got[0].flipped
    s = summarize(got)
    assert s["own_family_win_rate"] == 1.0 and s["ci95_clopper_pearson_decisive"] == (pytest.approx(0.025, abs=1e-4), 1.0)
    assert len(fake.prompts) == 2  # both orders, nothing else
