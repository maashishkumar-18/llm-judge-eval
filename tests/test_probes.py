import asyncio
import json
import re
from pathlib import Path

import pytest

from judge.probes import (
    BASES,
    FILLER,
    NAIVE_FOOLED_MIN,
    SPECS,
    VARIANTS,
    WRAPPERS,
    analyze,
    build_padded,
    build_terse,
    build_variants,
    build_verbose_wrong,
    build_wrong,
    hide_reference,
    is_correct,
    judge_naive_case,
    length_checks_pass,
    overall_noif,
    pick_sample,
    plan_items,
    run_stage,
    spearman,
    states_fact,
    word_count,
)
from judge.schemas import Case

SUITE = {c["id"]: c for c in json.loads(Path("data/outputs/A.json").read_text(encoding="utf-8"))}
BASE_CASES = [Case(**{**SUITE[i], "model_output": SUITE[i]["expected_output"]}) for i in BASES]
ORIGINALS = [Case(**SUITE[i]) for i in BASES]
BY_ID = {c.id: c for c in BASE_CASES}


def test_bases_are_the_suite_probe_bases():
    assert [i for i, c in SUITE.items() if c["meta"].get("probe_base")] == list(BASES)
    assert all(c.reference_based for c in BASE_CASES)


@pytest.mark.parametrize("cid", BASES)
def test_padded_is_the_correct_answer_plus_content_free_filler(cid):
    base, padded = BY_ID[cid], build_padded(BY_ID[cid])
    assert padded.startswith(base.expected_output)
    assert padded.endswith(" ".join(FILLER)) and padded.count(FILLER[0]) == 1
    assert not re.search(r"\d", " ".join(FILLER))  # no number, so no new factual claim
    assert is_correct(padded, cid)  # the facts are untouched
    assert word_count(padded) >= word_count(base.expected_output) + 30  # a real length defect
    assert not length_checks_pass(padded, base)  # padding breaks the case's own length limit (IF drop is expected)
    assert length_checks_pass(base.expected_output, base)  # ... which the unpadded reference respects


@pytest.mark.parametrize("cid", BASES)
def test_confidently_wrong_has_exactly_the_approved_false_fact(cid):
    base, wrong = BY_ID[cid], build_wrong(cid)
    assert wrong.startswith(WRAPPERS[BASES.index(cid) % 2])
    assert states_fact(wrong, cid, "wrong") and not states_fact(wrong, cid, "correct")
    assert not is_correct(wrong, cid)
    assert is_correct(base.expected_output, cid)
    assert length_checks_pass(wrong, base)  # only the fact is wrong: it respects the length limits, so IF is not the defect
    assert wrong != base.expected_output
    assert not re.search(r"\b(?:I think|maybe|perhaps|might|probably)\b", wrong, re.I)  # no hedging


def test_wrapper_alternates_and_lowercases_after_comma():
    assert [build_wrong(c).split()[0] for c in BASES] == ["Definitely:", "Without", "Definitely:", "Without", "Definitely:", "Without"]
    assert build_wrong("c19") == "Without question, the late registration fee today is 30."
    assert build_wrong("c01").startswith("Definitely: The fine is 2.50")
    assert build_wrong("c44").startswith("Without question, it has 124 seats")


@pytest.mark.parametrize("cid", BASES)
def test_terse_is_correct_and_much_shorter(cid):
    base, terse = BY_ID[cid], build_terse(cid)
    assert is_correct(terse, cid)
    assert word_count(terse) <= 0.6 * word_count(base.expected_output)
    assert length_checks_pass(terse, base)


@pytest.mark.parametrize("cid", BASES)
def test_verbose_wrong_is_wrong_and_padded_with_the_same_filler(cid):
    base, vw = BY_ID[cid], build_verbose_wrong(cid)
    assert vw == f"{build_wrong(cid)} {' '.join(FILLER)}"
    assert states_fact(vw, cid, "wrong") and not states_fact(vw, cid, "correct")
    assert word_count(vw) > word_count(build_wrong(cid)) + 30
    assert not length_checks_pass(vw, base)


def test_build_variants_keeps_task_and_reference():
    v = build_variants(BASE_CASES)
    assert len(v) == len(BASES) * len(VARIANTS)
    for (cid, name), case in v.items():
        b = BY_ID[cid]
        assert (case.id, case.input, case.system_prompt, case.expected_output) == (b.id, b.input, b.system_prompt, b.expected_output)
        assert case.meta["probe_variant"] == name


def test_original_correctness_matches_the_j5_scores():
    ok = [c.id for c in ORIGINALS if is_correct(c.model_output, c.id)]
    assert ok == ["c01", "c19", "c24", "c44"]  # c09 said "Yes", c16 said "Not stated": both wrong against the reference


def test_hide_reference_removes_it_from_every_prompt():
    from judge.prompts import render_naive_pointwise, render_pointwise

    case = build_variants(BASE_CASES)[("c24", "wrong")]
    free = hide_reference(case)
    assert free.expected_output is None and not free.reference_based
    ref = case.expected_output
    for rp in (render_pointwise(free), render_naive_pointwise(free)):
        assert ref not in rp.user and ref not in rp.system
    assert "pointwise_free" in render_pointwise(free).template_id
    assert ref in render_pointwise(case).user and ref in render_naive_pointwise(case).user


# ---------------------------------------------------------------- statistics

def test_overall_noif_hand_computed():
    # weights 0.40 faith, 0.25 completeness: (0.40*4 + 0.25*2) / 0.65 = 2.1 / 0.65
    assert overall_noif({"faithfulness": 4, "instruction_following": 1, "completeness": 2}) == pytest.approx(2.1 / 0.65)
    assert overall_noif({"faithfulness": 5, "instruction_following": 1, "completeness": 5}) == pytest.approx(5.0)


def test_spearman_hand_computed_with_ties():
    # ranks x 1..5; ranks y [1, 2, 3.5, 3.5, 5]; cov 9.5, var 10 and 9.5 -> 9.5 / sqrt(95)
    assert spearman([1, 2, 3, 4, 5], [5, 6, 7, 7, 8]) == pytest.approx(9.5 / 95 ** 0.5)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3], [7, 7, 7]) is None and spearman([1, 2], [1, 2]) is None


# ---------------------------------------------------------------- fake judges

def mit_json(f, i, c, quote="q"):
    crit = lambda s: {"evidence": quote, "rationale": "r", "score": s}  # noqa: E731
    return json.dumps({"faithfulness": crit(f), "instruction_following": crit(i), "completeness": crit(c)})


class FakeJudge:
    """Scores by what the prompt shows: mitigated -> fixed table by variant, naive -> a score by variant."""

    model, json_mode = "fake", False

    def __init__(self, table, naive):
        self.table, self.naive, self.prompts = table, naive, []

    async def complete(self, prompt, *, case_id="", system=None, nonce="", **_):
        self.prompts.append(prompt)
        variant = case_id.removesuffix("|reask").split("|")[-1].removeprefix("probe_ref_").removeprefix("probe_free_")
        if "Rate the overall quality" in prompt:
            text = json.dumps({"score": self.naive[variant], "reason": "x"})
        else:
            text = mit_json(*self.table[variant])
        return {"text": text, "tokens_in": 10, "tokens_out": 5, "cached": False}


TABLE = {"original": (5, 5, 4), "terse": (5, 5, 3), "padded": (5, 2, 5), "wrong": (2, 5, 4), "verbose_wrong": (4, 2, 5)}
NAIVE = {"original": 8, "terse": 7, "padded": 9, "wrong": 8, "verbose_wrong": 9}


def run_all(judge, stage="ref", tmp_path=None):
    items = plan_items(BASE_CASES, ORIGINALS, stage)
    got, left = asyncio.run(run_stage(judge, items, stage, checkpoint_path=tmp_path / f"{stage}.jsonl"))
    texts = {k: v[2].model_output for k, v in items.items()}
    return items, got, left, texts


def test_plan_sizes_match_the_budget():
    assert len(plan_items(BASE_CASES, ORIGINALS, "ref")) == 60  # 54 network calls + 6 mitigated originals that are cache hits
    free = plan_items(BASE_CASES, ORIGINALS, "free")
    assert len(free) == 12 and all(not v[2].reference_based for v in free.values())
    assert {v[1] for v in free.values()} == {"wrong"}


def test_ref_free_prompts_never_contain_the_reference(tmp_path):
    judge = FakeJudge(TABLE, NAIVE)
    items, got, left, _ = run_all(judge, "free", tmp_path)
    assert not left and len(judge.prompts) == 12
    for p in judge.prompts:
        assert not any(SUITE[c]["expected_output"] in p for c in BASES)


def test_end_to_end_with_a_fake_judge(tmp_path):
    judge = FakeJudge(TABLE, NAIVE)
    items, got, left, texts = run_all(judge, "ref", tmp_path)
    assert not left and len(got) == 60 and all(r["error_kind"] == "" for r in got.values())
    s = analyze(got, texts)
    assert s["n_invalid"] == 0
    # verbosity: padded - terse. mitigated faithfulness 0, completeness +2, noIF (0.4*0+0.25*2)/0.65, IF -3 (separate)
    pt = s["verbosity"]["mitigated"]["padded_minus_terse"]
    assert pt["faithfulness"]["mean"] == 0 and pt["completeness"]["mean"] == 2
    assert pt["overall_noif"]["mean"] == pytest.approx(0.5 / 0.65, abs=1e-4)
    assert pt["instruction_following"]["mean"] == -3
    assert s["verbosity"]["naive"]["padded_minus_terse"]["naive_score"]["mean"] == 2
    # vs original only where the original is correct: c01 c19 c24 c44
    assert s["verbosity"]["original_correct_on"] == ["c01", "c19", "c24", "c44"]
    assert s["verbosity"]["naive"]["padded_minus_original_where_correct"]["naive_score"]["n"] == 4
    # fooled: mitigated faithfulness 2 on wrong -> 0/6 (a); verbose_wrong noIF (0.4*4+0.25*5)/0.65=4.38 >= terse 4.23 -> 6/6 (b)
    m = s["sycophancy"]["mitigated"]["reference_mode"]
    assert m["a_wrong_answer_accepted"]["fooled"] == 0 and m["b_verbose_wrong_ge_terse_correct"]["fooled"] == 6
    assert m["a_or_b"]["fooled"] == 6 and m["a_or_b"]["n"] == 6
    # naive: wrong scored 8 == NAIVE_FOOLED_MIN -> fooled 6/6 (a); verbose 9 >= terse 7 -> 6/6 (b)
    n = s["sycophancy"]["naive"]["reference_mode"]
    assert NAIVE_FOOLED_MIN == 8 and n["a_wrong_answer_accepted"]["fooled"] == 6 and n["b_verbose_wrong_ge_terse_correct"]["fooled"] == 6
    # the length-score correlation is computed within each quality group
    assert s["verbosity"]["mitigated"]["spearman_words_vs_score"]["correct"]["n"] == 12


def test_reference_free_fooled_counts_separately(tmp_path):
    judge = FakeJudge({**TABLE, "wrong": (4, 5, 5)}, NAIVE)
    _, free, _, ftexts = run_all(judge, "free", tmp_path)
    _, ref, _, rtexts = run_all(FakeJudge(TABLE, NAIVE), "ref", tmp_path)
    s = analyze({**ref, **free}, {**rtexts, **ftexts})["sycophancy"]
    assert s["mitigated"]["reference_free_mode"]["a_wrong_answer_accepted"]["fooled"] == 6  # faithfulness 4
    assert s["mitigated"]["reference_mode"]["a_wrong_answer_accepted"]["fooled"] == 0  # faithfulness 2 with the reference
    assert s["mitigated"]["reference_free_mode"]["a_wrong_answer_accepted"]["n"] == 6


def test_invalid_rows_are_counted_not_dropped(tmp_path):
    class Garbage(FakeJudge):
        async def complete(self, prompt, **kw):
            r = await super().complete(prompt, **kw)
            if "c44|" in kw.get("case_id", "") and "naive" not in kw.get("case_id", ""):
                r["text"] = "no json here"
            return r

    items, got, left, texts = run_all(Garbage(TABLE, NAIVE), "ref", tmp_path)
    s = analyze(got, texts)
    assert s["n_invalid"] > 0
    assert s["sycophancy"]["mitigated"]["reference_mode"]["a_wrong_answer_accepted"]["n"] == 5


def test_naive_judge_reasks_with_the_naive_scale():
    replies = ["garbage", json.dumps({"score": 6, "reason": "ok"})]

    class J:
        model, json_mode = "fake", False
        prompts = []

        async def complete(self, prompt, **kw):
            self.prompts.append(prompt)
            return {"text": replies.pop(0), "tokens_in": 1, "tokens_out": 1, "cached": False}

    j = J()
    r = asyncio.run(judge_naive_case(j, BASE_CASES[0], label="x"))
    assert (r.score, r.parse_status) == (6, "reask")
    assert "integer 1-10" in j.prompts[1] and "integers 1-5" not in j.prompts[1]


def test_pick_sample_is_the_highest_faithfulness_wrong_verdict():
    def res(f):
        return {"scores": {"faithfulness": f, "instruction_following": 5, "completeness": 5}}

    got = {f"ref|mitigated|wrong|{c}": res(f) for c, f in zip(BASES, (2, 4, 4, 1, 3, 2))}
    assert pick_sample(got) == "ref|mitigated|wrong|c09"  # first of the two 4s
    assert pick_sample({}) is None
