import json

import pytest

from judge.parse import (
    ParseStatus, build_reask_prompt, extract, parse_text, parse_with_ladder, repair, strict, strip_reasoning,
)
from judge.schemas import PointwiseVerdict

from .corpus import GOOD_JSON, GOOD_SCORES, SAMPLES, Sample

LADDER = [s for s in SAMPLES if s.status != "needs_reask"]
UNFIXABLE = [s for s in SAMPLES if s.status == "needs_reask"]
ids = lambda samples: [s.name for s in samples]  # noqa: E731


@pytest.mark.parametrize("sample", LADDER, ids=ids(LADDER))
def test_corpus_parses_at_the_expected_rung(sample: Sample):
    r = parse_text(sample.raw, PointwiseVerdict)
    assert r.ok, r.error
    assert r.status.value == sample.status
    if sample.scores:
        assert tuple(r.value.scores.values()) == sample.scores
    for stage in sample.repairs:
        assert stage in r.repairs
    if sample.status != "repair":
        assert r.repairs == []


@pytest.mark.parametrize("sample", UNFIXABLE, ids=ids(UNFIXABLE))
def test_unfixable_replies_are_invalid_without_reask_and_name_the_problem(sample: Sample):
    r = parse_text(sample.raw, PointwiseVerdict)
    assert r.status is ParseStatus.INVALID and r.value is None
    assert r.error  # the re-ask needs something to quote
    assert [a.rung for a in r.attempts] == ["strict", "extract", "repair"]
    assert not any(a.ok for a in r.attempts)


def test_rungs_are_tried_in_order_and_stop_at_the_first_success():
    assert [a.rung for a in parse_text(GOOD_JSON, PointwiseVerdict).attempts] == ["strict"]
    fenced = parse_text(f"```json\n{GOOD_JSON}\n```", PointwiseVerdict)
    assert [(a.rung, a.ok) for a in fenced.attempts] == [("strict", False), ("extract", True)]
    trailing = next(s for s in SAMPLES if s.name == "trailing_comma")
    assert [(a.rung, a.ok) for a in parse_text(trailing.raw, PointwiseVerdict).attempts] == [
        ("strict", False), ("extract", False), ("repair", True)]


def test_error_messages_are_specific():
    by = {s.name: parse_text(s.raw, PointwiseVerdict).error for s in UNFIXABLE}
    assert "completeness" in by["missing_criterion"]
    assert "faithfulness.score" in by["score_out_of_range"] and "less than or equal to 5" in by["score_out_of_range"]
    assert "empty" in by["empty_reply"]
    assert "no usable JSON" in by["pure_prose_refusal"]


def test_each_rung_function_alone():
    # strict refuses anything but one bare object; extract refuses almost-JSON; repair needs the stages
    with pytest.raises(Exception):
        strict(f"```json\n{GOOD_JSON}\n```", PointwiseVerdict)
    assert extract(f"```json\n{GOOD_JSON}\n```", PointwiseVerdict).scores["faithfulness"] == 5
    trailing = next(s for s in SAMPLES if s.name == "trailing_comma").raw
    with pytest.raises(Exception):
        extract(trailing, PointwiseVerdict)
    value, applied = repair(trailing, PointwiseVerdict)
    assert applied == ["comments_trailing_commas"] and tuple(value.scores.values()) == GOOD_SCORES


def test_repair_never_invents_a_missing_field():
    truncated = '{"faithfulness": {"evidence": "x", "rationale": "y", "score": 5}, "instruction_following": {"evidence": "x"'
    r = parse_text(truncated, PointwiseVerdict)
    assert r.status is ParseStatus.INVALID  # closing the braces would still leave rationale/score/completeness out


def test_strip_reasoning():
    assert strip_reasoning("<think>a {b}</think>X") == "X"
    assert strip_reasoning("lost opening</think>X") == "X"
    assert strip_reasoning("<|channel|>analysis<|message|>a<|end|><|start|>assistant<|channel|>final<|message|>X<|return|>") == "X"
    assert strip_reasoning("plain") == "plain"


# ---- rung 4: the re-ask
async def test_reask_is_called_once_with_the_error_and_succeeds_as_reask():
    bad = next(s for s in UNFIXABLE if s.name == "score_out_of_range").raw
    calls = []

    async def reask(error: str) -> str:
        calls.append(error)
        return GOOD_JSON

    r = await parse_with_ladder(bad, PointwiseVerdict, reask=reask)
    assert r.status is ParseStatus.REASK and tuple(r.value.scores.values()) == GOOD_SCORES
    assert len(calls) == 1 and "faithfulness.score" in calls[0]
    assert [a.rung for a in r.attempts] == ["strict", "extract", "repair", "reask"]
    assert r.reask_raw == GOOD_JSON


async def test_reply_that_parses_never_triggers_a_reask():
    async def reask(error: str) -> str:
        raise AssertionError("must not be called")

    for s in LADDER:
        assert (await parse_with_ladder(s.raw, PointwiseVerdict, reask=reask)).ok


async def test_reask_reply_is_itself_run_through_the_lower_rungs():
    async def reask(error: str) -> str:
        return f"Sorry! ```json\n{GOOD_JSON}\n```"  # extract-able, still counts as a re-ask success

    r = await parse_with_ladder("nonsense", PointwiseVerdict, reask=reask)
    assert r.status is ParseStatus.REASK


async def test_one_reask_only_then_invalid_and_counted():
    n = {"calls": 0}

    async def reask(error: str) -> str:
        n["calls"] += 1
        return "still not json"

    r = await parse_with_ladder("nonsense", PointwiseVerdict, reask=reask)
    assert n["calls"] == 1  # never a loop
    assert r.status is ParseStatus.INVALID and r.value is None
    assert "after re-ask" in r.error and r.reask_raw == "still not json"


async def test_without_a_reask_callback_failure_is_invalid():
    r = await parse_with_ladder("nonsense", PointwiseVerdict)
    assert r.status is ParseStatus.INVALID


def test_reask_prompt_contains_task_error_and_previous_reply():
    p = build_reask_prompt("THE TASK", "garbage reply", "faithfulness.score: too big")
    assert p.startswith("THE TASK") and "faithfulness.score: too big" in p and "garbage reply" in p
    long = build_reask_prompt("T", "x" * 20000, "e")
    assert len(long) < 8000 and "[...truncated]" in long
    assert "(empty)" in build_reask_prompt("T", "", "e")
