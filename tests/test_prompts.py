"""Snapshot tests pin the exact rendered text of every prompt. Regenerate deliberately:
    UPDATE_SNAPSHOTS=1 python -m pytest tests/test_prompts.py
and review the diff (a prompt change also needs a VERSIONS bump in judge/prompts.py).
"""

import json
import os
from pathlib import Path

import pytest

from judge.prompts import (
    VERSIONS,
    pairwise_output_format,
    pointwise_output_format,
    render_naive_pairwise,
    render_naive_pointwise,
    render_pairwise,
    render_pointwise,
)
from judge.schemas import CRITERIA, Case, CriterionVerdict, PairwiseVerdict, load_anchors

SNAP = Path(__file__).parent / "snapshots"

REF = Case(
    id="ref-1",
    system_prompt="You are Northwind's support assistant. Answer in one sentence.",
    input="Policy: Gift cards never expire and cannot be redeemed for cash.\n\nQuestion: Can I cash out my gift card?",
    expected_output="No. Gift cards cannot be redeemed for cash.",
    model_output="No, gift cards cannot be exchanged for cash.",
)
FREE = Case(
    id="free-1",
    system_prompt="You are Northwind's support assistant. Be brief.",
    input="Policy: Gift cards never expire.\n\nQuestion: Does my gift card expire?",
    model_output="No, it never expires.",
    criteria=["Mention that gift cards never expire."],
)
ANS_A, ANS_B = "No, it never expires.", "Gift cards might expire after two years."


def check_snapshot(name: str, text: str) -> None:
    path = SNAP / f"{name}.txt"
    if os.environ.get("UPDATE_SNAPSHOTS"):
        SNAP.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    assert path.exists(), f"missing snapshot {path}; run with UPDATE_SNAPSHOTS=1"
    assert text == path.read_text(encoding="utf-8").replace("\r\n", "\n")


RENDERS = {
    "pointwise_ref": lambda: render_pointwise(REF),
    "pointwise_free": lambda: render_pointwise(FREE),
    "pairwise": lambda: render_pairwise(REF, ANS_A, ANS_B),
    "naive_pointwise": lambda: render_naive_pointwise(REF),
    "naive_pairwise": lambda: render_naive_pairwise(FREE, ANS_A, ANS_B),
}


@pytest.mark.parametrize("name", sorted(RENDERS))
def test_prompt_snapshot(name):
    check_snapshot(name, RENDERS[name]().as_snapshot())


def test_every_versioned_template_has_a_snapshot_test():
    assert set(RENDERS) == set(VERSIONS)


# ---- properties the mitigated prompts must keep (independent of exact wording)

MITIGATED = ["pointwise_ref", "pointwise_free", "pairwise"]


@pytest.mark.parametrize("name", MITIGATED)
def test_mitigated_prompts_state_length_earns_no_credit(name):
    text = RENDERS[name]().user
    assert "Length earns no credit" in text
    assert "Confident or authoritative tone earns no credit" in text


@pytest.mark.parametrize("name", ["pointwise_ref", "pointwise_free"])
def test_pointwise_requires_evidence_before_score(name):
    user = RENDERS[name]().user
    proc = user[user.index("# Procedure"):]
    assert proc.index("1. evidence") < proc.index("2. rationale") < proc.index("3. score")
    assert "verbatim span" in proc and "requires a quoted span" in proc
    fmt = pointwise_output_format()
    for c in CRITERIA:
        assert fmt.index('"evidence"', fmt.index(f'"{c}"')) < fmt.index('"score"', fmt.index(f'"{c}"'))


def test_pairwise_requires_evidence_for_both_answers_and_no_position_preference():
    user = RENDERS["pairwise"]().user
    assert "evidence_a" in user and "evidence_b" in user
    assert "carries no information" in user
    assert user.index("<answer_a>") < user.index("<answer_b>")


def test_output_format_examples_track_the_schema():
    fmt = json.loads(pointwise_output_format().replace("<integer 1-5>", '"x"'))
    assert list(fmt) == list(CRITERIA)
    assert list(fmt["faithfulness"]) == list(CriterionVerdict.model_fields)
    pfmt = json.loads(pairwise_output_format())
    assert list(pfmt) == list(PairwiseVerdict.model_fields)


def test_reference_selects_template_and_block():
    ref, free = RENDERS["pointwise_ref"](), RENDERS["pointwise_free"]()
    assert ref.template_id == "pointwise_ref.v1" and free.template_id == "pointwise_free.v1"
    assert "<reference_answer>" in ref.user and "<reference_answer>" not in free.user
    assert "No reference answer is provided" in free.user
    assert "<reference_answer>" in RENDERS["pairwise"]().user
    assert "<reference_answer>" not in render_pairwise(FREE, ANS_A, ANS_B).user


def test_anchors_included_by_default_and_switchable():
    n = len(load_anchors())
    n_free = sum(not a.case.reference_based for a in load_anchors())
    assert n == 6 and n_free == 5
    on = render_pointwise(REF).user
    assert on.count("Correct grading:") == n
    assert on.count("<reference_answer>") == 1 + (n - n_free)  # the case's own + the reference-based anchor
    free = render_pointwise(FREE).user
    assert free.count("Correct grading:") == n_free and "<reference_answer>" not in free
    assert "Correct grading:" not in render_pointwise(REF, anchors=()).user


def test_case_criteria_block_only_when_present():
    assert "Mention that gift cards never expire." in RENDERS["pointwise_free"]().user
    assert "Additional requirements" not in RENDERS["pointwise_ref"]().user


@pytest.mark.parametrize("name", ["naive_pointwise", "naive_pairwise"])
def test_naive_prompts_have_no_mitigations(name):
    text = RENDERS[name]().user.lower()
    for forbidden in ("evidence", "verbatim", "length", "rubric", "calibration", "confident"):
        assert forbidden not in text
    assert "1 to 10" in RENDERS["naive_pointwise"]().user


def test_case_text_is_not_reexpanded_as_a_template():
    tricky = Case(id="t", input="{{rubric}} {{unknown}}", model_output="{{model_output}}")
    user = render_pointwise(tricky, anchors=()).user
    assert "{{rubric}} {{unknown}}" in user and "{{model_output}}" in user
