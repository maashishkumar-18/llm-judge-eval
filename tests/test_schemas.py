import pytest
from pydantic import ValidationError

from judge.schemas import (
    CRITERIA,
    Case,
    CriterionVerdict,
    NaivePairVerdict,
    NaiveVerdict,
    PairwiseVerdict,
    PointwiseVerdict,
    Rubric,
    evidence_is_verbatim,
    load_anchors,
    load_rubric,
)


def verdict(f: int, i: int, c: int, **evidence) -> PointwiseVerdict:
    def crit(name, score):
        return {"evidence": evidence.get(name, "q"), "rationale": "r", "score": score}

    return PointwiseVerdict.model_validate(
        {"faithfulness": crit("faithfulness", f), "instruction_following": crit("instruction_following", i),
         "completeness": crit("completeness", c)}
    )


def test_rubric_loads_with_preregistered_values():
    r = load_rubric()
    assert r.weights == {"faithfulness": 0.40, "instruction_following": 0.35, "completeness": 0.25}
    assert (r.pass_rule.faithfulness_min, r.pass_rule.instruction_following_min, r.pass_rule.overall_min) == (4, 3, 3.5)
    assert all(sorted(c.levels) == [1, 2, 3, 4, 5] for c in r.criteria)


def test_overall_hand_computed():
    # 0.40*5 + 0.35*4 + 0.25*3 = 2.00 + 1.40 + 0.75 = 4.15
    assert verdict(5, 4, 3).overall() == pytest.approx(4.15)
    assert verdict(5, 5, 5).overall() == pytest.approx(5.0)
    assert verdict(1, 1, 1).overall() == pytest.approx(1.0)


def test_judge_reported_overall_is_ignored():
    raw = verdict(5, 4, 3).model_dump() | {"overall": 1.0, "overall_score": 1.0}
    v = PointwiseVerdict.model_validate(raw)
    assert v.overall() == pytest.approx(4.15)
    assert "overall" not in v.model_dump()


def test_pass_rule_boundaries():
    # (4,4,2) = 1.60 + 1.40 + 0.50 = 3.50 exactly: passes despite float error
    assert verdict(4, 4, 2).passes()
    assert verdict(4, 3, 3).overall() == pytest.approx(3.4) and not verdict(4, 3, 3).passes()  # overall < 3.5
    assert not verdict(3, 5, 5).passes()  # faithfulness < 4
    assert not verdict(5, 2, 5).passes()  # instruction_following < 3
    assert verdict(5, 5, 5).passes()


def test_rubric_rejects_bad_weights_and_names():
    raw = load_rubric().model_dump()
    raw["criteria"][0]["weight"] = 0.5
    with pytest.raises(ValidationError):
        Rubric.model_validate(raw)
    raw = load_rubric().model_dump()
    raw["criteria"][0]["name"] = "accuracy"
    with pytest.raises(ValidationError):
        Rubric.model_validate(raw)


@pytest.mark.parametrize("score", [0, 6, 4.5, "high"])
def test_score_out_of_range_or_not_integer_rejected(score):
    with pytest.raises(ValidationError):
        CriterionVerdict(evidence="e", rationale="r", score=score)


def test_string_integer_score_is_coerced():
    assert CriterionVerdict(evidence="e", rationale="r", score="4").score == 4


def test_missing_criterion_or_empty_evidence_rejected():
    with pytest.raises(ValidationError):
        PointwiseVerdict.model_validate({"faithfulness": {"evidence": "e", "rationale": "r", "score": 3}})
    with pytest.raises(ValidationError):
        CriterionVerdict(evidence="  ", rationale="r", score=3)


def test_evidence_must_precede_score_in_the_schema():
    assert list(CriterionVerdict.model_fields) == ["evidence", "rationale", "score"]


def test_evidence_verbatim_check():
    out = "The refund goes to the original payment method. It takes 5 business days."
    assert evidence_is_verbatim("original payment method", out)
    assert evidence_is_verbatim("THE  refund\ngoes", out)  # case / whitespace
    assert evidence_is_verbatim("The refund goes ... 5 business days", out)  # ellipsis joins fragments
    assert evidence_is_verbatim("The refund goes … 5 business days", out)
    assert not evidence_is_verbatim("takes 3 business days", out)
    assert not evidence_is_verbatim("NONE", out)
    assert not evidence_is_verbatim("The refund ... a gift card", out)  # one fragment missing


def test_unsupported_high_scores_flags_only_scores_of_4_plus():
    out = "Paris is the capital."
    v = verdict(5, 3, 4, faithfulness="Paris is the capital", instruction_following="made up", completeness="NONE")
    assert v.unsupported_high_scores(out) == ["completeness"]  # 4 with NONE; the 3 with made-up text is not flagged


def test_pairwise_winner_is_normalised_and_validated():
    crit = {"evidence_a": "a", "evidence_b": "b", "rationale": "r", "winner": "tie"}
    v = PairwiseVerdict.model_validate({c: crit for c in CRITERIA} | {"winner": " b ", "rationale": "r"})
    assert v.winner == "B" and v.faithfulness.winner == "TIE"
    with pytest.raises(ValidationError):
        PairwiseVerdict.model_validate({c: crit for c in CRITERIA} | {"winner": "C", "rationale": "r"})


def test_naive_verdicts():
    assert NaiveVerdict(score=7, reason="ok").score == 7
    with pytest.raises(ValidationError):
        NaiveVerdict(score=11)
    assert NaivePairVerdict(winner="a").winner == "A"


def test_case_blank_reference_means_reference_free():
    assert not Case(id="1", input="q", model_output="a", expected_output="  ").reference_based
    assert Case(id="1", input="q", model_output="a", expected_output="x").reference_based


def test_anchors_load_span_the_scale_and_quote_verbatim():
    anchors = load_anchors()
    overalls = sorted(a.verdict.overall() for a in anchors)
    # 0.40*1 + 0.35*2 + 0.25*1 = 1.35 ; 0.40*4 + 0.35*3 + 0.25*2 = 3.15 ; all fives = 5.0
    assert overalls == pytest.approx([1.35, 3.15, 5.0])
    assert [a.verdict.passes() for a in anchors].count(True) == 1
    for a in anchors:  # Anchor validation already enforces this; assert it stays enforced
        assert a.verdict.unsupported_high_scores(a.case.model_output) == []
