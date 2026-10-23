"""Hand-computed tests for aggregate.py and compare.py. Weights .40/.35/.25 from rubric.yaml:
(5,5,5)=5.00 pass | (4,4,4)=4.00 pass | (3,5,5)=1.2+1.75+1.25=4.20 FAIL (faithfulness 3 < 4)
(5,2,5)=2.0+0.7+1.25=3.95 FAIL (instruction-following 2 < 3) | (4,3,2)=1.6+1.05+0.5=3.15 FAIL (overall < 3.5)."""

import pytest

from judge.aggregate import (
    Prices, aggregate_pointwise, bootstrap_ci, generator_usage, mean_ci, percentile, score_rows, usage_from_calls, win_rate,
)
from judge.compare import NO_DIFF, compare_configs, config_report, decide, paired_rows, render_markdown
from judge.schemas import load_rubric

RUBRIC = load_rubric()
S555, S444, S355, S525, S432 = (
    dict(zip(("faithfulness", "instruction_following", "completeness"), t))
    for t in [(5, 5, 5), (4, 4, 4), (3, 5, 5), (5, 2, 5), (4, 3, 2)]
)


def call(tag="x", cached=False, tin=100, tout=50):
    return {"tag": tag, "prompt_hash": "h", "cached": cached, "tokens_in": tin, "tokens_out": tout}


def result(cid, scores, mode="reference", stored=True, calls=None):
    """A PointwiseResult as saved in results/pointwise_*.json; `stored` fills overall/passed like the live code."""
    ok = scores is not None
    return {
        "case_id": cid, "label": "t", "mode": mode, "parse_status": "strict" if ok else "invalid",
        "scores": scores, "overall": RUBRIC.overall(scores) if ok and stored else None,
        "passed": RUBRIC.passes(scores) if ok and stored else None,
        "unsupported_high_scores": [], "calls": calls if calls is not None else [call(cid)],
    }


def doc(rows, model="gem"):
    return {"label": "t", "judge_model": model, "json_mode": False, "cases_file": "x.json", "results": rows}


# ---------------------------------------------------------------- bootstrap

def test_percentile_hand_computed():
    xs = [1, 2, 3, 4]
    assert percentile(xs, 0) == 1 and percentile(xs, 1) == 4
    assert percentile(xs, 0.5) == 2.5  # (n-1)q = 1.5 -> halfway between 2 and 3
    assert percentile(xs, 0.25) == 1.75  # 0.75 of the way from 1 to 2


def test_bootstrap_constant_values_have_a_point_interval():
    assert bootstrap_ci([3.0, 3.0, 3.0, 3.0]) == (3.0, 3.0)


def test_bootstrap_needs_two_values_and_is_reproducible():
    assert bootstrap_ci([1.0]) is None and bootstrap_ci([]) is None
    xs = [0.0, 1.0, 1.0, 0.0, 1.0]
    assert bootstrap_ci(xs) == bootstrap_ci(xs)
    assert mean_ci([2.0]) == {"n": 1, "mean": 2.0, "ci_low": None, "ci_high": None}


def test_bootstrap_matches_the_normal_approximation_for_a_proportion():
    # 10 passes in 20: mean .5, se = sqrt(.25/20) = .1118, so 95% ~ .5 +- .219 = (.281, .719)
    lo, hi = bootstrap_ci([1.0] * 10 + [0.0] * 10)
    assert lo == pytest.approx(0.281, abs=0.04) and hi == pytest.approx(0.719, abs=0.04)
    assert lo < 0.5 < hi


def test_bootstrap_of_all_positive_values_excludes_zero_and_of_symmetric_values_spans_it():
    assert bootstrap_ci([1.0, 0.5, 1.5, 1.0])[0] > 0  # every resample mean is positive
    lo, hi = bootstrap_ci([1.0, -1.0] * 6)
    assert lo < 0 < hi


def test_bootstrap_float_noise_cannot_fake_a_nonzero_end():
    # 0.1 + 0.2 - 0.3 is 5.5e-17, not 0: a mean of such values must not produce lo > 0
    xs = [0.1 + 0.2 - 0.3] * 4
    assert bootstrap_ci(xs) == (0.0, 0.0)


# ---------------------------------------------------------------- pointwise aggregation

def mixed_doc():
    return doc([
        result("c1", S555), result("c2", S444, mode="free"), result("c3", S355), result("c4", None),
    ])


def test_aggregate_pointwise_hand_computed():
    a = aggregate_pointwise(mixed_doc(), RUBRIC)
    assert (a["n"], a["n_scored"], a["n_invalid"], a["invalid_ids"]) == (4, 3, 1, ["c4"])
    assert a["pass"]["n_pass"] == 2  # c1, c2 pass; c3 fails faithfulness 3 < 4 although overall 4.2 >= 3.5
    assert a["pass"]["rate_scored"]["mean"] == pytest.approx(2 / 3, abs=1e-4)
    assert a["pass"]["rate_invalid_as_fail"] == 0.5  # 2 passes over all 4, INVALID counted as a fail
    assert a["overall"]["mean"] == pytest.approx((5 + 4 + 4.2) / 3, abs=1e-4)  # 4.4
    assert a["criteria"]["faithfulness"]["mean"] == pytest.approx(4.0)  # (5+4+3)/3
    assert a["criteria"]["instruction_following"]["mean"] == pytest.approx(14 / 3, abs=1e-4)
    assert a["criteria"]["completeness"]["mean"] == pytest.approx(14 / 3, abs=1e-4)
    assert a["parse_status"] == {"strict": 3, "invalid": 1}
    assert a["by_mode"]["free"] == {"n": 1, "pass_rate": 1.0, "mean_overall": 4.0}
    assert a["rubric"]["weights"] == {"faithfulness": 0.4, "instruction_following": 0.35, "completeness": 0.25}


def test_pass_and_fail_reasons_each_come_from_their_own_threshold():
    rows = {r["case_id"]: r for r in score_rows(doc([result("a", S525), result("b", S432), result("c", S355)]), RUBRIC)}
    assert rows["a"]["overall"] == pytest.approx(3.95) and rows["a"]["passed"] is False  # inst 2 < 3
    assert rows["b"]["overall"] == pytest.approx(3.15) and rows["b"]["passed"] is False  # overall < 3.5
    assert rows["c"]["overall"] == pytest.approx(4.2) and rows["c"]["passed"] is False  # faith 3 < 4


def test_thresholds_come_from_the_rubric_object_and_rubric_yaml_is_unchanged():
    r = RUBRIC.pass_rule
    assert (r.faithfulness_min, r.instruction_following_min, r.overall_min) == (4, 3, 3.5)  # pre-registered values
    strict = RUBRIC.model_copy(update={"pass_rule": r.model_copy(update={"overall_min": 4.5})})
    d = doc([result("c2", S444)])
    assert aggregate_pointwise(d, RUBRIC)["pass"]["n_pass"] == 1
    assert aggregate_pointwise(d, strict)["pass"]["n_pass"] == 0  # 4.0 < 4.5: the rule is read, not baked in


def test_stored_values_are_recomputed_and_a_disagreement_is_flagged():
    tampered = result("c3", S355)
    tampered["passed"], tampered["overall"] = True, 4.9  # stored verdict claims a pass and a higher overall
    a = aggregate_pointwise(doc([result("c1", S555), tampered]), RUBRIC)
    assert a["stored_mismatch_cases"] == ["c3"]
    assert a["pass"]["n_pass"] == 1  # recomputed: c3 fails
    assert a["overall"]["mean"] == pytest.approx((5 + 4.2) / 2)


def test_all_invalid_suite_has_no_means_and_does_not_crash():
    a = aggregate_pointwise(doc([result("c1", None), result("c2", None)]), RUBRIC)
    assert a["n_scored"] == 0 and a["overall"]["mean"] is None and a["pass"]["rate_invalid_as_fail"] == 0.0


# ---------------------------------------------------------------- usage / cost

def test_usage_counts_network_vs_cached_and_prices_network_tokens():
    calls = [call("a", False, 1000, 200), call("b", True, 500, 100), call("c|reask", False, 2000, 300)]
    u = usage_from_calls(calls, Prices(1.0, 2.0))  # USD per 1M tokens
    assert (u["calls"], u["network_calls"], u["cached_calls"], u["reasks"]) == (3, 2, 1, 1)
    assert (u["tokens_in"], u["tokens_out"], u["network_tokens_in"], u["network_tokens_out"]) == (3500, 600, 3000, 500)
    assert u["cost_usd_list_equivalent"] == pytest.approx((3000 * 1 + 500 * 2) / 1e6)  # 0.004
    assert u["cost_usd_actual"] == pytest.approx(0.004)  # a paid key is billed the list price
    assert u["cost_usd_if_uncached"] == pytest.approx((3500 + 1200) / 1e6)  # 0.0047


def test_unpriced_usage_reports_null_cost_never_a_guess():
    u = usage_from_calls([call()], Prices())
    assert u["cost_usd_actual"] is None and u["cost_usd_list_equivalent"] is None and u["cost_usd_if_uncached"] is None
    assert u["priced"] is False
    assert Prices(1.0, None).known is False


def test_generator_usage_sums_meta_tokens():
    cases = [{"meta": {"gen_model": "m", "prompt_version": "v1", "tokens_in": 10, "tokens_out": 4}}, {"meta": {"tokens_in": 5, "tokens_out": 1}}]
    assert generator_usage(cases) == {"model": "m", "prompt_version": "v1", "calls": 2, "tokens_in": 15, "tokens_out": 5}


# ---------------------------------------------------------------- win rate

def pair(cid, verdict, identical=False, flipped=False):
    return {"case_id": cid, "verdict": verdict, "identical": identical, "flipped": None if verdict == "INVALID" else flipped, "x": "A", "y": "B", "orders": []}


def test_win_rate_hand_computed_with_ties_and_invalid():
    pairs = [pair("1", "A"), pair("2", "A"), pair("3", "B"), pair("4", "TIE", flipped=True), pair("5", "INVALID")]
    w = win_rate(pairs, "A")
    assert (w["n"], w["n_invalid"], w["wins_x"], w["wins_y"], w["ties"]) == (5, 1, 2, 1, 1)
    assert w["win_rate"] == 0.625  # (2 + 0.5) / 4 valid pairs
    assert (w["flips"], w["flip_rate"]) == (1, 0.25)  # 1 of the 4 pairs with both orders valid
    assert win_rate(pairs, "A", ids=["1", "3"])["win_rate"] == 0.5
    assert win_rate(pairs, "B")["win_rate"] == 0.375  # symmetric: (1 + 0.5) / 4


def test_win_rate_with_no_valid_pairs_is_none():
    assert win_rate([pair("1", "INVALID")], "A")["win_rate"] is None


# ---------------------------------------------------------------- decision rule

def ci(mean, lo, hi, n=20):
    return {"n": n, "mean": mean, "ci_low": lo, "ci_high": hi}


def wr(rate):
    return {"win_rate": rate}


def test_decision_ci_spanning_zero_is_no_significant_difference_whatever_the_win_rate():
    d = decide(ci(0.2, -0.1, 0.5), wr(0.9))
    assert d["winner"] is None and d["verdict"] == NO_DIFF and "spans 0" in d["reasons"][0]


def test_decision_ci_touching_zero_is_not_excluding_it():
    assert decide(ci(0.2, 0.0, 0.4), wr(0.9))["winner"] is None


def test_decision_winner_needs_ci_excluding_zero_and_a_consistent_win_rate():
    assert decide(ci(0.3, 0.1, 0.5), wr(0.6))["winner"] == "A"
    assert decide(ci(-0.3, -0.5, -0.1), wr(0.4))["winner"] == "B"
    assert decide(ci(0.3, 0.1, 0.5), wr(0.4))["winner"] is None  # pointwise says A, pairwise leans B
    assert decide(ci(0.3, 0.1, 0.5), wr(0.5))["winner"] is None  # exactly 0.5 is not a lean
    assert decide(ci(-0.3, -0.5, -0.1), wr(0.6))["winner"] is None


def test_decision_without_pairwise_or_with_too_few_cases_names_no_winner():
    d = decide(ci(0.3, 0.1, 0.5), None)
    assert d["winner"] is None and "cannot be made" in " ".join(d["reasons"])
    assert decide(ci(0.3, None, None, n=1), wr(0.9))["winner"] is None


# ---------------------------------------------------------------- compare end to end

def five(scores_by_case, cases=("c1", "c2", "c3", "c4", "c5")):
    return doc([result(c, scores_by_case[i]) for i, c in enumerate(cases)])


def pairs_for(verdicts, cases=("c1", "c2", "c3", "c4", "c5")):
    return [pair(c, v) for c, v in zip(cases, verdicts)]


def test_compare_clear_winner_a_minus_b_convention():
    a, b = five([S555] * 5), five([S444] * 5)  # every paired difference is 5.0 - 4.0 = +1.0
    c = compare_configs(a, b, pairs_for(["A", "A", "A", "TIE", "A"]))
    assert c["overall_diff"]["mean"] == 1.0 and c["overall_diff"]["ci_low"] == 1.0  # constant diffs -> point CI
    assert c["win_rate"]["win_rate"] == 0.9  # (4 + 0.5) / 5
    assert c["decision"]["winner"] == "A" and c["decision"]["verdict"] == "A is better"
    assert c["criteria_diff"]["faithfulness"]["mean"] == 1.0
    assert c["pass"] == {"n_pass_a": 5, "n_pass_b": 5, "a_only": 0, "b_only": 0, "rate_diff": mean_ci([0.0] * 5)}


def test_compare_swapped_configs_flips_the_sign_and_the_winner():
    a, b = five([S444] * 5), five([S555] * 5)
    c = compare_configs(a, b, pairs_for(["B", "B", "B", "B", "TIE"]))  # A over B: (0 + 0.5) / 5 = 0.1
    assert c["overall_diff"]["mean"] == -1.0 and c["win_rate"]["win_rate"] == 0.1
    assert c["decision"]["winner"] == "B"


def test_compare_symmetric_differences_report_no_significant_difference():
    # diffs: +0.8, -0.8, +0.8, -0.8 (S555 vs S355-ish): use overall 5.0 vs 4.2 and the reverse
    a = doc([result("c1", S555), result("c2", S355), result("c3", S555), result("c4", S355)])
    b = doc([result("c1", S355), result("c2", S555), result("c3", S355), result("c4", S555)])
    c = compare_configs(a, b, pairs_for(["A", "B", "A", "B"], cases=("c1", "c2", "c3", "c4")))
    assert c["overall_diff"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert c["overall_diff"]["ci_low"] < 0 < c["overall_diff"]["ci_high"]
    assert c["decision"]["winner"] is None and c["decision"]["verdict"] == NO_DIFF
    # A passes c1, c3 (S555) and fails c2, c4 (faithfulness 3); B is the mirror image
    assert (c["pass"]["a_only"], c["pass"]["b_only"]) == (2, 2)


def test_compare_drops_cases_not_scored_under_both_configs_and_lists_them():
    a = five([S555] * 5)
    b = doc([result("c1", S444), result("c2", None), result("c3", S444), result("c4", S444)])  # c2 INVALID, c5 missing
    both, dropped = paired_rows(a, b, RUBRIC)
    assert [x["case_id"] for x, _ in both] == ["c1", "c3", "c4"] and dropped == ["c2", "c5"]
    c = compare_configs(a, b, None)
    assert c["n_paired"] == 3 and c["dropped_ids"] == ["c2", "c5"]
    assert c["decision"]["winner"] is None  # no pairwise results -> rule 2 cannot be checked


def test_compare_usage_adds_pointwise_and_pairwise_judge_calls():
    a, b = doc([result("c1", S555, calls=[call(tin=10, tout=1), call(tin=20, tout=2, cached=True)])]), doc([result("c1", S444, calls=[call(tin=30, tout=3)])])
    p = pair("c1", "A")
    p["orders"] = [{"calls": [call(tin=100, tout=10)]}, {"calls": [call(tin=200, tout=20)]}]
    u = compare_configs(a, b, [p], prices=Prices(1.0, 1.0))["usage"]
    assert u["pairwise"]["calls"] == 2 and u["judge_total"]["calls"] == 5 and u["judge_total"]["cached_calls"] == 1
    assert u["judge_total"]["tokens_in"] == 360 and u["judge_total"]["network_tokens_in"] == 340
    assert u["judge_total"]["cost_usd_list_equivalent"] == pytest.approx((340 + 34) / 1e6)  # out: 1 + 3 + 10 + 20 = 34


def test_compare_win_rate_split_between_all_pairs_and_differing_pairs():
    a, b = five([S555] * 5), five([S444] * 5)
    ps = pairs_for(["A", "TIE", "TIE", "TIE", "TIE"])
    for p in ps[1:]:
        p["identical"] = True  # four identical-answer pairs carry no ranking information
    c = compare_configs(a, b, ps)
    assert c["win_rate"]["win_rate"] == 0.6  # (1 + 4 * 0.5) / 5
    d = c["win_rate_differing_pairs_only"]
    assert (d["n"], d["win_rate"], d["ci_low"]) == (1, 1.0, None)  # one pair: a rate but no CI


def test_config_report_and_markdown_render():
    d = mixed_doc()
    rep = config_report(d, [{"meta": {"tokens_in": 1, "tokens_out": 2}}], name="A", prices=Prices(1.0, 1.0))
    assert rep["pointwise"]["n_scored"] == 3 and rep["judge_usage"]["calls"] == 4 and rep["generator_usage"]["tokens_out"] == 2
    c = compare_configs(five([S555] * 5), five([S555] * 5), pairs_for(["TIE"] * 5))
    md = render_markdown(c)
    assert "no significant difference" in md and "Config A vs B" in md and "not priced" in md


def test_free_tier_reports_zero_actual_next_to_the_list_price_equivalent():
    p = Prices(0.30, 2.50, free_tier=True)
    u = usage_from_calls([call(tin=1_000_000, tout=100_000)], p)
    assert u["cost_usd_actual"] == 0.0
    assert u["cost_usd_list_equivalent"] == pytest.approx(0.30 + 0.25)  # 1M in at $0.30 + 0.1M out at $2.50


def test_recorded_price_file_has_a_source_and_date():
    from judge.aggregate import load_prices

    p = load_prices()
    assert (p.input_per_m, p.output_per_m, p.free_tier) == (0.30, 2.50, True)
    assert p.source_url.startswith("https://ai.google.dev/") and p.retrieved == "2026-10-23" and p.tier == "standard"
