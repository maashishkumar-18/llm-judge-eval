import copy
import json

import pytest

from judge.pointwise import judge_suite
from judge.schemas import Case, load_rubric
from judge.validate import (
    Entry, LabelError, adversarial_table, agreement_stats, analyze_retest, cohen_kappa, human_report,
    kappa_controls, load_human_labels, parse_label, retest_ids, run_retest, run_validation,
)

from .corpus import GOOD_JSON
from .test_pointwise import CASE, FREE, Script, client

RUBRIC = load_rubric()
GOOD = {"faithfulness": 5, "instruction_following": 5, "completeness": 5}
BAD = {"faithfulness": 1, "instruction_following": 5, "completeness": 5}


def lists(yy, yn, ny, nn):
    """(rater_a, rater_b) with yy items both yes, yn a-yes/b-no, ny a-no/b-yes, nn both no."""
    a = [True] * (yy + yn) + [False] * (ny + nn)
    b = [True] * yy + [False] * yn + [True] * ny + [False] * nn
    return a, b


# ---------------------------------------------------------------- kappa, hand-computed

def test_kappa_hand_example():
    # 50 items: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no. po = 35/50 = 0.7; rater A says yes on 25, B on 30;
    # pe = 0.5*0.6 + 0.5*0.4 = 0.5; kappa = (0.7 - 0.5) / (1 - 0.5) = 0.4
    a, b = lists(20, 5, 10, 15)
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_kappa_special_values():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, True, False, False], [False, False, True, True]) == -1.0  # po = 0, pe = 0.5
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0  # po = pe = 0.5
    assert cohen_kappa([True] * 4, [True] * 4) is None  # chance agreement 1: undefined
    assert cohen_kappa([], []) is None
    assert cohen_kappa([True] * 6, [True] * 4 + [False] * 2) == 0.0  # a constant rater carries no information (exact)
    with pytest.raises(ValueError):
        cohen_kappa([True], [True, False])


def test_kappa_three_classes():
    # agree on 3 of 6 (po = 0.5); each rater uses each class twice, pe = 3 * (1/3)^2 = 1/3; kappa = (1/2 - 1/3) / (2/3) = 0.25
    assert cohen_kappa(list("xxyyzz"), list("xyyzzx")) == pytest.approx(0.25)


def test_agreement_stats_hand():
    pred, label = lists(20, 5, 10, 15)  # pred = judge, label = reference
    s = agreement_stats(pred, label)
    assert (s["n"], s["agree"], s["agreement"], s["kappa"]) == (50, 35, 0.7, 0.4)
    assert s["confusion"] == {"tp": 20, "fp": 5, "fn": 10, "tn": 15}
    assert s["always_pass_agreement"] == 0.6  # 30 of 50 labels are pass: the raw agreement to beat
    assert s["false_pass"] == {"k": 5, "n": 20} and s["false_fail"] == {"k": 10, "n": 30}
    assert s["kappa_ci95"]["low"] < 0.4 < s["kappa_ci95"]["high"]
    lo, hi = s["agreement_ci95_exact"]
    assert lo < 0.7 < hi


def test_agreement_undefined_kappa_is_reported_not_hidden():
    s = agreement_stats([True] * 5, [True] * 5)
    assert s["kappa"] is None and "constant" in s["kappa_undefined_reason"] and s["agreement"] == 1.0
    assert agreement_stats([], [])["kappa_undefined_reason"] == "no items"


# ---------------------------------------------------------------- negative controls

def test_kappa_controls_on_a_good_judge():
    label = [True] * 6 + [False] * 6
    cs = {c["name"].split(" x")[0]: c for c in kappa_controls(list(label), label, perms=2000)}
    assert cs["always-pass judge"]["observed"] == 0 and cs["always-pass judge"]["agreement"] == 0.5
    assert cs["always-fail judge"]["observed"] == 0
    assert cs["inverted judge"]["observed"] == -1.0 and cs["inverted judge"]["ok"]
    shuffled = cs["labels shuffled"]
    assert abs(shuffled["observed"]) < 0.05 and shuffled["ok"] and shuffled["p_perm_ge_observed"] < 0.01


def test_kappa_controls_empty_when_labels_constant():
    assert kappa_controls([True, False], [True, True]) == []


def test_shuffle_control_does_not_flatter_a_random_judge():
    # a judge that is the labels shifted by one is uncorrelated here: its permutation p-value must not be tiny
    label = [True, True, False, False] * 6
    pred = label[1:] + label[:1]
    shuffled = kappa_controls(pred, label, perms=2000)[-1]
    assert shuffled["p_perm_ge_observed"] > 0.05


# ---------------------------------------------------------------- human labels

def test_load_human_labels(tmp_path):
    p = tmp_path / "h.jsonl"
    p.write_text('{"case_id":"c01","label":"pass"}\n\n{"case_id":"c02","config":"B","label":"FAIL","note":"x"}\n'
                 '{"id":"c03","label":1}\n{"case_id":"c04","label":null}\n{"case_id":"c05","label":"false"}\n', encoding="utf-8")
    got = load_human_labels(p)
    assert got[("A", "c01")]["label"] is True and got[("B", "c02")] == {"label": False, "note": "x"}
    assert got[("A", "c03")]["label"] is True and got[("A", "c04")]["label"] is None and got[("A", "c05")]["label"] is False


@pytest.mark.parametrize("line", ['{"case_id":"c01","label":"maybe"}', "not json", '{"label":"pass"}'])
def test_bad_label_rows_are_errors(tmp_path, line):
    p = tmp_path / "h.jsonl"
    p.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(LabelError):
        load_human_labels(p)


def test_duplicate_label_is_an_error(tmp_path):
    p = tmp_path / "h.jsonl"
    p.write_text('{"case_id":"c01","label":"pass"}\n{"case_id":"c01","config":"A","label":"fail"}\n', encoding="utf-8")
    with pytest.raises(LabelError, match="duplicate"):
        load_human_labels(p)


def test_parse_label_blank_is_none():
    assert parse_label("") is None and parse_label(None) is None and parse_label(" Pass ") is True


def entry(cid, scores, label, origin="suite:A", naive=None):
    return Entry(origin, cid, f"answer {cid}", "reference", label, "t", mit_scores=scores, mit_seen=True,
                 naive_score=naive, naive_seen=naive is not None)


def test_human_report_hand():
    es = [entry("c01", GOOD, True), entry("c02", GOOD, True), entry("c03", BAD, False), entry("c04", GOOD, False)]
    labels = {("A", "c01"): {"label": True}, ("A", "c02"): {"label": False}, ("A", "c03"): {"label": False},
              ("A", "c04"): {"label": None}, ("A", "c99"): {"label": True}}
    h = human_report(es, labels, RUBRIC)
    assert h["n_labelled"] == 3 and h["n_skipped_blank"] == 1 and h["unmatched"] == ["A:c99"]
    m = h["mitigated"]  # judge passes c01, c02, fails c03; humans: pass, fail, fail -> agree on 2 of 3, one false pass
    assert (m["n"], m["agree"], m["false_pass"]) == (3, 2, {"k": 1, "n": 2})
    assert [d["case_id"] for d in m["disagreements"]] == ["c02"]
    assert h["by_construction_vs_human"]["agree"] == 2  # construction: True, True, False vs human: True, False, False


def test_human_report_without_labels_says_so():
    assert human_report([entry("c01", GOOD, True)], {}, RUBRIC)["n_labelled"] == 0


# ---------------------------------------------------------------- test-retest

def doc(label, nonce, scores_by_case, cached=False):
    return {"label": label, "nonce": nonce, "results": [
        {"case_id": c, "scores": s, "calls": [{"tag": c, "cached": cached}]} for c, s in scores_by_case.items()]}


def test_retest_stable_judge_has_no_flips():
    same = {"c1": GOOD, "c2": BAD}
    r = analyze_retest([doc("A", "", same), doc("r1", "r1", same), doc("r2", "r2", same)])
    assert r["pass_flips"]["k"] == 0 and r["pass_flips"]["n"] == 2 and r["overall_sd"]["max"] == 0.0
    assert all(v["k"] == v["n"] == 2 for v in r["criterion_exact_agreement"].values())
    assert all(c["ok"] for c in r["negative_controls"])


def test_retest_flippy_judge_is_detected_with_hand_computed_sd():
    near = {"faithfulness": 4, "instruction_following": 5, "completeness": 5}  # overall 4.6, still a pass
    fail = {"faithfulness": 3, "instruction_following": 5, "completeness": 5}  # faithfulness 3 < 4: fail, overall 4.2
    runs = [doc("A", "", {"c1": GOOD, "c2": GOOD}), doc("r1", "r1", {"c1": GOOD, "c2": fail}), doc("r2", "r2", {"c1": near, "c2": GOOD})]
    r = analyze_retest(runs)
    assert (r["pass_flips"]["k"], r["pass_flips"]["n"], r["pass_flips"]["rate"], r["pass_flips"]["cases"]) == (1, 2, 0.5, ["c2"])
    # c1 overall 5.0, 5.0, 4.6: mean 4.8667, sample sd = sqrt(0.10667 / 2) = 0.2309; c2 5.0, 4.2, 5.0: sd = 0.4619
    sds = {row["case_id"]: row["overall_sd"] for row in r["per_case"]}
    assert sds["c1"] == pytest.approx(0.23094, abs=1e-4) and sds["c2"] == pytest.approx(0.46188, abs=1e-4)
    assert r["overall_sd"]["mean"] == pytest.approx(0.34641, abs=1e-4) and r["overall_sd"]["cases_with_any_difference"] == 2
    assert r["criterion_exact_agreement"]["faithfulness"] == {"k": 0, "n": 2}
    assert r["criterion_exact_agreement"]["completeness"] == {"k": 2, "n": 2}


def test_retest_invalid_run_is_excluded_and_counted():
    r = analyze_retest([doc("A", "", {"c1": GOOD, "c2": GOOD}), doc("r1", "r1", {"c1": None, "c2": GOOD})])
    assert r["n_invalid_per_run"] == {"A": 0, "r1": 1} and r["n_comparable"] == 1 and r["n_complete"] == 1


def test_retest_control_fails_when_the_cache_was_not_bypassed():
    r = analyze_retest([doc("A", "", {"c1": GOOD}), doc("r1", "r1", {"c1": GOOD}, cached=True)])
    ctrl = {c["name"]: c for c in r["negative_controls"]}["nonce runs bypass the cache"]
    assert not ctrl["ok"] and ctrl["observed"] == {"r1": 1}


def test_retest_ids_are_every_second_case():
    cases = [Case(id=f"c{i}", input="q", system_prompt="s", model_output="o") for i in range(5)]
    assert retest_ids(cases) == ["c0", "c2", "c4"]


@pytest.mark.asyncio
async def test_run_retest_bypasses_cache_with_distinct_nonces_and_reuses_saved_draws(tmp_path):
    cases = [CASE, FREE]
    script = Script(*[GOOD_JSON] * 6)  # baseline (2) + two reruns (2 each); a 7th request would raise IndexError
    async with client(tmp_path, script) as c:
        base, _ = await judge_suite(c, cases, "A", out_path=tmp_path / "pw_A.json", checkpoint_path=tmp_path / "cpA.jsonl")
        docs, left = await run_retest(c, cases, ["t1", "t2"], 2, out_dir=tmp_path, checkpoint_dir=tmp_path)
        assert left == [] and len(script.bodies) == 6  # every rerun call hit the network
        assert [d["nonce"] for d in docs] == ["retest1", "retest2"]
        analysis = analyze_retest([base, *docs])
        assert all(ctl["ok"] for ctl in analysis["negative_controls"]) and analysis["pass_flips"]["k"] == 0
        again, _ = await run_retest(c, cases, ["t1", "t2"], 2, out_dir=tmp_path, checkpoint_dir=tmp_path, fresh=True)
        assert len(script.bodies) == 6  # same nonce -> served from the cache: draws are reused, not re-sampled
        assert all(call["cached"] for d in again for r in d["results"] for call in r["calls"])
        assert not all(ctl["ok"] for ctl in analyze_retest([base, *again])["negative_controls"])  # control flags the reuse


# ---------------------------------------------------------------- adversarial table and the full report (saved files only)

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_adversarial_table_from_saved_results():
    a = adversarial_table(load("results/probes.json"), load("results/probes_hard.json"), load("results/sensitivity.json"))
    rows = {r["probe"]: r for r in a["rows"]}
    assert a["stored_mismatch"] == [] and all(c["ok"] for c in a["negative_controls"])
    w = rows["confidently wrong, reference shown"]
    assert (w["naive"]["k"], w["naive"]["n"], w["mitigated"]["k"], w["mitigated"]["n"]) == (0, 6, 0, 6)
    assert rows["padded minus terse-correct"]["naive"]["mean"] == -6.0
    acc = a["correct_accepted_terse"]
    assert (acc["mitigated"]["k"], acc["mitigated"]["n"]) == (6, 6)


def test_adversarial_table_detects_a_fooled_judge_and_a_reject_everything_judge():
    probes, hard, sens = load("results/probes.json"), load("results/probes_hard.json"), load("results/sensitivity.json")
    fooled = copy.deepcopy(probes)
    fooled["results"]["ref|mitigated|wrong|c01"]["scores"]["faithfulness"] = 5
    a = adversarial_table(fooled, hard, sens)
    assert a["rows"][0]["mitigated"]["k"] == 1 and a["rows"][0]["mitigated"]["cases"] == ["c01"]
    assert any("mitigated reference-mode fooled" in m for m in a["stored_mismatch"])  # recomputed != stored summary
    harsh = copy.deepcopy(probes)
    for k, r in harsh["results"].items():
        if k.startswith("ref|naive|terse|"):
            r["score"] = 2
    ctrl = {c["name"]: c for c in adversarial_table(harsh, hard, sens)["negative_controls"]}
    assert not ctrl["naive: terse-correct answers accepted"]["ok"]


def test_full_validation_on_saved_files_makes_no_judge_call_and_states_n():
    d = run_validation(labels_path="labels/__does_not_exist__.jsonl", retest_path="results/__none__.json")
    bc = d["by_construction"]
    assert bc["suite_A"]["n_items"] == 20 and bc["suite_B"]["n_items"] == 20 and bc["probes"]["n_items"] == 36
    assert bc["suite_distinct"]["n_items"] < 40 and bc["all_distinct"]["n_items"] == bc["suite_distinct"]["n_items"] + 36
    for s in bc.values():
        for j in ("mitigated", "naive"):
            assert s[j]["n"] <= s["n_items"] and s[j]["n_invalid"] == 0
            assert all(c["ok"] for c in s[j]["negative_controls"])
    assert d["human"]["n_labelled"] == 0 and d["test_retest"]["n_cases"] == 0 and d["caveats"]
