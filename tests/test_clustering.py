import asyncio
import json
from pathlib import Path

import pytest

from judge.clustering import analyze, auc, cluster_ids, run_naive_suite, separation, spread, suite_label
from judge.schemas import Case

SUITE = [Case(**c) for c in json.loads(Path("data/outputs/A.json").read_text(encoding="utf-8"))]


def run(coro):
    return asyncio.run(coro)


def test_spread_hand_computed():
    # [2, 2, 2, 4]: mean 2.5, population variance (3*0.25 + 2.25)/4 = 0.75
    s = spread([2, 2, 2, 4], (1, 10))
    assert s["mean"] == 2.5 and s["sd"] == pytest.approx(0.75 ** 0.5, abs=1e-4)
    assert s["sd_over_scale_width"] == pytest.approx(0.75 ** 0.5 / 9, abs=1e-4)
    assert s["distinct_values"] == 2 and s["modal_value"] == 2 and s["modal_share"] == 0.75
    assert s["share_at_scale_max"] == 0.0 and s["range_used_over_scale_width"] == pytest.approx(2 / 9, abs=1e-4)
    assert spread([5, 5, 5.0], (1, 5))["share_at_scale_max"] == 1.0 and spread([5, 5], (1, 5))["sd"] == 0.0
    assert spread([], (1, 5)) == {"n": 0}


def test_auc_hand_computed():
    assert auc([9, 9], [2, 3]) == 1.0
    assert auc([5], [5]) == 0.5
    assert auc([3, 5], [4]) == 0.5  # one win, one loss
    assert auc([1, 2, 3], [2]) == pytest.approx((0 + 0.5 + 1) / 3, abs=1e-4)  # 0.5
    assert auc([], [1]) is None


def test_separation_hand_computed():
    s = separation([9, 9, 8], [2, 3], (1, 10))
    assert s["auc"] == 1.0 and s["perfect_separation"] and s["min_good"] == 8 and s["max_corrupted"] == 3
    assert s["gap_over_scale_width"] == pytest.approx((26 / 3 - 2.5) / 9, abs=1e-4)
    assert separation([4], [4], (1, 5))["perfect_separation"] is False
    assert separation([], [1], (1, 5))["auc"] is None


def test_cluster_ids_is_the_first_twelve_in_suite_order():
    ids = cluster_ids(SUITE)
    assert ids == [c.id for c in SUITE][:12] and len(ids) == 12
    assert ids[:3] == ["c01", "c02", "c09"]


def test_suite_label_uses_reference_facts_and_constraints():
    by = {c.id: c for c in SUITE}
    assert suite_label(by["c09"]) is False and suite_label(by["c16"]) is False  # wrong answers from gpt-oss
    assert suite_label(by["c01"]) is True and suite_label(by["c28"]) is True


class FakeNaive:
    model, json_mode = "fake", False

    def __init__(self):
        self.n = 0

    async def complete(self, prompt, *, case_id="", system=None, nonce="", **_):
        self.n += 1
        return {"text": json.dumps({"score": 7, "reason": "ok"}), "tokens_in": 10, "tokens_out": 3, "cached": False}


def test_run_naive_suite_is_checkpointed(tmp_path):
    fake, ids = FakeNaive(), ["c01", "c02", "c09"]
    got, left = run(run_naive_suite(fake, SUITE, ids, checkpoint_path=tmp_path / "cp.jsonl"))
    assert not left and fake.n == 3 and {r["score"] for r in got.values()} == {7}
    got2, _ = run(run_naive_suite(fake, SUITE, ids, checkpoint_path=tmp_path / "cp.jsonl"))
    assert fake.n == 3 and set(got2) == set(ids)  # resumed, no new call


def _mit(cid, f, i, c):
    return {"case_id": cid, "scores": {"faithfulness": f, "instruction_following": i, "completeness": c}}


def test_analyze_end_to_end_on_fake_verdicts():
    ids = ["c01", "c09"]  # c01 correct, c09 wrong by the deterministic label
    naive = {"c01": {"score": 9}, "c09": {"score": 2}}
    mit = [_mit("c01", 5, 5, 5), _mit("c09", 1, 5, 2)]
    probes = {}
    for judge, good, wrong, vw in (("naive", 9, 2, 1), ("mitigated", None, None, None)):
        for cid in ("c01", "c09"):
            if judge == "naive":
                probes[f"ref|naive|terse|{cid}"] = {"score": good}
                probes[f"ref|naive|wrong|{cid}"] = {"score": wrong}
                probes[f"ref|naive|verbose_wrong|{cid}"] = {"score": vw}
            else:
                probes[f"ref|mitigated|terse|{cid}"] = {"scores": {"faithfulness": 5, "instruction_following": 5, "completeness": 5}}
                probes[f"ref|mitigated|wrong|{cid}"] = {"scores": {"faithfulness": 1, "instruction_following": 5, "completeness": 1}}
                probes[f"ref|mitigated|verbose_wrong|{cid}"] = {"scores": {"faithfulness": 5, "instruction_following": 5, "completeness": 5}}  # fooled
    hard = {"hard|naive|hard|c01": {"score": 3}, "hard|mitigated|hard|c01": {"scores": {"faithfulness": 3, "instruction_following": 3, "completeness": 5}}}
    out = analyze(cases=SUITE, ids=ids, naive=naive, mitigated_A=mit, mitigated_B=mit, probes=probes, hard=hard)
    sp = out["spread_same_cases"]
    assert sp["naive_1_10"]["distinct_values"] == 2 and sp["mitigated_faithfulness_1_5"]["mean"] == 3.0
    # mitigated overall: c01 5.0; c09 = .4*1 + .35*5 + .25*2 = 2.65
    assert sp["mitigated_overall_1_5"]["mean"] == pytest.approx((5.0 + 2.65) / 2, abs=1e-4)
    sep = out["separation_probes"]
    assert sep["naive"]["by_corruption"]["wrong"]["auc"] == 1.0 and sep["naive"]["all_corrupted"]["auc"] == 1.0
    assert sep["mitigated"]["by_corruption"]["wrong"]["auc"] == 1.0
    assert sep["mitigated"]["by_corruption"]["verbose_wrong"]["auc"] == 0.5  # tie with the good answers = chance
    assert sep["mitigated"]["by_corruption"]["hard"]["perfect_separation"] is True  # overall .4*3+.35*3+.25*5 = 3.5 < 5
    ss = out["separation_suite_A"]
    assert ss["good_ids"] == ["c01"] and ss["defective_ids"] == ["c09"] and ss["by_judge"]["naive"]["auc"] == 1.0
