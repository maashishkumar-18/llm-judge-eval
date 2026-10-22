import asyncio
import json
import re

import pytest

from judge.pairwise import (
    INVALID,
    TIE,
    ControlResult,
    OrderResult,
    PairResult,
    build_position_document,
    control_ids,
    judge_control,
    judge_controls,
    judge_pair,
    judge_pairs,
    naive_pair_ids,
    resolve,
    summarize_controls,
    summarize_pairs,
)
from judge.schemas import CRITERIA, Case

CASE = Case(id="p1", input="What is 2+2?", system_prompt="Answer briefly.", model_output="x", expected_output="4")
GOOD, BAD = "GOOD four", "meh five"


def mitigated_json(winner: str, criteria: str | None = None) -> str:
    c = criteria or winner
    crit = {"evidence_a": "a", "evidence_b": "b", "rationale": "r", "winner": c}
    return json.dumps({**{k: crit for k in CRITERIA}, "winner": winner, "rationale": "because"})


def naive_json(winner: str) -> str:
    return json.dumps({"winner": winner, "reason": "because"})


def shown(prompt: str) -> tuple[str, str]:
    """(text in position A, text in position B) from either pairwise template."""
    if "<answer_a>" in prompt:
        a = re.search(r"<answer_a>\n(.*?)\n</answer_a>", prompt, re.S).group(1)
        b = re.search(r"<answer_b>\n(.*?)\n</answer_b>", prompt, re.S).group(1)
    else:
        a = re.search(r"Answer A: (.*?)\n\nAnswer B:", prompt, re.S).group(1)
        b = re.search(r"Answer B: (.*?)\n\nReply with JSON only", prompt, re.S).group(1)
    return a, b


class FakeJudge:
    """Stands in for LLMClient.complete(). `pick(a, b)` returns "A", "B" or "TIE" for the displayed texts."""

    model, json_mode = "fake", False

    def __init__(self, pick, replies=()):
        self.pick, self.replies, self.prompts = pick, list(replies), []

    async def complete(self, prompt, *, case_id="", system=None, nonce="", **_):
        self.prompts.append(prompt)
        if self.replies:
            text = self.replies.pop(0)
        else:
            w = self.pick(*shown(prompt))
            text = mitigated_json(w) if "<answer_a>" in prompt else naive_json(w)
        return {"text": text, "tokens_in": 10, "tokens_out": 5, "cached": False}


def always_a(a, b):
    return "A"


def prefers_good(a, b):
    return "A" if "GOOD" in a else "B" if "GOOD" in b else "TIE"


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- order-swap rule (hand-computed)

def order(first, second, winner):
    pos = None if winner is None else TIE if winner == TIE else "first" if winner == first else "second"
    return OrderResult(first, second, "mitigated", "t", "s", "strict", position_winner=pos, winner=winner)


@pytest.mark.parametrize(
    "w1, w2, verdict, flipped, hard",
    [
        ("A", "A", "A", False, False),  # agree on the underlying answer
        ("A", "B", TIE, True, True),  # opposite decisive winners
        ("A", TIE, TIE, True, False),  # decisive in one order, tie in the other
        (TIE, "B", TIE, True, False),
        (TIE, TIE, TIE, False, False),
        (None, "A", INVALID, None, None),
        ("B", None, INVALID, None, None),
    ],
)
def test_resolve_table(w1, w2, verdict, flipped, hard):
    assert resolve(order("A", "B", w1), order("B", "A", w2)) == (verdict, flipped, hard)


# ---------------------------------------------------------------- mapping back to the underlying answers

def test_consistent_judge_maps_each_order_back_to_the_same_underlying_answer():
    j = FakeJudge(prefers_good)
    r = run(judge_pair(j, CASE, BAD, GOOD, mode="mitigated"))  # config A = BAD, config B = GOOD
    o1, o2 = r.orders
    assert (o1.first, o1.second, o1.position_winner, o1.winner) == ("A", "B", "second", "B")
    assert (o2.first, o2.second, o2.position_winner, o2.winner) == ("B", "A", "first", "B")
    assert (r.verdict, r.flipped, r.hard_flip, r.decided_by) == ("B", False, False, "both_orders")
    assert o1.criteria_winners == {c: "B" for c in CRITERIA}
    assert o2.criteria_winners == {c: "B" for c in CRITERIA}
    assert len(j.prompts) == 2
    assert shown(j.prompts[0]) == (BAD, GOOD) and shown(j.prompts[1]) == (GOOD, BAD)  # the order really swapped


def test_naive_consistent_judge_agrees_in_both_orders():
    r = run(judge_pair(FakeJudge(prefers_good), CASE, GOOD, BAD, mode="naive"))
    assert (r.verdict, r.two_order_verdict, r.flipped) == ("A", "A", False)


# ---------------------------------------------------------------- fake always-A judge

def test_always_first_judge_mitigated_gives_tie_and_a_flip_on_every_pair():
    r = run(judge_pair(FakeJudge(always_a), CASE, GOOD, BAD, mode="mitigated"))
    assert [o.winner for o in r.orders] == ["A", "B"]  # position A won twice = two different underlying answers
    assert (r.verdict, r.flipped, r.hard_flip) == (TIE, True, True)


def test_always_first_judge_naive_verdict_is_decided_by_the_single_order_and_is_wrong():
    j = FakeJudge(always_a)
    r = run(judge_pair(j, CASE, BAD, GOOD, mode="naive"))  # the better answer is config B
    assert r.verdict == "A" and r.decided_by == "first_order_only"  # naive baseline is fooled by position
    assert r.two_order_verdict == TIE and r.flipped and r.hard_flip  # the swapped call measured the flip
    assert len(j.prompts) == 2


def test_always_first_judge_on_an_identical_pair_is_still_a_flip():
    r = run(judge_pair(FakeJudge(always_a), CASE, GOOD, GOOD, mode="mitigated"))
    assert r.identical and r.flipped and r.verdict == TIE


def test_always_first_summary_flip_rate_and_position_preference():
    pairs = [run(judge_pair(FakeJudge(always_a), Case(id=f"p{i}", input="q", model_output="x"), GOOD, BAD, mode="naive"))
             for i in range(4)]
    s = summarize_pairs(pairs)
    assert (s["n"], s["flips"], s["flip_rate"], s["hard_flip_rate"]) == (4, 4, 1.0, 1.0)
    assert (s["first_position_wins"], s["second_position_wins"], s["first_position_share"]) == (8, 0, 1.0)
    assert (s["wins_x"], s["wins_y"], s["ties"], s["win_rate_x"]) == (4, 0, 0, 1.0)  # naive single order: x always wins


# ---------------------------------------------------------------- identical-answer controls

def test_controls_always_first_judge_prefers_first_every_time():
    res = [run(judge_control(FakeJudge(always_a), Case(id=f"c{i}", input="q", model_output=GOOD), GOOD, mode="naive"))
           for i in range(6)]
    s = summarize_controls(res)
    assert (s["n"], s["picked_first"], s["picked_second"], s["tie"]) == (6, 6, 0, 0)
    assert (s["first_rate"], s["tie_rate"]) == (1.0, 0.0)
    assert len(res[0].order.calls) == 1  # one order only


def test_controls_tie_judge_has_no_position_preference():
    res = [run(judge_control(FakeJudge(lambda a, b: "TIE"), CASE, GOOD, mode="mitigated")) for _ in range(6)]
    s = summarize_controls(res)
    assert (s["tie"], s["tie_rate"], s["first_rate"], s["first_rate_among_decisive"]) == (6, 1.0, 0.0, None)


def test_controls_summary_hand_computed():
    def ctl(w):
        return ControlResult("c", "naive", order("A", "A-copy", w))

    # 3 first, 1 second, 1 tie, 1 invalid -> first_rate 3/5, tie 1/5, among decisive 3/4
    s = summarize_controls([ctl("A"), ctl("A"), ctl("A"), ctl("A-copy"), ctl(TIE), ctl(None)])
    assert (s["n"], s["n_invalid"]) == (6, 1)
    assert s["first_rate"] == 0.6 and s["tie_rate"] == 0.2 and s["first_rate_among_decisive"] == 0.75


# ---------------------------------------------------------------- summary arithmetic

def test_pair_summary_hand_computed():
    def pr(cid, o1w, o2w, verdict, flipped, hard=False, identical=False, mode="mitigated"):
        return PairResult(
            cid, mode, "A", "B", identical, [order("A", "B", o1w), order("B", "A", o2w)],
            verdict, "both_orders", flipped, hard, verdict,
        )

    # p1: A wins in both orders (labels A,A); p2: B wins (B,B); p3: tie,tie; p4: hard flip -> TIE; p5: invalid
    rs = [
        pr("p1", "A", "A", "A", False),
        pr("p2", "B", "B", "B", False),
        pr("p3", TIE, TIE, TIE, False),
        pr("p4", "A", "B", TIE, True, True),
        pr("p5", None, "A", INVALID, None),
    ]
    s = summarize_pairs(rs)
    assert (s["n"], s["n_invalid"]) == (5, 1)
    assert (s["wins_x"], s["wins_y"], s["ties"]) == (1, 1, 2)
    assert s["win_rate_x"] == 0.5  # (1 + 0.5*2) / 4
    assert (s["n_flippable"], s["flips"], s["flip_rate"], s["hard_flip_rate"]) == (4, 1, 0.25, 0.25)
    # valid single judgements: p1 (A,A) p2 (B,B) p3 (T,T) p4 (A,B) p5 (second only: A) -> 9; decisive 7
    assert s["single_judgements"] == 9 and s["position_ties"] == 2
    assert (s["first_position_wins"], s["second_position_wins"]) == (4, 3)
    assert s["first_position_share"] == round(4 / 7, 4)
    assert summarize_pairs(rs, ids=["p1", "p4"])["flip_rate"] == 0.5


# ---------------------------------------------------------------- parse ladder reuse

def test_malformed_reply_is_reasked_with_pairwise_wording():
    j = FakeJudge(prefers_good, replies=["not json at all", mitigated_json("B")])
    o = run(__import__("judge.pairwise", fromlist=["judge_order"]).judge_order(
        j, CASE, {"A": BAD, "B": GOOD}, "A", "B", mode="mitigated"))
    assert o.parse_status == "reask" and o.winner == "B"
    assert len(j.prompts) == 2 and "winners as" in j.prompts[1] and "scores as integers" not in j.prompts[1]
    assert [c.tag.endswith("|reask") for c in o.calls] == [False, True]


def test_unparseable_pair_is_invalid_counted_and_not_in_the_flip_denominator():
    j = FakeJudge(prefers_good, replies=["junk", "junk", mitigated_json("B"), ])
    r = run(judge_pair(j, CASE, BAD, GOOD, mode="mitigated"))
    assert r.verdict == INVALID and r.flipped is None
    assert r.orders[0].error_kind == "parse" and r.orders[1].winner == "A"  # B-first order, position B = config A
    s = summarize_pairs([r])
    assert (s["n"], s["n_invalid"], s["n_flippable"], s["flip_rate"], s["reasks"]) == (1, 1, 0, None, 1)


def test_out_of_range_winner_is_a_parse_failure():
    j = FakeJudge(prefers_good, replies=[naive_json("C"), naive_json("C")])
    r = run(judge_pair(j, CASE, GOOD, BAD, mode="naive"))
    assert r.orders[0].error_kind == "parse" and r.verdict == INVALID


# ---------------------------------------------------------------- selection and checkpointing

def cases(outputs):
    return [Case(id=i, input="q", model_output=o, meta={"probe_base": i in ("c2", "c3")}) for i, o in outputs]


def test_naive_pair_rule_takes_first_n_differing_in_suite_order():
    xs = cases([("c1", "a"), ("c2", "b"), ("c3", "c"), ("c4", "d"), ("c5", "e")])
    ys = cases([("c1", "a"), ("c2", "B"), ("c3", "c"), ("c4", "D"), ("c5", "E")])
    assert naive_pair_ids(xs, ys, n=2) == ["c2", "c4"]
    assert naive_pair_ids(xs, ys) == ["c2", "c4", "c5"]
    assert control_ids(xs) == ["c2", "c3"]
    with pytest.raises(ValueError):
        naive_pair_ids(xs, ys[:3])


def test_judge_pairs_resumes_from_checkpoint_without_new_calls(tmp_path):
    xs = cases([("c1", GOOD), ("c2", BAD)])
    ys = cases([("c1", BAD), ("c2", GOOD)])
    j = FakeJudge(prefers_good)
    cp = tmp_path / "cp.jsonl"
    res, left = run(judge_pairs(j, xs, ys, ["c1", "c2"], mode="mitigated", checkpoint_path=cp))
    assert not left and [r.verdict for r in res] == ["A", "B"] and len(j.prompts) == 4
    res2, _ = run(judge_pairs(j, xs, ys, ["c1", "c2"], mode="mitigated", checkpoint_path=cp))
    assert len(j.prompts) == 4 and [r.verdict for r in res2] == ["A", "B"]
    assert isinstance(res2[0].orders[0], OrderResult) and res2[0].orders[0].calls[0].tokens_in == 10


def test_judge_controls_and_document(tmp_path):
    xs = cases([("c1", GOOD), ("c2", BAD), ("c3", "same")])
    ys = cases([("c1", BAD), ("c2", BAD), ("c3", "same")])
    j = FakeJudge(always_a)
    nv, _ = run(judge_pairs(j, xs, ys, ["c1"], mode="naive", checkpoint_path=tmp_path / "n.jsonl"))
    mt, _ = run(judge_pairs(j, xs, ys, ["c1", "c2", "c3"], mode="mitigated", checkpoint_path=tmp_path / "m.jsonl"))
    cn, _ = run(judge_controls(j, xs, ["c2", "c3"], mode="naive", checkpoint_path=tmp_path / "cn.jsonl"))
    cm, _ = run(judge_controls(j, xs, ["c2", "c3"], mode="mitigated", checkpoint_path=tmp_path / "cm.jsonl"))
    doc = build_position_document(
        judge_model="fake", json_mode=False, naive_ids=["c1"], control_case_ids=["c2", "c3"],
        pairs={"naive": nv, "mitigated": mt}, controls={"naive": cn, "mitigated": cm},
    )
    sm = doc["summary"]
    assert sm["pairwise_naive"]["flip_rate"] == 1.0 and sm["pairwise_mitigated_all"]["n"] == 3
    assert sm["pairwise_mitigated_same_pairs_as_naive"]["n"] == 1
    assert doc["selection"]["mitigated_identical_pair_ids"] == ["c2", "c3"]
    assert sm["pairwise_mitigated_differing_pairs"]["n"] == 1
    assert sm["controls"]["naive"]["first_rate"] == 1.0 and sm["controls"]["mitigated"]["first_rate"] == 1.0
    json.dumps(doc)  # serialisable
