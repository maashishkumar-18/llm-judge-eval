"""Pairwise judging with order-swap, and the position-bias measurements built on it (J4).

Two configs, both judged by the same code path:
* mitigated: every pair is judged in BOTH orders, each verdict is mapped back to the underlying answers
  (config labels, not display positions); winners that agree stand, winners that disagree become TIE and
  the pair is flagged as an order flip.
* naive: the baseline. The decision is the single order "first config shown first". The swapped order is
  still judged, but only to MEASURE the flip rate; it never changes the naive verdict.

Display positions are "first"/"second" inside this module (the judge sees them as A/B); the underlying
answers are named by config label (default "A" and "B"), so the two meanings of "A" never mix.

Identical-answer controls (both answers the same text) judge one order each: an unbiased judge picks
neither (TIE) or picks first about half the time over many pairs.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from judge.checkpoint import run_checkpointed
from judge.llm import LLMError
from judge.parse import ParseStatus, build_reask_prompt, parse_with_ladder
from judge.pointwise import CallRecord
from judge.prompts import render_naive_pairwise, render_pairwise
from judge.schemas import CRITERIA, Case, NaivePairVerdict, PairwiseVerdict

MODES = ("mitigated", "naive")
TIE, INVALID = "TIE", "INVALID"
_POSITION = {"A": "first", "B": "second", "TIE": TIE}
NAIVE_PAIR_COUNT = 12  # budget.yaml pairwise_naive: 12 pairs x both orders


@dataclass
class OrderResult:
    """One judge call: the pair shown in one order."""

    first: str  # config label shown in position A
    second: str  # config label shown in position B
    mode: str
    template_id: str
    template_sha: str
    parse_status: str
    error_kind: str = ""  # "" | "parse" | "call"
    error: str = ""
    repairs: list[str] = field(default_factory=list)
    position_winner: str | None = None  # "first" | "second" | "TIE"; None = INVALID
    winner: str | None = None  # config label | "TIE"; None = INVALID
    criteria_winners: dict[str, str] | None = None  # mitigated only, mapped to config labels
    rationale: str = ""
    calls: list[CallRecord] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.winner is not None


@dataclass
class PairResult:
    case_id: str
    mode: str
    x: str  # config label shown first in the primary order
    y: str
    identical: bool  # the two answers are the same text
    orders: list[OrderResult]  # [x first, y first]
    verdict: str  # config label | "TIE" | "INVALID"
    decided_by: str  # "both_orders" | "first_order_only"
    flipped: bool | None  # the two orders named different underlying winners (TIE vs a label counts); None = a side INVALID
    hard_flip: bool | None  # the two orders named opposite decisive winners (x vs y)
    two_order_verdict: str  # what the order-swap rule gives; equals verdict in mitigated mode


@dataclass
class ControlResult:
    case_id: str
    mode: str
    order: OrderResult  # first/second are two copies of the same answer


# ---------------------------------------------------------------- one judge call

def _label(position: str, first: str, second: str) -> str:
    return {"first": first, "second": second, TIE: TIE}[position]


async def judge_order(
    client: Any,
    case: Case,
    answers: Mapping[str, str],
    first: str,
    second: str,
    *,
    mode: str,
    nonce: str = "",
    tag: str = "",
) -> OrderResult:
    """Judge `answers[first]` shown as A and `answers[second]` shown as B. Parse failures become INVALID."""
    if mode not in MODES:
        raise ValueError(mode)
    a, b = answers[first], answers[second]
    rp = render_pairwise(case, a, b) if mode == "mitigated" else render_naive_pairwise(case, a, b)
    model = PairwiseVerdict if mode == "mitigated" else NaivePairVerdict
    tag = tag or f"{case.id}|{mode}|{first}>{second}"
    res = OrderResult(first, second, mode, rp.template_id, rp.sha, ParseStatus.INVALID.value)

    async def call(prompt: str, call_tag: str) -> str:
        from judge.cache import prompt_hash

        r = await client.complete(prompt, case_id=call_tag, system=rp.system, nonce=nonce)
        res.calls.append(CallRecord(call_tag, prompt_hash(prompt), bool(r["cached"]), r["tokens_in"], r["tokens_out"]))
        return r["text"]

    try:
        raw = await call(rp.user, tag)

        async def reask(error: str) -> str:
            return await call(build_reask_prompt(rp.user, raw, error, kind="pairwise"), f"{tag}|reask")

        parsed = await parse_with_ladder(raw, model, reask=reask)
    except LLMError as e:  # QuotaStop is not an LLMError: it propagates to the checkpointed runner
        res.error_kind, res.error = "call", f"judge call failed: {e}"
        return res

    res.parse_status, res.repairs = parsed.status.value, parsed.repairs
    if parsed.value is None:
        res.error_kind, res.error = "parse", parsed.error
        return res
    v = parsed.value
    res.position_winner = _POSITION[v.winner]
    res.winner = _label(res.position_winner, first, second)
    res.rationale = v.rationale if mode == "mitigated" else v.reason
    if mode == "mitigated":
        res.criteria_winners = {c: _label(_POSITION[getattr(v, c).winner], first, second) for c in CRITERIA}
    return res


# ---------------------------------------------------------------- one pair

def resolve(o1: OrderResult, o2: OrderResult) -> tuple[str, bool | None, bool | None]:
    """Order-swap rule -> (verdict, flipped, hard_flip). Agreeing winners stand; otherwise TIE + flipped.
    Either side INVALID -> INVALID with no flip statement."""
    if not (o1.valid and o2.valid):
        return INVALID, None, None
    if o1.winner == o2.winner:
        return o1.winner, False, False  # type: ignore[return-value]
    hard = TIE not in (o1.winner, o2.winner)
    return TIE, True, hard


async def judge_pair(
    client: Any,
    case: Case,
    answer_x: str,
    answer_y: str,
    *,
    mode: str,
    x: str = "A",
    y: str = "B",
    nonce: str = "",
) -> PairResult:
    answers = {x: answer_x, y: answer_y}
    o1 = await judge_order(client, case, answers, x, y, mode=mode, nonce=nonce)
    o2 = await judge_order(client, case, answers, y, x, mode=mode, nonce=nonce)
    two_order, flipped, hard = resolve(o1, o2)
    if mode == "mitigated":
        verdict, decided_by = two_order, "both_orders"
    else:  # naive baseline: the single order x-first decides; the swapped order only measures the flip
        verdict, decided_by = (o1.winner if o1.valid else INVALID), "first_order_only"
    return PairResult(
        case.id, mode, x, y, answer_x == answer_y, [o1, o2], verdict, decided_by, flipped, hard, two_order
    )


async def judge_control(client: Any, case: Case, answer: str, *, mode: str, nonce: str = "") -> ControlResult:
    """Identical-answer control: the same text as both answers, ONE order."""
    order = await judge_order(
        client, case, {"A": answer, "A-copy": answer}, "A", "A-copy", mode=mode, nonce=nonce,
        tag=f"{case.id}|{mode}|control",
    )
    return ControlResult(case.id, mode, order)


# ---------------------------------------------------------------- summaries

def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def summarize_pairs(results: Sequence[PairResult], ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Win rate (x over y) = (wins_x + 0.5 ties) / valid pairs. Flip rate = flipped / pairs with both orders
    valid. first_position_share looks at every valid single judgement, ties excluded."""
    rs = [r for r in results if ids is None or r.case_id in ids]
    n = len(rs)
    valid = [r for r in rs if r.verdict != INVALID]
    verdicts = Counter(r.verdict for r in valid)
    x = rs[0].x if rs else "A"
    flipsable = [r for r in rs if r.flipped is not None]
    orders = [o for r in rs for o in r.orders if o.valid]
    decisive = [o for o in orders if o.position_winner != TIE]
    first_wins = sum(1 for o in decisive if o.position_winner == "first")
    calls = [c for r in rs for o in r.orders for c in o.calls]
    return {
        "n": n,
        "n_invalid": n - len(valid),
        "n_identical": sum(r.identical for r in rs),
        "wins_x": verdicts.get(x, 0),
        "wins_y": sum(v for k, v in verdicts.items() if k not in (x, TIE)),
        "ties": verdicts.get(TIE, 0),
        "win_rate_x": _rate(verdicts.get(x, 0) * 2 + verdicts.get(TIE, 0), 2 * len(valid)),
        "n_flippable": len(flipsable),
        "flips": sum(1 for r in flipsable if r.flipped),
        "flip_rate": _rate(sum(1 for r in flipsable if r.flipped), len(flipsable)),
        "hard_flips": sum(1 for r in flipsable if r.hard_flip),
        "hard_flip_rate": _rate(sum(1 for r in flipsable if r.hard_flip), len(flipsable)),
        "single_judgements": len(orders),
        "first_position_wins": first_wins,
        "second_position_wins": len(decisive) - first_wins,
        "position_ties": len(orders) - len(decisive),
        "first_position_share": _rate(first_wins, len(decisive)),
        "parse_status": dict(Counter(o.parse_status for r in rs for o in r.orders)),
        "reasks": sum(1 for c in calls if c.tag.endswith("|reask")),
        "judge_calls": len(calls),
        "judge_calls_network": sum(1 for c in calls if not c.cached),
        "tokens_in": sum(c.tokens_in for c in calls),
        "tokens_out": sum(c.tokens_out for c in calls),
    }


def summarize_controls(results: Sequence[ControlResult]) -> dict[str, Any]:
    """Position preference on identical answers. first_rate counts ties as 'did not pick first'."""
    n = len(results)
    pw = Counter(r.order.position_winner for r in results)
    valid = sum(pw.get(k, 0) for k in ("first", "second", TIE))
    decisive = pw.get("first", 0) + pw.get("second", 0)
    calls = [c for r in results for c in r.order.calls]
    return {
        "n": n,
        "n_invalid": n - valid,
        "picked_first": pw.get("first", 0),
        "picked_second": pw.get("second", 0),
        "tie": pw.get(TIE, 0),
        "first_rate": _rate(pw.get("first", 0), valid),
        "tie_rate": _rate(pw.get(TIE, 0), valid),
        "first_rate_among_decisive": _rate(pw.get("first", 0), decisive),
        "judge_calls": len(calls),
        "judge_calls_network": sum(1 for c in calls if not c.cached),
        "reasks": sum(1 for c in calls if c.tag.endswith("|reask")),
    }


# ---------------------------------------------------------------- selection (fixed before any pairwise call)

def pair_cases(cases_x: Sequence[Case], cases_y: Sequence[Case]) -> list[tuple[Case, Case]]:
    by_y = {c.id: c for c in cases_y}
    missing = [c.id for c in cases_x if c.id not in by_y]
    if missing:
        raise ValueError(f"config outputs do not cover the same cases: {missing}")
    return [(c, by_y[c.id]) for c in cases_x]


def naive_pair_ids(cases_x: Sequence[Case], cases_y: Sequence[Case], n: int = NAIVE_PAIR_COUNT) -> list[str]:
    """Pre-registered rule: the first `n` pairs, in suite order, whose two answers differ (an identical pair
    carries no ranking information for the naive baseline)."""
    return [x.id for x, y in pair_cases(cases_x, cases_y) if x.model_output != y.model_output][:n]


def control_ids(cases: Sequence[Case]) -> list[str]:
    """The probe base cases (meta.probe_base): reference-based, answerable, the same six used by the probes."""
    return [c.id for c in cases if c.meta.get("probe_base")]


# ---------------------------------------------------------------- checkpointed suite runs

def _order_from_dict(d: dict[str, Any]) -> OrderResult:
    d = dict(d)
    d["calls"] = [CallRecord(**c) for c in d.get("calls", [])]
    return OrderResult(**d)


def _pair_from_dict(d: dict[str, Any]) -> PairResult:
    d = dict(d)
    d["orders"] = [_order_from_dict(o) for o in d["orders"]]
    return PairResult(**d)


def _control_from_dict(d: dict[str, Any]) -> ControlResult:
    return ControlResult(d["case_id"], d["mode"], _order_from_dict(d["order"]))


def _persist_pair(r: dict[str, Any]) -> bool:
    return all(o["error_kind"] != "call" for o in r["orders"])  # a failed call is retried on resume


async def judge_pairs(
    client: Any,
    cases_x: Sequence[Case],
    cases_y: Sequence[Case],
    ids: Sequence[str],
    *,
    mode: str,
    checkpoint_path: str | Path,
    nonce: str = "",
    fresh: bool = False,
    concurrency: int = 2,
    x_label: str = "A",
    y_label: str = "B",
) -> tuple[list[PairResult], list[str]]:
    """Judge the pairs `ids` (x = cases_x config shown first in the primary order). Checkpointed: a quota stop
    resumes without redoing finished pairs. Returns (results in `ids` order, ids not yet judged)."""
    cp = Path(checkpoint_path)
    if fresh and cp.exists():
        cp.unlink()
    pairs = {x.id: (x, y) for x, y in pair_cases(cases_x, cases_y)}

    async def one(case_id: str) -> dict[str, Any]:
        x, y = pairs[case_id]
        return asdict(await judge_pair(client, x, x.model_output, y.model_output, mode=mode, x=x_label, y=y_label, nonce=nonce))

    status = await run_checkpointed(list(ids), one, cp, concurrency=concurrency, persist=_persist_pair)
    return [_pair_from_dict(status.results[i]) for i in ids if i in status.results], status.remaining


async def judge_controls(
    client: Any,
    cases: Sequence[Case],
    ids: Sequence[str],
    *,
    mode: str,
    checkpoint_path: str | Path,
    nonce: str = "",
    fresh: bool = False,
    concurrency: int = 2,
) -> tuple[list[ControlResult], list[str]]:
    cp = Path(checkpoint_path)
    if fresh and cp.exists():
        cp.unlink()
    by_id = {c.id: c for c in cases}

    async def one(case_id: str) -> dict[str, Any]:
        c = by_id[case_id]
        return asdict(await judge_control(client, c, c.model_output, mode=mode, nonce=nonce))

    status = await run_checkpointed(
        list(ids), one, cp, concurrency=concurrency, persist=lambda r: r["order"]["error_kind"] != "call"
    )
    return [_control_from_dict(status.results[i]) for i in ids if i in status.results], status.remaining


# ---------------------------------------------------------------- the results document

def build_position_document(
    *,
    judge_model: str,
    json_mode: bool,
    naive_ids: Sequence[str],
    control_case_ids: Sequence[str],
    pairs: Mapping[str, Sequence[PairResult]],
    controls: Mapping[str, Sequence[ControlResult]],
    nonce: str = "",
) -> dict[str, Any]:
    """results/position_bias.json: per-call detail plus summaries for naive vs mitigated."""
    naive = summarize_pairs(pairs.get("naive", []))
    mit_all = summarize_pairs(pairs.get("mitigated", []))
    mit_same = summarize_pairs(pairs.get("mitigated", []), ids=list(naive_ids))
    differing = [r.case_id for r in pairs.get("mitigated", []) if not r.identical]
    mit_diff = summarize_pairs(pairs.get("mitigated", []), ids=differing)
    ctl = {m: summarize_controls(controls[m]) for m in controls}
    return {
        "judge_model": judge_model,
        "json_mode": json_mode,
        "nonce": nonce,
        "status": "exploratory: small N, counts reported with N, no significance claimed",
        "selection": {
            "naive_pair_rule": f"first {NAIVE_PAIR_COUNT} pairs in suite order whose A and B answers differ",
            "naive_pair_ids": list(naive_ids),
            "control_case_ids": list(control_case_ids),
            "control_rule": "probe base cases; config A's answer used as both answers",
            "mitigated_identical_pair_ids": [r.case_id for r in pairs.get("mitigated", []) if r.identical],
        },
        "summary": {
            "pairwise_naive": naive,
            "pairwise_mitigated_all": mit_all,
            "pairwise_mitigated_same_pairs_as_naive": mit_same,
            "pairwise_mitigated_differing_pairs": mit_diff,
            "controls": ctl,
        },
        "pairs": {m: [asdict(r) for r in rs] for m, rs in pairs.items()},
        "controls": {m: [asdict(r) for r in rs] for m, rs in controls.items()},
    }


def write_document(doc: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
