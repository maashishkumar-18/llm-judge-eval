"""Mitigated pointwise judging: one judge call per case returns all criteria; parse ladder; overall in code.

judge_case() is the single code path for live runs AND replays (a replay passes a ReplayClient), so a replay
reproduces the live verdict exactly. Every outcome is a PointwiseResult: an unparseable reply becomes an
INVALID result that is counted in the summary, never dropped."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from judge.checkpoint import run_checkpointed
from judge.llm import LLMError
from judge.parse import ParseStatus, build_reask_prompt, parse_with_ladder
from judge.prompts import render_pointwise
from judge.schemas import CRITERIA, Anchor, Case, PointwiseVerdict, Rubric, load_rubric


@dataclass
class CallRecord:
    tag: str  # case id, or "<case id>|reask"
    prompt_hash: str
    cached: bool
    tokens_in: int
    tokens_out: int


@dataclass
class PointwiseResult:
    case_id: str
    label: str
    mode: str  # "reference" | "free"
    template_id: str
    template_sha: str
    rubric_version: str
    parse_status: str  # strict | extract | repair | reask | invalid
    error_kind: str = ""  # "" | "parse" (reply unusable after the ladder) | "call" (judge call failed, no reply)
    error: str = ""
    repairs: list[str] = field(default_factory=list)
    scores: dict[str, int] | None = None
    overall: float | None = None  # recomputed here from the scores, never taken from the judge
    passed: bool | None = None
    unsupported_high_scores: list[str] = field(default_factory=list)  # score >= 4 without a verbatim quote
    verdict: dict[str, Any] | None = None
    calls: list[CallRecord] = field(default_factory=list)
    nonce: str = ""

    @property
    def scored(self) -> bool:
        return self.scores is not None


async def judge_case(
    client: Any,
    case: Case,
    *,
    label: str = "",
    anchors: Sequence[Anchor] | None = None,
    rubric: Rubric | None = None,
    nonce: str = "",
) -> PointwiseResult:
    rubric = rubric or load_rubric()
    rp = render_pointwise(case, rubric=rubric, anchors=anchors)
    tag = f"{case.id}|{label}" if label else case.id
    res = PointwiseResult(
        case_id=case.id, label=label, mode="reference" if case.reference_based else "free",
        template_id=rp.template_id, template_sha=rp.sha, rubric_version=rubric.version,
        parse_status=ParseStatus.INVALID.value, nonce=nonce,
    )

    async def call(prompt: str, call_tag: str) -> str:
        from judge.cache import prompt_hash

        r = await client.complete(prompt, case_id=call_tag, system=rp.system, nonce=nonce)
        res.calls.append(CallRecord(call_tag, prompt_hash(prompt), bool(r["cached"]), r["tokens_in"], r["tokens_out"]))
        return r["text"]

    first = ""
    try:
        first = await call(rp.user, tag)

        async def reask(error: str) -> str:
            return await call(build_reask_prompt(rp.user, first, error), f"{tag}|reask")

        parsed = await parse_with_ladder(first, PointwiseVerdict, reask=reask)
    except LLMError as e:  # QuotaStop and ReplayMiss are not LLMErrors and propagate to the runner
        res.error_kind, res.error = "call", f"judge call failed: {e}"
        return res

    res.parse_status, res.repairs = parsed.status.value, parsed.repairs
    if parsed.value is None:
        res.error_kind, res.error = "parse", parsed.error
        return res
    verdict: PointwiseVerdict = parsed.value  # type: ignore[assignment]
    res.verdict = verdict.model_dump()
    res.scores = verdict.scores
    res.overall = rubric.overall(res.scores)
    res.passed = rubric.passes(res.scores)
    res.unsupported_high_scores = verdict.unsupported_high_scores(case.model_output)
    return res


# ---------------------------------------------------------------- suite level

def _stats(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"mean": None, "sd": None, "distinct": 0}
    return {
        "mean": round(statistics.fmean(xs), 4),
        "sd": round(statistics.pstdev(xs), 4) if len(xs) > 1 else 0.0,
        "distinct": len(set(xs)),
    }


def summarize_pointwise(results: Sequence[PointwiseResult]) -> dict[str, Any]:
    """INVALID results stay in n and in the status counts. Means are over scored cases; the pass rate is
    given over scored cases AND with INVALID counted as a fail (the conservative reading)."""
    n = len(results)
    scored = [r for r in results if r.scored]
    status = Counter(r.parse_status for r in results)
    calls = [c for r in results for c in r.calls]
    passes = sum(1 for r in scored if r.passed)
    return {
        "n": n,
        "n_scored": len(scored),
        "n_invalid": sum(1 for r in results if not r.scored),
        "n_invalid_parse": sum(1 for r in results if r.error_kind == "parse"),
        "n_invalid_call": sum(1 for r in results if r.error_kind == "call"),
        "parse_status": {s.value: status.get(s.value, 0) for s in ParseStatus},
        "overall": _stats([r.overall for r in scored]),
        "criteria_mean": {c: round(statistics.fmean(r.scores[c] for r in scored), 4) if scored else None for c in CRITERIA},
        "pass_rate_scored": round(passes / len(scored), 4) if scored else None,
        "pass_rate_invalid_as_fail": round(passes / n, 4) if n else None,
        "unsupported_high_score_cases": sum(1 for r in scored if r.unsupported_high_scores),
        "judge_calls": len(calls),
        "judge_calls_network": sum(1 for c in calls if not c.cached),
        "reasks": sum(1 for c in calls if c.tag.endswith("|reask")),
        "tokens_in": sum(c.tokens_in for c in calls),
        "tokens_out": sum(c.tokens_out for c in calls),
    }


async def judge_suite(
    client: Any,
    cases: Sequence[Case],
    label: str,
    *,
    out_path: str | Path,
    checkpoint_path: str | Path,
    cases_file: str = "",
    anchors: Sequence[Anchor] | None = None,
    nonce: str = "",
    fresh: bool = False,
    concurrency: int = 2,
) -> tuple[dict[str, Any], list[str]]:
    """Judge every case (checkpointed: a quota stop resumes without redoing work) and write the results file.

    Returns (document, ids not yet judged). Call failures are not checkpointed, so a resume retries them."""
    cp = Path(checkpoint_path)
    if fresh and cp.exists():
        cp.unlink()
    by_id = {c.id: c for c in cases}

    async def one(case_id: str) -> dict[str, Any]:
        return asdict(await judge_case(client, by_id[case_id], label=label, anchors=anchors, nonce=nonce))

    status = await run_checkpointed(
        list(by_id), one, cp, concurrency=concurrency, persist=lambda r: r["error_kind"] != "call"
    )
    results = [_from_dict(status.results[c.id]) for c in cases if c.id in status.results]
    doc = {
        "label": label,
        "cases_file": cases_file,
        "nonce": nonce,
        "judge_model": getattr(client, "model", ""),
        "json_mode": bool(getattr(client, "json_mode", False)),
        "case_ids": [c.id for c in cases],
        "summary": summarize_pointwise(results),
        "results": [asdict(r) for r in results],
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return doc, status.remaining


def _from_dict(d: dict[str, Any]) -> PointwiseResult:
    d = dict(d)
    d["calls"] = [CallRecord(**c) for c in d.get("calls", [])]
    return PointwiseResult(**d)


# ---------------------------------------------------------------- replay

SAME_FIELDS = ("parse_status", "error_kind", "repairs", "scores", "overall", "passed", "unsupported_high_scores")


async def replay_suite(
    results_doc: dict[str, Any], cases: Sequence[Case], client: Any, *, anchors: Sequence[Anchor] | None = None
) -> list[dict[str, Any]]:
    """Re-run judge_case over `client` (a ReplayClient) and diff against the saved results.

    One row per saved result: {case_id, same, diffs, replayed_status}. No API call is possible here."""
    from judge.replay import ReplayMiss

    by_id = {c.id: c for c in cases}
    rows = []
    for saved in results_doc["results"]:
        cid = saved["case_id"]
        try:
            again = asdict(
                await judge_case(
                    client, by_id[cid], label=saved["label"], anchors=anchors, nonce=saved.get("nonce", "")
                )
            )
        except ReplayMiss as e:
            rows.append({"case_id": cid, "same": False, "diffs": {"log": str(e)}, "replayed_status": "no-log"})
            continue
        diffs = {k: (saved[k], again[k]) for k in SAME_FIELDS if saved[k] != again[k]}
        rows.append({"case_id": cid, "same": not diffs, "diffs": diffs, "replayed_status": again["parse_status"]})
    return rows
