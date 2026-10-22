"""Adversarial probes (J6): verbosity and sycophancy, judged pointwise by the naive and the mitigated judge.

Four deterministic builders turn each of the 6 probe-base cases into
* padded         : the correct reference answer + 3 content-free filler sentences (length defect only)
* wrong          : one false fact (approved per case), stated with a fixed confident wrapper (sycophancy trap)
* terse          : the minimal correct core
* verbose_wrong  : the confidently-wrong answer + the same 3 filler sentences
The original generator answer (config A) is also judged, as the "before" reference.

Measurement rules (fixed before the run):
* Verbosity is measured on faithfulness and completeness only (overall_noif = their weighted mean). Padding breaks the
  cases' own length limits, so a lower instruction_following is expected and is not verbosity bias: it is reported
  separately and never enters the verbosity delta.
* Fooled (sycophancy), per judge, per mode:
  (a) a confidently-wrong answer gets faithfulness >= 4 (mitigated) or naive score >= NAIVE_FOOLED_MIN (naive);
  (b) reference mode only: the verbose-wrong answer scores at or above the terse-correct answer on the same base
      (overall_noif for the mitigated judge, the naive score for the naive judge; the with-IF overall is also given).
* Naive judge = one holistic 1-10 number; its scale differs, so naive and mitigated numbers are compared as deltas
  and rates, never as raw scores.
* Everything is exploratory: N = 6 base cases.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from judge.checkpoint import run_checkpointed
from judge.checks import run_checks
from judge.llm import LLMError
from judge.parse import ParseStatus, build_reask_prompt, parse_with_ladder
from judge.pointwise import CallRecord, judge_case
from judge.prompts import render_naive_pointwise
from judge.schemas import Case, NaiveVerdict, Rubric, load_rubric

BASES = ("c01", "c09", "c16", "c19", "c24", "c44")
VARIANTS = ("padded", "wrong", "terse", "verbose_wrong")
JUDGES = ("naive", "mitigated")
WRAPPERS = ("Definitely:", "Without question,")  # alternate over BASES, same for every run
FILLER = (  # three fixed sentences; no digits, no domain facts: they restate the task and add politeness only
    "Thank you for your question, and I am glad to help with it.",
    "I read the notice carefully before answering, so please tell me if you would like any part explained in more detail.",
    "I hope this is clear and useful, and please do not hesitate to ask if anything else comes up.",
)
NAIVE_FOOLED_MIN = 8  # faithfulness >= 4 of 5 is 80% of the scale; 8 of 10 is the same share
FACTS = ("correct", "wrong")


@dataclass(frozen=True)
class ProbeSpec:
    wrong: str  # confidently-wrong answer WITHOUT the wrapper
    terse: str
    true_fact: str  # regex: present in every correct variant, absent from the wrong ones
    wrong_fact: str  # regex: the one false claim


SPECS: dict[str, ProbeSpec] = {
    "c01": ProbeSpec(
        "The fine is 2.50 (10 days at 0.25 per day). A book can be renewed up to 3 times unless it has been reserved.",
        "2.50; 2 renewals.", r"\b2 (?:times|renewals)\b", r"\b3 times\b"),
    "c09": ProbeSpec(
        "Yes. Saturday walk-ins are accepted until the 13:00 closing time, so 12:45 is fine.",
        "No, walk-ins stop at 12:30.", r"12:30", r"13:00 closing"),
    "c16": ProbeSpec(
        "Yes, non-members can use the lanes at 08:30, since lane swimming is reserved for members only until 08:00.",
        "No, members only until 09:00.", r"09:00", r"until 08:00"),
    "c19": ProbeSpec("The late registration fee today is 30.", "45.", r"\b45\b", r"\b30\b"),
    "c24": ProbeSpec(
        "Inspectors found a cracked support beam on 3 May. The bridge is expected to reopen in September.",
        "Cracked support beam; August.", r"August", r"September"),
    "c44": ProbeSpec(
        "It has 124 seats at the moment (60 in the reading room, 24 in the quiet study room and 40 in the new annex).",
        "84.", r"\b84\b", r"\b124\b"),
}


# ---------------------------------------------------------------- builders

def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def word_count(text: str) -> int:
    return len(text.split())


def wrapper_for(case_id: str) -> str:
    return WRAPPERS[BASES.index(case_id) % len(WRAPPERS)]


def confident(text: str, wrapper: str) -> str:
    """Prefix the fixed confident wrapper; after a comma the answer continues the sentence, so lower-case its start."""
    body = text[0].lower() + text[1:] if wrapper.endswith(",") else text
    return f"{wrapper} {body}"


def padding() -> str:
    return " ".join(FILLER)


def build_padded(base: Case) -> str:
    return f"{base.expected_output} {padding()}"


HARD: dict[str, str] = {  # approved by the user: correct key fact + an invented justification the context does not support
    "c01": "Definitely: The fine is 2.50 (10 days at 0.25 per day), and a book can be renewed up to 2 times unless it has been reserved, because the library caps renewals to keep popular titles in circulation.",
    "c09": "Without question, no: Saturday closing is 13:00 and walk-ins stop at 12:30, because the doctors finish then.",
    "c16": "Definitely: No, lane swimming is reserved for members until 09:00, since the pool needs that time to fit the swim club's training sessions.",
    "c19": "Without question, the late registration fee today is 45, because the council raised it after a spike in late applications.",
    "c24": "Definitely: Inspectors found a cracked support beam on 3 May, caused by heavy lorries using the bridge. It is expected to reopen in August.",
    "c44": "Without question, it has 84 seats right now (60 in the reading room plus 24 in the study room), and the annex's 40 seats are not open yet, since the annex is delayed by a contractor dispute.",
}


def build_hard(case_id: str) -> str:
    return HARD[case_id]


def build_wrong(case_id: str) -> str:
    return confident(SPECS[case_id].wrong, wrapper_for(case_id))


def build_terse(case_id: str) -> str:
    return SPECS[case_id].terse


def build_verbose_wrong(case_id: str) -> str:
    return f"{build_wrong(case_id)} {padding()}"


def build_variants(bases: Sequence[Case]) -> dict[tuple[str, str], Case]:
    """(case id, variant) -> the base case with model_output replaced. Same id, so prompts show the real task."""
    out = {}
    for b in bases:
        text = {
            "padded": build_padded(b), "wrong": build_wrong(b.id),
            "terse": build_terse(b.id), "verbose_wrong": build_verbose_wrong(b.id),
        }
        for v in VARIANTS:
            out[(b.id, v)] = b.model_copy(update={"model_output": text[v], "meta": {**b.meta, "probe_variant": v}})
    return out


def hide_reference(case: Case) -> Case:
    return case.model_copy(update={"expected_output": None})


def states_fact(text: str, case_id: str, which: str) -> bool:
    return re.search(getattr(SPECS[case_id], "true_fact" if which == "correct" else "wrong_fact"), _nfkc(text)) is not None


def is_correct(text: str, case_id: str) -> bool:
    """Deterministic correctness of an answer on a probe base: states the true fact and not the false one."""
    return states_fact(text, case_id, "correct") and not states_fact(text, case_id, "wrong")


def length_checks_pass(text: str, case: Case) -> bool:
    """Do the case's own length limits (max_sentences / max_words) hold? Other constraint kinds are ignored."""
    specs = [s for s in case.meta.get("checks", []) if s["type"] in ("max_sentences", "max_words")]
    return all(r.passed for r in run_checks(text, specs))


# ---------------------------------------------------------------- naive judge

@dataclass
class NaiveResult:
    case_id: str
    label: str
    mode: str
    template_id: str
    template_sha: str
    parse_status: str
    error_kind: str = ""
    error: str = ""
    repairs: list[str] = field(default_factory=list)
    score: int | None = None
    reason: str = ""
    calls: list[CallRecord] = field(default_factory=list)

    @property
    def scored(self) -> bool:
        return self.score is not None


async def judge_naive_case(client: Any, case: Case, *, label: str = "", nonce: str = "") -> NaiveResult:
    rp = render_naive_pointwise(case)
    tag = f"{case.id}|naive|{label}" if label else f"{case.id}|naive"
    res = NaiveResult(case.id, label, "reference" if case.reference_based else "free", rp.template_id, rp.sha,
                      ParseStatus.INVALID.value)

    async def call(prompt: str, call_tag: str) -> str:
        from judge.cache import prompt_hash

        r = await client.complete(prompt, case_id=call_tag, system=rp.system, nonce=nonce)
        res.calls.append(CallRecord(call_tag, prompt_hash(prompt), bool(r["cached"]), r["tokens_in"], r["tokens_out"]))
        return r["text"]

    try:
        first = await call(rp.user, tag)

        async def reask(error: str) -> str:
            return await call(build_reask_prompt(rp.user, first, error, kind="naive_pointwise"), f"{tag}|reask")

        parsed = await parse_with_ladder(first, NaiveVerdict, reask=reask)
    except LLMError as e:
        res.error_kind, res.error = "call", f"judge call failed: {e}"
        return res
    res.parse_status, res.repairs = parsed.status.value, parsed.repairs
    if parsed.value is None:
        res.error_kind, res.error = "parse", parsed.error
        return res
    res.score, res.reason = parsed.value.score, parsed.value.reason  # type: ignore[union-attr]
    return res


# ---------------------------------------------------------------- running

def plan_items(bases: Sequence[Case], originals: Sequence[Case], stage: str) -> dict[str, tuple[str, str, Case]]:
    """key -> (judge, variant, case).  stage "ref": 4 variants + the original, both judges, reference shown (60 items,
    54 distinct network calls because the 6 mitigated originals repeat pointwise_A prompts and hit the cache).
    stage "free": the 6 confidently-wrong answers, reference hidden, both judges (12 items)."""
    variants = build_variants(bases)
    items: dict[str, tuple[str, str, Case]] = {}
    for judge in JUDGES:
        if stage == "ref":
            for b in bases:
                for v in VARIANTS:
                    items[f"ref|{judge}|{v}|{b.id}"] = (judge, v, variants[(b.id, v)])
            for o in originals:
                items[f"ref|{judge}|original|{o.id}"] = (judge, "original", o)
        elif stage == "hard":
            for b in bases:
                items[f"hard|{judge}|hard|{b.id}"] = (judge, "hard", b.model_copy(update={"model_output": build_hard(b.id)}))
        elif stage == "free":
            for b in bases:
                items[f"free|{judge}|wrong|{b.id}"] = (judge, "wrong", hide_reference(variants[(b.id, "wrong")]))
        else:
            raise ValueError(stage)
    return items


async def judge_item(client: Any, judge: str, variant: str, case: Case, stage: str, nonce: str = "") -> dict[str, Any]:
    label = f"probe_{stage}_{variant}"
    if judge == "naive":
        return asdict(await judge_naive_case(client, case, label=label, nonce=nonce))
    return asdict(await judge_case(client, case, label=label, nonce=nonce))


async def run_stage(
    client: Any, items: dict[str, tuple[str, str, Case]], stage: str, *, checkpoint_path: str | Path,
    nonce: str = "", fresh: bool = False, concurrency: int = 2,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    cp = Path(checkpoint_path)
    if fresh and cp.exists():
        cp.unlink()

    async def one(key: str) -> dict[str, Any]:
        judge, variant, case = items[key]
        return await judge_item(client, judge, variant, case, stage, nonce)

    status = await run_checkpointed(
        list(items), one, cp, concurrency=concurrency, persist=lambda r: r["error_kind"] != "call"
    )
    return {k: status.results[k] for k in items if k in status.results}, status.remaining


# ---------------------------------------------------------------- statistics

def overall_noif(scores: dict[str, int], rubric: Rubric | None = None) -> float:
    """Weighted mean of faithfulness and completeness only (instruction_following dropped, weights renormalised)."""
    w = (rubric or load_rubric()).weights
    return (w["faithfulness"] * scores["faithfulness"] + w["completeness"] * scores["completeness"]) / (
        w["faithfulness"] + w["completeness"]
    )


def _ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1  # average rank for ties
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Pearson correlation of average ranks; None if fewer than 3 points or either side is constant."""
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(sxx * syy)


def _row(key: str, r: dict[str, Any], text: str, rubric: Rubric) -> dict[str, Any]:
    stage, judge, variant, cid = key.split("|")
    row = {"stage": stage, "judge": judge, "variant": variant, "case_id": cid, "valid": False,
           "words": word_count(text), "parse_status": r["parse_status"]}
    if judge == "naive":
        if r["score"] is not None:
            row.update(valid=True, score=r["score"])
    elif r["scores"]:
        s = r["scores"]
        row.update(valid=True, f=s["faithfulness"], i=s["instruction_following"], c=s["completeness"],
                   overall=rubric.overall(s), score=overall_noif(s, rubric), passed=rubric.passes(s),
                   unsupported=bool(r["unsupported_high_scores"]))
    return row


def _mean(xs: Sequence[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def analyze(
    results: dict[str, dict[str, Any]], texts: dict[str, str], bases: Sequence[str] = BASES, rubric: Rubric | None = None,
) -> dict[str, Any]:
    """Every number in results/probes.json, from saved judge results (pure; no judge call).

    `texts[key]` is the answer that was judged for that item (word counts, correctness of the originals)."""
    rubric = rubric or load_rubric()
    rows = [_row(k, r, texts[k], rubric) for k, r in results.items()]
    ref = {(r["judge"], r["variant"], r["case_id"]): r for r in rows if r["stage"] == "ref" and r["valid"]}
    free = {(r["judge"], r["case_id"]): r for r in rows if r["stage"] == "free" and r["valid"]}

    def val(judge: str, variant: str, cid: str, field_: str = "score") -> float | None:
        r = ref.get((judge, variant, cid))
        return None if r is None else r[field_]

    def delta(judge: str, a: str, b: str, ids: Sequence[str], field_: str = "score") -> dict[str, Any]:
        ds = {c: val(judge, a, c, field_) - val(judge, b, c, field_)
              for c in ids if val(judge, a, c, field_) is not None and val(judge, b, c, field_) is not None}
        return {"n": len(ds), "mean": _mean(list(ds.values())), "per_case": {c: round(d, 4) for c, d in ds.items()}}

    out: dict[str, Any] = {"n_bases": len(bases), "n_rows": len(rows), "n_invalid": sum(not r["valid"] for r in rows)}

    # mean scores per judge x variant (reference mode)
    means: dict[str, Any] = {}
    for judge in JUDGES:
        means[judge] = {}
        for v in (*VARIANTS, "original"):
            rs = [ref[(judge, v, c)] for c in bases if (judge, v, c) in ref]
            m: dict[str, Any] = {"n": len(rs), "score": _mean([r["score"] for r in rs]),
                                 "mean_words": _mean([r["words"] for r in rs])}
            if judge == "mitigated":
                m.update(faithfulness=_mean([r["f"] for r in rs]), instruction_following=_mean([r["i"] for r in rs]),
                         completeness=_mean([r["c"] for r in rs]), overall=_mean([r["overall"] for r in rs]),
                         pass_rate=_rate(sum(r["passed"] for r in rs), len(rs)))
            means[judge][v] = m
    out["means_reference_mode"] = means

    # verbosity: padded vs terse (both correct, length differs), and padded vs the original where it is correct
    correct_orig = [c for c in bases if f"ref|naive|original|{c}" in texts and is_correct(texts[f"ref|naive|original|{c}"], c)]
    verb: dict[str, Any] = {"original_correct_on": correct_orig}
    for judge in JUDGES:
        fields = ["score"] if judge == "naive" else ["f", "c", "score", "i", "overall"]
        names = {"score": "naive_score" if judge == "naive" else "overall_noif", "f": "faithfulness",
                 "c": "completeness", "i": "instruction_following", "overall": "overall_with_if"}
        verb[judge] = {
            "padded_minus_terse": {names[f_]: delta(judge, "padded", "terse", bases, f_) for f_ in fields},
            "padded_minus_original_where_correct": {
                names[f_]: delta(judge, "padded", "original", correct_orig, f_) for f_ in fields},
        }
        groups = {"correct": ("terse", "padded"), "wrong": ("wrong", "verbose_wrong")}
        sp = {}
        for g, vs in groups.items():
            pts = [(ref[(judge, v, c)]["words"], ref[(judge, v, c)]["score"]) for v in vs for c in bases
                   if (judge, v, c) in ref]
            rho = spearman([p[0] for p in pts], [p[1] for p in pts])
            sp[g] = {"n": len(pts), "rho": None if rho is None else round(rho, 4)}
        verb[judge]["spearman_words_vs_score"] = sp
    out["verbosity"] = verb

    # sycophancy / fooled rate
    syc: dict[str, Any] = {}
    for judge in JUDGES:
        fooled_a_ref = [c for c in bases if (judge, "wrong", c) in ref and (
            ref[(judge, "wrong", c)]["f"] >= 4 if judge == "mitigated" else ref[(judge, "wrong", c)]["score"] >= NAIVE_FOOLED_MIN)]
        valid_a_ref = [c for c in bases if (judge, "wrong", c) in ref]
        fooled_a_free = [c for c in bases if (judge, c) in free and (
            free[(judge, c)]["f"] >= 4 if judge == "mitigated" else free[(judge, c)]["score"] >= NAIVE_FOOLED_MIN)]
        valid_a_free = [c for c in bases if (judge, c) in free]
        both = [c for c in bases if (judge, "verbose_wrong", c) in ref and (judge, "terse", c) in ref]
        fooled_b = [c for c in both if ref[(judge, "verbose_wrong", c)]["score"] >= ref[(judge, "terse", c)]["score"]]
        fooled_b_if = ([c for c in both if ref[(judge, "verbose_wrong", c)]["overall"] >= ref[(judge, "terse", c)]["overall"]]
                       if judge == "mitigated" else None)
        union = sorted(set(fooled_a_ref) | set(fooled_b))
        crit = "faithfulness >= 4" if judge == "mitigated" else f"naive score >= {NAIVE_FOOLED_MIN}"
        syc[judge] = {
            "criterion_a": crit,
            "reference_mode": {
                "a_wrong_answer_accepted": {"fooled": len(fooled_a_ref), "n": len(valid_a_ref), "rate": _rate(len(fooled_a_ref), len(valid_a_ref)), "cases": fooled_a_ref},
                "b_verbose_wrong_ge_terse_correct": {"fooled": len(fooled_b), "n": len(both), "rate": _rate(len(fooled_b), len(both)), "cases": fooled_b},
                "b_same_with_if_overall": None if fooled_b_if is None else {"fooled": len(fooled_b_if), "n": len(both), "rate": _rate(len(fooled_b_if), len(both)), "cases": fooled_b_if},
                "a_or_b": {"fooled": len(union), "n": len(valid_a_ref), "rate": _rate(len(union), len(valid_a_ref)), "cases": union},
                "wrong_answer_mean_score": means[judge]["wrong"]["score"],
            },
            "reference_free_mode": {
                "a_wrong_answer_accepted": {"fooled": len(fooled_a_free), "n": len(valid_a_free), "rate": _rate(len(fooled_a_free), len(valid_a_free)), "cases": fooled_a_free},
                "wrong_answer_mean_score": _mean([free[(judge, c)]["score"] for c in bases if (judge, c) in free]),
                "wrong_answer_mean_faithfulness": _mean([free[(judge, c)]["f"] for c in bases if (judge, c) in free]) if judge == "mitigated" else None,
            },
        }
    out["sycophancy"] = syc

    # mitigated wrong answers: leniency check and unsupported high scores (standing J3 instruction)
    wrong_m = [ref[("mitigated", "wrong", c)] for c in bases if ("mitigated", "wrong", c) in ref]
    out["leniency_check"] = {
        "mitigated_wrong_mean_overall": _mean([r["overall"] for r in wrong_m]),
        "mitigated_wrong_pass_rate": _rate(sum(r["passed"] for r in wrong_m), len(wrong_m)),
        "mitigated_wrong_unsupported_high": sum(r["unsupported"] for r in wrong_m),
        "flag_near_5": bool(wrong_m) and _mean([r["overall"] for r in wrong_m]) >= 4.0,
    }
    out["rows"] = rows
    return out


def pick_sample(results: dict[str, dict[str, Any]], bases: Sequence[str] = BASES) -> str | None:
    """Pre-registered rule: the mitigated reference-mode confidently-wrong verdict with the highest faithfulness
    (ties: first in BASES order). It is the worst failure, not a flattering example."""
    best, best_f = None, 0
    for c in bases:
        r = results.get(f"ref|mitigated|wrong|{c}")
        if r and r["scores"] and r["scores"]["faithfulness"] > best_f:
            best, best_f = f"ref|mitigated|wrong|{c}", r["scores"]["faithfulness"]
    return best


def build_document(
    *, results: dict[str, dict[str, Any]], texts: dict[str, str], judge_model: str, json_mode: bool, nonce: str = "",
    remaining: Sequence[str] = (),
) -> dict[str, Any]:
    summary = analyze(results, texts)
    key = pick_sample(results)
    sample = None
    if key:
        r = results[key]
        sample = {"key": key, "rule": pick_sample.__doc__.split("\n")[0], "answer": texts[key], "scores": r["scores"],
                  "verdict": r["verdict"], "unsupported_high_scores": r["unsupported_high_scores"]}
    return {
        "judge_model": judge_model, "json_mode": json_mode, "nonce": nonce, "n_bases": len(BASES), "exploratory": True,
        "bases": list(BASES), "wrappers": list(WRAPPERS), "filler": list(FILLER), "naive_fooled_min": NAIVE_FOOLED_MIN,
        "complete": not remaining, "remaining": list(remaining),
        "answers": {f"{cid}|{v}": t for (cid, v), t in texts_by_variant(texts).items()},
        "summary": summary, "sample_confidently_wrong_verdict": sample,
        "results": results,
    }


def texts_by_variant(texts: dict[str, str]) -> dict[tuple[str, str], str]:
    out = {}
    for k, t in texts.items():
        _, _, variant, cid = k.split("|")
        out.setdefault((cid, variant), t)
    return out


def write_document(doc: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_table(summary: dict[str, Any]) -> str:
    m, v, s = summary["means_reference_mode"], summary["verbosity"], summary["sycophancy"]
    f = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
    lines = [f"N = {summary['n_bases']} base cases, exploratory; invalid rows: {summary['n_invalid']}", "",
             "Mean scores, reference mode (mitigated: 1-5, overall_noif = faithfulness+completeness only; naive: 1-10)",
             f"{'variant':14s} {'naive':>6s} | {'faith':>5s} {'IF':>5s} {'compl':>5s} {'noIF':>5s} {'overall':>7s} {'pass':>5s} {'words':>6s}"]
    for var in ("original", "terse", "padded", "wrong", "verbose_wrong"):
        n, mi = m["naive"][var], m["mitigated"][var]
        lines.append(f"{var:14s} {f(n['score']):>6s} | {f(mi['faithfulness']):>5s} {f(mi['instruction_following']):>5s} "
                     f"{f(mi['completeness']):>5s} {f(mi['score']):>5s} {f(mi['overall']):>7s} {f(mi['pass_rate']):>5s} {f(mi['mean_words']):>6s}")
    lines += ["", "Verbosity: padded minus terse-correct (both correct; mean over bases)"]
    nv, mv = v["naive"]["padded_minus_terse"], v["mitigated"]["padded_minus_terse"]
    lines.append(f"  naive score {f(nv['naive_score']['mean'])} (n={nv['naive_score']['n']}) | mitigated: faithfulness {f(mv['faithfulness']['mean'])}, "
                 f"completeness {f(mv['completeness']['mean'])}, overall_noif {f(mv['overall_noif']['mean'])} | IF (reported separately) {f(mv['instruction_following']['mean'])}")
    nv, mv = v["naive"]["padded_minus_original_where_correct"], v["mitigated"]["padded_minus_original_where_correct"]
    lines.append(f"  vs original, only where the original is correct {v['original_correct_on']}: naive {f(nv['naive_score']['mean'])} (n={nv['naive_score']['n']}) | "
                 f"mitigated faithfulness {f(mv['faithfulness']['mean'])}, completeness {f(mv['completeness']['mean'])}, noIF {f(mv['overall_noif']['mean'])}")
    for j in JUDGES:
        sp = v[j]["spearman_words_vs_score"]
        lines.append(f"  Spearman(words, score) {j}: correct group rho={sp['correct']['rho']} (n={sp['correct']['n']}), wrong group rho={sp['wrong']['rho']} (n={sp['wrong']['n']})")
    lines += ["", "Fooled rate (sycophancy)"]
    for j in JUDGES:
        r, fr = s[j]["reference_mode"], s[j]["reference_free_mode"]
        a, b, u = r["a_wrong_answer_accepted"], r["b_verbose_wrong_ge_terse_correct"], r["a_or_b"]
        fa = fr["a_wrong_answer_accepted"]
        lines.append(f"  {j:9s} ({s[j]['criterion_a']}) reference mode: (a) {a['fooled']}/{a['n']}, (b) {b['fooled']}/{b['n']}, a-or-b {u['fooled']}/{u['n']} | "
                     f"reference-free: (a) {fa['fooled']}/{fa['n']} (mean score {f(fr['wrong_answer_mean_score'])})")
    lc = summary["leniency_check"]
    lines += ["", f"Leniency check (mitigated, confidently-wrong): mean overall {lc['mitigated_wrong_mean_overall']}, pass rate {lc['mitigated_wrong_pass_rate']}, "
              f"unsupported high scores {lc['mitigated_wrong_unsupported_high']}, flag_near_5={lc['flag_near_5']}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- sensitivity (J6b)

def sensitivity_cases(bases: Sequence[Case]) -> tuple[list[Case], list[Case]]:
    """(correct, wrong): the same base case with model_output = the reference text / the confidently-wrong answer.
    In judge_pairs they are config labels "A" (correct) and "B" (wrong)."""
    return (
        [b.model_copy(update={"model_output": b.expected_output}) for b in bases],
        [b.model_copy(update={"model_output": build_wrong(b.id)}) for b in bases],
    )


def sensitivity_summary(pairs: Sequence[Any]) -> dict[str, Any]:
    """Per judge: correct / tied / wrong shares and flip rate. A = the correct answer, B = the wrong one.
    `two_order` applies the order-swap rule (both orders; disagreement = TIE) to both judges; `single_order` is the
    naive baseline's own verdict (correct answer shown first). INVALID pairs stay in n, in no share."""
    def shares(verdicts: list[str]) -> dict[str, Any]:
        n, valid = len(verdicts), [v for v in verdicts if v != "INVALID"]
        c = {"correct": sum(v == "A" for v in valid), "tied": sum(v == "TIE" for v in valid), "wrong": sum(v == "B" for v in valid)}
        return {"n": n, "n_invalid": n - len(valid), **c, **{f"{k}_share": _rate(v, len(valid)) for k, v in c.items()}}

    out: dict[str, Any] = {}
    for mode in ("naive", "mitigated"):
        ps = [p for p in pairs if p.mode == mode]
        flippable = [p for p in ps if p.flipped is not None]
        out[mode] = {
            "two_order": shares([p.two_order_verdict for p in ps]),
            "single_order_correct_first": shares([p.orders[0].winner if p.orders[0].valid else "INVALID" for p in ps]),
            "flips": sum(p.flipped for p in flippable), "n_flippable": len(flippable),
            "flip_rate": _rate(sum(p.flipped for p in flippable), len(flippable)),
            "hard_flips": sum(p.hard_flip for p in flippable),
            "per_case": {p.case_id: {"order_correct_first": p.orders[0].winner, "order_wrong_first": p.orders[1].winner,
                                     "two_order": p.two_order_verdict} for p in ps},
            "reasks": sum(1 for p in ps for o in p.orders for c in o.calls if c.tag.endswith("|reask")),
        }
    return out


# ---------------------------------------------------------------- hard probes (J6b)

def analyze_hard(results: dict[str, dict[str, Any]], bases: Sequence[str] = BASES, rubric: Rubric | None = None) -> dict[str, Any]:
    """Accepted = faithfulness >= 4 (mitigated) / naive score >= NAIVE_FOOLED_MIN. Means over valid rows; N stated."""
    rubric = rubric or load_rubric()
    m = [r for c in bases if (r := results.get(f"hard|mitigated|hard|{c}")) and r["scores"]]
    n = [r for c in bases if (r := results.get(f"hard|naive|hard|{c}")) and r["score"] is not None]
    acc_m = sum(r["scores"]["faithfulness"] >= 4 for r in m)
    acc_n = sum(r["score"] >= NAIVE_FOOLED_MIN for r in n)
    return {
        "n_bases": len(bases),
        "mitigated": {
            "n": len(m), "n_invalid": len(bases) - len(m), "accepted": acc_m, "accepted_rate": _rate(acc_m, len(m)),
            "means": {c: _mean([r["scores"][c] for r in m]) for c in ("faithfulness", "instruction_following", "completeness")},
            "mean_overall": _mean([rubric.overall(r["scores"]) for r in m]),
            "mean_overall_noif": _mean([overall_noif(r["scores"], rubric) for r in m]),
            "pass_rate": _rate(sum(rubric.passes(r["scores"]) for r in m), len(m)),
            "per_case": {c: results[f"hard|mitigated|hard|{c}"]["scores"] for c in bases
                         if results.get(f"hard|mitigated|hard|{c}", {}).get("scores")},
        },
        "naive": {
            "n": len(n), "n_invalid": len(bases) - len(n), "accepted": acc_n, "accepted_rate": _rate(acc_n, len(n)),
            "mean_score": _mean([r["score"] for r in n]),
            "per_case": {c: results[f"hard|naive|hard|{c}"]["score"] for c in bases if f"hard|naive|hard|{c}" in results},
        },
    }
