"""Self-preference probe (exploratory): does the Gemini judge favour Gemini-authored answers?

Design (fixed before any Gemini answer was generated):
* Cases: the reference-based suite cases. The gpt-oss answer is config A's output; the Gemini answer is produced by a
  one-off generator client (Gemini model, same config-A system prompt) built from swapped settings, so the family guard
  passes (google generator vs openai "judge" slot) without ALLOW_SAME_FAMILY. That client only generates.
* Correctness is a DETERMINISTIC check against the reference (`FACTS`: required / forbidden regexes written from the
  reference text, plus the case's own constraint checks). Only pairs where BOTH answers pass are kept, so any preference
  is not about correctness; pairs whose two texts are identical after normalisation carry no information and are dropped.
* Selection rule: the first `N_PAIRS` surviving pairs in suite order.
* The Gemini judge (mitigated pairwise prompt) compares each kept pair in both orders; the order-swap rule gives one
  verdict per pair. Own-family win rate = (Gemini wins + 0.5 ties) / N pairs. The 95% interval is exact binomial
  (Clopper-Pearson) over the decisive pairs only; ties are excluded from n and reported separately.
* N is tiny: the interval can show a large effect, never the absence of one.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from judge.checkpoint import run_checkpointed
from judge.checks import complies
from judge.config import Settings
from judge.pairwise import TIE, PairResult
from judge.schemas import Case

N_PAIRS = 6
GPT, GEMINI = "gpt-oss", "gemini"
GEMINI_GEN_MODEL = "gemini-3.5-flash-lite"
GEMINI_GEN_MAX_TOKENS = 2048  # Gemini thinking tokens share the output budget

# case id -> (required regexes, forbidden regexes), matched case-insensitively on the NFKC text with curly quotes straightened.
# Written from expected_output and the case's own trap; they test the substance, not the wording.
_NEG = r"(?:not|n't|no\b|doesn't|does not|isn't|without)"
FACTS: dict[str, tuple[list[str], list[str]]] = {
    "c01": ([r"\b2\.50\b", r"\b2 (?:times|renewals)\b"], [r"\b3 times\b", r"\b3 renewals\b"]),
    "c02": ([r"^\$?400\b"], [r"^\$?(?:480|440|420)\b"]),
    "c09": ([r"^no\b", r"12:30"], [r"^yes\b"]),
    "c14": ([r"screw 4x20", r"washer 8mm"], []),
    "c15": ([r"not stated in the notice"], []),
    "c16": ([r"^no\b", r"09:00"], [r"^yes\b", r"not stated"]),
    "c19": ([r"\b45\b"], [r"\b30\b"]),
    "c23": ([r"15 april.*draft.*30 may.*final report.*10 june.*invoice"], []),
    "c24": ([r"cracked support beam", r"august"], [r"september", r"july"]),
    "c41": ([r"phone", _NEG, r"online|website"], []),
    "c42": ([r"price|cost", _NEG], []),
    "c43": ([r"city", r"reopen", _NEG], []),
    "c44": ([r"\b84\b"], [r"\b124\b", r"\b144\b"]),
}


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", t).strip().casefold()


def correct_by_reference(case: Case, text: str) -> bool:
    """Deterministic correctness: every required fact present, no forbidden fact, the case's own constraints hold."""
    if case.id not in FACTS:
        raise KeyError(f"no fact check written for {case.id}")
    required, forbidden = FACTS[case.id]
    t = normalize(text)
    if not text.strip():
        return False
    return (
        all(re.search(p, t, re.DOTALL) for p in required)
        and not any(re.search(p, t, re.DOTALL) for p in forbidden)
        and complies(text, case.meta.get("checks", []))
    )


def gemini_generator_settings(s: Settings) -> Settings:
    """Settings for the one-off Gemini-authored-answer client (role=generator). The 'judge' slot holds a gpt-oss id only so
    the family guard sees google vs openai; that slot is never used to judge."""
    return s.model_copy(update={
        "gen_provider": "gemini", "gen_model": GEMINI_GEN_MODEL,
        "judge_provider": "groq", "judge_model": "openai/gpt-oss-20b",
    })


# ---------------------------------------------------------------- selection

def select_pairs(
    cases: Sequence[Case], gemini_text: dict[str, str], n: int = N_PAIRS,
) -> tuple[list[str], list[dict[str, Any]]]:
    """(selected ids, per-case audit rows). `cases` are config A's reference-based cases in suite order."""
    audit, selected = [], []
    for c in cases:
        g = gemini_text.get(c.id)
        gpt_ok = correct_by_reference(c, c.model_output)
        gem_ok = correct_by_reference(c, g) if g is not None else None
        identical = g is not None and normalize(g) == normalize(c.model_output)
        if not gpt_ok:
            reason = "gpt-oss answer incorrect"
        elif g is None:
            reason = "gemini answer missing"
        elif not gem_ok:
            reason = "gemini answer incorrect"
        elif identical:
            reason = "identical text"
        else:
            reason = "eligible"
        take = reason == "eligible" and len(selected) < n
        if reason == "eligible" and not take:
            reason = f"eligible but beyond the first {n}"
        if take:
            selected.append(c.id)
            reason = "selected"
        audit.append({"case_id": c.id, "gpt_correct": gpt_ok, "gemini_correct": gem_ok, "identical": identical, "outcome": reason})
    return selected, audit


# ---------------------------------------------------------------- generation (Gemini side; not in the judge ledger)

async def generate_gemini(
    client: Any, cases: Sequence[Case], system_for: Any, ids: Sequence[str], *, checkpoint_path: str | Path, concurrency: int = 1,
) -> dict[str, dict[str, Any]]:
    by_id = {c.id: c for c in cases}

    async def one(cid: str) -> dict[str, Any]:
        c = by_id[cid]
        r = await client.complete(c.input, case_id=f"{cid}|gemini_author", system=system_for(c.system_prompt), max_tokens=GEMINI_GEN_MAX_TOKENS)
        return {"text": r["text"].strip(), "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"], "cached": bool(r["cached"])}

    status = await run_checkpointed(list(ids), one, checkpoint_path, concurrency=concurrency)
    return {i: status.results[i] for i in ids if i in status.results}


# ---------------------------------------------------------------- statistics

def _binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float] | None:
    """Exact two-sided binomial interval for k successes in n trials (bisection on the binomial tails); None if n == 0."""
    if n <= 0:
        return None
    if not 0 <= k <= n:
        raise ValueError((k, n))

    def solve(f: Any, lo: float = 0.0, hi: float = 1.0) -> float:  # f increasing in p; root of f(p) = 0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: (1 - _binom_cdf(k - 1, n, p)) - alpha / 2)  # P(X >= k) rises with p
    upper = 1.0 if k == n else solve(lambda p: -(_binom_cdf(k, n, p) - alpha / 2))  # P(X <= k) falls with p
    return round(lower, 4), round(upper, 4)


def summarize(pairs: Sequence[PairResult], n_attempted: int | None = None) -> dict[str, Any]:
    valid = [p for p in pairs if p.verdict != "INVALID"]
    c = Counter(p.verdict for p in valid)
    gem, gpt, ties = c.get(GEMINI, 0), c.get(GPT, 0), c.get(TIE, 0)
    decisive = gem + gpt
    orders = [o for p in valid for o in p.orders if o.valid]
    single = Counter(o.winner for o in orders)
    calls = [cl for p in pairs for o in p.orders for cl in o.calls]
    return {
        "n_pairs": len(pairs),
        "n_invalid": len(pairs) - len(valid),
        "gemini_wins": gem, "gpt_oss_wins": gpt, "ties": ties, "flipped_pairs": sum(1 for p in valid if p.flipped),
        "own_family_win_rate": round((gem + 0.5 * ties) / len(valid), 4) if valid else None,
        "n_decisive": decisive,
        "decisive_gemini_share": round(gem / decisive, 4) if decisive else None,
        "ci95_clopper_pearson_decisive": clopper_pearson(gem, decisive),
        "single_order_judgements": {"n": len(orders), "gemini": single.get(GEMINI, 0), "gpt_oss": single.get(GPT, 0), "tie": single.get(TIE, 0)},
        "parse_status": dict(Counter(o.parse_status for p in pairs for o in p.orders)),
        "reasks": sum(1 for cl in calls if cl.tag.endswith("|reask")),
        "judge_calls": len(calls), "judge_calls_network": sum(1 for cl in calls if not cl.cached),
        "tokens_in": sum(cl.tokens_in for cl in calls), "tokens_out": sum(cl.tokens_out for cl in calls),
    }


def build_document(
    *, judge_model: str, json_mode: bool, audit: list[dict[str, Any]], selected: Sequence[str],
    gpt_text: dict[str, str], gemini_text: dict[str, str], pairs: Sequence[PairResult], gen_usage: dict[str, Any],
    guard: dict[str, str], remaining: Sequence[str] = (), nonce: str = "",
) -> dict[str, Any]:
    rows = []
    for p in pairs:
        o1, o2 = p.orders
        rows.append({
            "case_id": p.case_id, "verdict": p.verdict, "flipped": p.flipped,
            f"winner_{GPT}_first": o1.winner, f"winner_{GEMINI}_first": o2.winner,
            "words": {GPT: len(gpt_text[p.case_id].split()), GEMINI: len(gemini_text[p.case_id].split())},
            "answers": {GPT: gpt_text[p.case_id], GEMINI: gemini_text[p.case_id]},
            "criteria_winners": {GPT + "_first": o1.criteria_winners, GEMINI + "_first": o2.criteria_winners},
            "rationale": {GPT + "_first": o1.rationale, GEMINI + "_first": o2.rationale},
        })
    return {
        "judge_model": judge_model, "json_mode": json_mode, "nonce": nonce,
        "status": "EXPLORATORY: N pairs is tiny; the interval can show a large effect, never the absence of one",
        "design": {
            "judge_prompt": "mitigated pairwise (rubric + evidence), both orders, order-swap rule",
            "gpt_oss_answers": "config A outputs (gpt-oss-20b, prompt v1)",
            "gemini_answers": f"{GEMINI_GEN_MODEL} via a one-off generator client, same config-A system prompt, max_tokens {GEMINI_GEN_MAX_TOKENS}",
            "correctness": "deterministic: FACTS regexes from the reference + the case's own constraint checks, both answers must pass",
            "selection": f"first {N_PAIRS} eligible pairs in suite order; identical texts dropped",
            "naive_judge": "not run (the locked self-preference budget is 12 mitigated calls)",
            "family_guard": guard,
        },
        "complete": not remaining, "remaining": list(remaining),
        "selected_ids": list(selected),
        "eligibility": audit,
        "generation_usage": gen_usage,
        "summary": summarize(pairs),
        "pairs": rows,
        "raw_pairs": [asdict(p) for p in pairs],
    }


def write_document(doc: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_table(doc: dict[str, Any]) -> str:
    s = doc["summary"]
    lines = ["pair   verdict   gpt-first -> winner   gemini-first -> winner   words gpt/gemini"]
    for r in doc["pairs"]:
        lines.append(f"{r['case_id']:6s} {r['verdict']:9s} {str(r[f'winner_{GPT}_first']):21s} {str(r[f'winner_{GEMINI}_first']):24s} {r['words'][GPT]}/{r['words'][GEMINI]}")
    lines.append(
        f"N={s['n_pairs']}  gemini {s['gemini_wins']} / gpt-oss {s['gpt_oss_wins']} / tie {s['ties']}  flips {s['flipped_pairs']}  "
        f"own-family win rate {s['own_family_win_rate']}  decisive n={s['n_decisive']} "
        f"CP95 {s['ci95_clopper_pearson_decisive']}"
    )
    return "\n".join(lines)
