"""A/B comparison of two configs (J5): paired bootstrap on per-case differences, win rate from order-swapped
pairs, and a pre-registered decision rule.

Convention: every difference is A minus B (positive favours A) and the win rate is A over B, so both point the
same way. Pairing is by case id; a case enters the paired analysis only if BOTH configs were scored for it
(dropped ids are listed, never silently ignored).

Decision rule (CLAUDE.md section 3, "A/B winner"):
  1. the 95% bootstrap CI of the mean paired overall difference must EXCLUDE 0, else "no significant difference";
  2. the pairwise win rate must lean the same way (> 0.5 for A, < 0.5 for B); a win rate of exactly 0.5 or the
     opposite lean contradicts the pointwise result -> "no significant difference";
  3. no pairwise results -> no winner (rule 2 cannot be checked).
Whether the win-rate CI also excludes 0.5 is reported (`win_rate_ci_excludes_half`) but is not part of the rule.
The thresholds of the pass rule come from rubric.yaml via aggregate.score_rows, unchanged."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from judge.aggregate import (
    Prices, aggregate_pointwise, generator_usage, mean_ci, pairwise_calls, pointwise_calls, score_rows,
    usage_from_calls, win_rate,
)
from judge.schemas import CRITERIA, Rubric, load_rubric

NO_DIFF = "no significant difference"


def paired_rows(
    doc_a: Mapping[str, Any], doc_b: Mapping[str, Any], rubric: Rubric
) -> tuple[list[tuple[dict, dict]], list[str]]:
    """(A row, B row) for every case scored under both configs, in A's file order, and the ids dropped."""
    a = {r["case_id"]: r for r in score_rows(doc_a, rubric)}
    b = {r["case_id"]: r for r in score_rows(doc_b, rubric)}
    both = [(a[i], b[i]) for i in a if i in b and a[i]["scored"] and b[i]["scored"]]
    dropped = sorted(set(a) ^ set(b) | {i for i in a if i in b and not (a[i]["scored"] and b[i]["scored"])})
    return both, dropped


def decide(diff: Mapping[str, Any], wr: Mapping[str, Any] | None, name_a: str = "A", name_b: str = "B") -> dict[str, Any]:
    """Apply the decision rule to a mean_ci dict of paired overall differences (A - B) and a win_rate dict."""
    reasons: list[str] = []
    lo, hi = diff["ci_low"], diff["ci_high"]
    if lo is None:
        return {"winner": None, "verdict": NO_DIFF, "reasons": [f"fewer than 2 paired cases (n={diff['n']}): no CI"]}
    leader = name_a if lo > 0 else name_b if hi < 0 else None
    if leader is None:
        reasons.append(f"CI of the mean overall difference [{lo:+.3f}, {hi:+.3f}] spans 0")
    else:
        reasons.append(f"CI of the mean overall difference [{lo:+.3f}, {hi:+.3f}] excludes 0 in favour of {leader}")
        if wr is None or wr["win_rate"] is None:
            reasons.append("no valid pairwise results, so the win-rate consistency check cannot be made")
            leader = None
        else:
            w = wr["win_rate"]
            agrees = (w > 0.5) if leader == name_a else (w < 0.5)
            reasons.append(f"win rate of {name_a} over {name_b} is {w:.3f} ({'agrees' if agrees else 'does not agree'} with {leader})")
            if not agrees:
                leader = None
    if leader is None:
        return {"winner": None, "verdict": NO_DIFF, "reasons": reasons}
    return {"winner": leader, "verdict": f"{leader} is better", "reasons": reasons}


def compare_configs(
    doc_a: Mapping[str, Any],
    doc_b: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]] | None,
    *,
    name_a: str = "A",
    name_b: str = "B",
    rubric: Rubric | None = None,
    prices: Prices | None = None,
    pair_label_x: str = "A",
) -> dict[str, Any]:
    """The full comparison document. `pairs` are PairResult dicts (mitigated, x = config A shown first first)."""
    rubric = rubric or load_rubric()
    both, dropped = paired_rows(doc_a, doc_b, rubric)
    diffs = [a["overall"] - b["overall"] for a, b in both]
    overall = mean_ci(diffs)
    wr_all = win_rate(pairs, pair_label_x) if pairs else None
    differing = [p["case_id"] for p in (pairs or []) if not p["identical"]]
    wr_diff = win_rate(pairs, pair_label_x, ids=differing) if pairs else None
    decision = decide(overall, wr_all, name_a, name_b)
    pass_a, pass_b = sum(a["passed"] for a, _ in both), sum(b["passed"] for _, b in both)
    usage = {
        f"pointwise_{name_a}": usage_from_calls(pointwise_calls(doc_a), prices),
        f"pointwise_{name_b}": usage_from_calls(pointwise_calls(doc_b), prices),
    }
    if pairs:
        usage["pairwise"] = usage_from_calls(pairwise_calls(pairs), prices)
    usage["judge_total"] = _sum_usage(list(usage.values()), prices)
    return {
        "configs": {"a": name_a, "b": name_b},
        "judge_model": doc_a.get("judge_model", ""),
        "json_mode": bool(doc_a.get("json_mode", False)),
        "convention": "differences are A minus B; win rate is A over B",
        "status": f"N={len(both)} paired cases; exploratory at this size, a CI that spans 0 is not evidence of equality",
        "n_paired": len(both),
        "dropped_ids": dropped,
        "overall_diff": overall,
        "criteria_diff": {c: mean_ci([a["scores"][c] - b["scores"][c] for a, b in both]) for c in CRITERIA},
        "pass": {
            "n_pass_a": pass_a, "n_pass_b": pass_b,
            "a_only": sum(1 for a, b in both if a["passed"] and not b["passed"]),
            "b_only": sum(1 for a, b in both if b["passed"] and not a["passed"]),
            "rate_diff": mean_ci([float(a["passed"]) - float(b["passed"]) for a, b in both]),
        },
        "same_overall_cases": sum(1 for d in diffs if abs(d) < 1e-9),
        "win_rate": wr_all,
        "win_rate_differing_pairs_only": wr_diff,
        "win_rate_ci_excludes_half": (
            None if not wr_all or wr_all["ci_low"] is None else (wr_all["ci_low"] > 0.5 or wr_all["ci_high"] < 0.5)
        ),
        "decision": decision,
        "per_case": [
            {"case_id": a["case_id"], "overall_a": round(a["overall"], 4), "overall_b": round(b["overall"], 4),
             "diff": round(a["overall"] - b["overall"], 4), "pass_a": a["passed"], "pass_b": b["passed"]}
            for a, b in both
        ],
        "usage": usage,
        "reports": {name_a: aggregate_pointwise(doc_a, rubric), name_b: aggregate_pointwise(doc_b, rubric)},
    }


def _sum_usage(parts: Sequence[Mapping[str, Any]], prices: Prices | None) -> dict[str, Any]:
    keys = ("calls", "network_calls", "cached_calls", "reasks", "tokens_in", "tokens_out", "network_tokens_in", "network_tokens_out")
    tot = {k: sum(p[k] for p in parts) for k in keys}
    prices = prices or Prices()
    tot["cost_usd"] = prices.cost(tot["network_tokens_in"], tot["network_tokens_out"])
    tot["cost_usd_if_uncached"] = prices.cost(tot["tokens_in"], tot["tokens_out"])
    tot["priced"] = prices.known
    return tot


def config_report(
    doc: Mapping[str, Any], gen_cases: Sequence[Mapping[str, Any]], *, name: str, rubric: Rubric | None = None,
    prices: Prices | None = None,
) -> dict[str, Any]:
    """results/report_<config>.json: pointwise aggregates, judge usage and generator usage for one config."""
    return {
        "config": name,
        "judge_model": doc.get("judge_model", ""),
        "json_mode": bool(doc.get("json_mode", False)),
        "label": doc.get("label", ""),
        "cases_file": doc.get("cases_file", ""),
        "pointwise": aggregate_pointwise(doc, rubric),
        "judge_usage": usage_from_calls(pointwise_calls(doc), prices),
        "generator_usage": generator_usage(gen_cases),
    }


# ---------------------------------------------------------------- text

def _ci(d: Mapping[str, Any], places: int = 3, key: str = "mean") -> str:
    if d[key] is None:
        return "n/a"
    if d["ci_low"] is None:
        return f"{d[key]:.{places}f} (no CI)"
    return f"{d[key]:.{places}f} [{d['ci_low']:.{places}f}, {d['ci_high']:.{places}f}]"


def render_markdown(c: Mapping[str, Any]) -> str:
    a, b = c["configs"]["a"], c["configs"]["b"]
    ra, rb = c["reports"][a], c["reports"][b]
    d = c["decision"]
    lines = [
        f"# Config {a} vs {b}",
        "",
        f"Judge `{c['judge_model']}` (JSON mode {'on' if c['json_mode'] else 'off'}). {c['status']}.",
        f"Differences are {a} minus {b}; win rate is {a} over {b}. 95% percentile bootstrap CIs, 10,000 resamples.",
        "",
        f"**Decision: {d['verdict']}**" + (f" (winner {d['winner']})" if d["winner"] else ""),
        "",
        *[f"- {r}" for r in d["reasons"]],
        "",
        f"| | {a} | {b} | {a} - {b} (paired, N={c['n_paired']}) |",
        "|---|---|---|---|",
        f"| mean overall | {_ci(ra['overall'])} | {_ci(rb['overall'])} | {_ci(c['overall_diff'])} |",
    ]
    for k in CRITERIA:
        lines.append(f"| {k} | {_ci(ra['criteria'][k])} | {_ci(rb['criteria'][k])} | {_ci(c['criteria_diff'][k])} |")
    lines.append(
        f"| pass rate | {ra['pass']['n_pass']}/{ra['n_scored']} | {rb['pass']['n_pass']}/{rb['n_scored']} | {_ci(c['pass']['rate_diff'])} |"
    )
    wr = c["win_rate"]
    if wr:
        lines += [
            "",
            f"Win rate of {a} over {b} (order-swapped, N={wr['n']} pairs): {_ci(wr, key='win_rate')}; "
            f"{a} {wr['wins_x']} / {b} {wr['wins_y']} / tie {wr['ties']}; flip rate {wr['flip_rate']}.",
        ]
        wd = c["win_rate_differing_pairs_only"]
        lines.append(f"Differing answers only (N={wd['n']}): {_ci(wd, key='win_rate')}; {a} {wd['wins_x']} / {b} {wd['wins_y']} / tie {wd['ties']}.")
    u = c["usage"]["judge_total"]
    cost = f"${u['cost_usd']:.4f}" if u["priced"] else "not priced (free tier, no price configured)"
    lines += [
        "",
        f"Judge usage: {u['calls']} calls ({u['network_calls']} network, {u['cached_calls']} cached, {u['reasks']} re-asks), "
        f"{u['tokens_in']} tokens in / {u['tokens_out']} out; cost {cost}.",
    ]
    if c["dropped_ids"]:
        lines.append(f"Dropped from the paired analysis (not scored under both configs): {c['dropped_ids']}.")
    return "\n".join(lines) + "\n"
