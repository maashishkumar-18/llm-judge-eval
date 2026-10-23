"""Suite-level aggregation: per-case pass/fail, pass rate, per-criterion means with bootstrap CIs,
win rate from order-swapped pairs, and call / token / cost tracking.

Everything here works on the saved results documents (results/pointwise_<label>.json, the `pairs` block of
results/position_bias.json), so a report can be rebuilt offline with zero judge calls. Pass/fail and overall are
RE-COMPUTED from the stored criterion scores with the thresholds in rubric.yaml (unchanged); a stored value that
disagrees is flagged, not trusted. INVALID cases stay in every denominator that says so."""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from judge.schemas import CRITERIA, Rubric, load_rubric

PRICING_PATH = Path(__file__).with_name("pricing.yaml")

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 5150  # fixed so a report is reproducible; changing it moves CI ends by a few thousandths
CI_LEVEL = 0.95
_TIE, _INVALID = "TIE", "INVALID"


# ---------------------------------------------------------------- bootstrap

def percentile(sorted_xs: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an ascending list, q in [0, 1]."""
    k = (len(sorted_xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (k - lo)


def bootstrap_ci(
    xs: Sequence[float], *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED, level: float = CI_LEVEL
) -> tuple[float, float] | None:
    """Percentile bootstrap CI for the mean. None for fewer than 2 values (no spread to resample).
    Ends are rounded to 6 places so float noise cannot make an interval that touches 0 look like it excludes 0."""
    if len(xs) < 2:
        return None
    rng, n = random.Random(seed), len(xs)
    means = sorted(math.fsum(rng.choices(xs, k=n)) / n for _ in range(resamples))
    a = (1 - level) / 2
    return round(percentile(means, a), 6), round(percentile(means, 1 - a), 6)


def mean_ci(xs: Sequence[float]) -> dict[str, Any]:
    ci = bootstrap_ci(xs)
    return {
        "n": len(xs),
        "mean": round(statistics.fmean(xs), 4) if xs else None,
        "ci_low": ci[0] if ci else None,
        "ci_high": ci[1] if ci else None,
    }


# ---------------------------------------------------------------- pointwise

def score_rows(doc: Mapping[str, Any], rubric: Rubric) -> list[dict[str, Any]]:
    """One row per judged case, in file order. overall / passed are recomputed from the scores with `rubric`."""
    rows = []
    for r in doc["results"]:
        sc = r.get("scores")
        row: dict[str, Any] = {
            "case_id": r["case_id"], "mode": r["mode"], "parse_status": r["parse_status"],
            "scored": sc is not None, "scores": sc, "overall": None, "passed": None,
            "unsupported_high_scores": r.get("unsupported_high_scores", []), "stored_mismatch": False,
        }
        if sc is not None:
            row["overall"], row["passed"] = rubric.overall(sc), rubric.passes(sc)
            row["stored_mismatch"] = (
                r.get("overall") is None or abs(r["overall"] - row["overall"]) > 1e-9 or r.get("passed") != row["passed"]
            )
        rows.append(row)
    return rows


def aggregate_pointwise(doc: Mapping[str, Any], rubric: Rubric | None = None) -> dict[str, Any]:
    rubric = rubric or load_rubric()
    rows = score_rows(doc, rubric)
    scored = [r for r in rows if r["scored"]]
    passes = [1.0 if r["passed"] else 0.0 for r in scored]
    overall = [r["overall"] for r in scored]
    return {
        "n": len(rows),
        "n_scored": len(scored),
        "n_invalid": len(rows) - len(scored),
        "invalid_ids": [r["case_id"] for r in rows if not r["scored"]],
        "parse_status": dict(Counter(r["parse_status"] for r in rows)),
        "pass": {
            "n_pass": int(sum(passes)),
            "rate_scored": mean_ci(passes),
            "rate_invalid_as_fail": round(sum(passes) / len(rows), 4) if rows else None,
            "rule": rubric.pass_rule.model_dump(),
        },
        "overall": {**mean_ci(overall), "sd": round(statistics.pstdev(overall), 4) if len(overall) > 1 else 0.0},
        "criteria": {c: mean_ci([r["scores"][c] for r in scored]) for c in CRITERIA},
        "by_mode": {
            m: {"n": len(g), "pass_rate": round(sum(1 for r in g if r["passed"]) / len(g), 4), "mean_overall": round(statistics.fmean(r["overall"] for r in g), 4)}
            for m in ("reference", "free")
            if (g := [r for r in scored if r["mode"] == m])
        },
        "unsupported_high_score_cases": sum(1 for r in scored if r["unsupported_high_scores"]),
        "stored_mismatch_cases": [r["case_id"] for r in rows if r["stored_mismatch"]],
        "rubric": {"version": rubric.version, "weights": rubric.weights},
        "per_case": [
            {k: r[k] for k in ("case_id", "mode", "parse_status", "scores", "overall", "passed")} for r in rows
        ],
    }


# ---------------------------------------------------------------- calls, tokens, cost

@dataclass(frozen=True)
class Prices:
    """USD per 1M tokens. None = no price configured: cost is reported as null, never guessed.
    `free_tier` = the key bills nothing, so the actual cost is 0 and the list price is only an equivalent."""

    input_per_m: float | None = None
    output_per_m: float | None = None
    free_tier: bool = False
    source_url: str = ""
    retrieved: str = ""
    tier: str = ""

    @property
    def known(self) -> bool:
        return self.input_per_m is not None and self.output_per_m is not None

    def cost(self, tokens_in: int, tokens_out: int) -> float | None:
        """List-price cost of these tokens (what a paid key would be billed)."""
        if not self.known:
            return None
        return round((tokens_in * self.input_per_m + tokens_out * self.output_per_m) / 1_000_000, 6)  # type: ignore[operator]

    def describe(self) -> dict[str, Any]:
        return {
            "input_usd_per_1m": self.input_per_m, "output_usd_per_1m": self.output_per_m, "free_tier": self.free_tier,
            "source_url": self.source_url, "retrieved": self.retrieved, "tier": self.tier,
        }


def load_prices(path: str | Path = PRICING_PATH) -> Prices:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Prices(
        d["input_usd_per_1m"], d["output_usd_per_1m"], d["billing"] == "free_tier",
        d["source_url"], str(d["retrieved"]), d["tier"],
    )


def cost_fields(prices: Prices, net_in: int, net_out: int, all_in: int, all_out: int) -> dict[str, Any]:
    """actual = what this key was billed (0 on a free tier); list_equivalent = the same network tokens at list price;
    if_uncached = every call, cache hits included, at list price."""
    eq = prices.cost(net_in, net_out)
    return {
        "cost_usd_actual": 0.0 if prices.free_tier else eq,
        "cost_usd_list_equivalent": eq,
        "cost_usd_if_uncached": prices.cost(all_in, all_out),
        "priced": prices.known,
    }


def usage_from_calls(calls: Sequence[Mapping[str, Any]], prices: Prices | None = None) -> dict[str, Any]:
    """Calls are CallRecord dicts. `network` = calls that really hit the API (what the quota and a bill count);
    cache hits are reported separately. Costs: see cost_fields."""
    prices = prices or Prices()
    net = [c for c in calls if not c["cached"]]
    t_in, t_out = sum(c["tokens_in"] for c in calls), sum(c["tokens_out"] for c in calls)
    n_in, n_out = sum(c["tokens_in"] for c in net), sum(c["tokens_out"] for c in net)
    return {
        "calls": len(calls),
        "network_calls": len(net),
        "cached_calls": len(calls) - len(net),
        "reasks": sum(1 for c in calls if c["tag"].endswith("|reask")),
        "tokens_in": t_in, "tokens_out": t_out,
        "network_tokens_in": n_in, "network_tokens_out": n_out,
        **cost_fields(prices, n_in, n_out, t_in, t_out),
    }


def pointwise_calls(doc: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [c for r in doc["results"] for c in r["calls"]]


def pairwise_calls(pairs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [c for p in pairs for o in p["orders"] for c in o["calls"]]


def generator_usage(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Generator tokens as recorded by `judge generate` in each case's meta (generator-side, separate ledger)."""
    meta = [c.get("meta", {}) for c in cases]
    return {
        "model": next((m["gen_model"] for m in meta if m.get("gen_model")), ""),
        "prompt_version": next((m["prompt_version"] for m in meta if m.get("prompt_version")), ""),
        "calls": len(cases),
        "tokens_in": sum(m.get("tokens_in", 0) for m in meta),
        "tokens_out": sum(m.get("tokens_out", 0) for m in meta),
    }


# ---------------------------------------------------------------- pairwise win rate

def win_rate(pairs: Sequence[Mapping[str, Any]], x: str = "A", *, ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Win rate of `x` over its opponent from order-swapped pair verdicts: (wins + 0.5 ties) / valid pairs.
    A pair's verdict is already the order-swap result (disagreeing orders = TIE). INVALID pairs are counted
    and excluded from the rate. The CI resamples the per-pair scores (1 / 0.5 / 0)."""
    ps = [p for p in pairs if ids is None or p["case_id"] in ids]
    valid = [p for p in ps if p["verdict"] != _INVALID]
    scores = [1.0 if p["verdict"] == x else 0.5 if p["verdict"] == _TIE else 0.0 for p in valid]
    flippable = [p for p in ps if p.get("flipped") is not None]
    out = mean_ci(scores)
    return {
        "n": len(ps), "n_invalid": len(ps) - len(valid),
        "wins_x": sum(1 for p in valid if p["verdict"] == x),
        "wins_y": sum(1 for p in valid if p["verdict"] not in (x, _TIE)),
        "ties": sum(1 for p in valid if p["verdict"] == _TIE),
        "win_rate": out["mean"], "ci_low": out["ci_low"], "ci_high": out["ci_high"],
        "flips": sum(1 for p in flippable if p["flipped"]),
        "flip_rate": round(sum(1 for p in flippable if p["flipped"]) / len(flippable), 4) if flippable else None,
    }
