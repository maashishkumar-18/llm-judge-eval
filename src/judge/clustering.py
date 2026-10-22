"""Score clustering (J7, exploratory): naive holistic 1-10 judge vs the anchored per-criterion 1-5 judge.

Part 1, spread on the same 12 suite cases (config A outputs, the first 12 in suite order; fixed before the run):
  naive score (1-10) vs mitigated overall (1-5, recomputed) and mitigated faithfulness (integers 1-5).
  Metrics: mean, population sd, sd / scale width (so the two scales compare), distinct values used, modal value and its
  share, share at the scale maximum. A narrow spread = clustering. The mitigated overall is a weighted mean of three
  integers, so its distinct-value count is inflated relative to an integer scale; faithfulness is the like-for-like row.
Part 2, separation of good vs corrupted outputs, from verdicts already on disk (no new judge calls):
  good = terse-correct answers (J6), corrupted = confidently wrong, verbose wrong (J6) and right-answer-with-invented-reasoning
  (J6b), each by construction. Measures: AUC = P(good scores above corrupted) with ties counted half (scale-free, so 1-10
  and 1-5 compare), mean gap / scale width, perfect separation (lowest good above highest corrupted).
  Also on the 12 suite cases: good = passes the deterministic check (reference facts + the case's own constraints).
All of it is exploratory: N is 6 per probe group and 12 suite cases.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from judge.checkpoint import run_checkpointed
from judge.probes import NaiveResult, judge_naive_case
from judge.schemas import Case, Rubric, load_rubric
from judge.selfpref import correct_by_reference
from judge.checks import complies

N_CASES = 12
NAIVE_SCALE = (1, 10)
MIT_SCALE = (1, 5)
CORRUPTED = ("wrong", "verbose_wrong", "hard")
EPS = 1e-9


def cluster_ids(cases: Sequence[Case], n: int = N_CASES) -> list[str]:
    """Pre-registered rule: the first `n` cases of config A in suite order."""
    return [c.id for c in cases][:n]


# ---------------------------------------------------------------- spread

def spread(values: Sequence[float], scale: tuple[float, float]) -> dict[str, Any]:
    lo, hi = scale
    n = len(values)
    if n == 0:
        return {"n": 0}
    sd = statistics.pstdev(values) if n > 1 else 0.0
    counts = Counter(round(v, 6) for v in values)
    mode, mode_n = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    return {
        "n": n,
        "scale": [lo, hi],
        "mean": round(statistics.fmean(values), 4),
        "sd": round(sd, 4),
        "sd_over_scale_width": round(sd / (hi - lo), 4),
        "min": min(values), "max": max(values),
        "range_used_over_scale_width": round((max(values) - min(values)) / (hi - lo), 4),
        "distinct_values": len(counts),
        "modal_value": mode, "modal_share": round(mode_n / n, 4),
        "share_at_scale_max": round(sum(1 for v in values if v >= hi - EPS) / n, 4),
        "value_counts": {str(k): v for k, v in sorted(counts.items())},
    }


# ---------------------------------------------------------------- separation

def auc(good: Sequence[float], bad: Sequence[float]) -> float | None:
    """P(a random good output outscores a random corrupted one), ties count half. 1.0 = perfect, 0.5 = chance."""
    if not good or not bad:
        return None
    wins = sum(1.0 if g > b else 0.5 if g == b else 0.0 for g in good for b in bad)
    return round(wins / (len(good) * len(bad)), 4)


def separation(good: Sequence[float], bad: Sequence[float], scale: tuple[float, float]) -> dict[str, Any]:
    lo, hi = scale
    if not good or not bad:
        return {"n_good": len(good), "n_corrupted": len(bad), "auc": None}
    return {
        "n_good": len(good), "n_corrupted": len(bad),
        "mean_good": round(statistics.fmean(good), 4), "mean_corrupted": round(statistics.fmean(bad), 4),
        "gap_over_scale_width": round((statistics.fmean(good) - statistics.fmean(bad)) / (hi - lo), 4),
        "auc": auc(good, bad),
        "perfect_separation": min(good) > max(bad),
        "min_good": min(good), "max_corrupted": max(bad),
    }


# ---------------------------------------------------------------- naive run on the suite

async def run_naive_suite(
    client: Any, cases: Sequence[Case], ids: Sequence[str], *, checkpoint_path: str | Path, nonce: str = "",
    fresh: bool = False, concurrency: int = 2,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    cp = Path(checkpoint_path)
    if fresh and cp.exists():
        cp.unlink()
    by_id = {c.id: c for c in cases}

    async def one(cid: str) -> dict[str, Any]:
        r: NaiveResult = await judge_naive_case(client, by_id[cid], label="cluster", nonce=nonce)
        return asdict(r)

    status = await run_checkpointed(list(ids), one, cp, concurrency=concurrency, persist=lambda r: r["error_kind"] != "call")
    return {i: status.results[i] for i in ids if i in status.results}, status.remaining


# ---------------------------------------------------------------- the analysis (pure; from saved verdicts)

def suite_label(case: Case) -> bool:
    """Deterministic by-construction label: reference facts (reference-based cases) and the case's own constraints."""
    if case.reference_based:
        return correct_by_reference(case, case.model_output)
    return complies(case.model_output, case.meta.get("checks", []))


def _mit_scores(r: Mapping[str, Any], rubric: Rubric) -> tuple[float, float] | None:
    sc = r.get("scores")
    return (rubric.overall(sc), float(sc["faithfulness"])) if sc else None


def analyze(
    *, cases: Sequence[Case], ids: Sequence[str], naive: Mapping[str, Mapping[str, Any]],
    mitigated_A: Sequence[Mapping[str, Any]], mitigated_B: Sequence[Mapping[str, Any]],
    probes: Mapping[str, Mapping[str, Any]], hard: Mapping[str, Mapping[str, Any]], rubric: Rubric | None = None,
) -> dict[str, Any]:
    rubric = rubric or load_rubric()
    by_case = {c.id: c for c in cases}
    mit_a = {r["case_id"]: r for r in mitigated_A}
    got = [i for i in ids if i in naive and naive[i].get("score") is not None and _mit_scores(mit_a.get(i, {}), rubric)]
    nv = [float(naive[i]["score"]) for i in got]
    ov = [_mit_scores(mit_a[i], rubric)[0] for i in got]  # type: ignore[index]
    fa = [_mit_scores(mit_a[i], rubric)[1] for i in got]  # type: ignore[index]
    all_scored = [r for r in list(mitigated_A) + list(mitigated_B) if r.get("scores")]
    spread_doc = {
        "cases": got,
        "naive_1_10": spread(nv, NAIVE_SCALE),
        "mitigated_overall_1_5": spread(ov, MIT_SCALE),
        "mitigated_faithfulness_1_5": spread(fa, MIT_SCALE),
        "mitigated_overall_1_5_all_A_and_B": spread([rubric.overall(r["scores"]) for r in all_scored], MIT_SCALE),
        "mitigated_faithfulness_1_5_all_A_and_B": spread([float(r["scores"]["faithfulness"]) for r in all_scored], MIT_SCALE),
    }

    def probe_vals(stage: str, judge: str, variant: str) -> list[float]:
        out = []
        for k in sorted(probes):
            st, j, v, _ = k.split("|")
            if (st, j, v) != (stage, judge, variant):
                continue
            r = probes[k]
            if judge == "naive" and r.get("score") is not None:
                out.append(float(r["score"]))
            elif judge == "mitigated" and r.get("scores"):
                out.append(rubric.overall(r["scores"]))
        return out

    def hard_vals(judge: str) -> list[float]:
        out = []
        for k in sorted(hard):
            _, j, _, _ = k.split("|")
            r = hard[k]
            if j != judge:
                continue
            if judge == "naive" and r.get("score") is not None:
                out.append(float(r["score"]))
            elif judge == "mitigated" and r.get("scores"):
                out.append(rubric.overall(r["scores"]))
        return out

    sep: dict[str, Any] = {}
    for judge, scale in (("naive", NAIVE_SCALE), ("mitigated", MIT_SCALE)):
        good = probe_vals("ref", judge, "terse")
        groups = {"wrong": probe_vals("ref", judge, "wrong"), "verbose_wrong": probe_vals("ref", judge, "verbose_wrong"), "hard": hard_vals(judge)}
        sep[judge] = {
            "scale": list(scale),
            "good_definition": "terse-correct answers on the 6 probe bases (J6)",
            "by_corruption": {g: separation(good, v, scale) for g, v in groups.items()},
            "all_corrupted": separation(good, [x for v in groups.values() for x in v], scale),
            "good_spread": spread(good, scale),
            "corrupted_spread": spread([x for v in groups.values() for x in v], scale),
        }

    lab = {i: suite_label(by_case[i]) for i in got}
    suite_sep = {}
    for judge, vals, scale in (("naive", dict(zip(got, nv)), NAIVE_SCALE), ("mitigated", dict(zip(got, ov)), MIT_SCALE)):
        suite_sep[judge] = separation([vals[i] for i in got if lab[i]], [vals[i] for i in got if not lab[i]], scale)
    return {
        "spread_same_cases": spread_doc,
        "separation_probes": sep,
        "separation_suite_A": {
            "label": "deterministic: reference facts (reference-based) / the case's own constraints (reference-free)",
            "good_ids": [i for i in got if lab[i]], "defective_ids": [i for i in got if not lab[i]],
            "by_judge": suite_sep,
        },
        "per_case": [
            {"case_id": i, "label_good": lab[i], "naive_1_10": naive[i]["score"],
             "mitigated_overall": round(_mit_scores(mit_a[i], rubric)[0], 4), "mitigated_scores": mit_a[i]["scores"]}  # type: ignore[index]
            for i in got
        ],
    }


def render_tables(summary: Mapping[str, Any]) -> str:
    sp = summary["spread_same_cases"]
    rows = [("naive 1-10", sp["naive_1_10"]), ("mitigated overall 1-5", sp["mitigated_overall_1_5"]),
            ("mitigated faithfulness 1-5", sp["mitigated_faithfulness_1_5"]),
            ("mitigated overall, all A+B", sp["mitigated_overall_1_5_all_A_and_B"]),
            ("mitigated faithfulness, all A+B", sp["mitigated_faithfulness_1_5_all_A_and_B"])]
    out = ["Spread (same 12 cases unless marked)", f"{'':32s} {'n':>3s} {'mean':>6s} {'sd':>6s} {'sd/width':>8s} {'distinct':>8s} {'modal':>6s} {'modal%':>7s} {'at max%':>8s}"]
    for name, s in rows:
        out.append(f"{name:32s} {s['n']:3d} {s['mean']:6.2f} {s['sd']:6.2f} {s['sd_over_scale_width']:8.3f} {s['distinct_values']:8d} {s['modal_value']:6} {s['modal_share']*100:6.0f}% {s['share_at_scale_max']*100:7.0f}%")
    out += ["", "Separation, terse-correct (good) vs corrupted, AUC [gap/scale width] (probes N=6 per group)"]
    out.append(f"{'':10s} {'wrong':>16s} {'verbose_wrong':>16s} {'hard':>16s} {'all corrupted':>16s}")
    for judge in ("naive", "mitigated"):
        s = summary["separation_probes"][judge]
        cells = [s["by_corruption"][g] for g in ("wrong", "verbose_wrong", "hard")] + [s["all_corrupted"]]
        out.append(f"{judge:10s} " + " ".join(f"{c['auc']!s:>7s} [{c['gap_over_scale_width']:.2f}]".rjust(16) for c in cells))
    ss = summary["separation_suite_A"]
    out += ["", f"Suite A, deterministic label: good {ss['good_ids']}, defective {ss['defective_ids']}"]
    for judge in ("naive", "mitigated"):
        s = ss["by_judge"][judge]
        out.append(f"{judge:10s} AUC {s['auc']}  mean good {s.get('mean_good')}  mean defective {s.get('mean_corrupted')}  perfect {s.get('perfect_separation')}")
    return "\n".join(out)


def write_document(doc: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
