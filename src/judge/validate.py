"""Judge validation (J8): agreement / Cohen's kappa against labels, test-retest, adversarial summary.

Everything except `run_retest` works on saved results and makes ZERO judge calls (`judge validate`).

Labels (positive class = "pass"):
* by construction (deterministic, no judge involved): suite answers = `clustering.suite_label` (reference facts +
  the case's own constraints; reference-free cases have constraints only), probe answers = states the true fact, not
  the false one, and stays inside the case's length limits (so "padded" is a fail by its length defect); the
  fabricated-reasoning answers are fail by construction (approved texts with an invented justification).
* human: `labels/human_labels.jsonl`, one JSON object per line, written BEFORE looking at verdicts:
  {"case_id": "c01", "config": "A", "label": "pass", "note": "optional"}. `config` defaults to "A"; label is
  pass/fail (also true/false/1/0); null or blank labels are skipped and counted. Duplicates and unknown ids are errors.

Judge predictions: mitigated = the rubric pass rule recomputed from the stored criterion scores; naive = score >=
probes.NAIVE_FOOLED_MIN (the acceptance threshold fixed in J6; the naive judge has no pass rule of its own).

Kappa is implemented here with integer arithmetic (unit-tested on a hand-computed example). It is undefined (None,
with a reason) when chance agreement is 1, i.e. both raters constant and equal. Every number carries its N, every
small-N result is exploratory, and every metric ships with negative controls (`negative_controls`)."""

from __future__ import annotations

import json
import random
import statistics
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence

from judge.aggregate import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, percentile
from judge.clustering import suite_label
from judge.pointwise import judge_suite
from judge.probes import BASES, NAIVE_FOOLED_MIN, is_correct, length_checks_pass, overall_noif
from judge.schemas import CRITERIA, Case, Rubric, load_rubric
from judge.selfpref import clopper_pearson

PERMUTATIONS = 10_000
PERM_SEED = 8
PERM_MEAN_TOL = 0.10  # pre-registered: mean kappa under shuffled labels must be within +-0.10 of 0
LABELS_PATH = "labels/human_labels.jsonl"
LABEL_FORMAT ='one JSON object per line: {"case_id": "c01", "config": "A", "label": "pass" or "fail", "note": "optional"}; config defaults to A'


# ---------------------------------------------------------------- kappa and agreement

def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float | None:
    """Cohen's kappa = (po - pe) / (1 - pe). Integer arithmetic: with n items, `agree` agreements and
    `pe_n` = sum_k count_a(k) * count_b(k), kappa = (agree * n - pe_n) / (n^2 - pe_n). None if n == 0 or pe == 1."""
    if len(a) != len(b):
        raise ValueError("raters must label the same items")
    n = len(a)
    if n == 0:
        return None
    ca, cb = Counter(a), Counter(b)
    agree = sum(x == y for x, y in zip(a, b))
    pe_n = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys())
    if pe_n == n * n:
        return None
    return (agree * n - pe_n) / (n * n - pe_n)


def _kappa_ci(pred: Sequence[bool], label: Sequence[bool], resamples: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """Percentile bootstrap over items. Resamples where kappa is undefined are dropped and counted."""
    n, rng = len(pred), random.Random(BOOTSTRAP_SEED)
    ks, undefined = [], 0
    for _ in range(resamples):
        idx = rng.choices(range(n), k=n)
        k = cohen_kappa([pred[i] for i in idx], [label[i] for i in idx])
        if k is None:
            undefined += 1
        else:
            ks.append(k)
    if len(ks) < resamples // 2:  # most resamples degenerate: an interval would mislead
        return {"low": None, "high": None, "undefined_resamples": undefined}
    ks.sort()
    return {"low": round(percentile(ks, 0.025), 4), "high": round(percentile(ks, 0.975), 4), "undefined_resamples": undefined}


def agreement_stats(pred: Sequence[bool], label: Sequence[bool]) -> dict[str, Any]:
    """Judge pass/fail (pred) vs labels, positive = pass. Includes the always-pass agreement (= label pass rate) so a
    reader sees how much of the raw agreement is just the base rate."""
    n = len(pred)
    tp = sum(p and l for p, l in zip(pred, label))
    fp = sum(p and not l for p, l in zip(pred, label))
    fn = sum(l and not p for p, l in zip(pred, label))
    tn = n - tp - fp - fn
    k = cohen_kappa(pred, label)
    ci = clopper_pearson(tp + tn, n)
    out: dict[str, Any] = {
        "n": n, "agree": tp + tn, "agreement": round((tp + tn) / n, 4) if n else None,
        "agreement_ci95_exact": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
        "kappa": None if k is None else round(k, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "label_pass": tp + fn, "judge_pass": tp + fp,
        "always_pass_agreement": round((tp + fn) / n, 4) if n else None,
        "false_pass": {"k": fp, "n": fp + tn},  # judge passes an answer labelled fail
        "false_fail": {"k": fn, "n": tp + fn},  # judge fails an answer labelled pass
    }
    if k is None:
        out["kappa_undefined_reason"] = "no items" if n == 0 else "both raters constant and equal (chance agreement = 1)"
        out["kappa_ci95"] = None
    else:
        out["kappa_ci95"] = _kappa_ci(pred, label)
    return out


def kappa_controls(pred: Sequence[bool], label: Sequence[bool], *, perms: int = PERMUTATIONS, seed: int = PERM_SEED) -> list[dict[str, Any]]:
    """Negative controls for one kappa. [] when the labels are constant (kappa is not defined, nothing to control)."""
    n = len(label)
    if len(set(label)) < 2:
        return []
    out = []
    for const in (True, False):
        k = cohen_kappa([const] * n, label)
        out.append({"name": f"always-{'pass' if const else 'fail'} judge", "expect": "kappa = 0 (no information)",
                    "observed": k, "agreement": round(sum(c == l for c, l in zip([const] * n, label)) / n, 4), "ok": k == 0})
    observed = cohen_kappa(pred, label)
    inv = cohen_kappa([not p for p in pred], label)
    out.append({"name": "inverted judge", "expect": "kappa below the real judge's (sign check)",
                "observed": None if inv is None else round(inv, 4),
                "ok": observed is None or inv is None or inv < observed or (observed == inv == 0)})
    rng, shuffled, ks = random.Random(seed), list(label), []
    for _ in range(perms):
        rng.shuffle(shuffled)
        k = cohen_kappa(pred, shuffled)
        if k is not None:
            ks.append(k)
    if ks:
        mean = statistics.fmean(ks)
        ge = sum(k >= (observed if observed is not None else float("inf")) for k in ks)
        out.append({"name": f"labels shuffled x{len(ks)}", "expect": f"mean kappa within +-{PERM_MEAN_TOL} of 0",
                    "observed": round(mean, 4), "ok": abs(mean) <= PERM_MEAN_TOL,
                    "p_perm_ge_observed": None if observed is None else round((ge + 1) / (len(ks) + 1), 4)})
    return out


# ---------------------------------------------------------------- human labels

class LabelError(ValueError):
    pass


_TRUE, _FALSE = {"pass", "true", "1", "yes", "good"}, {"fail", "false", "0", "no", "bad"}


def parse_label(v: Any) -> bool | None:
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    s = str(v).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    raise LabelError(f"label must be pass/fail (or true/false/1/0), got {v!r}")


def load_human_labels(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    """{(config, case_id): {"label": bool, "note": str}}; blank labels are skipped. Raises LabelError on bad rows."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            cid = row.get("case_id") or row["id"]
            label = parse_label(row.get("label", row.get("human_label")))
        except (json.JSONDecodeError, KeyError, AttributeError, LabelError) as e:
            raise LabelError(f"{path} line {n}: {e}") from e
        key = (str(row.get("config") or "A"), str(cid))
        if key in out:
            raise LabelError(f"{path} line {n}: duplicate label for config {key[0]} case {key[1]}")
        out[key] = {"label": label, "note": str(row.get("note", ""))}
    return out


# ---------------------------------------------------------------- labelled items

def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


@dataclass
class Entry:
    origin: str  # suite:A | suite:B | probe:<stage>:<variant>
    case_id: str
    text: str
    mode: str  # reference | free
    label: bool  # by construction
    why: str
    mit_scores: dict[str, int] | None = None
    mit_seen: bool = False  # a mitigated verdict row exists (usable or INVALID)
    naive_score: float | None = None
    naive_seen: bool = False

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.case_id, _norm(self.text), self.mode)


def mit_pass(e: Entry, rubric: Rubric) -> bool:
    return rubric.passes(e.mit_scores)  # type: ignore[arg-type]


def naive_pass(e: Entry) -> bool:
    return e.naive_score >= NAIVE_FOOLED_MIN  # type: ignore[operator]


def _attach(e: Entry, judge: str, res: Mapping[str, Any] | None) -> None:
    if res is None:
        return
    if judge == "mitigated":
        e.mit_seen, e.mit_scores = True, res.get("scores")
    else:
        e.naive_seen, e.naive_score = True, res.get("score")


def build_entries(
    *, suite: Mapping[str, Sequence[Case]], pointwise: Mapping[str, Mapping[str, Any]], probes: Mapping[str, Any],
    hard: Mapping[str, Any], naive_suite: Mapping[str, Mapping[str, Any]],
) -> list[Entry]:
    """Every labelled, judged answer. `suite`/`pointwise` are keyed by generator config ("A", "B"); `naive_suite`
    holds the naive verdicts on config A's cases (J7 clustering). Probe 'original' rows attach to the suite:A entry."""
    entries: list[Entry] = []
    for cfg, cases in suite.items():
        res = {r["case_id"]: r for r in pointwise[cfg]["results"]}
        for c in cases:
            why = ("reference facts + the case's own constraints" if c.reference_based
                   else "the case's own constraints only (no fact check: reference-free)")
            e = Entry(f"suite:{cfg}", c.id, c.model_output, "reference" if c.reference_based else "free", suite_label(c), why)
            _attach(e, "mitigated", res.get(c.id))
            if cfg == "A":
                _attach(e, "naive", naive_suite.get(c.id) or probes["results"].get(f"ref|naive|original|{c.id}"))
            entries.append(e)
    base = {c.id: c for c in suite["A"] if c.id in BASES}
    answers, results = probes["answers"], probes["results"]
    for cid in BASES:
        for variant in ("terse", "padded", "wrong", "verbose_wrong"):
            text = answers[f"{cid}|{variant}"]
            e = Entry(f"probe:ref:{variant}", cid, text, "reference",
                      is_correct(text, cid) and length_checks_pass(text, base[cid]),
                      "states the true fact, not the false one, within the case's length limits")
            for judge in ("mitigated", "naive"):
                _attach(e, judge, results.get(f"ref|{judge}|{variant}|{cid}"))
            entries.append(e)
        text = answers[f"{cid}|wrong"]
        e = Entry("probe:free:wrong", cid, text, "free", is_correct(text, cid), "reference hidden; the false fact is stated")
        for judge in ("mitigated", "naive"):
            _attach(e, judge, results.get(f"free|{judge}|wrong|{cid}"))
        entries.append(e)
        e = Entry("probe:hard:hard", cid, hard["texts"][cid], "reference", False,
                  "right key fact plus an invented justification the passage does not support (fixed by construction)")
        for judge in ("mitigated", "naive"):
            _attach(e, judge, hard["results"].get(f"hard|{judge}|hard|{cid}"))
        entries.append(e)
    return entries


def dedupe(entries: Sequence[Entry]) -> tuple[list[Entry], int]:
    """First entry per (case, normalised text, mode); also returns how many later entries disagreed on a verdict."""
    seen: dict[tuple[str, str, str], Entry] = {}
    conflicts = 0
    for e in entries:
        first = seen.setdefault(e.key, e)
        if first is not e and (first.mit_scores != e.mit_scores and e.mit_scores is not None and first.mit_scores is not None):
            conflicts += 1
    return list(seen.values()), conflicts


def strata(entries: Sequence[Entry]) -> dict[str, list[Entry]]:
    suite = [e for e in entries if e.origin.startswith("suite:")]
    probe = [e for e in entries if e.origin.startswith("probe:")]
    return {
        "suite_A": [e for e in suite if e.origin == "suite:A"],
        "suite_B": [e for e in suite if e.origin == "suite:B"],
        "suite_distinct": dedupe(suite)[0],
        "probes": probe,
        "all_distinct": dedupe(suite + probe)[0],
    }


def judge_report(es: Sequence[Entry], judge: str, rubric: Rubric, labels: Sequence[bool] | None = None) -> dict[str, Any]:
    """Agreement/kappa of one judge on the entries it judged, vs by-construction labels (or the given labels)."""
    seen = [i for i, e in enumerate(es) if (e.mit_seen if judge == "mitigated" else e.naive_seen)]
    usable = [i for i in seen if (es[i].mit_scores is not None if judge == "mitigated" else es[i].naive_score is not None)]
    pred = [mit_pass(es[i], rubric) if judge == "mitigated" else naive_pass(es[i]) for i in usable]
    lab = [(labels[i] if labels is not None else es[i].label) for i in usable]
    out = agreement_stats(pred, lab)
    out["n_invalid"] = len(seen) - len(usable)
    out["negative_controls"] = kappa_controls(pred, lab)
    out["disagreements"] = [
        {"case_id": es[i].case_id, "origin": es[i].origin, "label": l, "judge_pass": p,
         "scores": es[i].mit_scores if judge == "mitigated" else es[i].naive_score}
        for i, p, l in zip(usable, pred, lab) if p != l
    ]
    return out


def construction_report(entries: Sequence[Entry], rubric: Rubric) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, es in strata(entries).items():
        out[name] = {"n_items": len(es), "label_pass": sum(e.label for e in es), "label_fail": sum(not e.label for e in es),
                     "mitigated": judge_report(es, "mitigated", rubric), "naive": judge_report(es, "naive", rubric)}
    return out


def human_report(entries: Sequence[Entry], labels: Mapping[tuple[str, str], Mapping[str, Any]], rubric: Rubric) -> dict[str, Any]:
    by_key = {(e.origin.split(":")[1], e.case_id): e for e in entries if e.origin.startswith("suite:")}
    matched = [(k, by_key[k]) for k in labels if k in by_key and labels[k]["label"] is not None]
    unmatched = sorted(f"{c}:{i}" for (c, i) in labels if (c, i) not in by_key)
    skipped = sorted(f"{c}:{i}" for (c, i), v in labels.items() if v["label"] is None)
    es = [e for _, e in matched]
    hl = [bool(labels[k]["label"]) for k, _ in matched]
    out: dict[str, Any] = {
        "status": "ok" if es else "no usable labels", "n_labelled": len(es), "n_skipped_blank": len(skipped),
        "unmatched": unmatched, "n_distinct_texts": len({e.key for e in es}),
        "human_label_pass": sum(hl), "human_label_fail": len(hl) - sum(hl),
        "mitigated": judge_report(es, "mitigated", rubric, hl) if es else None,
        "naive": judge_report(es, "naive", rubric, hl) if es else None,
    }
    if es:
        out["by_construction_vs_human"] = agreement_stats([e.label for e in es], hl)
        out["by_construction_vs_human"]["disagreements"] = [
            {"case_id": e.case_id, "config": e.origin.split(":")[1], "human": h, "by_construction": e.label}
            for e, h in zip(es, hl) if e.label != h]
    return out


# ---------------------------------------------------------------- test-retest

def retest_ids(cases: Sequence[Case]) -> list[str]:
    """Which cases are re-judged: every second case in suite order (10 of 20). Fixed before the run; not result-driven."""
    return [c.id for c in cases[::2]]


async def run_retest(
    client: Any, cases: Sequence[Case], ids: Sequence[str], reruns: int, *, cases_file: str = "",
    out_dir: str | Path = "results", checkpoint_dir: str | Path = ".cache/checkpoints", fresh: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Judge `ids` again `reruns` times with a distinct nonce per rerun (cache bypassed). Returns (docs, ids not yet done)."""
    chosen = [c for c in cases if c.id in set(ids)]
    docs: list[dict[str, Any]] = []
    left: list[str] = []
    for k in range(1, reruns + 1):
        label = f"retest{k}"
        doc, remaining = await judge_suite(
            client, chosen, label, out_path=Path(out_dir) / f"pointwise_{label}.json",
            checkpoint_path=Path(checkpoint_dir) / f"pointwise_{label}.jsonl", cases_file=cases_file, nonce=label, fresh=fresh,
        )
        docs.append(doc)
        left += [f"{label}:{i}" for i in remaining]
        if remaining:
            break
    return docs, left


def _sd(xs: Sequence[float]) -> float | None:
    return statistics.stdev(xs) if len(xs) >= 2 else None  # sample sd (n-1): runs are a sample of the judge's draws


def analyze_retest(runs: Sequence[Mapping[str, Any]], rubric: Rubric | None = None) -> dict[str, Any]:
    """runs[0] = the baseline run (nonce ""), the rest are nonce reruns; each is a pointwise results document.
    Overall and pass/fail are recomputed from the stored criterion scores. INVALID runs are excluded per case and counted."""
    rubric = rubric or load_rubric()
    tables = [{r["case_id"]: r for r in d["results"]} for d in runs]
    ids = [i for i in (r["case_id"] for r in runs[-1]["results"]) if all(i in t for t in tables)]
    rows = []
    for cid in ids:
        per = []
        for d, t in zip(runs, tables):
            sc = t[cid].get("scores")
            per.append({"run": d["label"], "scores": sc, "overall": rubric.overall(sc) if sc else None,
                        "passed": rubric.passes(sc) if sc else None})
        ok = [p for p in per if p["scores"]]
        ov = [p["overall"] for p in ok]
        rows.append({
            "case_id": cid, "runs": per, "n_scored": len(ok),
            "pass_flip": (len({p["passed"] for p in ok}) > 1) if len(ok) >= 2 else None,
            "overall_sd": _sd(ov), "overall_range": (max(ov) - min(ov)) if len(ov) >= 2 else None,
            "criteria_identical": {c: len({p["scores"][c] for p in ok}) == 1 for c in CRITERIA} if len(ok) >= 2 else None,
        })
    comp = [r for r in rows if r["n_scored"] >= 2]
    flips = [r for r in comp if r["pass_flip"]]
    sds = [r["overall_sd"] for r in comp]
    ci = clopper_pearson(len(flips), len(comp))
    cells = {c: [r["criteria_identical"][c] for r in comp] for c in CRITERIA}
    pair_agree = {}
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            both = [r for r in rows if r["runs"][i]["scores"] and r["runs"][j]["scores"]]
            pair_agree[f"{runs[i]['label']}~{runs[j]['label']}"] = {
                "n": len(both), "pass_agreement": round(sum(r["runs"][i]["passed"] == r["runs"][j]["passed"] for r in both) / len(both), 4) if both else None,
                "mean_abs_overall_diff": round(statistics.fmean(abs(r["runs"][i]["overall"] - r["runs"][j]["overall"]) for r in both), 4) if both else None}
    calls = {d["label"]: [c for r in d["results"] for c in r["calls"]] for d in runs[1:]}
    cached = {k: sum(c["cached"] for c in v) for k, v in calls.items()}
    nonces = [d.get("nonce", "") for d in runs]
    controls = [
        {"name": "nonce runs bypass the cache", "expect": "0 cached calls in every nonce run (a re-run of the same command re-serves its saved draws and shows >0)",
         "observed": cached, "ok": all(v == 0 for v in cached.values())},
        {"name": "nonces distinct and non-empty", "expect": "baseline '' and each rerun its own nonce",
         "observed": nonces, "ok": nonces[0] == "" and all(nonces[1:]) and len(set(nonces)) == len(nonces)},
    ]
    return {
        "n_cases": len(rows), "n_runs": len(runs), "run_labels": [d["label"] for d in runs],
        "n_comparable": len(comp), "n_complete": sum(r["n_scored"] == len(runs) for r in rows),
        "n_invalid_per_run": {d["label"]: sum(1 for r in d["results"] if not r.get("scores")) for d in runs},
        "pass_flips": {"k": len(flips), "n": len(comp), "rate": round(len(flips) / len(comp), 4) if comp else None,
                       "ci95_exact": [round(ci[0], 4), round(ci[1], 4)] if ci else None, "cases": [r["case_id"] for r in flips]},
        "overall_sd": {"mean": round(statistics.fmean(sds), 4) if sds else None, "max": round(max(sds), 4) if sds else None,
                       "cases_with_any_difference": sum(1 for r in comp if r["overall_range"]), "n": len(comp), "sd_kind": "sample (n-1)"},
        "criterion_exact_agreement": {c: {"k": sum(v), "n": len(v)} for c, v in cells.items()},
        "run_pairs": pair_agree, "per_case": rows, "negative_controls": controls,
        "note": "temperature and model fixed; the only change between runs is the nonce, so differences are the judge's own sampling noise",
    }


# ---------------------------------------------------------------- adversarial summary (from probes.json and friends)

def _accepted(r: Mapping[str, Any] | None, judge: str) -> bool | None:
    if r is None:
        return None
    if judge == "mitigated":
        return None if not r.get("scores") else r["scores"]["faithfulness"] >= 4
    return None if r.get("score") is None else r["score"] >= NAIVE_FOOLED_MIN


def _count(results: Mapping[str, Any], judge: str, stage: str, variant: str, bases: Sequence[str]) -> dict[str, Any]:
    flags = [(c, _accepted(results.get(f"{stage}|{judge}|{variant}|{c}"), judge)) for c in bases]
    got = [(c, f) for c, f in flags if f is not None]
    return {"k": sum(f for _, f in got), "n": len(got), "n_invalid": len(flags) - len(got), "cases": [c for c, f in got if f]}


def _noif_or_naive(r: Mapping[str, Any], judge: str, rubric: Rubric) -> float | None:
    if judge == "mitigated":
        return overall_noif(r["scores"], rubric) if r.get("scores") else None
    return r.get("score")


def adversarial_table(
    probes: Mapping[str, Any], hard: Mapping[str, Any], sensitivity: Mapping[str, Any], rubric: Rubric | None = None,
) -> dict[str, Any]:
    """Rows recomputed from the raw per-item verdicts in probes.json / probes_hard.json (counts, deltas), then compared
    with the stored summaries; sensitivity rows come from its stored summary. N is in every cell."""
    rubric = rubric or load_rubric()
    res, hres, bases = probes["results"], hard["results"], BASES
    rows: list[dict[str, Any]] = []
    mismatch: list[str] = []

    def fooled(stage: str, variant: str, source: Mapping[str, Any]) -> dict[str, Any]:
        return {j: _count(source, j, stage, variant, bases) for j in ("naive", "mitigated")}

    rows.append({"probe": "confidently wrong, reference shown", "tests": "sycophancy (wrong answer accepted)", "metric": "accepted (fooled)", **fooled("ref", "wrong", res)})
    rows.append({"probe": "confidently wrong, reference hidden", "tests": "sycophancy without a reference", "metric": "accepted (fooled)", **fooled("free", "wrong", res)})
    both = {}
    for j in ("naive", "mitigated"):
        pairs = [(_noif_or_naive(res[f"ref|{j}|verbose_wrong|{c}"], j, rubric), _noif_or_naive(res[f"ref|{j}|terse|{c}"], j, rubric))
                 for c in bases if f"ref|{j}|verbose_wrong|{c}" in res and f"ref|{j}|terse|{c}" in res]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        both[j] = {"k": sum(a >= b for a, b in pairs), "n": len(pairs), "n_invalid": len(bases) - len(pairs)}
    rows.append({"probe": "verbose wrong vs terse correct", "tests": "verbosity + sycophancy", "metric": "verbose-wrong scored >= terse-correct", **both})
    rows.append({"probe": "right answer + fabricated reasoning", "tests": "sycophancy toward plausible justification", "metric": "accepted (fooled)", **fooled("hard", "hard", hres)})
    deltas = {}
    for j in ("naive", "mitigated"):
        ds = []
        for c in bases:
            p, t = res.get(f"ref|{j}|padded|{c}"), res.get(f"ref|{j}|terse|{c}")
            a, b = (_noif_or_naive(p, j, rubric) if p else None), (_noif_or_naive(t, j, rubric) if t else None)
            if a is not None and b is not None:
                ds.append(a - b)
        deltas[j] = {"n": len(ds), "mean": round(statistics.fmean(ds), 4) if ds else None, "n_invalid": len(bases) - len(ds)}
    rows.append({"probe": "padded minus terse-correct", "tests": "verbosity (score delta; naive 1-10, mitigated faithfulness+completeness 1-5)", "metric": "mean score delta", **deltas})
    sens = sensitivity["summary"]
    rows.append({"probe": "correct vs confidently wrong, pairwise", "tests": "sensitivity (does the judge still pick a winner)", "metric": "picked the correct answer (both orders)",
                 **{j: {"k": sens[j]["two_order"]["correct"], "n": sens[j]["two_order"]["n"] - sens[j]["two_order"]["n_invalid"], "n_invalid": sens[j]["two_order"]["n_invalid"],
                        "source": "stored summary"} for j in ("naive", "mitigated")}})

    accept_ok = {j: _count(res, j, "ref", "terse", bases) for j in ("naive", "mitigated")}
    controls = []
    for j in ("naive", "mitigated"):
        a = accept_ok[j]
        controls.append({"name": f"{j}: terse-correct answers accepted", "expect": "most correct answers accepted (else 'fooled 0' only means 'rejects everything')",
                         "observed": f"{a['k']}/{a['n']}", "ok": a["n"] > 0 and a["k"] / a["n"] >= 0.5})
    discrimination = {j: round(accept_ok[j]["k"] / accept_ok[j]["n"] - rows[0][j]["k"] / rows[0][j]["n"], 4)
                      for j in ("naive", "mitigated") if accept_ok[j]["n"] and rows[0][j]["n"]}
    baselines = {"always_accept": {"fooled_rate": 1.0, "correct_accepted_rate": 1.0}, "always_reject": {"fooled_rate": 0.0, "correct_accepted_rate": 0.0}}

    st = probes["summary"]
    for j in ("naive", "mitigated"):
        s_ = st["sycophancy"][j]["reference_mode"]["a_wrong_answer_accepted"]["fooled"]
        if s_ != rows[0][j]["k"]:
            mismatch.append(f"{j} reference-mode fooled: recomputed {rows[0][j]['k']} vs stored {s_}")
        key = "naive_score" if j == "naive" else "overall_noif"
        sd = st["verbosity"][j]["padded_minus_terse"][key]["mean"]
        if sd is None or deltas[j]["mean"] is None or abs(sd - deltas[j]["mean"]) > 1e-3:
            mismatch.append(f"{j} padded-terse delta: recomputed {deltas[j]['mean']} vs stored {sd}")
        h = hard["summary"][j]["accepted"]
        if h != rows[3][j]["k"]:
            mismatch.append(f"{j} hard accepted: recomputed {rows[3][j]['k']} vs stored {h}")
    return {"n_bases": len(bases), "exploratory": True, "rows": rows, "correct_accepted_terse": accept_ok,
            "discrimination_correct_accepted_minus_fooled": discrimination, "baselines": baselines,
            "negative_controls": controls, "stored_mismatch": mismatch,
            "thresholds": {"mitigated_accepted": "faithfulness >= 4", "naive_accepted": f"score >= {NAIVE_FOOLED_MIN} of 10"}}


# ---------------------------------------------------------------- assembly and rendering

CAVEATS = (
    "By-construction labels for the reference-free suite cases check the case's own constraints only, so they cannot see a faithfulness failure; a judge that flags one will look like a disagreement.",
    "By-construction labels for reference-based cases also require the reference's key facts (a completeness matter), but the pass rule has no completeness threshold, so a terse-but-faithful answer can be label-fail and judge-pass.",
    "The 'probes' stratum is a set of designed extremes (6 correct, 30 wrong or defective): kappa there measures separation of constructed corruptions, not agreement on natural answers; the headline is suite_distinct.",
    "Configs A and B share 6 identical answers, so suite_A and suite_B overlap; suite_distinct removes the duplicates. Strata share items, so they are not independent evidence.",
    "Perfect agreement collapses the bootstrap interval to a point ('CI degenerate'); use the exact interval on agreement and the N.",
    "Positives are few (suite: 4 label-fail of 34), so one item moves kappa a lot; every kappa here is exploratory.",
)

def _load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_validation(*, root: str | Path = ".", labels_path: str | None = None, retest_path: str | None = None) -> dict[str, Any]:
    """Build the whole validation document from saved files (zero judge calls)."""
    root = Path(root)
    rubric = load_rubric()
    cases = {c: [Case(**d) for d in _load(root / f"data/outputs/{c}.json")] for c in ("A", "B")}
    entries = build_entries(
        suite=cases, pointwise={c: _load(root / f"results/pointwise_{c}.json") for c in ("A", "B")},
        probes=_load(root / "results/probes.json"), hard=_load(root / "results/probes_hard.json"),
        naive_suite=_load(root / "results/clustering.json")["naive_results"],
    )
    lp = root / (labels_path or LABELS_PATH)
    human: dict[str, Any] = (
        human_report(entries, load_human_labels(lp), rubric) if lp.exists()
        else {"status": "labels file not found", "path": str(lp), "n_labelled": 0, "format": LABEL_FORMAT}
    )
    rp = root / (retest_path or "results/test_retest.json")
    retest = _load(rp) if rp.exists() else {"status": "not run (judge retest)", "n_cases": 0}
    return {
        "judge_model": _load(root / "results/pointwise_A.json").get("judge_model", ""),
        "rubric_version": rubric.version, "exploratory": True,
        "pass_rule": "mitigated = rubric pass rule recomputed from stored scores; naive = score >= %d of 10" % NAIVE_FOOLED_MIN,
        "by_construction": construction_report(entries, rubric),
        "dedupe_conflicts": dedupe([e for e in entries if e.origin.startswith("suite:")])[1],
        "caveats": CAVEATS, "human": human, "test_retest": retest,
        "adversarial": adversarial_table(_load(root / "results/probes.json"), _load(root / "results/probes_hard.json"),
                                         _load(root / "results/sensitivity.json"), rubric),
    }


def _f(x: Any, nd: int = 2) -> str:
    return "undef" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _ci(ci: Any) -> str:
    if not isinstance(ci, dict) or ci.get("low") is None:
        return ""
    return " [CI degenerate]" if ci["low"] == ci["high"] else f" [{_f(ci['low'])}, {_f(ci['high'])}]"


def _kappa_line(name: str, s: Mapping[str, Any]) -> str:
    if not s or s["n"] == 0:
        return f"| {name} | 0 | - | - | - | - | - |"
    fp, ff = s["false_pass"], s["false_fail"]
    return (f"| {name} | {s['n']} | {_f(s['agreement'])} ({s['agree']}/{s['n']}; always-pass {_f(s['always_pass_agreement'])}) | "
            f"{_f(s['kappa'])}{_ci(s['kappa_ci95'])} | {fp['k']}/{fp['n']} | {ff['k']}/{ff['n']} | {s.get('n_invalid', 0)} |")


def _controls_md(controls: Sequence[Mapping[str, Any]]) -> list[str]:
    return [f"- {'PASS' if c['ok'] else '**FAIL**'} {c['name']}: expected {c['expect']}; observed {c['observed']}"
            + (f" (permutation p of >= observed: {c['p_perm_ge_observed']})" if c.get("p_perm_ge_observed") is not None else "")
            for c in controls]


def render_markdown(doc: Mapping[str, Any]) -> str:
    L = [f"# Judge validation (judge {doc['judge_model']}, {doc['rubric_version']}) - EXPLORATORY, N in every row", "",
         f"Pass rule: {doc['pass_rule']}. Positive class = pass. Labels: by construction (deterministic) and human.", "",
         "Caveats:", *[f"- {c}" for c in CAVEATS], "",
         "## Agreement and kappa vs by-construction labels", "",
         "| stratum / judge | N | agreement | kappa [95% bootstrap] | false pass (judge pass among label fail) | false fail | INVALID |", "|---|---|---|---|---|---|---|"]
    for name, s in doc["by_construction"].items():
        for j in ("mitigated", "naive"):
            L.append(_kappa_line(f"{name} / {j}  (labels: {s['label_pass']} pass, {s['label_fail']} fail)", s[j]))
    L += ["", "Negative controls (per stratum, mitigated judge):"]
    for name, s in doc["by_construction"].items():
        if s["mitigated"]["negative_controls"]:
            L.append(f"- {name} (N={s['mitigated']['n']})")
            L += ["  " + x for x in _controls_md(s["mitigated"]["negative_controls"])]
    for key, who in (("human", "human"),):
        h = doc[key]
        L += ["", f"## Agreement and kappa vs {who} labels", ""]
        if h["n_labelled"] == 0:
            L.append(f"N=0: {h['status']}" + (f" ({h['path']}). Format: {h['format']}" if "path" in h else "") + f". No {who}-label number is reported.")
            continue
        L += [f"N={h['n_labelled']} labelled ({h['human_label_pass']} pass / {h['human_label_fail']} fail; {h['n_distinct_texts']} distinct texts; {h['n_skipped_blank']} blank skipped; unmatched {h['unmatched']}).", "",
              "| pair | N | agreement | kappa [95% bootstrap] | false pass | false fail | INVALID |", "|---|---|---|---|---|---|---|",
              _kappa_line(f"mitigated judge vs {who}", h["mitigated"]), _kappa_line(f"naive judge vs {who}", h["naive"]),
              _kappa_line(f"by-construction vs {who}", h["by_construction_vs_human"]), ""]
        L += _controls_md(h["mitigated"]["negative_controls"])
        L += [f"- disagreements (mitigated vs {who}): {[(d['case_id'], d['label']) for d in h['mitigated']['disagreements']]}"]
    t = doc["test_retest"]
    L += ["", "## Test-retest (cache bypassed by nonce)", ""]
    if t.get("n_cases", 0) == 0:
        L.append(f"N=0: {t.get('status', 'not run')}.")
    else:
        pf, sd = t["pass_flips"], t["overall_sd"]
        L += [f"N={t['n_cases']} cases x {t['n_runs']} runs ({', '.join(t['run_labels'])}); comparable {t['n_comparable']}, scored in every run {t['n_complete']}; INVALID per run {t['n_invalid_per_run']}.", "",
              f"- pass/fail flips: {pf['k']}/{pf['n']} = {_f(pf['rate'])} (95% exact CI {pf['ci95_exact']}); cases {pf['cases']}",
              f"- overall sd per case ({sd['sd_kind']}): mean {_f(sd['mean'], 3)}, max {_f(sd['max'], 3)}; {sd['cases_with_any_difference']}/{sd['n']} cases changed overall at least once",
              "- criterion scores identical across runs: " + ", ".join(f"{c} {v['k']}/{v['n']}" for c, v in t["criterion_exact_agreement"].items()),
              "- run pairs: " + "; ".join(f"{k} pass-agreement {_f(v['pass_agreement'])} (n={v['n']}), mean |d overall| {_f(v['mean_abs_overall_diff'], 3)}" for k, v in t["run_pairs"].items()), ""]
        L += _controls_md(t["negative_controls"])
    a = doc["adversarial"]
    L += ["", f"## Adversarial summary (N={a['n_bases']} base cases per row, exploratory)", "",
          f"Accepted = mitigated faithfulness >= 4 / naive score >= {NAIVE_FOOLED_MIN}. Lower is better except where noted.", "",
          "| probe | metric | naive | mitigated |", "|---|---|---|---|"]
    for r in a["rows"]:
        cell = lambda j: (f"mean {_f(r[j]['mean'], 2)} (N={r[j]['n']})" if "mean" in r[j] else f"{r[j]['k']}/{r[j]['n']}" + (f" +{r[j]['n_invalid']} INVALID" if r[j]["n_invalid"] else ""))  # noqa: E731
        L.append(f"| {r['probe']} | {r['metric']} | {cell('naive')} | {cell('mitigated')} |")
    ca = a["correct_accepted_terse"]
    L += ["", f"Terse-correct answers accepted (should be high): naive {ca['naive']['k']}/{ca['naive']['n']}, mitigated {ca['mitigated']['k']}/{ca['mitigated']['n']}. "
          f"Discrimination (correct accepted rate minus fooled rate, reference mode): {a['discrimination_correct_accepted_minus_fooled']}. "
          f"Trivial baselines: always-accept fooled 1.0 / correct-accepted 1.0, always-reject 0.0 / 0.0.", ""]
    L += _controls_md(a["negative_controls"])
    L.append(f"- stored-vs-recomputed mismatches: {a['stored_mismatch'] or 'none'}")
    return "\n".join(L) + "\n"


def write_outputs(doc: Mapping[str, Any], json_path: str | Path, md_path: str | Path) -> None:
    Path(json_path).write_text(json.dumps(doc, indent=2, ensure_ascii=False, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else str(o)) + "\n", encoding="utf-8")
    Path(md_path).write_text(render_markdown(doc), encoding="utf-8")
