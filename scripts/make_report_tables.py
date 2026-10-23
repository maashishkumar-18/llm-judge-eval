"""Generate the README result tables from results/*.json (no judge calls, stdlib only).

    python scripts/make_report_tables.py            # rewrite the tables block in README.md
    python scripts/make_report_tables.py --check    # exit 1 if README.md is out of date
    python scripts/make_report_tables.py --stdout   # print the block, touch nothing

Every number comes from a saved result file; a missing key raises KeyError on purpose, so a renamed
field breaks the build instead of printing a blank. The block sits between the two markers below.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
README = ROOT / "README.md"
BEGIN = "<!-- BEGIN GENERATED TABLES (scripts/make_report_tables.py) -->"
END = "<!-- END GENERATED TABLES -->"


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def f2(x: float) -> str:
    return f"{x:.2f}"


def f3(x: float) -> str:
    return f"{x:.3f}"


def ci(lo: float, hi: float, fmt=f2) -> str:
    return f"[{fmt(lo)}, {fmt(hi)}]"


def signed(x: float, fmt=f2) -> str:
    return f"{x:+.{2 if fmt is f2 else 3}f}"


def sci(lo: float, hi: float, fmt=f2) -> str:
    """CI of a signed difference: both ends carry their sign."""
    return f"[{signed(lo, fmt)}, {signed(hi, fmt)}]"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def kappa_cell(m: dict) -> str:
    k = m["kappa"]
    c = m["kappa_ci95"]
    if k is None:
        return "undefined"
    return f"{k:.2f} [{c['low']:.2f}, {c['high']:.2f}]"


# --------------------------------------------------------------------------- sections


def setup() -> str:
    rep = load("report_A")
    pw = rep["pointwise"]
    return table(
        ["item", "value"],
        [
            ["judge", f"`{rep['judge_model']}` (Google), JSON mode {'on' if rep['json_mode'] else 'off'} (prompt-only JSON)"],
            ["generator", f"`{rep['generator_usage']['model']}` via Groq (OpenAI family)"],
            ["suite", f"{pw['n']} cases: {pw['by_mode']['reference']['n']} reference-based, {pw['by_mode']['free']['n']} reference-free"],
            ["pass rule", "faithfulness >= {faithfulness_min}, instruction-following >= {instruction_following_min}, overall >= {overall_min}".format(**pw["pass"]["rule"])],
            ["parse status, config A", ", ".join(f"{k}: {v}" for k, v in pw["parse_status"].items())],
        ],
    )


def generation() -> str:
    g = load("gen_check")
    names = {"A": "A: gpt-oss-20b, prompt v1", "B": "B: gpt-oss-20b, prompt v2", "C": "C: qwen3.8-27b, prompt v1"}
    rows = [
        [
            names[k],
            f"{g[k]['constraint_pass']}/{g[k]['n']}",
            f"{g[k]['exact_match']}/{g[k]['n_ref']}",
            f"{g[k]['mean_tokens_out']:.0f}",
        ]
        for k in ("A", "B", "C")
    ]
    t = table(["generator config", "constraint pass", "exact match (reference-based)", "mean output tokens"], rows)
    d = g["C_minus_A_pts"]
    rule = "prompt v1 vs v2 (A vs B)" if max(abs(d["constraint"]), abs(d["exact_match"])) < 10 else "model A vs model C"
    return t + (
        f"\n\nC minus A: {d['constraint']:+.1f} pts constraint, {d['exact_match']:+.1f} pts exact match. "
        f"Pre-registered rule (10-point threshold on either check) selects **{rule}**."
    )


def ab() -> str:
    a, b, c = load("report_A")["pointwise"], load("report_B")["pointwise"], load("compare_A_vs_B")

    def cell(m: dict) -> str:
        return f"{f3(m['mean'])} {ci(m['ci_low'], m['ci_high'], f3)}"

    def crit(key: str, label: str) -> list[str]:
        d = c["criteria_diff"][key]
        return [label, cell(a["criteria"][key]), cell(b["criteria"][key]), f"{signed(d['mean'])} {sci(d['ci_low'], d['ci_high'])}"]

    od = c["overall_diff"]
    pr = c["pass"]["rate_diff"]
    rows = [
        ["mean overall (1-5)", cell(a["overall"]), cell(b["overall"]), f"{signed(od['mean'], f3)} {sci(od['ci_low'], od['ci_high'], f3)}"],
        crit("faithfulness", "faithfulness"),
        crit("instruction_following", "instruction-following"),
        crit("completeness", "completeness"),
        [
            "pass rate",
            f"{a['pass']['n_pass']}/{a['n']} = {f2(a['pass']['rate_scored']['mean'])}",
            f"{b['pass']['n_pass']}/{b['n']} = {f2(b['pass']['rate_scored']['mean'])}",
            f"{signed(pr['mean'])} {sci(pr['ci_low'], pr['ci_high'])}",
        ],
    ]
    t = table(["", "A (prompt v1)", "B (prompt v2)", "A - B, paired"], rows)
    w, dw = c["win_rate"], c["win_rate_differing_pairs_only"]
    dec = c["decision"]
    return (
        t
        + f"\n\nPairwise, mitigated, both orders: win rate A over B **{f3(w['win_rate'])}** {ci(w['ci_low'], w['ci_high'], f3)} "
        f"(N={w['n']}: A {w['wins_x']} / B {w['wins_y']} / tie {w['ties']}; flip rate {pct(w['flip_rate'])}); "
        f"on the {dw['n']} pairs whose answers differ: {f3(dw['win_rate'])} {ci(dw['ci_low'], dw['ci_high'], f3)}.\n\n"
        f"**Decision: {dec['verdict']}** ({dec['reasons'][0]}). "
        f"{c['same_overall_cases']} of {c['n_paired']} cases have an identical overall under A and B."
    )


def bias() -> str:
    pb = load("position_bias")["summary"]
    nv, mt, mt12 = pb["pairwise_naive"], pb["pairwise_mitigated_same_pairs_as_naive"], pb["pairwise_mitigated_all"]
    ctl = pb["controls"]
    pr = load("probes")["summary"]
    hard = load("probes_hard")["summary"]
    sens = load("sensitivity")["summary"]
    sp = load("self_preference")["summary"]
    cl = load("clustering")["summary"]
    sp_same = cl["spread_same_cases"]
    v_n, v_m = pr["verbosity"]["naive"], pr["verbosity"]["mitigated"]
    sy_n, sy_m = pr["sycophancy"]["naive"], pr["sycophancy"]["mitigated"]

    def ctl_cell(c: dict) -> str:
        return f"first {c['picked_first']} / second {c['picked_second']} / tie {c['tie']} (N={c['n']})"

    def fooled(d: dict) -> str:
        return f"{d['fooled']}/{d['n']}"

    pos = table(
        ["position bias", "naive", "mitigated"],
        [
            ["flip rate, same 12 pairs, both orders", f"{nv['flips']}/{nv['n_flippable']} = {pct(nv['flip_rate'])}", f"{mt['flips']}/{mt['n_flippable']} = {pct(mt['flip_rate'])}"],
            ["hard flips (opposite decisive winners)", f"{nv['hard_flips']}/{nv['n_flippable']}", f"{mt['hard_flips']}/{mt['n_flippable']}"],
            ["share choosing the first-listed answer (decisive)", pct(nv["first_position_share"]), pct(mt["first_position_share"])],
            ["verdicts A / B / tie", f"{nv['wins_x']} / {nv['wins_y']} / {nv['ties']}", f"{mt['wins_x']} / {mt['wins_y']} / {mt['ties']}"],
            ["identical-answer controls, one order", ctl_cell(ctl["naive"]), ctl_cell(ctl["mitigated"])],
            ["flip rate, all 20 A/B pairs", "not run (naive was 12 pairs)", f"{mt12['flips']}/{mt12['n_flippable']} = {pct(mt12['flip_rate'])}"],
            ["sensitivity: picked the correct answer of correct-vs-wrong, both orders (N=6)", f"{sens['naive']['two_order']['correct']}/6", f"{sens['mitigated']['two_order']['correct']}/6"],
        ],
    )

    ver = table(
        ["verbosity and sycophancy probes (N=6 bases)", "naive", "mitigated"],
        [
            ["padded minus terse-correct, mean score delta", f"{v_n['padded_minus_terse']['naive_score']['mean']:+.2f} (1-10)", f"{v_m['padded_minus_terse']['overall_noif']['mean']:+.2f} (faith+compl, 1-5)"],
            ["padded minus terse-correct, instruction-following", "n/a", f"{v_m['padded_minus_terse']['instruction_following']['mean']:+.2f}"],
            ["Spearman(words, score), correct group, n=12", f"{v_n['spearman_words_vs_score']['correct']['rho']:+.2f}", f"{v_m['spearman_words_vs_score']['correct']['rho']:+.2f}"],
            ["confidently wrong accepted, reference shown", fooled(sy_n["reference_mode"]["a_wrong_answer_accepted"]), fooled(sy_m["reference_mode"]["a_wrong_answer_accepted"])],
            ["confidently wrong accepted, reference hidden", fooled(sy_n["reference_free_mode"]["a_wrong_answer_accepted"]), fooled(sy_m["reference_free_mode"]["a_wrong_answer_accepted"])],
            ["verbose-wrong scored >= terse-correct", fooled(sy_n["reference_mode"]["b_verbose_wrong_ge_terse_correct"]), fooled(sy_m["reference_mode"]["b_verbose_wrong_ge_terse_correct"])],
            ["right answer + fabricated reasoning accepted", f"{hard['naive']['accepted']}/{hard['naive']['n']}", f"{hard['mitigated']['accepted']}/{hard['mitigated']['n']}"],
        ],
    )

    nsp = sp_same["naive_1_10"]
    msp = sp_same["mitigated_overall_1_5"]
    mfa = sp_same["mitigated_faithfulness_1_5"]
    ci_sp = sp["ci95_clopper_pearson_decisive"]
    clu = table(
        ["score clustering and self-preference (exploratory)", "naive", "mitigated"],
        [
            ["distinct score values, same 12 cases", f"{nsp['distinct_values']}", f"{msp['distinct_values']} (overall); {mfa['distinct_values']} (faithfulness)"],
            ["share at the modal value", pct(nsp["modal_share"]), f"{pct(msp['modal_share'])} (overall); {pct(mfa['modal_share'])} (faithfulness)"],
            ["sd / scale width", f3(nsp["sd_over_scale_width"]), f"{f3(msp['sd_over_scale_width'])} (overall); {f3(mfa['sd_over_scale_width'])} (faithfulness)"],
            ["own-family win rate, Gemini-authored vs gpt-oss, N=6 pairs", "not run", f"{f2(sp['own_family_win_rate'])}: Gemini {sp['gemini_wins']} / gpt-oss {sp['gpt_oss_wins']} / tie {sp['ties']}; 95% CI on decisive share {ci(*ci_sp, fmt=f2)}"],
        ],
    )
    return (
        pos
        + "\n\n"
        + ver
        + "\n\n"
        + clu
        + "\n\nRead these tables with the stated N. The mitigations' measured effect is clear for position bias on the naive "
        "single-order baseline (flip rate 41.7% to 0%). For verbosity, sycophancy and clustering the naive judge already "
        "sat at the floor or ceiling, so no before-to-after improvement is claimed (see Known limitations)."
    )


def validation() -> str:
    v = load("validation")
    bc = v["by_construction"]
    hm = v["human"]
    rt = v["test_retest"]
    crit = v["criterion_vs_deterministic"]["suite_all_outputs"]["instruction_following"]

    def row(label: str, m: dict) -> list[str]:
        return [label, str(m["n"]), f"{f2(m['agreement'])} ({f2(m['always_pass_agreement'])})", kappa_cell(m), f"{m['false_pass']['k']}/{m['false_pass']['n']}", f"{m['false_fail']['k']}/{m['false_fail']['n']}"]

    rows = [
        row("mitigated vs by-construction labels, suite distinct (headline)", bc["suite_distinct"]["mitigated"]),
        row("naive vs by-construction labels, suite distinct", bc["suite_distinct"]["naive"]),
        row("mitigated vs human labels (c09, c16 not blind)", hm["mitigated"]),
        row("naive vs human labels", hm["naive"]),
        row("mitigated vs by-construction, probes (designed extremes)", bc["probes"]["mitigated"]),
        row("naive vs by-construction, probes", bc["probes"]["naive"]),
        row("mitigated, instruction-following vs `checks.py` (40 outputs)", crit),
    ]
    t = table(["judge vs labels", "N", "agreement (always-pass baseline)", "Cohen kappa [95% bootstrap]", "false pass", "false fail"], rows)

    flips, sd, ag = rt["pass_flips"], rt["overall_sd"], rt["criterion_exact_agreement"]
    rt_t = table(
        ["test-retest (temperature 0, nonce bypasses the cache)", "value"],
        [
            ["design", f"{rt['n_cases']} cases x {rt['n_runs']} runs (saved run + {rt['reruns']} nonce reruns)"],
            ["pass/fail flips", f"{flips['k']}/{flips['n']} (95% exact CI {ci(*flips['ci95_exact'])})"],
            ["overall sd per case, mean / max", f"{f3(sd['mean'])} / {f3(sd['max'])}; {sd['cases_with_any_difference']} of {sd['n']} cases changed at least once"],
            ["criterion scores identical across all runs", ", ".join(f"{k.replace('_', '-')} {a['k']}/{a['n']}" for k, a in ag.items())],
        ],
    )

    adv = v["adversarial"]
    adv_t = table(
        ["negative control", "naive", "mitigated"],
        [
            ["terse-correct answers accepted (N=6)", f"{adv['correct_accepted_terse']['naive']['k']}/6", f"{adv['correct_accepted_terse']['mitigated']['k']}/6"],
            ["discrimination = correct accepted rate - fooled rate", f2(adv["discrimination_correct_accepted_minus_fooled"]["naive"]), f2(adv["discrimination_correct_accepted_minus_fooled"]["mitigated"])],
        ],
    )
    return t + "\n\n" + rt_t + "\n\n" + adv_t


def usage() -> str:
    c = load("compare_A_vs_B")
    u = c["usage"]
    p = c["pricing"]
    rows = []
    for key, label in (("pointwise_A", "pointwise A"), ("pointwise_B", "pointwise B"), ("pairwise", "pairwise (both orders)"), ("judge_total", "A/B comparison total")):
        x = u[key]
        rows.append([label, str(x["calls"]), str(x["network_calls"]), str(x["reasks"]), f"{x['tokens_in']:,} / {x['tokens_out']:,}", f"${x['cost_usd_list_equivalent']:.4f}"])
    t = table(["judge usage", "calls", "network calls", "re-asks", "tokens in / out", "list-price equivalent"], rows)
    return t + (
        f"\n\nActual cost ${u['judge_total']['cost_usd_actual']:.2f} (free tier). List price ${p['input_usd_per_1m']}/1M in, "
        f"${p['output_usd_per_1m']}/1M out (`{p['tier']}` tier, retrieved {p['retrieved']}, {p['source_url']})."
    )


def block() -> str:
    parts = [
        ("Setup", setup()),
        ("Generator check and the pre-registered A/B rule", generation()),
        ("Main A/B: prompt v1 vs prompt v2 (pointwise, mitigated judge)", ab()),
        ("Bias: naive vs mitigated judge", bias()),
        ("Judge validation", validation()),
        ("Judge cost", usage()),
    ]
    out = [BEGIN, "", "_Generated by `python scripts/make_report_tables.py` from `results/*.json`; do not edit by hand._", ""]
    for title, body in parts:
        out += [f"#### {title}", "", body, ""]
    out.append(END)
    return "\n".join(out)


def splice(readme: str, new_block: str) -> str:
    if BEGIN not in readme or END not in readme:
        raise SystemExit(f"README.md is missing the markers:\n{BEGIN}\n{END}")
    head, rest = readme.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    return head + new_block + tail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if README.md differs from the generated block")
    ap.add_argument("--stdout", action="store_true", help="print the block instead of writing README.md")
    args = ap.parse_args()
    new_block = block()
    if args.stdout:
        sys.stdout.buffer.write((new_block + "\n").encode("utf-8"))
        return 0
    old = README.read_text(encoding="utf-8")
    new = splice(old, new_block)
    if args.check:
        if new != old:
            print("README.md tables are out of date: run `python scripts/make_report_tables.py`", file=sys.stderr)
            return 1
        print("README.md tables are up to date")
        return 0
    if new != old:
        README.write_text(new, encoding="utf-8", newline="\n")
        print("README.md tables updated")
    else:
        print("README.md tables already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
