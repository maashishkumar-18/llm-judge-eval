"""judge CLI: `budget`, `generate`/`genreport`, `pointwise`/`replay`, `position`, `run`/`compare`, `probes`, `selfpref`, `clustering`, `retest`, `validate`."""

from __future__ import annotations

import argparse
import asyncio
import sys

from judge.budget import BudgetExceeded, load_budget
from judge.config import MissingKeyError, load_settings
from judge.ledger import JudgeLedger


def cmd_budget(args: argparse.Namespace) -> int:
    s = load_settings()
    budget = load_budget()
    ledger = JudgeLedger(s.ledger_path, s.judge_daily_cap, budget.caps)

    problems = budget.validate()
    if args.check:
        try:
            budget.check_run(args.check, args.calls, spent=ledger.used_experiment(args.check))
            if args.calls > ledger.remaining_today():
                raise BudgetExceeded(
                    f"refused: {args.calls} calls exceed the {ledger.remaining_today()} left of today's cap of {s.judge_daily_cap}"
                )
        except (BudgetExceeded, KeyError) as e:
            print(e.args[0] if isinstance(e, KeyError) else e)
            return 1
        print(f"ok: {args.check} may run {args.calls} calls (spent {ledger.used_experiment(args.check)}/{budget.get(args.check).cap})")
        return 0

    print(f"Judge: {s.judge_model} | limiter {s.judge_rpm_limit} req/min | concurrency {s.judge_concurrency} | daily cap {s.judge_daily_cap} non-cached calls")
    print(f"{'experiment':22s} {'planned':>7s} {'cap':>5s} {'spent':>6s}  description")
    for e in budget.experiments.values():
        print(f"{e.name:22s} {e.planned:7d} {e.cap:5d} {ledger.used_experiment(e.name):6d}  {e.desc}")
    print(f"{'TOTAL':22s} {budget.total_planned:7d} {budget.total_cap:5d}")
    mins = budget.total_planned / s.judge_rpm_limit
    print(f"\nused today: {ledger.used_today()}/{s.judge_daily_cap} | planned total uses {budget.total_planned / s.judge_daily_cap:.0%} of one day's cap"
          f" | min wall time at {s.judge_rpm_limit} RPM: {mins:.0f} min")
    if budget.total_planned > ledger.remaining_today():
        problems.append(f"total planned {budget.total_planned} > calls left today {ledger.remaining_today()} (run will span days)")
    for p in problems:
        print(f"REFUSED: {p}")
    return 1 if budget.validate() else 0


def cmd_generate(args: argparse.Namespace) -> int:
    from pathlib import Path

    from judge.generate import generate_config
    from judge.suite import load_candidates, load_config, summarize

    sys.stdout.reconfigure(errors="replace")  # model text may hold characters the Windows console cannot encode
    s = load_settings()
    rows = load_candidates(Path(args.suite))
    if args.limit:
        rows = rows[: args.limit]
    print(f"suite {args.suite}: {summarize(rows)}")
    for name in args.config:
        cases, status = asyncio.run(generate_config(s, rows, load_config(name)))
        empty = sum(1 for c in cases if not c.model_output)
        ok = sum(all(r["passed"] for r in c.meta["constraint_results"]) for c in cases)
        print(f"config {name}: {len(cases)}/{len(rows)} outputs (new {status.ran}, resumed {status.skipped}), "
              f"empty {empty}, constraint-compliant {ok}/{len(cases)}")
        for c in cases[: args.sample]:
            first = (c.model_output.splitlines() or [""])[0][:90]
            print(f"  {c.id} [{'ok ' if all(r['passed'] for r in c.meta['constraint_results']) else 'BAD'}] {first}")
    return 0


def cmd_genreport(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from judge.generate import summarize_outputs
    from judge.schemas import Case

    rep = {}
    for name in args.config:
        cases = [Case(**d) for d in json.loads(Path(f"data/outputs/{name}.json").read_text(encoding="utf-8"))]
        r = rep[name] = summarize_outputs(cases)
        print(f"{name}: constraint {r['constraint_pass']}/{r['n']} ({r['constraint_pass_rate']:.0%}) | ref-based: "
              f"constraint {r['constraint_pass_ref']}/{r['n_ref']}, exact-match {r['exact_match']}/{r['n_ref']} "
              f"({r['exact_match_rate']:.0%}) | empty {r['empty']} | mean out tokens {r['mean_tokens_out']:.0f}")
    for name in (n for n in list(rep) if n != "A" and "A" in rep):
        dc = 100 * (rep[name]["constraint_pass_rate"] - rep["A"]["constraint_pass_rate"])
        de = 100 * (rep[name]["exact_match_rate"] - rep["A"]["exact_match_rate"])
        print(f"{name} - A: constraint {dc:+.1f} pts, exact-match {de:+.1f} pts")
        rep[f"{name}_minus_A_pts"] = {"constraint": dc, "exact_match": de}
    Path("results").mkdir(exist_ok=True)
    Path("results/gen_check.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    return 0


def _load_cases(path: str):
    import json
    from pathlib import Path

    from judge.schemas import Case

    return [Case(**d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]


def _fmt_row(r: dict) -> str:
    sc = r["scores"]
    scores = "/".join(str(sc[c]) for c in ("faithfulness", "instruction_following", "completeness")) if sc else "-"
    overall = f"{r['overall']:.2f}" if r["overall"] is not None else "  - "
    verdict = {True: "PASS", False: "fail", None: " -- "}[r["passed"]]
    flags = ("unsupported:" + ",".join(r["unsupported_high_scores"])) if r["unsupported_high_scores"] else ""
    if r["repairs"]:
        flags += " repairs:" + ",".join(r["repairs"])
    calls = f"{sum(1 for c in r['calls'] if not c['cached'])}net/{len(r['calls'])}"
    return (f"{r['case_id']:5s} {r['mode']:9s} {r['parse_status']:8s} f/i/c={scores:6s} overall={overall} {verdict}  "
            f"calls={calls} {flags}{('  ERROR: ' + r['error'][:120]) if r['error'] else ''}")


def _pointwise_core(args: argparse.Namespace, config: str, label: str):
    """Judge one config's outputs (checkpointed, budgeted). Returns (doc, remaining, n_cases) or an int exit code."""
    from pathlib import Path

    from judge.llm import LLMClient
    from judge.pointwise import judge_suite

    s = load_settings()
    cases_file = args.cases_file or f"data/outputs/{config}.json"
    cases = _load_cases(cases_file)
    if args.ids:
        cases = [c for c in cases if c.id in args.ids]
        missing = set(args.ids) - {c.id for c in cases}
        if missing:
            print(f"unknown case ids: {sorted(missing)}")
            return 1
    if args.limit:
        cases = cases[: args.limit]
    if args.experiment:
        budget = load_budget()
        ledger = JudgeLedger(s.ledger_path, s.judge_daily_cap, budget.caps)
        try:
            budget.check_run(args.experiment, len(cases), spent=ledger.used_experiment(args.experiment))
        except BudgetExceeded as e:
            print(e)
            return 1

    async def run():
        async with LLMClient(s, "judge", log_name=f"pointwise_{label}.jsonl", experiment=args.experiment or None) as client:
            return await judge_suite(
                client, cases, label, out_path=f"results/pointwise_{label}.json",
                checkpoint_path=Path(".cache/checkpoints") / f"pointwise_{label}.jsonl",
                cases_file=cases_file, nonce=args.nonce, fresh=args.fresh,
            )

    doc, remaining = asyncio.run(run())
    return doc, remaining, len(cases)


def _print_pointwise(doc: dict, remaining: list, n_cases: int, label: str) -> None:
    print(f"judge {doc['judge_model']} (json_mode={doc['json_mode']}), {len(doc['results'])}/{n_cases} cases, label {label}")
    for r in doc["results"]:
        print(_fmt_row(r))
    sm = doc["summary"]
    print(f"parse_status {sm['parse_status']} | invalid {sm['n_invalid']} (parse {sm['n_invalid_parse']}, call {sm['n_invalid_call']}) | "
          f"mean overall {sm['overall']['mean']} | pass rate {sm['pass_rate_scored']} (invalid as fail: {sm['pass_rate_invalid_as_fail']}) | "
          f"calls {sm['judge_calls']} ({sm['judge_calls_network']} network, {sm['reasks']} re-ask) | tokens {sm['tokens_in']} in / {sm['tokens_out']} out")
    if remaining:
        print(f"STOPPED with {len(remaining)} cases left (quota cap); re-run the same command to resume: {remaining}")


def cmd_pointwise(args: argparse.Namespace) -> int:
    sys.stdout.reconfigure(errors="replace")
    label = args.label or args.config
    out = _pointwise_core(args, args.config, label)
    if isinstance(out, int):
        return out
    doc, remaining, n = out
    _print_pointwise(doc, remaining, n, label)
    return 2 if remaining else 0


def _prices(args: argparse.Namespace):
    """Recorded list price (src/judge/pricing.yaml); --price-in/--price-out override it (and drop the source)."""
    from dataclasses import replace

    from judge.aggregate import load_prices

    base = load_prices()
    if args.price_in is None and args.price_out is None:
        return base
    if args.price_in is None or args.price_out is None:
        raise SystemExit("give both --price-in and --price-out")
    return replace(base, input_per_m=args.price_in, output_per_m=args.price_out, source_url="cli override", retrieved="", tier="")


def cmd_run(args: argparse.Namespace) -> int:
    """Pointwise-judge each config and write results/report_<config>.json (aggregates, CIs, usage)."""
    import json
    from pathlib import Path

    from judge.compare import config_report

    sys.stdout.reconfigure(errors="replace")
    rc = 0
    for config in args.config:
        if args.report_only:  # rebuild from the saved results: no judge call, no budget use
            doc, remaining = json.loads(Path(f"results/pointwise_{config}.json").read_text(encoding="utf-8")), []
        else:
            out = _pointwise_core(args, config, config)
            if isinstance(out, int):
                return out
            doc, remaining, n = out
            _print_pointwise(doc, remaining, n, config)
        if remaining:
            rc = 2
            continue  # an incomplete run gets no report: a partial suite must not look like a result
        gen = json.loads(Path(doc["cases_file"]).read_text(encoding="utf-8"))
        rep = config_report(doc, gen, name=config, prices=_prices(args))
        Path("results").mkdir(exist_ok=True)
        path = Path(f"results/report_{config}.json")
        path.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        pw = rep["pointwise"]
        o, p = pw["overall"], pw["pass"]["rate_scored"]
        print(f"report {config}: overall {o['mean']} [{o['ci_low']}, {o['ci_high']}] | pass {pw['pass']['n_pass']}/{pw['n_scored']} "
              f"= {p['mean']} [{p['ci_low']}, {p['ci_high']}] | " + " ".join(f"{k[:4]}={v['mean']}" for k, v in pw["criteria"].items())
              + f" | wrote {path}")
    return rc


def cmd_compare(args: argparse.Namespace) -> int:
    """Offline (zero judge calls): paired bootstrap + win rate + decision from the saved results files."""
    import json
    from pathlib import Path

    from judge.compare import compare_configs, render_markdown

    sys.stdout.reconfigure(errors="replace")
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
    a_path = args.a_results or f"results/pointwise_{args.a}.json"
    b_path = args.b_results or f"results/pointwise_{args.b}.json"
    for p in (a_path, b_path):
        if not Path(p).exists():
            print(f"missing {p}: run `judge run --config {Path(p).stem.removeprefix('pointwise_')}` first")
            return 1
    doc_a, doc_b = load(a_path), load(b_path)
    pairs = None
    if Path(args.pairs).exists():
        pairs = load(args.pairs)["pairs"].get("mitigated")
        if pairs and (pairs[0]["x"], pairs[0]["y"]) != (args.a, args.b):
            print(f"{args.pairs} pairs are {pairs[0]['x']} vs {pairs[0]['y']}, not {args.a} vs {args.b}; win rate skipped")
            pairs = None
    else:
        print(f"no {args.pairs}: win rate skipped (run `judge position` for the pairwise half)")
    if doc_a.get("judge_model") != doc_b.get("judge_model"):
        print(f"WARNING: configs were judged by different models: {doc_a.get('judge_model')} vs {doc_b.get('judge_model')}")
    c = compare_configs(doc_a, doc_b, pairs, name_a=args.a, name_b=args.b, prices=_prices(args))
    stem = f"results/compare_{args.a}_vs_{args.b}"
    Path("results").mkdir(exist_ok=True)
    Path(stem + ".json").write_text(json.dumps(c, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = render_markdown(c)
    Path(stem + ".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {stem}.json and {stem}.md")
    return 0


def cmd_position(args: argparse.Namespace) -> int:
    from pathlib import Path

    from judge.llm import LLMClient
    from judge.pairwise import (
        build_position_document, control_ids, judge_controls, judge_pairs, naive_pair_ids, write_document,
    )

    sys.stdout.reconfigure(errors="replace")
    s = load_settings()
    cx, cy = _load_cases("data/outputs/A.json"), _load_cases("data/outputs/B.json")
    all_ids = [c.id for c in cx]
    naive_ids, ctl_ids = naive_pair_ids(cx, cy), control_ids(cx)
    if args.ids:
        unknown = set(args.ids) - set(all_ids)
        if unknown:
            print(f"unknown case ids: {sorted(unknown)}")
            return 1
    pick = lambda ids: [i for i in ids if not args.ids or i in args.ids]  # noqa: E731
    plan = {  # stage -> (budget experiment, case ids, judge calls = ids x orders)
        "naive": ("pairwise_naive", pick(naive_ids), 2),
        "mitigated": ("pairwise_mitigated", pick(all_ids), 2),
        "controls_naive": ("controls", pick(ctl_ids), 1),
        "controls_mitigated": ("controls", pick(ctl_ids), 1),
    }
    budget = load_budget()
    ledger = JudgeLedger(s.ledger_path, s.judge_daily_cap, budget.caps)
    for exp in {plan[st][0] for st in args.stages}:
        need = sum(len(plan[st][1]) * plan[st][2] for st in args.stages if plan[st][0] == exp)
        try:
            budget.check_run(exp, need, spent=ledger.used_experiment(exp))
        except BudgetExceeded as e:
            print(e)
            return 1

    label = args.label
    pairs, controls, stopped = {}, {}, []

    async def run():
        for st in args.stages:
            exp, ids, _ = plan[st]
            mode = st.removeprefix("controls_")
            async with LLMClient(s, "judge", log_name=f"position_{label}.jsonl", experiment=exp) as client:
                cp = Path(".cache/checkpoints") / f"position_{label}_{st}.jsonl"
                if st.startswith("controls"):
                    controls[mode], left = await judge_controls(
                        client, cx, ids, mode=mode, checkpoint_path=cp, nonce=args.nonce, fresh=args.fresh
                    )
                else:
                    pairs[mode], left = await judge_pairs(
                        client, cx, cy, ids, mode=mode, checkpoint_path=cp, nonce=args.nonce, fresh=args.fresh
                    )
                stopped.extend(f"{st}:{i}" for i in left)
                if left:
                    break
        return client.model, client.json_mode

    model, json_mode = asyncio.run(run())
    doc = build_position_document(
        judge_model=model, json_mode=json_mode, naive_ids=naive_ids, control_case_ids=ctl_ids,
        pairs=pairs, controls=controls, nonce=args.nonce,
    )
    out = args.out or ("results/position_bias.json" if label == "main" else f"results/position_bias_{label}.json")
    write_document(doc, out)
    sm = doc["summary"]
    for name in ("pairwise_naive", "pairwise_mitigated_same_pairs_as_naive", "pairwise_mitigated_differing_pairs", "pairwise_mitigated_all"):
        r = sm[name]
        if r["n"]:
            print(f"{name:42s} n={r['n']:2d} flips={r['flips']}/{r['n_flippable']} (hard {r['hard_flips']}) flip_rate={r['flip_rate']} "
                  f"first-position share={r['first_position_share']} win_rate_A={r['win_rate_x']} "
                  f"[A {r['wins_x']} / B {r['wins_y']} / tie {r['ties']}] invalid={r['n_invalid']} reasks={r['reasks']}")
    for m, r in sm["controls"].items():
        print(f"controls {m:9s} n={r['n']} picked first {r['picked_first']}, second {r['picked_second']}, tie {r['tie']} "
              f"(first_rate={r['first_rate']}) invalid={r['n_invalid']} reasks={r['reasks']}")
    print(f"wrote {out}")
    if stopped:
        print(f"STOPPED at a quota cap with {len(stopped)} items left; re-run the same command to resume: {stopped}")
        return 2
    return 0


def cmd_probes(args: argparse.Namespace) -> int:
    from pathlib import Path

    from judge.llm import LLMClient
    from judge.pairwise import control_ids
    from judge.probes import BASES, build_document, plan_items, render_table, run_stage, write_document

    sys.stdout.reconfigure(errors="replace")
    s = load_settings()
    originals_all = _load_cases("data/outputs/A.json")
    if tuple(control_ids(originals_all)) != BASES:
        print(f"probe bases in the suite {control_ids(originals_all)} differ from probes.BASES {list(BASES)}")
        return 1
    originals = [c for c in originals_all if c.id in BASES]
    bases = [c.model_copy(update={"model_output": c.expected_output}) for c in originals]
    exp = {"ref": "probes", "free": "probes_ref_free"}
    plans = {st: plan_items(bases, originals, st) for st in args.stages}
    budget = load_budget()
    ledger = JudgeLedger(s.ledger_path, s.judge_daily_cap, budget.caps)
    for st, items in plans.items():  # upper bound: items not yet checkpointed (cache hits cost nothing)
        cp = Path(".cache/checkpoints") / f"probes_{st}_{args.label}.jsonl"
        done = 0 if args.fresh or not cp.exists() else len(cp.read_text(encoding="utf-8").splitlines())
        try:
            budget.check_run(exp[st], max(len(items) - done, 0), spent=ledger.used_experiment(exp[st]))
        except BudgetExceeded as e:
            print(e)
            return 1

    results, remaining, model, json_mode = {}, [], "", False

    async def run():
        nonlocal model, json_mode
        for st, items in plans.items():
            async with LLMClient(s, "judge", log_name=f"probes_{st}_{args.label}.jsonl", experiment=exp[st]) as client:
                got, left = await run_stage(client, items, st, checkpoint_path=Path(".cache/checkpoints") / f"probes_{st}_{args.label}.jsonl",
                                            nonce=args.nonce, fresh=args.fresh)
                results.update(got)
                remaining.extend(left)
                model, json_mode = client.model, client.json_mode
                if left:
                    break

    asyncio.run(run())
    texts = {k: v[2].model_output for st in plans for k, v in plans[st].items()}
    doc = build_document(results=results, texts=texts, judge_model=model, json_mode=json_mode, nonce=args.nonce, remaining=remaining)
    out = args.out or "results/probes.json"
    write_document(doc, out)
    print(render_table(doc["summary"]))
    print(f"wrote {out}")
    if remaining:
        print(f"STOPPED at a quota cap with {len(remaining)} items left; re-run the same command to resume: {remaining}")
        return 2
    return 0


def cmd_sensitivity(args: argparse.Namespace) -> int:
    from dataclasses import asdict
    from pathlib import Path

    from judge.llm import LLMClient
    from judge.pairwise import judge_pairs
    from judge.probes import BASES, sensitivity_cases, sensitivity_summary, write_document

    sys.stdout.reconfigure(errors="replace")
    s = load_settings()
    base = [c for c in _load_cases("data/outputs/A.json") if c.id in BASES]
    correct, wrong = sensitivity_cases(base)
    ids = [c.id for c in base]
    budget = load_budget()
    ledger = JudgeLedger(s.ledger_path, s.judge_daily_cap, budget.caps)
    try:
        budget.check_run("sensitivity", 2 * 2 * len(ids), spent=ledger.used_experiment("sensitivity"))
    except BudgetExceeded as e:
        print(e)
        return 1
    pairs, stopped = [], []

    async def run():
        async with LLMClient(s, "judge", log_name="sensitivity_main.jsonl", experiment="sensitivity") as client:
            for mode in ("mitigated", "naive"):
                got, left = await judge_pairs(client, correct, wrong, ids, mode=mode, nonce=args.nonce, fresh=args.fresh,
                                              checkpoint_path=Path(".cache/checkpoints") / f"sensitivity_{mode}.jsonl")
                pairs.extend(got)
                stopped.extend(f"{mode}:{i}" for i in left)
            return client.model, client.json_mode

    model, json_mode = asyncio.run(run())
    doc = {"judge_model": model, "json_mode": json_mode, "n_pairs": len(ids), "exploratory": True, "bases": ids,
           "design": "A = correct answer (the expected_output text), B = its confidently-wrong version; both orders; reference shown",
           "summary": sensitivity_summary(pairs), "pairs": [asdict(p) for p in pairs]}
    write_document(doc, args.out)
    for mode, r in doc["summary"].items():
        for rule in ("two_order", "single_order_correct_first"):
            x = r[rule]
            print(f"{mode:9s} {rule:27s} n={x['n']} correct {x['correct']} tied {x['tied']} wrong {x['wrong']} invalid {x['n_invalid']}")
        print(f"{mode:9s} flip rate {r['flip_rate']} ({r['flips']}/{r['n_flippable']}, hard {r['hard_flips']}) reasks {r['reasks']}")
    print(f"wrote {args.out}")
    return 2 if stopped else 0


def cmd_hard(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from judge.llm import LLMClient
    from judge.probes import BASES, HARD, NAIVE_FOOLED_MIN, analyze_hard, plan_items, run_stage, write_document

    sys.stdout.reconfigure(errors="replace")
    s = load_settings()
    originals = [c for c in _load_cases("data/outputs/A.json") if c.id in BASES]
    bases = [c.model_copy(update={"model_output": c.expected_output}) for c in originals]
    items = plan_items(bases, originals, "hard")
    budget = load_budget()
    ledger = JudgeLedger(s.ledger_path, s.judge_daily_cap, budget.caps)
    try:
        budget.check_run("probes_hard", len(items), spent=ledger.used_experiment("probes_hard"))
    except BudgetExceeded as e:
        print(e)
        return 1

    async def run():
        async with LLMClient(s, "judge", log_name="probes_hard_main.jsonl", experiment="probes_hard") as client:
            got, left = await run_stage(client, items, "hard", checkpoint_path=Path(".cache/checkpoints") / "probes_hard_main.jsonl",
                                        nonce=args.nonce, fresh=args.fresh)
            return got, left, client.model, client.json_mode

    results, left, model, json_mode = asyncio.run(run())
    doc = {"judge_model": model, "json_mode": json_mode, "exploratory": True, "n_bases": len(BASES), "mode": "reference",
           "texts": HARD, "naive_accepted_min": NAIVE_FOOLED_MIN, "summary": analyze_hard(results), "complete": not left,
           "remaining": left, "results": results}
    write_document(doc, args.out)
    print(json.dumps(doc["summary"], indent=1))
    print(f"wrote {args.out}")
    return 2 if left else 0


def cmd_selfpref(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from judge.llm import LLMClient
    from judge.pairwise import judge_pairs
    from judge.selfpref import (GEMINI, GPT, build_document, correct_by_reference, gemini_generator_settings,
                                generate_gemini, render_table, select_pairs, write_document)
    from judge.suite import load_config

    sys.stdout.reconfigure(errors="replace")
    s = load_settings()
    cfg = load_config("A")
    ref_cases = [c for c in _load_cases("data/outputs/A.json") if c.reference_based]
    gpt_ok_ids = [c.id for c in ref_cases if correct_by_reference(c, c.model_output)]
    gs = gemini_generator_settings(s)
    guard = dict(zip(("generator_family", "judge_slot_family"), gs.enforce_family_guard()))
    cp = Path(".cache/checkpoints")

    async def gen():
        async with LLMClient(gs, "generator", run_id="selfpref_gemini", log_name="selfpref_gemini_author.jsonl") as g:
            got = await generate_gemini(g, ref_cases, cfg.system_for, gpt_ok_ids, checkpoint_path=cp / "selfpref_gemini_author.jsonl")
            return got, g.model

    gem, gen_model = asyncio.run(gen())
    gemini_text = {i: r["text"] for i, r in gem.items()}
    Path("data/outputs/selfpref_gemini.json").write_text(
        json.dumps({"model": gen_model, "config": "A (system prompt only)", "answers": gem}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    selected, audit = select_pairs(ref_cases, gemini_text)
    gen_usage = {"model": gen_model, "calls": len(gem), "tokens_in": sum(r["tokens_in"] for r in gem.values()),
                 "tokens_out": sum(r["tokens_out"] for r in gem.values()),
                 "note": "generator-side Gemini calls; not in the judge ledger, but they use the same key's free-tier quota"}
    print(f"gemini answers: {len(gem)}  selected: {selected}")
    for a in audit:
        print("  ", a)
    budget = load_budget()
    ledger = JudgeLedger(s.ledger_path, s.judge_daily_cap, budget.caps)
    pairs_cp = cp / "selfpref_pairs.jsonl"
    done = 0 if args.fresh or not pairs_cp.exists() else len(pairs_cp.read_text(encoding="utf-8").splitlines())
    try:  # upper bound: pairs not yet checkpointed (cache hits cost nothing)
        budget.check_run("self_preference", 2 * max(len(selected) - done, 0), spent=ledger.used_experiment("self_preference"))
    except BudgetExceeded as e:
        print(e)
        return 1
    by_id = {c.id: c for c in ref_cases}
    cases_gpt = [by_id[i] for i in selected]
    cases_gem = [by_id[i].model_copy(update={"model_output": gemini_text[i]}) for i in selected]
    gpt_text = {i: by_id[i].model_output for i in selected}

    async def run():
        async with LLMClient(s, "judge", log_name="selfpref_main.jsonl", experiment="self_preference") as client:
            got, left = await judge_pairs(client, cases_gpt, cases_gem, selected, mode="mitigated", nonce=args.nonce, fresh=args.fresh,
                                          checkpoint_path=pairs_cp, x_label=GPT, y_label=GEMINI)
            return got, left, client.model, client.json_mode

    pairs, left, model, json_mode = asyncio.run(run())
    doc = build_document(judge_model=model, json_mode=json_mode, audit=audit, selected=[p.case_id for p in pairs], gpt_text=gpt_text,
                         gemini_text=gemini_text, pairs=pairs, gen_usage=gen_usage, guard=guard, remaining=left, nonce=args.nonce)
    write_document(doc, args.out)
    print(render_table(doc))
    print(f"wrote {args.out}")
    return 2 if left else 0


def cmd_clustering(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from judge.clustering import analyze, cluster_ids, render_tables, run_naive_suite, write_document
    from judge.llm import LLMClient

    sys.stdout.reconfigure(errors="replace")
    s = load_settings()
    cases = _load_cases("data/outputs/A.json")
    ids = cluster_ids(cases)
    budget = load_budget()
    ledger = JudgeLedger(s.ledger_path, s.judge_daily_cap, budget.caps)
    try:  # upper bound: probe-base cases were already judged naive by the probes run and hit the cache
        budget.check_run("clustering_naive", len(ids), spent=ledger.used_experiment("clustering_naive"))
    except BudgetExceeded as e:
        print(e)
        return 1

    async def run():
        async with LLMClient(s, "judge", log_name="clustering_main.jsonl", experiment="clustering_naive") as client:
            got, left = await run_naive_suite(client, cases, ids, checkpoint_path=Path(".cache/checkpoints") / "clustering_naive.jsonl",
                                              nonce=args.nonce, fresh=args.fresh)
            return got, left, client.model, client.json_mode

    naive, left, model, json_mode = asyncio.run(run())
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
    summary = analyze(
        cases=cases, ids=ids, naive=naive, mitigated_A=load("results/pointwise_A.json")["results"],
        mitigated_B=load("results/pointwise_B.json")["results"], probes=load("results/probes.json")["results"],
        hard=load("results/probes_hard.json")["results"],
    )
    calls = [c for r in naive.values() for c in r["calls"]]
    doc = {
        "judge_model": model, "json_mode": json_mode, "exploratory": True, "nonce": args.nonce,
        "design": {
            "cases": f"first {len(ids)} cases of config A in suite order: {ids}",
            "naive": "holistic 1-10, no rubric / anchors / evidence / length rule, pointwise, reference shown when the case has one",
            "mitigated": "anchored per-criterion 1-5 pointwise verdicts already on disk (results/pointwise_A.json, _B.json)",
            "separation_sources": "results/probes.json (terse, wrong, verbose_wrong) and results/probes_hard.json, no new judge calls",
            "caveat": "the two judges differ in prompt, scale and criteria at once, so a spread difference cannot be attributed to anchors alone",
        },
        "complete": not left, "remaining": left,
        "naive_usage": {"calls": len(calls), "network": sum(1 for c in calls if not c["cached"]),
                        "tokens_in": sum(c["tokens_in"] for c in calls), "tokens_out": sum(c["tokens_out"] for c in calls),
                        "reasks": sum(1 for c in calls if c["tag"].endswith("|reask")),
                        "parse_status": {k: sum(1 for r in naive.values() if r["parse_status"] == k)
                                         for k in sorted({r["parse_status"] for r in naive.values()})}},
        "summary": summary,
        "naive_results": naive,
    }
    write_document(doc, args.out)
    print(render_tables(summary))
    print(f"naive usage {doc['naive_usage']}")
    print(f"wrote {args.out}")
    return 2 if left else 0


def cmd_replay(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from judge.pointwise import replay_suite
    from judge.replay import ReplayClient

    sys.stdout.reconfigure(errors="replace")
    doc = json.loads(Path(args.results).read_text(encoding="utf-8"))
    log = args.log or f"logs/pointwise_{doc['label']}.jsonl"
    cases = _load_cases(args.cases_file or doc["cases_file"])
    client = ReplayClient(log)
    rows = asyncio.run(replay_suite(doc, cases, client))
    for r in rows:
        print(f"{r['case_id']:5s} {'SAME' if r['same'] else 'DIFF'}  replayed parse_status={r['replayed_status']}"
              + ("" if r["same"] else f"  {r['diffs']}"))
    same = sum(r["same"] for r in rows)
    print(f"replayed {len(rows)} verdicts from {log} with 0 API calls ({client.served} logged responses served): {same} identical, {len(rows) - same} different")
    return 0 if same == len(rows) else 1


def cmd_retest(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from judge.llm import LLMClient
    from judge.validate import analyze_retest, retest_ids, run_retest

    sys.stdout.reconfigure(errors="replace")
    s = load_settings()
    cases_file = f"data/outputs/{args.config}.json"
    cases = _load_cases(cases_file)
    ids = args.ids or retest_ids(cases)
    missing = set(ids) - {c.id for c in cases}
    if missing:
        print(f"unknown case ids: {sorted(missing)}")
        return 1
    baseline = json.loads(Path(f"results/pointwise_{args.config}.json").read_text(encoding="utf-8"))
    budget = load_budget()
    ledger = JudgeLedger(s.ledger_path, s.judge_daily_cap, budget.caps)
    try:  # upper bound: nonce draws already in the cache from an earlier run cost nothing
        budget.check_run("test_retest", len(ids) * args.reruns, spent=ledger.used_experiment("test_retest"))
    except BudgetExceeded as e:
        print(e)
        return 1

    async def run():
        async with LLMClient(s, "judge", log_name="retest_main.jsonl", experiment="test_retest") as client:
            docs, left = await run_retest(client, cases, ids, args.reruns, cases_file=cases_file, fresh=args.fresh)
            return docs, left, client.model, client.json_mode

    docs, left, model, json_mode = asyncio.run(run())
    if left:
        print(f"STOPPED with {len(left)} cases left (quota cap); re-run the same command to resume: {left}")
        return 2
    keep = set(ids)
    base = {**baseline, "results": [r for r in baseline["results"] if r["case_id"] in keep]}
    analysis = analyze_retest([base, *docs])
    doc = {"judge_model": model, "json_mode": json_mode, "judge_temperature": s.judge_temperature, "exploratory": True,
           "config": args.config, "case_ids": ids, "reruns": args.reruns,
           "design": "run 1 = the saved config-%s pointwise run (nonce ''); runs 2.. = reruns with nonce retest<k> (cache bypassed)" % args.config,
           **analysis}
    Path(args.out).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pf, sd = analysis["pass_flips"], analysis["overall_sd"]
    print(f"{analysis['n_cases']} cases x {analysis['n_runs']} runs | pass flips {pf['k']}/{pf['n']} {pf['cases']} | overall sd mean {sd['mean']} max {sd['max']} | invalid {analysis['n_invalid_per_run']}")
    for c in analysis["negative_controls"]:
        print(f"  control {'PASS' if c['ok'] else 'FAIL'}: {c['name']} -> {c['observed']}")
    print(f"wrote {args.out}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from judge.validate import LabelError, run_validation, write_outputs

    sys.stdout.reconfigure(errors="replace")
    try:
        doc = run_validation(labels_path=args.labels)
    except LabelError as e:
        print(f"label file error: {e}")
        return 1
    write_outputs(doc, args.out, args.out.replace(".json", ".md"))
    from judge.validate import render_markdown

    print(render_markdown(doc))
    print(f"wrote {args.out} and {args.out.replace('.json', '.md')} (0 judge calls)")
    return 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="judge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("budget", help="dry-run: planned judge calls per experiment; refuses runs over a cap")
    b.add_argument("--check", metavar="EXPERIMENT", help="refuse (exit 1) if EXPERIMENT cannot run --calls more calls")
    b.add_argument("--calls", type=int, default=0)
    b.set_defaults(fn=cmd_budget)
    g = sub.add_parser("generate", help="generate model_output for the suite under generator config(s) A/B")
    g.add_argument("--config", nargs="+", default=["A", "B"])
    g.add_argument("--suite", default="data/suite.yaml")
    g.add_argument("--limit", type=int, default=0, help="only the first N cases (smoke)")
    g.add_argument("--sample", type=int, default=5, help="print this many first lines per config")
    g.set_defaults(fn=cmd_generate)
    r = sub.add_parser("genreport", help="constraint pass rate and exact-match per generator config (no LLM calls)")
    r.add_argument("--config", nargs="+", default=["A", "B", "C"])
    r.set_defaults(fn=cmd_genreport)
    p = sub.add_parser("pointwise", help="mitigated pointwise judging of one config's outputs (live judge calls)")
    p.add_argument("--config", default="A", help="generator config whose outputs are judged (data/outputs/<config>.json)")
    p.add_argument("--cases-file", default="", help="override the outputs file")
    p.add_argument("--ids", nargs="+", help="only these case ids")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--label", default="", help="names results/pointwise_<label>.json and logs/pointwise_<label>.jsonl (default: config)")
    p.add_argument("--experiment", default="pointwise_mitigated", help="budget experiment to charge ('' = none)")
    p.add_argument("--nonce", default="", help="bypass the cache (test-retest)")
    p.add_argument("--fresh", action="store_true", help="discard the checkpoint and judge every case again")
    p.set_defaults(fn=cmd_pointwise)
    po = sub.add_parser("position", help="pairwise order-swap, flip rate and identical-answer controls (live judge calls)")
    po.add_argument("--stages", nargs="+", default=["naive", "mitigated", "controls_naive", "controls_mitigated"],
                    choices=["naive", "mitigated", "controls_naive", "controls_mitigated"])
    po.add_argument("--ids", nargs="+", help="restrict every stage to these case ids (smoke)")
    po.add_argument("--label", default="main", help="names the checkpoints and logs/position_<label>.jsonl; 'main' writes results/position_bias.json")
    po.add_argument("--out", default="")
    po.add_argument("--nonce", default="")
    po.add_argument("--fresh", action="store_true")
    po.set_defaults(fn=cmd_position)
    ru = sub.add_parser("run", help="mitigated pointwise judging + aggregate report per config (live judge calls)")
    ru.add_argument("--config", nargs="+", default=["A", "B"])
    ru.add_argument("--cases-file", default="")
    ru.add_argument("--ids", nargs="+")
    ru.add_argument("--limit", type=int, default=0)
    ru.add_argument("--experiment", default="pointwise_mitigated")
    ru.add_argument("--nonce", default="")
    ru.add_argument("--fresh", action="store_true")
    ru.add_argument("--report-only", action="store_true", help="rebuild results/report_<cfg>.json from saved results (no judge calls)")
    ru.add_argument("--price-in", type=float, default=None, help="USD per 1M judge input tokens (default: pricing.yaml)")
    ru.add_argument("--price-out", type=float, default=None, help="USD per 1M judge output tokens")
    ru.set_defaults(fn=cmd_run)
    cp = sub.add_parser("compare", help="A/B decision from saved results: paired bootstrap, win rate (no judge calls)")
    cp.add_argument("--a", default="A")
    cp.add_argument("--b", default="B")
    cp.add_argument("--a-results", default="")
    cp.add_argument("--b-results", default="")
    cp.add_argument("--pairs", default="results/position_bias.json")
    cp.add_argument("--price-in", type=float, default=None)
    cp.add_argument("--price-out", type=float, default=None)
    cp.set_defaults(fn=cmd_compare)
    pr = sub.add_parser("probes", help="verbosity + sycophancy probes, naive vs mitigated, reference and reference-free (live judge calls)")
    pr.add_argument("--stages", nargs="+", default=["ref", "free"], choices=["ref", "free"])
    pr.add_argument("--label", default="main", help="names the checkpoints and logs/probes_<stage>_<label>.jsonl")
    pr.add_argument("--out", default="")
    pr.add_argument("--nonce", default="")
    pr.add_argument("--fresh", action="store_true")
    pr.set_defaults(fn=cmd_probes)
    se = sub.add_parser("sensitivity", help="correct vs confidently-wrong pairs, both orders, naive + mitigated (live judge calls)")
    se.add_argument("--out", default="results/sensitivity.json")
    se.add_argument("--nonce", default="")
    se.add_argument("--fresh", action="store_true")
    se.set_defaults(fn=cmd_sensitivity)
    ha = sub.add_parser("hard", help="hard probes: right answer + fabricated reasoning, naive + mitigated pointwise (live judge calls)")
    ha.add_argument("--out", default="results/probes_hard.json")
    ha.add_argument("--nonce", default="")
    ha.add_argument("--fresh", action="store_true")
    ha.set_defaults(fn=cmd_hard)
    sp = sub.add_parser("selfpref", help="self-preference probe: gpt-oss vs Gemini-authored correct answers, both orders (live calls)")
    sp.add_argument("--out", default="results/self_preference.json")
    sp.add_argument("--nonce", default="")
    sp.add_argument("--fresh", action="store_true")
    sp.set_defaults(fn=cmd_selfpref)
    cl = sub.add_parser("clustering", help="score clustering: naive 1-10 vs anchored 1-5, plus good-vs-corrupted separation (live calls)")
    cl.add_argument("--out", default="results/clustering.json")
    cl.add_argument("--nonce", default="")
    cl.add_argument("--fresh", action="store_true")
    cl.set_defaults(fn=cmd_clustering)
    rt = sub.add_parser("retest", help="test-retest: re-judge a subset with a nonce per rerun (cache bypassed; live judge calls, budget test_retest)")
    rt.add_argument("--config", default="A")
    rt.add_argument("--ids", nargs="+", help="default: every second case in suite order (10 of 20)")
    rt.add_argument("--reruns", type=int, default=2, help="nonce reruns; the saved run counts as run 1, so 2 reruns = 3 runs (the budget); 3 reruns on 10 cases exceeds the cap")
    rt.add_argument("--out", default="results/test_retest.json")
    rt.add_argument("--fresh", action="store_true")
    rt.set_defaults(fn=cmd_retest)
    va = sub.add_parser("validate", help="kappa/agreement vs by-construction and human labels, test-retest, adversarial table (no judge calls)")
    va.add_argument("--labels", default=None, help="human labels JSONL (default labels/human_labels.jsonl)")
    va.add_argument("--out", default="results/validation.json")
    va.set_defaults(fn=cmd_validate)
    rp = sub.add_parser("replay",help="reproduce saved pointwise verdicts from the call log, zero API calls")
    rp.add_argument("--results", default="results/pointwise_A.json")
    rp.add_argument("--log", default="", help="call log (default logs/pointwise_<label>.jsonl)")
    rp.add_argument("--cases-file", default="")
    rp.set_defaults(fn=cmd_replay)
    args = ap.parse_args(argv)
    try:
        code = args.fn(args)
    except MissingKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        code = 2
    sys.exit(code)


if __name__ == "__main__":
    main()
