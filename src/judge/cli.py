"""judge CLI: `budget` (j0c) and `generate` (j2); later sessions add run/compare/replay/probes/validate."""

from __future__ import annotations

import argparse
import asyncio
import sys

from judge.budget import BudgetExceeded, load_budget
from judge.config import load_settings
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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="judge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("budget", help="dry-run: planned judge calls per experiment; refuses runs over a cap")
    b.add_argument("--check", metavar="EXPERIMENT", help="refuse (exit 1) if EXPERIMENT cannot run --calls more calls")
    b.add_argument("--calls", type=int, default=0)
    b.set_defaults(fn=cmd_budget)
    g = sub.add_parser("generate", help="generate model_output for the suite under generator config(s) A/B")
    g.add_argument("--config", nargs="+", default=["A", "B"])
    g.add_argument("--suite", default="data/suite/candidates.yaml")
    g.add_argument("--limit", type=int, default=0, help="only the first N cases (smoke)")
    g.add_argument("--sample", type=int, default=5, help="print this many first lines per config")
    g.set_defaults(fn=cmd_generate)
    args = ap.parse_args(argv)
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
