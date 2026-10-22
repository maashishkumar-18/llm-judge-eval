"""Generate model_output for each suite case under a generator config (A / B) with the GENERATOR client.

One call per case, checkpointed (resumes without redoing finished cases), cached by the client. Output file:
data/outputs/<config>.json = list of Case-shaped dicts (model_output filled) + meta recording config, prompt
version and the deterministic constraint-check results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from judge.checks import run_checks
from judge.checkpoint import RunStatus, run_checkpointed
from judge.config import Settings
from judge.llm import LLMClient
from judge.schemas import Case
from judge.suite import GenConfig

GEN_MAX_TOKENS = 1024  # reasoning (effort=low) shares this budget


async def generate_config(
    settings: Settings,
    rows: list[dict[str, Any]],
    config: GenConfig,
    out_dir: Path = Path("data/outputs"),
    *,
    client: LLMClient | None = None,
) -> tuple[list[Case], RunStatus]:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {r["id"]: r for r in rows}
    own = client is None
    client = client or LLMClient(settings, "generator", run_id=f"gen_{config.name}", log_name=f"generate_{config.name}.jsonl")

    async def one(case_id: str) -> dict[str, Any]:
        r = by_id[case_id]
        res = await client.complete(
            r["input"], case_id=case_id, system=config.system_for(r.get("system_prompt", "")), max_tokens=GEN_MAX_TOKENS
        )
        return {"model_output": res["text"].strip(), "tokens_in": res["tokens_in"], "tokens_out": res["tokens_out"]}

    try:
        status = await run_checkpointed(
            list(by_id), one, out_dir / f".{config.name}.checkpoint.jsonl", concurrency=settings.gen_concurrency
        )
    finally:
        if own:
            await client.aclose()

    cases: list[Case] = []
    for cid, r in by_id.items():
        if cid not in status.results:
            continue
        out = status.results[cid]["model_output"]
        meta = {k: v for k, v in r.get("meta", {}).items()}
        meta.update(
            config=config.name, prompt_version=config.prompt_version, gen_model=settings.gen_model,
            tokens_in=status.results[cid]["tokens_in"], tokens_out=status.results[cid]["tokens_out"],
            constraint_results=[
                {"type": c.type, "passed": c.passed, "detail": c.detail} for c in run_checks(out, meta["checks"])
            ],
        )
        cases.append(Case(
            id=cid, input=r["input"], system_prompt=r.get("system_prompt", ""), model_output=out,
            expected_output=r.get("expected_output"), criteria=r.get("criteria", []), meta=meta,
        ))
    (out_dir / f"{config.name}.json").write_text(
        json.dumps([c.model_dump() for c in cases], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return cases, status
