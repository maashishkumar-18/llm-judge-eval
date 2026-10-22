"""make smoke: fails loudly if the judge/generator model ids are not actually served, then makes
one generator call and one judge call and prints tokens, cache status and the family check."""

from __future__ import annotations

import asyncio
import sys

from judge.config import FamilyGuardError, load_settings
from judge.llm import LLMClient, gemini_model_exists, list_groq_models

PROMPT = "Reply with exactly one word: pong"


async def run() -> int:
    s = load_settings()
    try:
        gf, jf = s.enforce_family_guard()
    except FamilyGuardError as e:
        print(f"FAIL family guard: {e}")
        return 1
    print(f"family guard OK: generator={s.gen_model} ({gf}) | judge={s.judge_model} ({jf})")

    # 1. the ids must actually be served (checked per role's provider)
    served_groq: list[str] | None = None
    for role, provider, model in (("generator", s.gen_provider, s.gen_model), ("judge", s.judge_provider, s.judge_model)):
        if provider == "groq":
            served_groq = served_groq if served_groq is not None else await list_groq_models(s)
            if model not in served_groq:
                print(f"FAIL {role} model {model!r} not served by Groq. Served: {sorted(served_groq)}")
                return 1
            print(f"{role} model served by Groq ({len(served_groq)} models listed)")
        elif provider == "gemini":
            if not await gemini_model_exists(s, model):
                print(f"FAIL {role} model {model!r} not found on Gemini")
                return 1
            print(f"{role} model served by Gemini")

    # 2. one call each
    ok = True
    for role in ("generator", "judge"):
        async with LLMClient(s, role, run_id="smoke", log_name=f"smoke_{role}.jsonl") as c:
            r = await c.complete(PROMPT, case_id="smoke", max_tokens=1024)
            print(
                f"{role:9s} {c.model:24s} json_mode={c.json_mode!s:5s} cached={r['cached']!s:5s} "
                f"tokens_in={r['tokens_in']} tokens_out={r['tokens_out']} reply={r['text'].strip()[:40]!r}"
            )
            # plumbing check only: a non-empty reply with counted tokens (models may not echo exactly)
            if not r["text"].strip() or r["tokens_in"] <= 0 or r["tokens_out"] <= 0:
                print(f"FAIL {role}: empty reply or missing token counts")
                ok = False
    print("SMOKE PASSED" if ok else "SMOKE FAILED")
    return 0 if ok else 1


def main() -> None:
    sys.exit(asyncio.run(run()))


if __name__ == "__main__":
    main()
