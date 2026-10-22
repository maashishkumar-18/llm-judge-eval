# PROGRESS.md — session log

| Session | Status | Commit | Time |
|---|---|---|---|
| J0 scaffold, config, async client, family guard | done | `j0: scaffold` | ~35 min (fill in real time) |
| J0b role swap (gen=gpt-oss/Groq, judge=Gemini) | done | `j0b: swap generator and judge roles` | ~30 min (estimate) |
| J1–J10 | todo | | |

## J0 — public interfaces
- `judge.config.Settings` (pydantic-settings, `.env`): GEN_PROVIDER/GEN_MODEL/GEN_TEMPERATURE/GEN_CONCURRENCY/GEN_REASONING_EFFORT/GEN_STRIP_REASONING, JUDGE_PROVIDER/JUDGE_MODEL/JUDGE_TEMPERATURE/JUDGE_CONCURRENCY/JUDGE_JSON_MODE (changed in j0b), GEMINI_API_KEY, GROQ_API_KEY, ALLOW_SAME_FAMILY, MAX_RETRIES, BACKOFF_BASE_SECONDS, BACKOFF_MAX_SECONDS, REQUEST_TIMEOUT_SECONDS, CACHE_DIR, LOG_DIR. `load_settings()`, `Settings.enforce_family_guard() -> (gen_family, judge_family)`, `family_of(model_id)`, `FamilyGuardError`.
- `judge.llm.LLMClient(settings, role="judge"|"generator", run_id=, log_name=, transport=, sleep=)`; `await client.complete(prompt, case_id=, system=, temperature=, max_tokens=, nonce="") -> {text, reasoning, tokens_in, tokens_out, cached}`; `client.counters` (calls, network_calls, cached_calls, tokens_in, tokens_out); `client.log_path`. Client construction runs the family guard.
- `judge.llm.CallLog` fields: run_id, role, model, case_id, prompt_hash, prompt, raw_response, tokens_in, tokens_out, latency_ms, cached, nonce, reasoning, system, temperature, ts. Default log file `logs/calls_<role>_<run_id>.jsonl`.
- `judge.cache`: `make_key(model, params, payload, nonce)`, `prompt_hash`, `DiskCache` (atomic writes).
- `make smoke` / `make test` / `make install`. Windows: `make ... PY=.venv/Scripts/python` if `python` isn't the venv.

## J0 — evidence
- `make test`: 12 passed (cache hit + log rows, nonce bypass, Retry-After honoured, 401 not retried, give-up after max retries, Gemini token parsing, family guard blocks/overrides/unknown, independent GEN/JUDGE settings).
- Smoke, cold cache: both calls `cached=False`, SMOKE PASSED; immediately again: both `cached=True`, SMOKE PASSED. Judge tokens 78 in / 55 out.
- No key patterns (`AIza`, `gsk_`) in the tree outside `.env`.

## Bugs and fixes (raw material for 2.5)
- First smoke run "failed": Gemini answered `ping` to "Reply with exactly one word: pong" (model quirk, plumbing fine). My check was over-strict; changed to non-empty reply + counted tokens. Not a client bug. (On the cold re-run it did answer `pong` — the model is not deterministic on a trivial prompt even at temperature 0.)
- Decision: Gemini docs give no free-tier limits (AI Studio only) — not hard-coded anywhere.

## Open items / risks carried forward
- (updated j0b) Groq now only generates, so its 8K TPM / 200K TPD limits matter less; confirm plan limits in the Groq console anyway.
- Gemini (judge) free RPM/TPM/RPD must be recorded in CLAUDE.md section 5 before J3 (first judge calls); the user was to paste them.
- Keys were copied from Problem 1's `.env` into this repo's gitignored `.env`.
- Time spent above is an estimate; record the real number.

## J0b — role swap (generator = Groq gpt-oss-120b, judge = Gemini gemini-3.8-flash)
- Config: gen defaults now groq / openai/gpt-oss-120b / temp 0 / reasoning low / strip on; judge defaults gemini / temp 0 / `JUDGE_JSON_MODE=False`; `JUDGE_MODEL` has no default. `JUDGE_REASONING_EFFORT` removed (replaced by `GEN_REASONING_EFFORT`).
- `LLMClient.complete(..., response_schema=None)`: schema + `responseMimeType` are sent only when role=judge and JUDGE_JSON_MODE=1; the json and reasoning flags are part of the cache key. New properties `client.json_mode`, `client.strip_reasoning`.
- Judge model choice by probe (1 tiny call each): argon / 3.1-pro / pro-latest 429 quota; 2.5-pro 404; 3.8-flash 200 (503 on about half of attempts); 3.7 mostly 503; 3.6 and 3.5 200. Chose 3.8-flash; fallback 3.6-flash.
- Tests: 18 passed (was 12). New: reasoning stripped from result/log/cache and `reasoning_effort=low` + `include_reasoning=false` sent; strip can be disabled; JSON mode off by default, on only via flag, never for generator; token ledger (`Counters`) sums match call-log rows incl. cached calls; swapped role defaults. Family guard tests re-pointed (same-family is now two OpenAI ids) and green. No separate ledger module exists; `Counters` + CallLog rows are the ledger and are tested as such.
- Smoke (cold cache, then warm): generator 78 in / 23 out, judge 8 in / 98 out (thinking); second run `cached=True` for both. `logs/smoke_*.jsonl` hold both runs (cached False then True); reasoning field empty in the generator log.
- **Not done:** the Gemini free-tier limits were to be pasted by the user; none arrived. Placeholders are in CLAUDE.md section 5.
- The J0 evidence above (12 tests, Gemini-generator / gpt-oss-judge smoke) describes the pre-swap configuration.
- Risk: the 600–700 judge calls now hit the Gemini free tier with unknown RPM/RPD and frequent 503s on 3.8; checkpoint/resume and the cache matter more. A judge change mid-project invalidates validation numbers.
