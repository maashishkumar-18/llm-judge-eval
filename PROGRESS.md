# PROGRESS.md — session log

| Session | Status | Commit | Time |
|---|---|---|---|
| J0 scaffold, config, async client, family guard | done | `j0: scaffold` | ~35 min (fill in real time) |
| J1–J10 | todo | | |

## J0 — public interfaces
- `judge.config.Settings` (pydantic-settings, `.env`): GEN_PROVIDER/GEN_MODEL/GEN_TEMPERATURE/GEN_CONCURRENCY, JUDGE_PROVIDER/JUDGE_MODEL/JUDGE_TEMPERATURE/JUDGE_CONCURRENCY/JUDGE_REASONING_EFFORT, GEMINI_API_KEY, GROQ_API_KEY, ALLOW_SAME_FAMILY, MAX_RETRIES, BACKOFF_BASE_SECONDS, BACKOFF_MAX_SECONDS, REQUEST_TIMEOUT_SECONDS, CACHE_DIR, LOG_DIR. `load_settings()`, `Settings.enforce_family_guard() -> (gen_family, judge_family)`, `family_of(model_id)`, `FamilyGuardError`.
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
- Groq TPM/TPD (8K/200K) likely limits the ~650-call plan more than RPM; confirm plan limits in the Groq console and consider `JUDGE_REASONING_EFFORT=low` for bulk runs. Possibly trim prompts/anchors.
- Look up Gemini's actual free RPM/RPD in AI Studio before J2.
- Keys were copied from Problem 1's `.env` into this repo's gitignored `.env`.
- Time spent above is an estimate; record the real number.
