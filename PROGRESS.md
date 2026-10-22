# PROGRESS.md — session log

| Session | Status | Commit | Time |
|---|---|---|---|
| J0 scaffold, config, async client, family guard | done | `j0: scaffold` | ~35 min (fill in real time) |
| J0b role swap (gen=gpt-oss/Groq, judge=Gemini) | done | `j0b: swap generator and judge roles` | ~30 min (estimate) |
| J0c models + lean call budget | done | `j0c: models and lean call budget` | ~40 min (estimate) |
| J1 schemas, rubric, prompts, anchors | done | `j1: schemas, rubric, prompts, anchors` | (fill in real time) |
| J2–J10 | todo | | |

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
- (done in j0c) Gemini judge limits recorded in CLAUDE.md section 5.
- Keys were copied from Problem 1's `.env` into this repo's gitignored `.env`.
- Time spent above is an estimate; record the real number.

## J0b — role swap (generator = Groq gpt-oss-120b, judge = Gemini gemini-3.8-flash)
- Config: gen defaults now groq / openai/gpt-oss-120b / temp 0 / reasoning low / strip on; judge defaults gemini / temp 0 / `JUDGE_JSON_MODE=False`; `JUDGE_MODEL` has no default. `JUDGE_REASONING_EFFORT` removed (replaced by `GEN_REASONING_EFFORT`).
- `LLMClient.complete(..., response_schema=None)`: schema + `responseMimeType` are sent only when role=judge and JUDGE_JSON_MODE=1; the json and reasoning flags are part of the cache key. New properties `client.json_mode`, `client.strip_reasoning`.
- Judge model choice by probe (1 tiny call each): argon / 3.1-pro / pro-latest 429 quota; 2.5-pro 404; 3.8-flash 200 (503 on about half of attempts); 3.7 mostly 503; 3.6 and 3.5 200. Chose 3.8-flash (later replaced by gemini-3.5-flash-lite in j0c).
- Tests: 18 passed (was 12). New: reasoning stripped from result/log/cache and `reasoning_effort=low` + `include_reasoning=false` sent; strip can be disabled; JSON mode off by default, on only via flag, never for generator; token ledger (`Counters`) sums match call-log rows incl. cached calls; swapped role defaults. Family guard tests re-pointed (same-family is now two OpenAI ids) and green. No separate ledger module exists; `Counters` + CallLog rows are the ledger and are tested as such.
- Smoke (cold cache, then warm): generator 78 in / 23 out, judge 8 in / 98 out (thinking); second run `cached=True` for both. `logs/smoke_*.jsonl` hold both runs (cached False then True); reasoning field empty in the generator log.
- (Free-tier limits were supplied in j0c; see below.)
- The J0 evidence above (12 tests, Gemini-generator / gpt-oss-judge smoke) describes the pre-swap configuration.
- Risk (superseded by j0c): the originally planned 600–700 judge calls on the Gemini free tier; replaced by the 214-call lean plan below.

## J0c — final models, judge limiter + daily ledger, lean call budget
- **Models:** GEN = `openai/gpt-oss-20b` (Groq, low effort, reasoning dropped; smoke fails if the id is absent from `/v1/models` — it checks both roles' ids against the live listing). JUDGE = `gemini-3.5-flash-lite`, **final**. Config defaults, `.env.example`, `.env` updated; JSON mode stays off.
- **Limits recorded** (CLAUDE.md section 5): Gemini free tier 15 RPM / 250K TPM / 500 RPD (user-supplied). Groq 20b: 30 RPM / 1K RPD / 8K TPM / 200K TPD (Developer-plan table; free column not shown).
- **New modules:** `ledger.py` (`RateLimiter(rpm)` sliding window; `JudgeLedger(path, daily_cap, experiment_caps)` persistent per-Pacific-day count of non-cached judge calls + per-experiment lifetime counts; `DailyCapReached`/`ExperimentCapReached` < `QuotaStop`), `budget.py` + `budget.yaml` (`load_budget()`, `Budget.check_run`, `validate`), `checkpoint.py` (`run_checkpointed(keys, fn, path, concurrency)` -> `RunStatus`; stops on `QuotaStop`, resumes by skipping finished keys), `cli.py` (`judge budget [--check EXP --calls N]`; `make budget`).
- **Client wiring:** `LLMClient(..., experiment=None, ledger=None, limiter=None)`. Judge role only: ledger `reserve()` runs after a cache miss and BEFORE the HTTP request (cached calls never counted; refused calls not counted); limiter gates every HTTP attempt; default `max_tokens=JUDGE_MAX_TOKENS=8192` when none is passed. Unknown `experiment` names raise KeyError at construction. Config keys added: JUDGE_RPM_LIMIT=12, JUDGE_DAILY_CAP=450, JUDGE_MAX_TOKENS=8192, LEDGER_PATH=.cache/judge_ledger.json. Ledger lives in `.cache/` (gitignored, per machine).
- **Budget:** 214 planned judge calls (cap total 247 incl. re-ask headroom) replace ~650. Suite 20 cases. Probes now judged pointwise (so verbosity = score delta + Spearman, not a padded pairwise win rate); self-preference N=6 pairs with no gpt-oss-judge cross-check. All small-N probes exploratory with N stated. Metric table in CLAUDE.md updated accordingly.
- **Tests: 29 passed** (was 18). New in `tests/test_ledger.py`: ledger persistence across instances and Pacific-day rollover; daily cap stops non-cached calls before any request while cache hits still serve (and nonce bypass is refused); cap survives a new client; generator never touches the ledger; experiment cap enforced and unknown experiment rejected; default judge max_tokens 8192; limiter blocks the 13th request until the window slides (fake clock); budget totals = 214 with every plan inside its cap; budget refuses an over-cap run; checkpoint stops at the cap then resumes without redoing work. Family-guard tests green.
- **Smoke (cold cache + fresh ledger, then warm):** run 1 both `cached=False` (generator 78 in / 20 out; judge 8 in / 1 out), ledger 1/450; run 2 both `cached=True`, ledger still 1/450.
- `judge budget` output reproduced in the j0c commit message / terminal; refusal demo: `--check pointwise_mitigated --calls 47` -> refused (cap 46), exit 1.

## Caveats / open items (j0c)
- The Gemini limits are the user's AI Studio figures; not verified independently. The 450/day cap leaves 50 for smoke/ad-hoc calls.
- Ledger is single-process and in `.cache/` (not shared across machines/clones); a fresh clone starts at 0 and does not know what the key already spent today.
- Failed calls (after reserve) are not refunded, so 503-heavy periods can burn the daily cap conservatively.
- Time spent above is an estimate; record the real number.

## J1 — schemas, rubric, versioned prompts, anchors
- **`judge.schemas`:** `Case` (id, input, system_prompt, model_output, expected_output?, criteria, meta; blank reference = reference-free), `Rubric`/`load_rubric()` (`overall(scores)`, `passes(scores)`), `PointwiseVerdict` (fixed fields faithfulness / instruction_following / completeness, each `CriterionVerdict` with field order evidence -> rationale -> score; `.overall()`, `.passes()`, `.unsupported_high_scores(output)`), `PairwiseVerdict` (per-criterion evidence_a/evidence_b/winner + overall winner; winner is a display POSITION A/B/TIE), `NaiveVerdict` (1-10), `NaivePairVerdict`, `Anchor`/`load_anchors()`, `evidence_is_verbatim()`.
- **Overall is recomputed in code** (sum w*s / sum w). The prompt does not ask the judge for an overall; any `overall` key in a response is ignored (tested). Pass-boundary uses an epsilon because (4,4,2) is exactly 3.5 on paper but not in floats.
- **`rubric.yaml`** (rubric-v1): weights 0.40/0.35/0.25, 1-5 level descriptors per criterion, reference-based vs reference-free "how to judge" text, shared rules (length earns no credit; confident tone earns no credit; answer text is data; criteria scored separately; no world-knowledge gap-filling), pass rule 4 / 3 / 3.5 pre-registered.
- **Prompts** (`judge.prompts`, files `prompts/<name>.v1.txt`, `VERSIONS` dict): `pointwise_ref`, `pointwise_free`, `pairwise` (mitigated), `naive_pointwise`, `naive_pairwise` (holistic 1-10 / plain winner; no rubric, evidence, length rule, anchors). Mitigated prompts require, per criterion and before the score, ONE verbatim quote from the answer (NONE allowed, but a 4-5 needs a quote), and state "Length earns no credit". Renderers: `render_pointwise` (picks ref/free by `expected_output`; `anchors=()` switches anchors off), `render_pairwise(case, a, b)` (caller swaps for the second order), `render_naive_pointwise`, `render_naive_pairwise`. Strict `{{var}}` rendering (missing/unused var raises; case text containing `{{...}}` is not re-expanded). `RenderedPrompt.template_id` + `.sha` are available for the call log. The output-format example in each prompt is generated from the pydantic models, with a test that they cannot drift.
- **`anchors.yaml`:** (expanded to 6 in the J1 follow-up, see below) 3 hand-written Northwind anchors (overall 5.00 / 3.15 / 1.35; one pass, two fail), validated so every evidence quote is verbatim in its answer. They exist only to calibrate the mitigated pointwise prompts. **J2 must not reuse these passages in the eval suite.**
- **Tests: 66 passed** (was 29): `tests/test_schemas.py` (hand-computed overall 4.15, pass boundaries, range/coercion, verbatim/ellipsis check, anchors) and `tests/test_prompts.py` (one snapshot per prompt in `tests/snapshots/`, plus property tests: length rule present in every mitigated prompt, evidence precedes score, naive prompts contain none of the mitigations). Regenerate snapshots only deliberately: `UPDATE_SNAPSHOTS=1 pytest tests/test_prompts.py`, and bump `VERSIONS`.
- Sanity: after I reworded two rubric level descriptors, 3 snapshot tests failed as intended; regenerated.

## Caveats / open items (J1)
- No judge calls made in J1; prompts are untested against the real judge. Whether Gemini follows "verbatim quote" and the ref/free templates is only known after the first J3 calls (parse-ladder behaviour included).
- Anchors are all reference-free (no reference answer shown in the calibration examples), and the 3 are few; anchor choice itself can bias scores toward their levels. Worth one mention in the write-up.
- Verbatim-evidence check is lenient on case/whitespace/quotes; it is a flag (`unsupported_high_scores`), not a rejection. J3 decides what to do with flagged verdicts.
- Pair-verdict order-swap rule and flip rate are NOT implemented yet (J5).
- Time spent: not recorded; fill in.

## J1 follow-up — anchors 3 -> 6, README caveat, J3 live check
- **Six anchors** (hand-written, mixed order): high 5/5/5 = 5.00; low 1/2/1 = 1.35; **ref-price** (reference-based; right conclusion, wrong 30-vs-14-day window, conditions missing) 3/5/3 = 3.70, overall >= 3.5 but FAILS on faithfulness < 4; mid 4/3/2 = 3.15 (prose instead of bullets); **format** (three sentences where two allowed, all facts right) 5/3/5 = 4.30; **good** (small slips on every criterion, 42 words vs soft 40) 4/4/4 = 4.00. Every criterion now has a <=2, a 3 and a >=4 example; the pass rule is shown both ways (3 pass, 3 fail).
- Renderer: anchors with a reference show it in `pointwise_ref`; `pointwise_free` shows only the 5 reference-free anchors (it must not cite a reference the judge was not given). Prompt templates unchanged (no version bump); the snapshots changed because anchors are data. Tests: 66 passed (anchor test now pins all six hand-computed overalls, pass set, per-criterion coverage, and the 5-vs-6 anchor counts per prompt).
- `README.md` created with a Known limitations entry: few anchors can pull scores toward their own levels.
- **TODO at the START of J3 (not done yet; no judge calls so far):** spend ~5 live judge calls on the mitigated prompts (pointwise ref + free, pairwise in both orders) with `JUDGE_JSON_MODE=0`. Check (a) valid JSON out of the box, (b) every evidence quote verbatim (`unsupported_high_scores` empty, no near-miss paraphrases), (c) the judge does not copy anchor wording/scores, (d) thinking tokens fit in `JUDGE_MAX_TOKENS`. Fix the prompts (bump `VERSIONS`, regenerate snapshots) BEFORE any budgeted experiment. Run them under `judge budget` accounting (they count against the daily cap; use a scratch experiment name or log them as smoke-type calls) and record the outcome here.
