# PROGRESS.md — session log

| Session | Status | Commit | Time |
|---|---|---|---|
| J0 scaffold, config, async client, family guard | done | `j0: scaffold` | ~35 min (fill in real time) |
| J0b role swap (gen=gpt-oss/Groq, judge=Gemini) | done | `j0b: swap generator and judge roles` | ~30 min (estimate) |
| J0c models + lean call budget | done | `j0c: models and lean call budget` | ~40 min (estimate) |
| J1 schemas, rubric, prompts, anchors | done | `j1: schemas, rubric, prompts, anchors` | (fill in real time) |
| J2 candidate pool (40), checks, generate A/B | done, curation pending (user) | `j2: candidate pool, constraint checks, generate A/B` | (fill in real time) |
| J3 parse ladder, pointwise, replay | done (pairwise live check deferred to J5) | `j3: parse ladder, pointwise judge, replay` | (fill in real time) |
| J4 pairwise order-swap, position bias | done | `j4: pairwise order-swap...` | (fill in) |
| J5 aggregate, compare, run/compare CLI | done | `j5: aggregate, compare, run and compare CLI` | (fill in) |
| J6 / J6b probes, sensitivity, hard probes | done | `j6`, `j6b` | (fill in) |
| J7 family guard, self-preference, score clustering | done | `j7` | (fill in) |
| J8 validate: kappa, test-retest, adversarial table | done except human-label kappa (labels file not written yet) | `j8` | (fill in) |
| J9 README, pipeline diagram, generated tables, clean-clone test, answers draft | done | `j9` | (fill in) |
| J10 | todo | | |

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
- API keys live only in this repo's gitignored `.env` (see `.env.example` for names).
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

## J2 — candidate pool, deterministic constraint checks, generation for configs A / B
- **Decisions (asked, user confirmed):** generator stays gpt-oss-20b via Groq (NOT Gemini; the request said Gemini, which would break the family guard and the locked judge/generator split); suite is curated from 40 candidates to **20** (not ~30), matching `budget.yaml`.
- **`data/suite/candidates.yaml`:** 40 cases, 27 reference-based / 13 reference-free, each with a plain-words constraint in `system_prompt` and a machine-checkable `meta.checks`. To curate: set `meta.drop: true` (or delete). Target ~13 reference-based + ~7 reference-free. All context inline; no anchor passage reused (tested). **Curation is the user's job and is not done.**
- **`judge.checks`:** 14 deterministic check types (max/min words, max sentences, exact bullets/lines, regex, all_lines_match, one_of, json_keys, starts_with, all_lowercase, must_include/must_not_include). They test hard constraints only, not correctness; they will give by-construction instruction-following labels later.
- **`data/configs/{A,B}.yaml`:** same model and settings, only the generator preamble differs (A = v1 "helpful assistant"; B = v2 stricter: exact format, use only given info, do arithmetic first). The preamble is prepended to the case system_prompt for generation; the saved `Case.system_prompt` is the case's own, so the judge is held to the case requirements only.
- **`judge.generate` + `judge generate` / `make generate`:** one generator call per case, checkpointed (`data/outputs/.<cfg>.checkpoint.jsonl`, gitignored), cached, logged to `logs/generate_<cfg>.jsonl`; writes `data/outputs/<cfg>.json` (Case-shaped, `meta` has config, prompt_version, tokens, constraint results). Generator calls are not in the judge ledger.
- **Results (all 40 candidates, both configs; cached, so curating later costs no calls):** A 40/40 outputs, 0 empty, constraint-compliant 40/40. B 40/40 outputs, 0 empty, compliant 39/40 (c23: 4 bullets instead of 3). Output tokens 2466 (A) vs 2954 (B). Zero judge calls made.
- **Tests: 70 passed** (was 66): checks hand-computed, bad spec rejected, candidate-pool shape (40/27/13, anchors not leaked), generate prompt assembly + resume + file contents.

## Bugs and fixes / observations (raw material for 2.5)
- My first B preamble said "output only the final answer", which overrode c02's request for a sentence of working (B returned just "400" plus a parenthetical). Caught from the 5-line sample, reworded the preamble, regenerated B. Lesson: a generic preamble can silently conflict with case instructions.
- First check run falsely failed c31 ("350 ml") because gpt-oss emits U+202F (narrow no-break space) between number and unit (4 of 40 A outputs, 2 of 40 B). The check now normalises whitespace for `must_include`. Worth remembering for the judge's verbatim-quote check (lenient on whitespace already) and for any exact-match logic.
- Windows console (cp1252) crashed on those characters while printing samples; CLI now prints with `errors="replace"`.

## Caveats / open items (J2)
- **Constraint compliance is near ceiling for both configs (40/40 vs 39/40), so A vs B differ little on hard constraints.** Any A/B difference will come from correctness/faithfulness/completeness (reference-based cases) or style; the judge may find "no significant difference", which is a legitimate result. Not tuned post-hoc.
- Correctness against `expected_output` was not checked deterministically (only constraints); the judge and the ~20 hand labels do that.
- The curated suite (20) is not frozen yet; after curation, record the final ids here and commit `candidates.yaml`.
- Time spent: not recorded; fill in.

## J2 (curation) — locked suite, config C probe, decision rule
- **Interrupted work:** a previous request (j2b: temptation cases c41-c50 + identical A/B pair handling + budget notes) was interrupted by a shell error before anything was applied and then superseded by this request. It is NOT done: no c41-c50, no pair script, `budget.yaml` unchanged (still 214 planned). Re-request it if wanted.
- **Curation applied (user's list):** reference-based c01 c02 c03 c05 c09 c14 c15 c16 c19 c23 c24 c26 c27 (13); reference-free c28 c31 c32 c33 c38 c39 c40 (7). All other candidates have `meta.drop: true` in `candidates.yaml`. Probe bases (`meta.probe_base`): c01 c09 c16 c19 c24 c27. **Locked suite: `data/suite.yaml`** (20 cases). `generate` and `judge generate` now default to it; `data/outputs/{A,B}.json` were rewritten for the 20 cases (the other 20 candidates' outputs remain in the cache/checkpoint).
- **Config C = qwen/qwen3.8-27b, config A's prompt, no judge calls.** Findings: docs table 30 RPM / 1K RPD / 8K TPM / 200K TPD, but the live key also has **OTPM 1000/min** (max_tokens 1024 rejected; parallel requests hit 429, one run failed after 6 retries) -> C uses concurrency 1, max_tokens 600 (`GenConfig.concurrency/max_tokens/gen_model`). Thinking: no params = answers without thinking; `reasoning_effort=low` + `reasoning_format=hidden` drops it server-side (reasoning tokens still in `completion_tokens`); we also strip `<think>` client-side (tested). 0 `<think>` tags and 0 non-empty reasoning fields in C's outputs/logs.
- **Result (`judge genreport` -> `results/gen_check.json`):**

| config | constraint pass (20) | ref-based constraint (13) | exact-match (13) | mean out tokens |
|---|---|---|---|---|
| A gpt-oss-20b v1 | 20/20 (100%) | 13/13 | 6/13 (46%) | 64 |
| B gpt-oss-20b v2 | 19/20 (95%) | 12/13 | 6/13 (46%) | 76 |
| C qwen3.8-27b v1 | 19/20 (95%) | 12/13 | 6/13 (46%) | 225 |

  Both B and C fail c23 (4 bullets instead of 3). C - A = -5.0 pts constraint, +0.0 pts exact-match, below the 10-pt threshold -> **per the rule in CLAUDE.md section 2b the main A/B is prompt v1 vs v2 on gpt-oss-20b.** C is dropped from the main experiment.
- **Caveats:** the check set is coarse (N=20 / N=13; one case = 5 / 7.7 pts) and exact-match is deliberately strict, so many correct free-text answers count as misses (7 of 13 ref-based). "Either check" is my reading of "those checks" (documented). Qwen was run at temperature 0, below its recommended 0.7-1.0. C is much more verbose (225 vs 64 tokens, includes hidden reasoning tokens); that is itself relevant to the verbosity probe later. Both A and B sit at ceiling on hard constraints, so the A/B contrast will come from correctness/faithfulness judged by the judge.
- **Tests: 74 passed** (was 70): locked suite = curation, exact_match hand-computed, config C prompt == A, qwen request params + think-stripping.

## J2 (swap) — temptation cases c41-c44, recorded BEFORE any judge call
- **Design decision (user-proposed, pre-judging, not result-driven):** A and B sat at the ceiling on constraints (20/20 vs 19/20, exact-match equal), so a tie was likely with nothing for the faithfulness criterion to catch. Swapped out c03, c05, c26, c27 for four temptation cases: c41 (clinic notice, no phone number), c42 (museum notice, no ticket price), c43 (footbridge passage names only the street, asks the city and reopening date), c44 (library seats: 60 + 24 now, annex of 40 not yet open; answer 84). All reference-based, each with a machine-checkable constraint (max sentences; c41 / c42 also forbid invented numbers; c44 must contain 84). Suite stays 20 (13 ref / 7 free), judge-call budget unchanged at 214.
- **Probe bases:** c27 was one; **c44 replaces it** -> c01 c09 c16 c19 c24 c44 (my call, flagged to the user: a probe base must be reference-based and answerable; c44's terse-correct is "84", its confidently-wrong is e.g. "124").
- **Deliberate asymmetry:** the temptation prompts do not instruct "say not stated". Prompt v2's preamble ("if the answer is not given, say so") does, so this slice favours B by design. Documented in CLAUDE.md 2b and README.
- **Regenerated A, B, C for the 4 new cases** (generator calls only; 0 judge calls). Suite-level numbers: A constraint 20/20, B 19/20, C 19/20; exact-match 3/13 each (down from 6/13 because the swapped-out cases were the short exact-answer ones; the metric is strict). C - A = -5.0 / +0.0 pts -> rule outcome unchanged: **prompt v1 vs v2**.
- **Observation (honest): the temptation cases did not tempt in generation.** All three configs said the fact is not provided on c41-c43 (A even without being told to; A's c41 and c43 answers also add the available part, B's are terser) and all answered 84 on c44 (C showed the sum). So A and B do NOT differ on faithfulness in these 4 cases at the generation level; any A/B difference must come from the judge's finer distinctions (completeness, how the abstention is phrased). The expected v1-vs-v2 separation did not show up. Not tuned further; changing the cases now, after seeing outputs, would be result-driven.
- Tests: 74 passed (shape 44/31/13 pool, locked-suite probes c44, swapped ids).

## J3 — parse ladder, pointwise judge, replay
- **`judge.parse`** (explicit ladder, one function per rung): `strict` (whole reply is one object that validates) -> `extract` (strip `<think>` / gpt-oss harmony channels, ignore fences and prose, try every balanced `{...}` span string-aware, LAST valid span wins so a draft in the reasoning loses to the final answer) -> `repair` (cumulative named stages: smart_quotes, comments_trailing_commas, unquoted_keys, inner_quotes, control_chars, python_literals, truncation; syntax only, never invents a field) -> `reask` (ONE extra judge call quoting the validation error and the previous reply; its reply goes through rungs 1-3) -> `invalid`. Validation is the pydantic schema, so a score of 7 or a missing criterion is a parse failure. `ParseResult` carries status, attempts per rung, repairs applied, last error. `parse_with_ladder(raw, model, reask=cb)` is generic (pairwise reuses it in J5).
- **`judge.pointwise`**: `judge_case(client, case, label=, anchors=, nonce=)` -> `PointwiseResult` (parse_status, error_kind `""|parse|call`, repairs, scores, **overall recomputed from the weights**, pass flag, `unsupported_high_scores` = score >= 4 without a verbatim quote, per-call records incl. the re-ask). INVALID results are kept: `summarize_pointwise` counts them in n and in `parse_status`, reports means over scored cases, and gives the pass rate both over scored cases and with INVALID as fail. A judge call that fails after retries is `invalid` + `error_kind=call` and is NOT checkpointed (a resume retries it); an unparseable reply IS checkpointed (its replies are cached, a retry would repeat it). `judge_suite` is checkpointed (`.cache/checkpoints/pointwise_<label>.jsonl`, `--fresh` drops it) and writes `results/pointwise_<label>.json`; `checkpoint.run_checkpointed` gained a `persist` predicate for this.
- **Replay** (`judge.replay`, `judge replay --results results/pointwise_<label>.json`): `ReplayClient` serves raw responses from the JSONL call log keyed by (system, prompt, nonce); it has no HTTP client and no keys. `replay_suite` re-runs the SAME `judge_case` code over it (so parse ladder, re-ask and overall recomputation are replayed, not stored) and diffs parse_status / scores / overall / pass / repairs against the saved results. A changed prompt shows up as `no-log`, an edited verdict as a diff; exit code 1 on any difference.
- **Corpus:** `tests/corpus.py`, 46 replies, each pinned to the rung it must reach (6 strict, 12 extract incl. gpt-oss reasoning prose, `<think>`, harmony leak, draft-then-final; 12 repair; 16 that no local repair may rescue and must go to the re-ask).
- **Tests: 145 passed** (was 74): corpus parametrised per sample, rung order, error messages, re-ask called once and quotes the error, re-ask failure -> INVALID counted, summary hand-computed (overall mean 11/3, pass rates 1/3 and 1/4), flagged paraphrase evidence, suite replay of strict/extract/reask/invalid with 0 requests, tamper/no-log detection, call-failure resume, cache.
- CLI: `judge pointwise --config A [--ids ...] [--label L] [--experiment pointwise_mitigated] [--nonce N] [--fresh]`, `judge replay`. Both default to the lean budget accounting (`--experiment ''` for none).

### J3 live check (5 calls, label `live5`, config A, mitigated prompts, `JUDGE_JSON_MODE=0`)
| case | mode | parse_status | f/i/c | overall | pass |
|---|---|---|---|---|---|
| c01 | reference | strict | 5/5/4 | 4.75 | PASS |
| c41 | reference (temptation) | strict | 5/5/5 | 5.00 | PASS |
| c44 | reference (temptation) | strict | 5/5/4 | 4.75 | PASS |
| c28 | free | strict | 5/5/5 | 5.00 | PASS |
| c31 | free | strict | 5/5/5 | 5.00 | PASS |
- 5 network calls, 0 re-asks, 17,790 tokens in / 1,361 out (3.3-3.8K in per call because of the anchors; outputs 270-360 incl. thinking, far under `JUDGE_MAX_TOKENS`). Files: `results/pointwise_live5.json`, `logs/pointwise_live5.jsonl`. `judge replay` over it: 5/5 identical, 0 API calls.
- Checks from the J1 TODO: (a) valid JSON out of the box: yes, 5/5 strict with prompt-only JSON; (b) `unsupported_high_scores` empty for all 5 (every quote verbatim); (c) no anchor wording seen in the rationales I read (c01, c28, c31); (d) thinking tokens fit. No prompt change needed, `VERSIONS` unchanged, so these 5 are cache hits in the full run (they stay charged to `pointwise_mitigated`: 5/46 spent).
- **Not done:** the pairwise half of the J1 TODO (both orders) was not run; the pairwise verdict is not wired to the ladder yet (J5).

## Caveats / open items (J3)
- **The lower rungs were never exercised live**: Gemini returned bare JSON 5/5, so extract / repair / reask are proven only by the corpus and mock-transport tests. The gpt-oss "reasoning before the JSON" case is a synthetic sample, not an observed Gemini behaviour. The live strict rate says nothing about the other 15 cases.
- **Possible leniency**: 5/5 cases PASS, 17 of 20 criterion scores are 5 (mean overall 4.90), and the cases are the generator's easy ones. This is what a score-clustering/leniency problem would look like but N=5 cannot tell it from genuinely good outputs; J4/J7 probes (confidently-wrong, padded) are the real test. Not tuned.
- The judge counts words (c31 "44 words", c28 "13, 9, 14") from reading; counts were plausible here but LLM word counting is unreliable, and the deterministic checks in `meta.checks` are the ground truth for those constraints.
- Evidence rule: the prompt asks for a quote of at most 40 words; c01's quote is the whole 2-sentence answer and was accepted (the code only checks verbatim, not length).
- `repair` heuristics (inner quotes, unquoted keys) are tuned on the corpus, not on real judge failures; the status and the stages are logged per result so a repair-heavy run is visible in the summary.
- Time spent: not recorded; fill in.

## Standing instructions from the J3 review (user)
- Leniency flag (5/5 pass, 17/20 scores at 5) is NOT a finding. J4/J7 probes and J8 negative controls decide it. **If the probes also score near 5, investigate (prompt, anchors, evidence rule) before reporting any number.**
- The lower parse rungs are test-proven only; stated in README Known limitations. **Report the real INVALID count and re-ask count from the full pointwise run** (they are in `results/pointwise_<label>.json` -> `summary.parse_status`, `n_invalid`, `reasks`).


## J4 — pairwise order-swap, naive single-order baseline, position bias
- **Code:** `judge.pairwise` (`judge_order`, `judge_pair`, `judge_control`, `resolve`, summaries, checkpointed `judge_pairs`/`judge_controls`), CLI `judge position` (`make position`), results `results/position_bias.json`, log `logs/position_main.jsonl`. `build_reask_prompt` gained `kind="pairwise"` wording (the pointwise text asked for 1-5 scores). Smoke run (pair c01, 4 calls) moved to `results/smoke/`.
- **Rule:** mitigated = both orders, verdict mapped back to config A/B, disagreement -> TIE + `flipped`. Naive = verdict from the single order "A first"; the swapped order is judged only to measure the flip. Flip = the two orders name different underlying winners (decisive vs TIE counts); hard flip = opposite decisive winners. Naive pairs = first 12 suite-order pairs whose A and B texts differ (rule fixed in code before the run). Controls = the 6 probe bases, A's answer used as both answers, one order each.
- **Tests:** 168 passed (was 145; +23): resolve table, order mapping, fake always-A judge (mitigated: TIE + hard flip every pair; naive: A wins every pair, flip 100%; controls: first 6/6), hand-computed summary, re-ask wording, INVALID excluded from the flip denominator, checkpoint resume, selection rule.
- **Live results (judge gemini-3.5-flash-lite, JSON mode off; 0 INVALID, 0 re-asks; exploratory, N stated):**

| | N | flips | hard flips | flip rate | A / B / tie | win rate A | first-position share (decisive) |
|---|---|---|---|---|---|---|---|
| naive | 12 | 5 | 2 | 41.7% | 7 / 4 / 1 | 0.625 | 66.7% |
| mitigated, same 12 | 12 | 0 | 0 | 0% | 3 / 1 / 8 | 0.583 | 50% |
| mitigated, 14 differing | 14 | 0 | 0 | 0% | 3 / 1 / 10 | 0.571 | 50% |
| mitigated, all 20 | 20 | 0 | 0 | 0% | 3 / 1 / 16 | 0.55 | 50% |

  Controls (6 identical pairs, one order): naive 0 first / 0 second / 6 TIE; mitigated 0 / 0 / 6 TIE.
- **Mitigated decisive pairs (4):** c09 -> B, c23 -> A, c40 -> A, c41 -> A, each stable across both orders. The other 16 are TIE in both orders; 6 of those 16 are identical-text pairs (c14 c15 c16 c19 c24 c44).
- **Pair c28 (for the template screenshot):** naive picks the first-listed answer in both orders (A first -> A; B first -> B; verdict A, hard flip). Mitigated: TIE in both orders. Detail in `results/position_bias.json` -> `pairs.naive` / `pairs.mitigated`, case_id c28.
- **Caveat (open):** a judge that always answers TIE also has a 0% flip rate; 0% is meaningful only if the mitigated judge picks a winner when there is a real quality gap. Not yet tested. **Added to J6:** sensitivity check, 6 probe bases, correct answer vs its confidently-wrong version, both orders, mitigated + naive (24 calls, cap 28; budget.yaml `sensitivity`, planned total now 238, cap 275). Report share decided correctly and flip rate on those pairs; if the mitigated judge ties them too, state that the mitigation trades bias for indecision.
- **Budget spent:** pairwise_naive 24/28, pairwise_mitigated 34/46 (second order of the 6 identical pairs is a cache hit), controls 8/14 (4 mitigated controls duplicate identical-pair prompts and were cache hits).
- Template 2.5 reflection: left for the user.


## J5 — aggregate, compare, `run` / `compare` CLI, reports for A and B
- **`judge.aggregate`:** `bootstrap_ci` (percentile, 10,000 resamples, seed 5150, ends rounded to 6 places so float noise cannot fake a CI that excludes 0; None for N<2), `mean_ci`, `score_rows` / `aggregate_pointwise` (per-case pass/fail, pass rate with CI and with INVALID-as-fail, overall and per-criterion means with CIs, by-mode split), `Prices` + `usage_from_calls` (calls, network vs cached, re-asks, tokens, cost), `generator_usage`, `win_rate` (order-swapped pair verdicts, (wins + 0.5 ties)/valid pairs, bootstrap CI over per-pair scores). **Overall and pass/fail are recomputed from the stored scores with `rubric.yaml`** (pass rule 4 / 3 / 3.5 untouched, pinned by a test); a stored value that disagrees is listed in `stored_mismatch_cases`.
- **`judge.compare`:** pairs by case id (only cases scored under BOTH configs; dropped ids listed), differences are **A minus B**, win rate is **A over B**. Decision rule (docstring + `decide`): (1) 95% CI of the mean paired overall difference must exclude 0, else "no significant difference"; (2) the pairwise win rate must lean the same way (>0.5 for A, <0.5 for B; exactly 0.5 or the opposite lean -> no significant difference); (3) no pairwise results -> no winner. My reading of "win rate consistent" is a direction check; whether the win-rate CI excludes 0.5 is reported (`win_rate_ci_excludes_half`) but not part of the rule. Also reports per-criterion paired differences (no multiplicity correction; descriptive), paired pass counts (A-only / B-only), win rate over all 20 pairs and over the 14 differing pairs.
- **CLI:** `judge run [--config A B]` (= `pointwise` per config + `results/report_<cfg>.json`; an incomplete run writes no report), `judge compare --a A --b B` (offline, zero judge calls; writes `results/compare_A_vs_B.{json,md}`). `--price-in/--price-out` (USD per 1M tokens) turn on cost; **unset = cost null**, because the free-tier judge has no bill and I did not invent a list price. `make run`, `make compare`. `cmd_pointwise` was refactored into `_pointwise_core` (same behaviour).
- **Tests: 195 passed** (was 168; +27 in `tests/test_aggregate.py`): percentile and normal-approximation CI hand checks, float-noise guard, pass/fail by each threshold ((5,2,5)=3.95 fails on instruction-following, (4,3,2)=3.15 on overall, (3,5,5)=4.20 on faithfulness), thresholds read from the rubric object, tampered stored verdict flagged, cost arithmetic, win rate 0.625, every branch of the decision rule, A-minus-B sign convention, dropped-case handling, usage totals.
- **Live run (judge gemini-3.5-flash-lite, JSON mode off):** 29 new network calls, 0 INVALID, 0 re-asks (A: 20 strict; B: 19 strict + 1 repair `inner_quotes`, c33). pointwise_mitigated spent 34/46. `judge replay` over both: 20/20 identical, 0 API calls.

| | A (prompt v1) | B (prompt v2) | A - B paired, N=20 |
|---|---|---|---|
| mean overall | 4.610 [4.133, 4.947] | 4.612 [4.175, 4.938] | -0.003 [-0.362, +0.325] |
| faithfulness | 4.60 [4.00, 5.00] | 4.50 [3.90, 5.00] | +0.10 [-0.50, +0.70] |
| instruction-following | 4.70 [4.30, 5.00] | 4.75 [4.35, 5.00] | -0.05 [-0.35, +0.25] |
| completeness | 4.50 [3.90, 4.95] | 4.60 [4.10, 4.95] | -0.10 [-0.60, +0.30] |
| pass rate | 18/20 = 0.90 [0.75, 1.00] | 17/20 = 0.85 [0.70, 1.00] | +0.05 [-0.10, +0.20] |

  Win rate A over B (mitigated, order-swapped, N=20): 0.550 [0.450, 0.650], A 3 / B 1 / tie 16, flip rate 0 (14 differing pairs: 0.571 [0.429, 0.714]). **Decision: no significant difference** (CI of the overall difference spans 0). Judge usage for the comparison (pointwise A+B plus the J4 pairwise calls): 80 calls, 200,400 tokens in / 26,185 out, cost not priced.
- **What drives it:** 14 of 20 cases have identical overall under A and B. The six that differ: c02 -0.70, c09 -2.60 (A fails faithfulness 1), c23 +1.50 (B's 4 bullets instead of 3), c39 -0.35, c40 +1.60 (B faithfulness 1), c41 +0.50. They cancel (mean -0.003). c09/c23/c40/c41 agree in direction with the pairwise winners from J4.

## Caveats / open items (J5)
- **A tie, not equality:** N=20 and a mean overall CI of about +-0.34 can only detect a large difference; per-case effects are large and in both directions (-2.6 to +1.6), so it is mostly two or three cases deciding the sign. The temptation cases (c41-c44) did not separate the configs in generation or judging, as recorded in J2.
- **Judge leniency still unresolved:** 14 of 20 cases under A and 14 of 20 under B score exactly 5/5/5; the J3 standing instruction (investigate if probes also score near 5) stays open for J6.
- **Identical-text cases are not independent evidence:** 6 pairs (c14 c15 c16 c19 c24 c44) have the same answer under A and B; the judge prompt is identical, so the second pointwise call is a cache hit with the same verdict (difference exactly 0 by construction). They still count in N=20 and shrink the CI width; the 14-pair numbers are in the compare JSON.
- Pairwise numbers come from the J4 run (`results/position_bias.json`); `compare` does not run judge calls.
- Pass-rate CI is a bootstrap on 20 binary values (ends can hit 1.0); no Wilson interval.
- Time spent: not recorded; fill in.

## J5b — recorded price, six-case review, first repair-rung event (user follow-up)
- **Price recorded from the official page, not memory.** Fetched https://ai.google.dev/gemini-api/docs/pricing on 2026-10-23 (live fetch, `maxAge 0`). Stored in `src/judge/pricing.yaml` with URL, retrieval date and tier: **gemini-3.5-flash-lite Standard paid tier $0.30 per 1M input, $2.50 per 1M output (output includes thinking tokens, which our `tokens_out` counts); Free Tier column "Free of charge".** Caveat: the scraped page had four tables for this model (tabs Standard / Batch / Flex / Priority) with no tab labels in the text; I mapped by order (Standard 0.30/2.50, Batch and Flex 0.15/1.25, Priority 0.54/4.50). Standard = 2x Batch matches Google's usual batch discount and is the synchronous tier this project uses, but the mapping is inferred.
- **Cost row:** usage now has `cost_usd_actual` (0.0 on the free tier), `cost_usd_list_equivalent` (network tokens at list price) and `cost_usd_if_uncached` (every call). `compare`/`report` print both and the source. A vs B comparison (80 judge calls: pointwise A+B + the J4 pairwise): **$0.00 actual, $0.0964 list-price equivalent** (148,833 network tokens in, 20,711 out). `--price-in/--price-out` override the file. New `judge run --report-only` rebuilds the reports from saved results with no judge call (a plain re-run is refused by the budget pre-check, which counts planned calls, not cache hits). Tests: 197 passed (+2: free-tier actual vs equivalent, price file has source and date).
- **Six-case review:** `results/evidence/six_case_review.md` (c02 c09 c23 c39 c40 c41). Summary: direction right in 5 of 6; all 12 evidence quotes verbatim. Two judge errors: **c23** B's "2 March: Applications open" charged to both faithfulness (3) and instruction-following (3), which flips B's pass to fail (faithfulness is supported; only scope is wrong), and **c39** the judge counted A as 42 words (actual 39 by whitespace, 41 counting dash-joined pairs) and docked instruction-following to 4. Correcting both by hand would give a mean paired difference of about -0.03 and B 18/20 passes: still a tie. Nothing was changed or re-scored; these are observations for J8 and the write-up (judge misapplies "score criteria separately" once; LLM word counting unreliable). Also found: scratch reasoning ("Wait, let's count ...") inside `rationale` in 5 verdicts (4 distinct replies).
- **First real repair-rung event kept as evidence for template 2.4:** `results/evidence/repair_event_c33_B.json` has the raw judge reply, the strict parse error (`Expecting ',' delimiter`, char 844), the text around it, the ladder trace (strict fail, extract fail, repair `inner_quotes` ok) and the repaired verdict (5/5/5). Cause: the judge wrote its word-count self-check ("Wait, 53 words ... apologise once (met: "I'm sorry"), no "unfortunately" ...") inside the `rationale` string with unescaped double quotes. The extract rung also failed (the first balanced object it found was a nested criterion, missing required fields), so repair was the rung that rescued it. This is the only repair event in 40 verdicts; 0 re-asks, 0 INVALID. The repaired score is a faithful reading of the reply (syntax-only repair, nothing invented).


## J6 — adversarial probes (verbosity + sycophancy), `src/judge/probes.py`
- **Approved before building (user):** 6 bases (c01 c09 c16 c19 c24 c44), one false fact per base, fixed wrapper alternating "Definitely:" / "Without question,", 3 fixed content-free filler sentences, terse-correct = minimal core, verbose-wrong = wrong + same filler. Verbosity is measured on faithfulness + completeness only (`overall_noif` = their renormalised weighted mean); instruction-following is reported separately because padding breaks the cases' own length limits. Added by the user: the 6 wrong answers are also judged reference-free (expected_output hidden), naive and mitigated (+12 calls, `probes_ref_free` in budget.yaml; planned total 250, cap 289; CLAUDE.md section 6 updated).
- **Code:** builders (`build_padded/wrong/terse/verbose_wrong`), `judge_naive_case` (no naive pointwise runner existed; re-ask kind `naive_pointwise` added to parse.py), checkpointed `run_stage`, `analyze` (pure, from saved results), `judge probes` / `make probes`, `results/probes.json`, logs `logs/probes_{ref,free}_main.jsonl`. Pre-registered in code before the run: naive "fooled" threshold = score >= 8 of 10 (the same 80% of scale as faithfulness >= 4 of 5); fooled (b) compares overall_noif (mitigated) / the naive score; the with-IF overall is also given; sample verdict = mitigated reference-mode wrong answer with the highest faithfulness.
- **Tests: 235 passed** (+37 in tests/test_probes.py): each builder produces its intended defect (padded keeps the correct facts, adds >= 30 words, no digits, breaks the case's length limit; wrong states exactly the approved false fact, no true one, respects the length limits so IF is not the defect; terse is correct and <= 60% of the reference; verbose-wrong = wrong + same filler), reference hidden from every free-mode prompt, hand-computed overall_noif and Spearman (ties), fake-judge end-to-end for deltas / fooled counts / invalid rows, naive re-ask wording.
- **Live run (judge gemini-3.5-flash-lite, JSON mode off; N=6 bases, EXPLORATORY):** 66 network calls (probes 54/62, probes_ref_free 12/14), 0 INVALID, 0 re-asks; 6 mitigated originals were cache hits (pointwise_A prompts). Parse status: 36 strict, 36 extract (fenced or wrapped replies).

| variant (reference mode) | naive (1-10) | faith | IF | compl | noIF | overall | pass | words |
|---|---|---|---|---|---|---|---|---|
| original (A's answer) | 6.83 | 3.67 | 4.50 | 3.33 | 3.54 | 3.88 | 4/6 | 11 |
| terse-correct | 9.00 | 5.00 | 4.67 | 4.83 | 4.94 | 4.84 | 6/6 | 3 |
| padded | 3.00 | 5.00 | 1.67 | 5.00 | 5.00 | 3.83 | 0/6 | 69 |
| confidently wrong | 2.17 | 1.17 | 4.00 | 1.67 | 1.36 | 2.28 | 0/6 | 19 |
| verbose wrong | 1.00 | 1.17 | 1.50 | 2.00 | 1.49 | 1.49 | 0/6 | 72 |

  - **Verbosity, padded minus terse-correct:** mitigated faithfulness 0.00, completeness +0.17, noIF +0.06 (IF -3.00, separate, expected); naive -6.00. Against the original, only where the original is correct (c01 c19 c24 c44; c09 and c16 originals are wrong): mitigated noIF +0.19, naive -7.25. Spearman(words, score), pooled over bases: mitigated correct group +0.44 / wrong group +0.16; naive -0.74 / -0.31 (n=12 each; pooled across bases, so confounded by the case).
  - **Fooled rate (naive -> mitigated):** reference mode (a) wrong answer accepted 0/6 -> 0/6, (b) verbose-wrong >= terse-correct 0/6 -> 0/6; reference-free (a) 0/6 -> 0/6 (mitigated faithfulness 1 on all six, naive scores 1-2).
- **Reading (not over-claimed):** (1) **No verbosity bias detected in either judge at N=6.** The mitigated judge's faithfulness and completeness do not move with padding. The naive judge scores the padded answer far LOWER (3.0 vs 9.0) because its prompt shows the system prompt and the padded answer breaks it; its single holistic number cannot separate that from length preference, so the naive before-number is a length-limit penalty, not evidence of an anti-verbosity judge. (2) **No sycophancy detected, and no before/after improvement is claimable:** the naive judge is already at 0/6, so there is nothing for the mitigation to reduce. The probe was not discriminative: one false fact is caught whenever the reference is shown, and reference-free the judge still caught all six from the passage in `input`. (3) Naive c01 gave the confidently-wrong answer 7/10 (partial credit for the right fine); the mitigated judge gave faithfulness 2. (4) The standing J3 leniency question is answered for this slice: the mitigated judge is not lenient on wrong answers (mean overall 2.28, 0/6 pass, 0 unsupported high scores). It is still not shown that 5/5/5 on correct answers is calibrated.
- **Sample verdict (rule: highest faithfulness among mitigated reference-mode wrong answers):** c01, faithfulness 2, evidence "A book can be renewed up to 3 times" (verbatim), rationale cites the 2-times limit. Full verdict in `results/probes.json` -> `sample_confidently_wrong_verdict`.
- **Caveats / open:** the unpadded control (the reference text judged on its own) was not run (+12 calls, over the probes cap); terse-correct vs padded is the length-only comparison, and terse can legitimately lose completeness (c01, c09). Wrong answers are single, easily checked facts; harder traps (plausible near-misses, wrong reasoning with correct final number) would be needed to find sycophancy, and the sensitivity check (correct vs wrong pairwise, 24 calls) is still to run. Time spent: not recorded.


## J6b — sensitivity check and hard probes (frozen prompts and rubric; judge gemini-3.5-flash-lite, JSON mode off; EXPLORATORY, N=6)
- **Code:** `probes.sensitivity_cases/sensitivity_summary`, `probes.HARD/build_hard/analyze_hard`, CLI `judge sensitivity` (-> `results/sensitivity.json`, log `logs/sensitivity_main.jsonl`) and `judge hard` (-> `results/probes_hard.json`, log `logs/probes_hard_main.jsonl`). `budget.yaml` gained `probes_hard` (12 calls, cap 14); planned total 262. Tests: 244 passed (+9 in tests/test_probes.py: sensitivity summary hand-computed, hard texts keep the key fact, comply with every case constraint, add no new number, keep the reference qualifiers, stage plan and analysis).
- **Sensitivity (24 network calls of cap 28, 0 INVALID, 0 re-asks):** A = the expected_output text (correct), B = the confidently-wrong answer from J6, both orders, reference shown in the prompt (the correct answer is identical to the reference).

| judge | rule | N | correct | tied | wrong | flips |
|---|---|---|---|---|---|---|
| mitigated pairwise | both orders, order-swap rule | 6 | 6 | 0 | 0 | 0/6 |
| naive pairwise | both orders, same rule | 6 | 6 | 0 | 0 | 0/6 |
| naive pairwise | its own verdict (correct answer first) | 6 | 6 | 0 | 0 | - |

  The J4 open caveat (a judge that always says TIE has a 0% flip rate) is answered for the mitigated judge: it picked the correct answer in all 6 pairs in both orders. The naive judge did the same, so this check does not separate the two judges.
- **Hard probes: right final answer + invented justification** (texts approved by the user, in `results/probes_hard.json` -> `texts`; each passes its case constraints in `checks.py`: c01 1/2 sentences, c09 17/20 words, c16 1/2, c19 1/1, c24 2/2 and no "very", c44 1/2 and includes "84"). Reference mode, pointwise, 12 network calls of cap 14, 0 INVALID, 0 re-asks (parse: 6 strict, 6 extract).
  - **Accepted rate:** mitigated (faithfulness >= 4) **0/6**; naive (score >= 8 of 10) **0/6**.
  - **Mitigated per-criterion means (1-5):** faithfulness 2.33 (c01 3, c09 3, c16 3, c19 2, c24 2, c44 1), instruction-following 3.00 (3 on all six), completeness 5.00 (5 on all six); mean overall 3.23, mean overall without IF 3.36, pass rate 0/6.
  - **Naive scores (1-10):** c01 7, c09 7, c16 1, c19 1, c24 3, c44 2; mean 3.5.
  - **Judge errors seen in the rationales:** instruction-following was 3 on all six; the c01 rationale says the answer "is written as a single long sentence" and violates the two-sentence limit (it is 1 sentence, limit 2); the c44 rationale lowers instruction-following for the fabricated detail, which is a faithfulness matter. Not corrected or re-scored. Faithfulness evidence quotes were the invented clauses (e.g. c44 "...since the annex is delayed by a contractor dispute.").
  - **Reading:** both judges rejected all six fabricated-reasoning answers, so neither probe round (J6, J6b) found an answer the mitigated judge accepts wrongly, and the two judges have not been separated on sycophancy. Faithfulness 3 on c01, c09, c16 means the mitigated judge partially credits an unsupported clause attached to a correct answer.



## J7 — family guard test, self-preference probe, score clustering (judge gemini-3.5-flash-lite, JSON mode off; EXPLORATORY, N stated)
- **Code:** `judge.selfpref` (deterministic fact checks `FACTS`, `select_pairs`, one-off Gemini generator settings, `clopper_pearson`, summary), `judge.clustering` (`spread`, `auc`, `separation`, `run_naive_suite`, `analyze`), CLI `judge selfpref` / `judge clustering` (`make selfpref`, `make clustering`), `judge_pairs` gained `x_label`/`y_label`. Results: `results/self_preference.json`, `results/clustering.json`; Gemini answers in `data/outputs/selfpref_gemini.json`; logs `logs/selfpref_*.jsonl`, `logs/clustering_main.jsonl`. **Tests: 271 passed** (+27: guard, Clopper-Pearson hand values, fact check, selection, summary, spread/AUC hand-computed, fake-judge end to end).
- **Family guard (tests, no live call):** the project pair passes (generator openai, judge google); the Gemini-authoring client built from swapped settings passes (google vs openai) with `ALLOW_SAME_FAMILY` off; a Gemini or Gemma generator with the Gemini judge, `gpt-oss-120b` as judge, and an unknown family are all refused; the guard runs at `LLMClient` construction; the swapped generator client has no ledger or limiter. This is structural only: it keeps same-family grading out of the main A/B and measures nothing.

### Self-preference (6 pairs, mitigated pairwise, both orders; 13 network judge calls of cap 14: 12 + 1 re-ask)
- **Fixed before generating:** 13 reference-based cases; gpt-oss answer = config A output; Gemini answer = `gemini-3.5-flash-lite` as generator with the same config-A system prompt (11 generator calls, 1,018 in / 264 out, same Gemini key but outside the judge ledger). Correct = deterministic: required/forbidden regexes written from the reference (`FACTS`) plus the case's own constraint checks. Keep pairs where both are correct and the texts differ; take the first 6 in suite order. Per-case audit is in the JSON (`eligibility`): c09 and c16 dropped (gpt-oss answer wrong), c14 and c15 dropped (identical text), c42 c43 c44 eligible but beyond the first 6.
- **Result (N=6 pairs: c01 c02 c19 c23 c24 c41):**

| Gemini-authored wins | gpt-oss wins | ties | flips | own-family win rate (wins + 0.5 ties) / N | decisive n | 95% CI on the decisive share (Clopper-Pearson) |
|---|---|---|---|---|---|---|
| 3 | 0 | 3 | 0 | 0.75 | 3 (Gemini 3/3) | [0.292, 1.000] |

  Single orders: 6 Gemini, 0 gpt-oss, 6 tie of 12. Every decisive pair was stable across both orders. 0 INVALID, 1 re-ask.
- **Honest reading:** the point estimate is above 0.5, but the interval on 3 decisive pairs runs from 0.29 to 1.0, so this can show a large effect at most and cannot show absence of one. **It also does not demonstrate self-preference**, because each of the 3 decisive wins has a content-level reason in the judge's own rationale that I checked against the texts (post-hoc review, not a pre-registered check): **c01** Gemini kept the reference's "unless reserved" qualifier and gpt-oss dropped it (judge: more complete; my `FACTS` for c01 did not require the qualifier, so "both correct" was weaker than "equally good"); **c02** Gemini wrote one sentence of working after "400", gpt-oss put the working in parentheses (judge: instruction-following); **c23** Gemini copied the reference wording, gpt-oss appended "due" / "submission deadline" (judge: instruction-following). Authorship and answer quality are confounded in these pairs; the 3 ties (c19 c24 c41) are pairs the judge calls equal. Length does not explain it: Gemini was longer in 5 of 6 pairs but shorter in a decisive one (c23, 14 vs 18 words) and tied on 3.
- **Not measured:** the naive judge (the locked budget is 12 mitigated calls; a naive row costs 12 more, so rule A6 "report both" is not met for this row), a gpt-oss judge cross-check (no budget), and the pairs excluded by the correctness filter.

### Score clustering (naive 1-10 vs anchored per-criterion 1-5) and good-vs-corrupted separation
- **Design (fixed in code before the run):** 12 cases = the first 12 of config A in suite order (c01 c02 c09 c14 c15 c16 c19 c23 c24 c28 c31 c32); naive pointwise on them: 12 calls, 7 network (5 probe bases were already naive-judged in J6 and hit the cache), 0 re-asks, parse status `extract` on all 12. Mitigated side = the existing A verdicts on the same cases (no new calls). Separation reuses J6/J6b verdicts, no new calls: good = terse-correct (6); corrupted = confidently wrong (6), verbose-wrong (6), fabricated reasoning (6), all by construction. On the 12 suite cases a deterministic label is used (defective: c09, c16, where gpt-oss answered wrongly).
- **Spread (same 12 cases):**

| score | n | mean | sd | sd / scale width | distinct values | modal share | at scale max |
|---|---|---|---|---|---|---|---|
| naive holistic 1-10 | 12 | 8.33 | 3.30 | 0.367 | 3 (1, 9, 10) | 67% (10) | 67% |
| mitigated overall 1-5 | 12 | 4.40 | 1.17 | 0.291 | 5 | 67% (5.0) | 67% |
| mitigated faithfulness 1-5 | 12 | 4.33 | 1.49 | 0.373 | 2 (1, 5) | 83% | 83% |
| mitigated overall, all 40 verdicts (A+B) | 40 | 4.61 | 0.91 | 0.228 | 9 | 70% | 70% |

- **Separation, terse-correct vs corrupted, AUC (mean gap / scale width), N=6 per group:**

| judge | confidently wrong | verbose wrong | fabricated reasoning | all corrupted (n=18) |
|---|---|---|---|---|
| naive 1-10 | 0.972 (0.76) | 1.000 (0.89) | 0.944 (0.61) | 0.972 (0.75) |
| mitigated 1-5 | 1.000 (0.64) | 1.000 (0.84) | 1.000 (0.40) | 1.000 (0.63) |

  Suite A (10 good vs 2 defective): both judges AUC 1.0 (naive means 9.8 vs 1.0, mitigated overall 4.91 vs 1.88).
- **Honest reading:** **no clustering improvement is demonstrated, and none should be claimed.** On these cases both judges pile up at the top (8 of 12 at the maximum: naive 10, mitigated overall 5.0) because 10 of the 12 answers are good, so the spread is set by the good/bad mix rather than by the judge. The naive judge is not narrow here (sd / width 0.37) but bimodal (1, 9, 10 only). The mitigated overall has 5 distinct values vs 3, but it is a weighted mean of three integers, which flatters that count; the like-for-like row (faithfulness) uses 2 values with 83% at 5, so it is more clustered than the naive score. Where the mitigated judge does add resolution is among good answers (c02 4.30 for an instruction-following 3, c01 4.75 for completeness 4, where naive gave 10 and 9): a per-criterion effect, with the ceiling at 5 intact. Separation: the mitigated judge separates perfectly on all three corruption types with a margin (lowest terse-correct 4.30 vs highest corrupted 3.50); the naive judge is near-perfect and its misses are ties at 7 (terse c01/c09 got 7, and some corrupted c01/c09 answers got 7). One or two pairs at N=6 are not evidence that mitigated separates better.
- **Caveats:** the two judges differ in prompt, scale, criteria and anchors at once, so a spread difference cannot be attributed to anchors (no anchors-off ablation, no budget). The 12-case sample is a bimodal mix with two wrong answers; measuring clustering proper needs a sample of near-equal answers. Prompts were frozen at J1 and not changed after seeing the probe texts.
- **Budget:** self_preference 13/14, clustering_naive 7/14. Time spent: not recorded.

### J7 template entries (agreed in review)
- **Self-preference / family guard:** "family guard in code; exploratory probe inconclusive (N=6, CI [0.29, 1.00])". The naive self-preference row is deliberately NOT run (no budget; would not change the conclusion). The c01 caveat above (fact check weaker than "equally good") stays with the result.
- **Score clustering:** "no improvement". Both judges pile up at the top because most answers are good, the faithfulness row is more clustered than the naive score, and the judges differ in prompt, scale, criteria and anchors at once, so no gap is attributed to the anchors.



## J8 - judge validation (`src/judge/validate.py`; judge gemini-3.5-flash-lite, JSON mode off; EXPLORATORY, N in every row)
- **Code:** `judge validate` (offline, zero judge calls; writes `results/validation.{json,md}`), `judge retest` (live, budget `test_retest`; writes `results/test_retest.json`, `results/pointwise_retest{1,2}.json`, `logs/retest_main.jsonl`), `make validate` / `make retest`. Kappa is my own implementation (integer arithmetic; undefined = None with a reason when chance agreement is 1). **Tests: 296 passed** (+25 in tests/test_validate.py): kappa hand-computed (50-item example = 0.4, perfect/-1/0/undefined, 3-class = 0.25), confusion and rates, label file parsing and errors, retest sd hand-computed (0.2309, 0.4619), nonce cache-bypass through a real client with a mock transport, adversarial table recomputation. Mutation check: breaking the kappa denominator fails 5 tests.
- **Human labels (N=20, config-A answers; 17 pass / 3 fail: c01, c09, c16).** `labels/human_labels.jsonl` was filled in outside the tooling. Its first version (commit 26ac903) equalled an LLM-rater draft (commit 567bfd8, removed later) on 20/20; the author then revised it after re-verifying (reported by the author, not verifiable here) and the current version differs from that draft on c01 only (pass -> fail: the answer drops the 'unless reserved' qualifier). c09 and c16 are NOT blind labels (the errors were known from earlier reports). Results against the revised labels: mitigated judge agreement 19/20 = 0.95 vs always-pass 0.85, kappa 0.77, bootstrap 95% [0.00, 1.00]; naive judge (13 items) 12/13, kappa 0.75 [0.00, 1.00]; by-construction labels vs human 19/20, kappa 0.77. The single disagreement is c01 for the judge, the naive judge and the by-construction label alike: the judge scored completeness 4 and passed it, the author fails it for the missing qualifier (the by-construction fact check never required it). Negative controls pass (shuffled-label mean kappa 0.003, permutation p 0.017). N=20 with 3 negatives: exploratory, reported as the secondary row, not the headline.
- **Headline agreement (by-construction labels, suite distinct, N=34; 30 label-pass, 4 label-fail):** agreement 0.94 (32/34) vs always-pass baseline 0.88; kappa 0.72, bootstrap 95% interval [0.00, 1.00]. Both numbers go together; the interval includes 0.
- **Per-criterion rows vs deterministic labels (mitigated judge):** *instruction-following vs checks.py over the 40 suite outputs:* agreement 37/40 = 0.93 vs always-pass 0.97, kappa -0.03; only 1 of 40 outputs violates a countable constraint (c23 under B, 4 bullets), the judge gave it IF 3 which still passes (false pass 1/1). Judge IF < 3 on c16 (A and B) while checks pass: the checks do not cover the 'abstain only if not stated' rule the judge applied, so that is a label limit, not a counting error. Stricter reading (IF >= 4): 37/40, kappa 0.38 [0.00, 1.00], extra misses c02 A (IF 3), c16 A/B. No sentence/word-count error shows up in this slice (the c39 word-count slip from J5b is not a disagreement here because the answer is within its limit either way); the evidence on countable-constraint errors is thin (N=1 negative). *Faithfulness vs support label* (reference facts present and no contradicting fact; reference-based outputs only, N=26, 20 distinct): agreement 24/26 = 0.92 vs 0.85, kappa 0.70 [0.00, 1.00]; misses c23 B (judge F 3, scope charge; label supported) and c41 B (judge F 5, label unsupported because the fact check wants the online-booking detail). The support label also encodes correctness/completeness, so it is coarser than 'every claim is supported'; the label definition is my choice, flagged for review.
- **By-construction labels (deterministic, judge not involved):** suite answers = `clustering.suite_label` (reference facts + the case's constraints; reference-free cases = constraints only); probes = states the true fact, not the false one, within the case's length limits (so padded is a fail by its length defect); fabricated-reasoning answers = fail by construction. Naive judge "pass" = score >= 8 (the J6 acceptance threshold; it has no pass rule of its own).

| stratum | judge | N | agreement (always-pass baseline) | kappa [95% bootstrap] | false pass | false fail |
|---|---|---|---|---|---|---|
| suite A | mitigated | 20 | 1.00 (0.90) | 1.00 (CI degenerate) | 0/2 | 0/18 |
| suite B | mitigated | 20 | 0.90 (0.85) | 0.61 [-0.07, 1.00] | 1/3 | 1/17 |
| suite distinct (A+B deduped) | mitigated | 34 | 0.94 (0.88) | 0.72 [0.00, 1.00] | 1/4 | 1/30 |
| suite distinct | naive | 13 | 1.00 (0.85) | 1.00 (CI degenerate) | 0/2 | 0/11 |
| probes (designed extremes) | mitigated | 36 | 1.00 (0.17) | 1.00 (CI degenerate) | 0/30 | 0/6 |
| probes | naive | 36 | 0.94 (0.17) | 0.77 [0.30, 1.00] | 0/30 | 2/6 |
| all distinct | mitigated | 70 | 0.97 (0.51) | 0.94 [0.85, 1.00] | 1/34 | 1/36 |
| all distinct | naive | 49 | 0.96 (0.35) | 0.91 [0.76, 1.00] | 0/32 | 2/17 |

  - **Headline is suite distinct (N=34: 30 label-pass, 4 label-fail): mitigated kappa 0.72, interval lower bound 0.00.** The probes stratum is constructed (6 correct, 30 wrong/defective) so its kappa measures separation of designed corruptions, not agreement on natural answers. Strata share items and are not independent.
  - **The two mitigated disagreements are both config B and both look like label limits, not judge errors** (my reading, post-hoc, from the saved scores and answers): **c40** (reference-free) label = constraints met, judge faithfulness 1 (its rationale: B invents troubleshooting steps, auto-save and "File > Recover Unsaved Documents", that the customer message does not support); constraint-only labels cannot see faithfulness. **c41** label = fail because the fact check requires the "online booking" detail (B's answer only says no phone number), judge = pass with faithfulness 5, completeness 3; the pass rule has no completeness threshold. Not re-labelled. Naive misses are two terse-correct answers scored 7 (c16, c19) under the >= 8 threshold.
  - **Negative controls (per stratum, mitigated, all PASS):** always-pass and always-fail judges have kappa exactly 0 (while their raw agreement equals the base rate, e.g. 0.88 on suite distinct, which is why agreement alone is not reported as a result); the inverted judge scores lower (suite distinct -0.19); shuffled labels x10,000 give mean kappa 0.001 (tolerance +-0.10, set before running) and permutation p of kappa >= observed: suite A 0.0045, suite B 0.045, suite distinct 0.003, probes 0.0001.
- **Test-retest (live, 20 network calls of cap 23, 0 INVALID, 0 re-asks; N=10 cases, every second case in suite order, rule fixed in code):** run 1 = the saved config-A run (nonce ""), runs 2-3 = nonce `retest1`, `retest2` (cache bypassed; controls: 0 cached calls in both nonce runs, nonces distinct). Pass/fail flips **0/10** (95% exact CI [0.00, 0.31]). Overall sd per case (sample, n-1): mean 0.095, max 0.404; 4 of 10 cases changed overall at least once (c01 completeness 4/3/4, c09 IF 5/5/4, c33 IF 5/5/3, c39 IF 4/3/3). Criterion scores identical across all three runs: faithfulness 10/10, completeness 9/10, instruction-following 7/10. Pairwise run agreement on pass/fail 1.00 (n=10) for all three pairs; mean |d overall| 0.06 / 0.14 / 0.13.
  - **Reading:** at temperature 0 the verdicts are stable on pass/fail in this sample, and the noise is in instruction-following (the criterion where the judge counts words/sentences). The zero flips are partly luck of the subset: c33 and c39 sit on the IF = 3 boundary (still a pass), so one more point of IF noise would flip them. N=10 cannot show that flips are rare, only that none were seen.
  - **Deviation to flag:** the request said "rerun the suite 3 times"; `budget.yaml` (test_retest = 10 cases x 2 reruns, cap 23) plans 3 runs per case in total, counting the saved run. I followed the budget: 3 runs of 10 cases, not 3 nonce reruns of all 20 (60 calls). `judge retest --reruns 3` on 10 cases is refused by the budget check unless `budget.yaml` is raised.
- **Adversarial summary (from probes.json, probes_hard.json, sensitivity.json; N=6 bases per row; counts and deltas are recomputed from the raw per-item verdicts and cross-checked against the stored summaries: 0 mismatches; the sensitivity row is taken from its stored summary):**

| probe | metric | naive | mitigated |
|---|---|---|---|
| confidently wrong, reference shown | accepted (fooled) | 0/6 | 0/6 |
| confidently wrong, reference hidden | accepted (fooled) | 0/6 | 0/6 |
| verbose wrong vs terse correct | verbose-wrong scored >= terse-correct | 0/6 | 0/6 |
| right answer + fabricated reasoning | accepted (fooled) | 0/6 | 0/6 |
| padded minus terse-correct | mean score delta | -6.00 (naive 1-10) | +0.06 (faithfulness+completeness 1-5) |
| correct vs wrong, pairwise, both orders | picked the correct answer | 6/6 | 6/6 |

  - **Negative control that matters here: terse-correct answers accepted** naive **4/6**, mitigated **6/6** (a judge that rejects everything also has 0 fooled). So the naive judge's 0/6 is partly strictness (it rejects two correct terse answers); the mitigated judge's 0/6 comes with full acceptance of correct answers. Discrimination (correct accepted rate minus fooled rate): naive 0.67, mitigated 1.00. Trivial baselines: always-accept = fooled 1.0, always-reject = correct accepted 0.0. The table also detects a planted failure (a test sets one wrong answer's faithfulness to 5; the fooled count and the stored-vs-recomputed check both move).
  - **No before/after improvement can be claimed from the adversarial probes** (naive already 0/6); the naive padded delta is a length-limit penalty, not an anti-verbosity judge (see J6).
- **Caveats (also printed in `results/validation.md`):** reference-free labels are constraint-only; reference-based labels include completeness facts the pass rule ignores; A and B share 6 identical answers; bootstrap CIs collapse under perfect agreement; positives are few so one item moves kappa a lot. All numbers exploratory.
- **Budget:** test_retest 20/23. Time spent: not recorded. Reflection sections left for the user.

### Template table 2.3.3 rows (four)
| row | value |
|---|---|
| Agreement and kappa vs by-construction gold labels (suite distinct, N=34) | agreement 0.94 vs always-pass 0.88; kappa 0.72, bootstrap 95% [0.00, 1.00]; exploratory |
| Agreement and kappa vs human labels (N=20; c09, c16 not blind; author-revised) | agreement 0.95 vs always-pass 0.85; kappa 0.77, bootstrap 95% [0.00, 1.00]; one disagreement (c01, missing 'unless reserved' qualifier); exploratory |
| Test-retest, judge temperature 0.0, N=10 cases x 3 runs (2 nonce reruns) | 0/10 pass/fail flips (95% exact CI up to 0.31); noise in instruction-following; c33 and c39 sit on the IF = 3 boundary, so the zero is partly luck |
| Adversarial probes, N=6 bases | fooled 0/6 for naive and mitigated (floor effect, no before/after claim); terse-correct accepted naive 4/6, mitigated 6/6 |


## J9 - README, generated tables, clean-clone test, answers draft
- **`scripts/make_report_tables.py`** builds every result table in README.md (between the GENERATED markers) from `results/*.json` only; a missing key raises. `--check` exits 1 if the README is stale, `--stdout` prints. README also has a Mermaid pipeline diagram (not rendered here; GitHub renders it), a "when each judging mode fits" table with the project's own evidence per mode, a bias-at-a-glance table, setup and offline/live command lists.
- **Clean-clone test** (fresh `git clone`, new venv, `pip install -e ".[dev]"`, README commands in order): 299 tests pass; `budget`, `genreport`, `run --report-only`, `compare`, `validate`, both `replay` (20/20 identical, 0 API calls) and the table generator all exit 0 with no `.env`; regenerating `results/` left `git status` empty, so the committed results reproduce byte-for-byte from the committed data. Live commands were not run (no keys in the clone).
- **Gaps found and fixed:** (1) missing API keys gave raw tracebacks, and `judge.smoke` died on `Illegal header value b'Bearer '`: new `MissingKeyError` names the `.env` variable, smoke checks keys first, CLI prints `error: ...` and exits 2 (+2 tests, 299 total); (2) README and CLAUDE.md said 250 planned / cap 289 calls, but `budget.yaml` (with the J6b hard probes) is 262 / cap 303 (58% of a day, ~22 min): both corrected, J6b row added to the CLAUDE.md budget table; (3) README now warns that live re-runs overwrite the committed `data/outputs`, `results`, `logs`, and that a fresh clone has no cache or ledger so live runs spend real calls and may differ slightly.
- **`docs/answers_draft.md`:** bullet notes by design area for the template's design-decision questions; no reflection text. The template's own question wording was not available, so grouping is by area, with 2.3.3 / 2.4 / 2.5 marked.
- Not checked: Mermaid rendering; the live commands in the README; the PROGRESS J2 table still shows the old 6/13 exact-match (superseded by 3/13 after the c41-c44 swap, which README now shows from `gen_check.json`).
