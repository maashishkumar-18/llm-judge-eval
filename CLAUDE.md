# CLAUDE.md — llm-judge-eval (Problem 2: LLM-as-Judge Evaluation Pipeline)

Memory across sessions lives here (rarely changed) and in PROGRESS.md (updated every session).
Read both first. One session = one sub-phase (J0–J10). Commit at the end of every session.

## 1. Assignment (condensed)
Given a test case `{input, system_prompt, model_output, expected_output?, criteria?}`, use an LLM judge to
produce a structured verdict (per-criterion score + rationale + overall), with concrete measures to detect
and reduce judge bias.

Requirements: (1) accept JSON/YAML suite, structured judging prompt, parse structured verdict with robust
malformed-JSON handling; (2) explicit rubric, not a bare number; (3) aggregate per-case -> suite report
(pass rate, mean scores, win rate), compare TWO configs, declare a winner; (4) log every judge prompt + raw
response, track judge tokens/calls, judge and generator configurable independently.

Bias handling — name it, mitigate in code, measure before -> after (a claim without a number scores zero):
position (both orders, flip rate) · verbosity (length rule, padded probe) · self-enhancement (different
family judge / family guard) · sycophancy (per-criterion quoted evidence, confidently-wrong probe) · score
clustering (few-shot anchors / pairwise).
Validate the judge (>= 1, aim for all): agreement/kappa vs labels · test-retest · adversarial probes.
Scoring: pipeline 20 · judging design 20 · bias handling 25 · judge validation 20 · comparison/engineering 15.

## 2. Locked decisions (with amendments A1–A8)
**Roles were swapped in j0b** (supersedes the original roadmap: generator Gemini / judge gpt-oss).
- **Generator: `openai/gpt-oss-120b` via Groq** (OpenAI family), `reasoning_effort=low`, reasoning stripped (`include_reasoning=false`, `reasoning` field dropped; only final `content` is cached/logged). Env: GEN_REASONING_EFFORT, GEN_STRIP_REASONING.
- **Judge: Gemini `gemini-3.8-flash`** (Google family) — the strongest model that works on this key's free tier (probed 2026-10-23 with one tiny call each: `gemini-4-argon`, `gemini-3.1-pro-preview`, `gemini-pro-latest` -> 429 quota on first call; `gemini-2.5-pro` -> 404 no longer available to new users; `gemini-3.8-flash` -> 200 but intermittent 503 "high demand" (2 of 4 attempts OK; retried by backoff); `gemini-3.7-flash` mostly 503; `gemini-3.6-flash` and `gemini-3.5-flash` -> 200). Fallback if 3.8 stays flaky: `gemini-3.6-flash` (change JUDGE_MODEL only; any judge change invalidates validation numbers, so re-run J8).
- **JSON mode OFF for judge calls** (`JUDGE_JSON_MODE=0`, default): prompt-only JSON, so the parse ladder is genuinely exercised. When set to 1: Gemini `responseMimeType=application/json` (+ `responseJsonSchema` if a schema is passed), Groq `response_format=json_object`. The flag never applies to the generator. Any experiment run with the flag ON must say so.
- Python 3.11, Pydantic v2. One async client layer: **httpx for both providers** (A5). No LangGraph/FastAPI.
- Parse ladder in code (A4): strict -> extract (fences/braces) -> repair -> one re-ask with the error -> INVALID (counted, never dropped). Instructor optional for the re-ask rung only. Gemini is the judge now, so expect fenced JSON / chatter rather than gpt-oss reasoning prose; the ladder must handle both.
- Rubric (A2): faithfulness 0.40, instruction-following 0.35, completeness 0.25; length/confident tone earn no credit.
- Modes (A3): pointwise (reference-based if expected_output else reference-free) + pairwise (order swap).
- Two judge configs (A6): **naive** (single order, holistic 1–10, no anchors/length rule/grounding) and **mitigated**. Every bias row reports both.
- Suite quality known by construction (A7); context = Northwind passages inside `input`; anchors in a separate file from cases NOT in the eval suite.

**Amendments revised by the swap**
- **A1 (judge choice) — superseded.** Original: judge = gpt-oss-120b because Llama is not served on the free Groq tier. Now: judge = Gemini (`gemini-3.8-flash`), generator = gpt-oss-120b. The rationale that matters is unchanged: judge and generator are different families (Google vs OpenAI). Llama remains unavailable (Groq listed 11 models at J0, no Llama chat model).
- **A8 (repo/generator) — revised.** Separate repo `llm-judge-eval`, copy (don't import) Problem 1 modules, both links in the template field. The original clause "keep generator A/B on Gemini so the judge never grades its own family" becomes **"keep generator A/B on gpt-oss (Groq) so the Gemini judge never grades Google-family output"**. Consequence: the main A/B experiments (J2–J6) judge only OpenAI-family answers with a Google-family judge.

**Family guard rationale.** Self-enhancement bias: a judge tends to prefer outputs from its own family. `Settings.enforce_family_guard()` runs at every `LLMClient` construction and refuses judge family == generator family (and any unidentifiable family) unless `ALLOW_SAME_FAMILY=1` (which logs a warning). Gemma counts as Google. It is a structural mitigation, not a measurement; J7 measures the residual preference. After the swap the guard protects against a Gemini judge grading Gemini/Gemma output instead of a gpt-oss judge grading gpt-oss output; the logic is symmetric.

## 3. Metric definitions (lock before any experiment; thresholds pre-registered in rubric.yaml)
| Quantity | Definition |
|---|---|
| Overall score | Σ wᵢ·scoreᵢ / Σ wᵢ on 1–5, **recomputed in code** (never trust the judge's arithmetic) |
| Pass rule | faithfulness ≥ 4 AND instruction-following ≥ 3 AND overall ≥ 3.5 (adjust once, before the A/B run) |
| Pair verdict | judge both orders; winners agree -> that winner; else TIE + flagged as order flip |
| Win rate (A over B) | (winsA + 0.5·ties) / N pairs, after the order-swap rule |
| Flip rate | share of pairs whose winner changes when order is swapped (before the tie rule); report naive and mitigated |
| Position preference | on identical-answer control pairs (A=B): share where judge picks first-listed; unbiased ≈ 50% or tie |
| Length bias | Spearman(token length, overall) within a quality group + padded-vs-original win rate |
| Self-preference (J7, exploratory) | Judge = Gemini (Google). Build ~10 reference-checkable cases; author one answer with **gpt-oss** (the normal generator) and one with **Gemini** (own family) per case. Keep only pairs where **both are correct by a deterministic check** against the reference. The Gemini judge compares each pair in both orders (order-swap rule applies). Own-family win rate = (wins for the Gemini-authored answer + 0.5·ties) / N pairs, with an exact binomial (Clopper–Pearson) 95% CI; 50% = no self-preference. State N plainly. The Gemini-authored answers are made with a one-off generator client built from `settings.model_copy(update={gen_provider:'gemini', gen_model:<gemini>, judge_provider:'groq', judge_model:'openai/gpt-oss-120b'})`, so the family guard passes without ALLOW_SAME_FAMILY (that client only generates; it never judges). Optional cross-check: repeat with gpt-oss as judge; if the preference flips toward the gpt-oss answer, that supports self-preference over answer quality. |
| Fooled rate | share of adversarial probes with faithfulness ≥ 4 on a confidently-wrong answer, or verbose-wrong preferred over terse-correct |
| Score spread | std dev, # distinct values used, share at modal value; clustering = narrow spread |
| Agreement, kappa | agreement rate + Cohen's kappa on pass/fail vs hand labels and by-construction labels; kappa own implementation, unit-tested on a hand-computed example |
| Test-retest | rerun suite 3× with cache bypassed (nonce); share of cases whose pass/fail flips + score std dev |
| A/B winner | paired per case; bootstrap 95% CI on mean overall diff and on win rate; winner only if CI excludes 0 and win rate consistent, else "no significant difference"; state N |

## 4. Conventions
- Config is env-only (`.env`, never committed; `.env.example` lists names). GEN_* and JUDGE_* settings share no keys/defaults; model ids are required (no silent fallback).
- **Every LLM call goes through `judge.llm.LLMClient.complete()`** — cached, rate-limited, logged. No raw HTTP elsewhere (except the smoke model-listing helpers).
- Every call appends a CallLog row to JSONL (cached calls too). Retests use a `nonce` to bypass the cache.
- Every metric has a hand-computed unit test. Thresholds pre-registered before the A/B run; post-hoc changes are labelled post-hoc in PROGRESS.md.
- Commit results/*.json, logs/*.jsonl, labels/. Ignore only secrets, caches, scratch.
- Distrust results that look too good (all 5s / 1.0); run negative controls. Reflection sections (template 2.5) and the ~20 hand labels are written by the user, not Claude.
- Never paste logs/suites into chat; write to files and read summaries.

## 5. Provider facts (verified 2026-10-23 — re-check; they change)
**Groq / `openai/gpt-oss-120b` (now the GENERATOR)** (console.groq.com/docs/rate-limits): 30 RPM, 1K RPD, 8K TPM, 200K TPD (page labels these "Developer plan" base limits; free-plan column ambiguous — confirm at console.groq.com/settings/limits). 429 carries `retry-after` (seconds); `x-ratelimit-*` always present; cached tokens don't count. Groq now only generates (roughly 60–150 calls over the project), so the 8K TPM / 200K TPD cap binds much less than when it judged ~650 calls; long passages + reasoning can still hit 8K TPM, so keep GEN_CONCURRENCY=2 and lean on the cache. Smoke: 78 in / 23 out.
- Reasoning is returned in message field `reasoning` (separate from `content`); `reasoning_format` NOT supported for gpt-oss; `include_reasoning` true/false; `reasoning_effort` low|medium|high. We send `include_reasoning=false` and also drop the field client-side. Reasoning tokens still count in `completion_tokens`.
- Served-model check: `GET https://api.groq.com/openai/v1/models` (11 models at J0; no Llama).

**Gemini (now the JUDGE) — `gemini-3.8-flash`**: listed Stable at ai.google.dev/gemini-api/docs/models. The docs publish **no free-tier numbers**; limits are per-project, per-model, visible only in AI Studio (aistudio.google.com/rate-limit); daily quota resets midnight Pacific. The ~600–700 judge calls in the roadmap budget now fall on Gemini's RPM/RPD, which is the constraint to plan around. Observed: intermittent 503 "high demand" on 3.8/3.7-flash (retryable; MAX_RETRIES=6). Thinking tokens come back in `usageMetadata.thoughtsTokenCount` and are counted in tokens_out (smoke: 8 in / 98 out for "pong"); set `max_tokens` generously on judge calls or thinking can truncate the answer.
- Usable for generateContent on this key (live `GET /v1beta/models`, 2026-10-23): gemini-3.8/3.7/3.6/3.5-flash, gemini-3.5-flash-lite, gemini-3.1-flash-lite, gemini-2.5-flash(-lite), gemini-flash(-lite)-latest, gemma-4-26b/31b-it, plus pro/preview/image/TTS variants. Pro models and `gemini-4-argon` are listed but have no free quota on this key.
- **Free-tier limits for the judge key (RPM / TPM / RPD): NOT YET RECORDED.** The user said they would paste these from AI Studio; nothing was pasted by j0b. Fill in:
  - gemini-3.8-flash: RPM = ?, TPM = ?, RPD = ?
  - (fallback) gemini-3.6-flash: RPM = ?, TPM = ?, RPD = ?
