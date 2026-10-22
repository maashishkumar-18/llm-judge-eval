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
- Generator: Gemini (`gemini-3.5-flash-lite`). Judge: `openai/gpt-oss-120b` via Groq (A1; Llama not served on this account). Different family from Gemini.
- Python 3.11, Pydantic v2. One async client layer: **httpx for both providers** (A5). No LangGraph/FastAPI.
- Parse ladder in code (A4): strict -> extract (fences/braces) -> repair -> one re-ask with the error -> INVALID (counted, never dropped). Instructor optional for the re-ask rung only.
- Rubric (A2): faithfulness 0.40, instruction-following 0.35, completeness 0.25; length/confident tone earn no credit.
- Modes (A3): pointwise (reference-based if expected_output else reference-free) + pairwise (order swap).
- Two judge configs (A6): **naive** (single order, holistic 1–10, no anchors/length rule/grounding) and **mitigated**. Every bias row reports both.
- Suite quality known by construction (A7); context = Northwind passages inside `input`; anchors in a separate file from cases NOT in the eval suite.
- Separate repo `llm-judge-eval`; copy (don't import) Problem 1 modules (A8). Generator A/B stays on Gemini.

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

## 5. Provider facts (verified from official docs 2026-10-23 — re-check; they change)
**Groq / `openai/gpt-oss-120b`** (console.groq.com/docs/rate-limits): 30 RPM, 1K RPD, 8K TPM, 200K TPD (page labels these "Developer plan" base limits; free-plan column ambiguous — confirm at console.groq.com/settings/limits). 429 carries `retry-after` (seconds); `x-ratelimit-*` always present (requests = per-day, tokens = per-minute); cached tokens don't count.
- **8K TPM / 200K TPD is the binding constraint**, not RPM: ~650 calls × (prompt + reasoning) could exceed 200K tokens/day. Observed smoke: 78 in / 55 out for a trivial prompt (reasoning tokens count in completion). Plan long runs across days, keep prompts tight, lean on the cache.
- Reasoning: returned in message field `reasoning`, separate from `content` by default; `reasoning_format` NOT supported for gpt-oss; `include_reasoning` true/false; `reasoning_effort` low|medium|high (env `JUDGE_REASONING_EFFORT`). Parse ladder must still tolerate prose/fences in `content`.
- JSON mode: docs do not confirm `response_format` (json_object/json_schema) for this model — not relied on; JSON enforced by prompt + parse ladder.
- Served-model check: `GET https://api.groq.com/openai/v1/models` (11 models listed at J0).

**Gemini / `gemini-3.5-flash-lite`**: listed Stable at ai.google.dev/gemini-api/docs/models; `GET /v1beta/models/{id}` confirms it is served. Docs publish **no free-tier numbers** — limits are per-project and per-model, visible only in AI Studio (aistudio.google.com/rate-limit); daily quota resets midnight Pacific. Record the observed RPM/RPD there before J2 generation. Token usage: `usageMetadata` (`promptTokenCount`, `candidatesTokenCount`, `thoughtsTokenCount` — thoughts are added to tokens_out).
