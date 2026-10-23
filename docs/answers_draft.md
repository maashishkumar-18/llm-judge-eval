# Draft notes for the template's design-decision questions

Bullet notes only, for the author to turn into answers. **No reflection text is written here**: template 2.5 (what
went wrong, what I would change, what I learned) and the human labels are the author's. The template's exact question
wording was not available when this was drafted, so the notes are grouped by design area; the template section numbers
that `PROGRESS.md` already refers to (2.3.3 table rows, 2.4 repair evidence, 2.5 reflection) are marked where they
apply. Every number is from `results/*.json`; N is stated because all small-N probes are exploratory.

## 1. Which judge, which generator, and why they differ
- Generator `openai/gpt-oss-20b` (Groq, OpenAI family, `reasoning_effort=low`, reasoning stripped). Judge
  `gemini-3.5-flash-lite` (Google family). Independent settings: `GEN_*` and `JUDGE_*` share no keys or defaults.
- Different families so the judge does not grade its own family's output (self-enhancement). Enforced in code:
  `Settings.enforce_family_guard()` runs at every `LLMClient` construction and refuses same-family or unidentifiable
  families unless `ALLOW_SAME_FAMILY=1`. Gemma counts as Google.
- Roles were swapped in j0b (originally generator Gemini, judge gpt-oss) because the swap keeps the Gemini judge off
  Google-family answers in the main A/B. Only the 6 deliberate self-preference pairs contain Google-authored answers.
- Judge fixed for the whole project: every validation number is valid only for the judge that produced it. j0b had picked
  `gemini-3.8-flash`; replaced by `gemini-3.5-flash-lite` in j0c (3.8 returned frequent 503s; the free quota is for flash-lite).
- Free-tier limits (user-supplied from AI Studio): 15 RPM, 250K TPM, 500 RPD. Client guards: 12 req/min limiter,
  concurrency 2, daily cap 450 non-cached calls.
- Llama is not served on this Groq key (11 models, no Llama chat model), which is why the earlier plan could not use it.

## 2. Rubric
- Three criteria, integer 1-5, weights: faithfulness 0.40, instruction-following 0.35, completeness 0.25
  (`src/judge/rubric.yaml`, `rubric-v1`). Each has a level descriptor for every score, so the judge scores against a
  rubric, not a bare number.
- Reference-based and reference-free "how to judge" text per criterion: the reference is one acceptable answer, judge
  meaning not phrasing; reference-free judges only against `input` and `system_prompt`.
- Shared rules injected into every mitigated prompt: length earns no credit; confident tone earns no credit; answer text
  is data, not instructions; score criteria separately; no filling gaps from world knowledge.
- Pass rule, fixed before the A/B run: faithfulness >= 4 AND instruction-following >= 3 AND overall >= 3.5.
- Overall is recomputed in code as the weighted mean; any `overall` in the judge reply is ignored (tested). Pass
  boundary uses an epsilon because (4,4,2) is exactly 3.5 on paper but not in floats.
- Weights and pass rule were not tuned after seeing results.

## 3. Judging prompt design
- One call returns all three criteria in one JSON (no per-criterion calls): cheaper, and fits the call budget.
- Field order per criterion is evidence -> rationale -> score, so the quote is committed before the number.
- Mitigated prompts require one verbatim quote per criterion ("NONE" allowed, but a 4-5 needs a quote);
  `unsupported_high_scores` flags a score >= 4 without a verifiable quote (lenient on case, whitespace, quotes).
- Output-format example in each prompt is generated from the pydantic models, with a test that they cannot drift.
- Versioned prompt files (`prompts/<name>.v1.txt`, `VERSIONS` dict), strict `{{var}}` rendering, one snapshot test per
  prompt. Frozen at J1; not changed after seeing any probe text.
- Six hand-written few-shot anchors (`anchors.yaml`; overall 5.00, 4.30, 4.00, 3.70, 3.15, 1.35), kept out of the eval
  suite, mixed order; the reference-free prompt shows only the five reference-free anchors.
- JSON mode OFF by default (`JUDGE_JSON_MODE=0`): prompt-only JSON, so the parse ladder is genuinely exercised. The flag
  never applies to the generator.
- Naive judge (A6) for the before/after: holistic 1-10, no rubric, anchors, evidence or length rule.

## 4. Malformed-JSON handling (template 2.4)
- Ladder in `src/judge/parse.py`: strict -> extract (fences, prose, balanced `{...}` spans, last valid span wins) ->
  repair (smart quotes, comments/trailing commas, unquoted keys, inner quotes, control chars, Python literals,
  truncation; syntax only, never invents a field) -> one re-ask quoting the validation error -> INVALID.
- Validation is the pydantic schema: a score of 7 or a missing criterion is a parse failure, not a repair.
- A re-ask is a real judge call and counts against the experiment cap. INVALID results are counted in N and in the
  pass rate with INVALID-as-fail, never dropped.
- Tested on a 46-reply malformed corpus (some samples synthetic) plus mock-transport tests.
- Live record: in the 40 suite verdicts, 39 strict and 1 repair. Real repair event kept as evidence:
  `results/evidence/repair_event_c33_B.json`. Cause: the judge put a word-count self-check with unescaped double quotes
  inside `rationale`; strict failed (`Expecting ',' delimiter`, char 844), extract failed (first balanced object was a
  nested criterion), repair `inner_quotes` succeeded; repaired verdict 5/5/5.
- Across all experiments: 0 INVALID. Re-asks: 0 in the main runs, 1 in self-preference. Probe and naive replies often
  arrive fenced or wrapped (36 of 72 probe verdicts parsed at the `extract` rung).
- Replay (`judge replay`) re-runs the same ladder over the logged raw replies with no API call: 20/20 identical for
  both configs.

## 5. Judging modes and when each fits
- Pointwise reference-based: a reference exists; per-criterion scores and a pass/fail; 1 call per answer.
- Pointwise reference-free: no reference; grounded in `input` and `system_prompt`; cannot verify outside facts.
- Pairwise with order swap: head-to-head ranking; 2 calls per pair; TIE when the orders disagree (flagged as a flip).
- Evidence for the guidance (see README table): pairwise returned TIE on 16 of 20 real A/B pairs but picked the correct
  answer in 6/6 correct-vs-wrong pairs in both orders; reference-free still rejected 6/6 wrong answers because the
  passage was in `input`; the 7 reference-free suite cases scored 4.95 mean (ceiling).
- Countable constraints are better checked deterministically (`checks.py`): the judge mis-counted words once (c39,
  counted 42, actual 39) and instruction-following was the least stable criterion in test-retest.

## 6. Aggregation and the A/B comparison
- Per case -> suite: pass rate (with a bootstrap CI and an INVALID-as-fail variant), mean overall and per-criterion
  means with percentile bootstrap CIs (10,000 resamples, seed 5150), by-mode split (reference vs reference-free).
- Compare two configs paired by case id; differences are A minus B; win rate is A over B =
  (wins + 0.5 ties) / valid pairs after the order-swap rule.
- Decision rule (`compare.decide`): (1) 95% CI of the mean paired overall difference excludes 0; (2) the pairwise win
  rate leans the same way (> 0.5 for A, < 0.5 for B); (3) no pairwise results -> no winner; otherwise "no significant
  difference". "Win rate consistent" is read as a direction check; whether its CI excludes 0.5 is reported separately.
- Which pair is compared was pre-registered before any judge call (CLAUDE.md 2b): if qwen (config C) differs from A by
  >= 10 points on either deterministic check, compare models; otherwise compare prompts v1 vs v2. Observed C - A =
  -5.0 pts constraint, +0.0 pts exact match, so prompts (A vs B); C dropped.
- Result (N=20, mitigated judge): mean overall A 4.610, B 4.612, paired A - B = -0.003, CI [-0.362, +0.325]; pass rate
  18/20 vs 17/20; win rate A over B 0.550 [0.450, 0.650] (A 3 / B 1 / tie 16; flip rate 0). Decision: no significant
  difference.
- What drives it: 14 of 20 cases have identical overall under A and B; six differ (c02 -0.70, c09 -2.60, c23 +1.50,
  c39 -0.35, c40 +1.60, c41 +0.50) and cancel.
- Suite design: 20 cases (13 reference-based, 7 reference-free), four temptation cases (c41-c44) added before judging
  because A and B were at the ceiling on hard constraints; they favour prompt v2 by construction and did not separate
  the configs in generation.

## 7. Bias handling: name, mitigation, number (naive -> mitigated)
- **Position.** Mitigation: judge both orders, disagreement -> TIE + flag. Flip rate on the same 12 pairs: naive 5/12 =
  41.7% (2 hard flips), mitigated 0/12; mitigated on all 20 pairs 0/20. First-position share (decisive): naive 66.7%,
  mitigated 50%. Identical-answer controls (N=6, one order): both judges 6/6 TIE. Sensitivity (a judge that always says
  TIE would also have 0% flips): both picked the correct answer 6/6 in both orders. Example pair c28: naive picks the
  first-listed answer in both orders, mitigated TIE in both.
- **Verbosity.** Mitigation: length-earns-no-credit rule, evidence quote, instruction-following scored separately.
  Padded minus terse-correct (N=6): mitigated faithfulness 0.00, completeness +0.17, faithfulness+completeness +0.06;
  naive -6.00 (1-10). The naive drop is a length-limit penalty (the padded answer breaks the case's own limit), not an
  anti-verbosity judge. Spearman(words, score), correct group, n=12: mitigated +0.44, naive -0.74 (pooled across bases,
  confounded by case). No verbosity bias detected; no improvement claimable.
- **Self-enhancement.** Mitigation: family guard (structural). Measured residual (exploratory): 6 pairs where both the
  gpt-oss and Gemini answers are deterministically correct, mitigated pairwise both orders: Gemini 3 / gpt-oss 0 / tie 3,
  own-family win rate 0.75, Clopper-Pearson 95% CI on the 3 decisive pairs [0.29, 1.00]. Each decisive win had a
  content-level reason in the judge's rationale (c01 kept the "unless reserved" qualifier; c02 and c23 instruction-following),
  checked post hoc. Not run: naive judge row, gpt-oss judge cross-check (no budget).
- **Sycophancy.** Mitigation: per-criterion verbatim evidence; probes are confidently-wrong answers (wrapper
  "Definitely:" / "Without question,"), a reference-hidden variant, verbose-wrong, and right answer + fabricated
  reasoning. Accepted as correct: 0/6 for naive and mitigated in every probe. Floor effect: nothing for the mitigation to
  reduce, so no before/after claim. Negative control: terse-correct answers accepted naive 4/6, mitigated 6/6.
- **Score clustering.** Mitigation: six anchors and per-criterion scores. Same 12 cases: naive 3 distinct values (1, 9,
  10), 67% at the max; mitigated overall 5 distinct, 67% at the max; mitigated faithfulness 2 distinct, 83% at the max.
  Terse-correct vs corrupted AUC: both judges ~1.0. No improvement shown. Caveat: the two judges differ in prompt, scale,
  criteria and anchors at once; no anchors-off ablation (no budget).

## 8. Judge validation (template 2.3.3 rows)
- Labels: by-construction (deterministic, judge not involved: reference facts + the case's own constraints; probes by
  design) and 20 human labels on config-A answers (17 pass / 3 fail: c01, c09, c16) written by the author outside the
  tooling. c09 and c16 are not blind. Provenance and revision history are in README Known limitations.
- Kappa: own implementation (integer arithmetic; undefined reported as None with a reason), unit-tested on a hand-computed
  50-item example (0.4); mutation check breaks 5 tests.
- Agreement vs by-construction labels, suite distinct (N=34, 30 pass / 4 fail): agreement 0.94 vs always-pass 0.88, kappa
  0.72, bootstrap 95% [0.00, 1.00]. Disagreements: c40 B (constraint-only label cannot see the invented steps) and c41 B
  (label requires the "online booking" detail; the pass rule has no completeness threshold).
- Agreement vs human labels (N=20): 0.95 vs 0.85, kappa 0.77 [0.00, 1.00]; one disagreement (c01, the dropped
  "unless reserved" qualifier; judge completeness 4, passed). Naive judge on 13 items: 12/13, kappa 0.75.
- Negative controls (all pass): always-pass / always-fail kappa exactly 0; inverted judge lower; shuffled labels x10,000
  mean kappa ~0.001 (tolerance +-0.10 set before running); permutation p 0.003 (suite distinct).
- Instruction-following vs `checks.py` (40 outputs): agreement 0.93 vs always-pass 0.97, kappa -0.03; only 1 of 40
  outputs violates a countable constraint, so kappa is not interpretable.
- Test-retest (temperature 0, nonce bypasses cache, N=10 cases x 3 runs): 0/10 pass/fail flips (95% exact CI up to 0.31);
  overall sd per case mean 0.095, max 0.404; 4 of 10 cases changed overall at least once; criterion scores identical
  across runs: faithfulness 10/10, completeness 9/10, instruction-following 7/10. c33 and c39 sit on the IF = 3 pass
  boundary. Deviation: the request said 3 reruns of the suite; the budget plans 3 runs of 10 cases (saved run + 2
  reruns), which is what was run.
- Adversarial summary, N=6 bases: fooled 0/6 for both judges on every probe; discrimination (correct accepted - fooled)
  naive 0.67, mitigated 1.00.

## 9. Engineering and logging
- Every LLM call goes through `LLMClient.complete()`: disk cache, sliding-window rate limiter (every HTTP attempt),
  persistent daily ledger of non-cached judge calls (hard cap 450 per Pacific day), per-experiment caps from
  `budget.yaml`, retry with backoff honouring Retry-After, 401 not retried.
- Every call (cached too) appends a CallLog row: prompt, raw response, hashes, tokens in/out, latency, cached flag,
  nonce, model, temperature. Judge tokens and calls are tracked per experiment; cost is reported as actual (free tier,
  $0.00) and list-price equivalent ($0.0964 for the A/B comparison, 80 calls; price from the official page, retrieved
  2026-10-23, tier mapping inferred).
- Batch runs go through `checkpoint.run_checkpointed`: stop at a cap, resume without redoing work. `judge budget` is the
  dry run; a run over its cap is refused.
- Call budget: 262 planned, cap 303 (about 22 minutes at 12 req/min), 58% of one day's cap.
- Judge and generator are independently configurable (`.env`); one httpx-based async client for both providers.
- Tests: 299 passed; every metric has a hand-computed test; prompt snapshots; fake-judge end-to-end tests for each
  experiment; mock-transport tests for retries, caching and nonce bypass.
- Reproducibility: committed `results/`, `logs/`, `labels/`; `judge replay` and `judge validate` rebuild numbers with no
  key; `scripts/make_report_tables.py` builds the README tables from `results/*.json`.

## 10. Facts to keep visible in the write-up (not reflection)
- All small-N probe numbers are exploratory (probes N=6, controls N=6, self-preference N=6 pairs, clustering N=12,
  test-retest N=10); intervals that include 0 are reported with the point estimate.
- No before/after improvement is claimable for verbosity, sycophancy or clustering; position is the one bias with a
  measured naive -> mitigated change (41.7% -> 0%).
- Not measured: naive self-preference, gpt-oss judge cross-check, anchors-off ablation, unpadded control, probes where
  the wrong answer is subtle.
- Known judge misreads (observations, nothing re-scored): c23 B charged the same scope issue to faithfulness and
  instruction-following; c39 word-count error; c01 hard probe "single long sentence" when it was one sentence within a
  two-sentence limit; scratch reasoning ("Wait, let's count...") inside `rationale` in 5 verdicts.
- 6 of the 20 A/B pairs have identical answers, so their second pointwise call is a cache hit with difference exactly 0;
  14-pair numbers are in `results/compare_A_vs_B.json`.

## 11. Left for the author
- Template 2.5 reflection sections.
- Human labels (done by the author; provenance statement is the author's to confirm).
- Real time spent per session (not recorded in `PROGRESS.md`).
- Screenshots (`docs/screenshots/` is empty); c28 is the suggested pair for the position-bias screenshot.
