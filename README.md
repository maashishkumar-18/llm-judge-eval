# llm-judge-eval

LLM-as-judge evaluation pipeline with measured bias mitigations. Work in progress; see `PROGRESS.md` for
status and `CLAUDE.md` for the locked design decisions, metric definitions and call budget.

## Known limitations

- **Few-shot anchors can pull scores toward their own levels.** The mitigated pointwise prompts include six
  hand-written calibration anchors (`src/judge/anchors.yaml`; overall scores 1.35, 3.15, 3.70, 4.00, 4.30 and
  5.00) to fight score clustering. With so few examples, the judge may copy their score profile or their
  wording instead of grading the case, so the anchors themselves are a bias source. They are written by hand,
  kept out of the eval suite, listed in mixed order, and the reference-free prompt only sees the five
  reference-free ones. Their effect is measured in the clustering experiment (naive vs mitigated spread), not
  assumed.
- **The lower parse-ladder rungs are proven by tests, not by live use.** The ladder is strict -> extract -> repair ->
  one re-ask -> INVALID (`src/judge/parse.py`). In the J3 live check (5 cases) Gemini returned bare JSON every time,
  so only `strict` was exercised against the real judge; `extract`, `repair` and `reask` are covered by a 46-reply
  malformed corpus and mock-transport tests (some samples, such as gpt-oss-style reasoning before the JSON, are
  synthetic). The real INVALID and re-ask counts come from the full run and are reported in the results, not assumed.
- **A near-ceiling judge is a flag, not a finding.** The first 5 live verdicts were all PASS with 17 of 20 criterion
  scores at 5. That cannot separate leniency from good outputs. The adversarial probes (confidently-wrong, padded,
  verbose-wrong) and the negative controls are the real test; if they also score near 5, the judge is investigated
  before any number is reported.
- **The generator decision rule is a coarse screen.** Whether the main A/B contrasts models (gpt-oss-20b vs qwen) or
  prompts (v1 vs v2) is decided by a 10-percentage-point threshold on two cheap checks (constraint pass rate over 20
  cases, strict exact-match over 13). One case is worth 5 and 7.7 points, so two cases flip it; it is not a
  significance test. It was fixed before any judge call (CLAUDE.md section 2b).
- **The suite contains deliberate temptation cases (c41-c44) that favour prompt v2 by construction.** Their system
  prompts do not tell the model to abstain, and v2's preamble does. That is a designed probe of grounding, chosen
  before judging because A and B were both at the ceiling on hard constraints; it is not a neutral sample of tasks.
- **Human-label provenance is unverified, and two labels are not blind.** `labels/human_labels.jsonl` (20 config-A
  answers) was not produced by the project tooling; the tooling only generated the blank template. Its labels are
  identical on all 20 items to an LLM-rater draft that has since been removed from the tree (still in git history,
  commit 567bfd8), and that draft was not blind either. For c09 and c16 the labels are certainly not blind: the
  original answers' errors were already known from earlier reports. Treat judge-vs-human agreement as unusable
  until the author confirms who wrote the labels or relabels from a blind sheet. The headline agreement is against
  the deterministic by-construction labels (suite, N=34).
- **The instruction-following check has almost no negatives.** Only 1 of the 40 suite outputs (c23 under config B)
  violates a countable constraint in `checks.py`, so judge-vs-checks agreement (0.93) is below the always-pass
  baseline (0.97) and its kappa is not interpretable. The adversarial probes are a floor effect: both judges are
  fooled 0/6, so no mitigation effect can be shown.
- **Test-retest zero flips is partly luck.** 0/10 pass/fail flips at judge temperature 0.0, with noise in
  instruction-following; two cases (c33, c39) sit on the instruction-following = 3 pass boundary, so one more point
  of noise would flip them. N=10 shows none were seen, not that flips are rare.

