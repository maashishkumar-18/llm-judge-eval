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
- **The generator decision rule is a coarse screen.** Whether the main A/B contrasts models (gpt-oss-20b vs qwen) or
  prompts (v1 vs v2) is decided by a 10-percentage-point threshold on two cheap checks (constraint pass rate over 20
  cases, strict exact-match over 13). One case is worth 5 and 7.7 points, so two cases flip it; it is not a
  significance test. It was fixed before any judge call (CLAUDE.md section 2b).
- **The suite contains deliberate temptation cases (c41-c44) that favour prompt v2 by construction.** Their system
  prompts do not tell the model to abstain, and v2's preamble does. That is a designed probe of grounding, chosen
  before judging because A and B were both at the ceiling on hard constraints; it is not a neutral sample of tasks.

