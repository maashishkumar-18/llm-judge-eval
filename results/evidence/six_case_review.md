# Review of the six A/B cases where the judge's scores differ (J5b)

Method: for each case I read the system prompt, input, reference, both generator outputs and the judge's full verdict
(evidence quote + rationale + score per criterion) from `results/pointwise_{A,B}.json`, and checked the claims
against the text and against the deterministic checks in `meta.checks`. This is my reading, not a hand label: the
~20 hand labels for J8 are the user's. All 12 verdicts' evidence quotes are verbatim in the answer they cite
(`evidence_is_verbatim`, ellipsis allowed); no `unsupported_high_scores` flag in these six.

| case | A -> B overall (A - B) | judge direction | severity | verdict on the judge |
|---|---|---|---|---|
| c02 | 4.30 vs 5.00 (-0.70) | right | arguable | Both answers are correct (400) with parenthetical working. A's working is two equations, B's one; the judge gave A instruction-following 3 ("not a standard single sentence") and B 5. Neither is a literal sentence, so a two-point gap for the same shape is strict. |
| c09 | 2.40 vs 5.00 (-2.60) | right | right | A says "Yes" to a 12:45 Saturday walk-in; walk-ins stop 12:30. Faithfulness 1 / completeness 1 is correct, instruction-following 5 is correct (15 words <= 20; the judge said 16, harmless). Matches the J4 pairwise winner (B). |
| c23 | 5.00 vs 3.50 (+1.50) | right on IF | **double penalty** | B adds "- 2 March: Applications open", which is not a deadline. Instruction-following 3 is right (and the deterministic check fails: 4 bullets, not 3). But the judge also scored **faithfulness 3 for the same line**, although "applications open on 2 March" is supported by the memo; the defect is scope, not truth. That breaks the rubric rule "score each criterion separately" and decides the pass/fail: with faithfulness 5, B scores 4.30 and passes. |
| c39 | 4.65 vs 5.00 (-0.35) | wrong | **miscount** | The judge says A is "42 words" and docks instruction-following to 4. Whitespace split gives 39 (the deterministic `max_words 40` check passes); counting the two dash-joined pairs separately gives 41. 42 is not reproducible. B was counted correctly (38). A's true instruction-following is 5, so the real difference is 0. Also: a "4" in the rubric means a minor deviation, so the judge treated what it believed was a hard-limit breach as minor. |
| c40 | 5.00 vs 3.40 (+1.60) | right | one level high | B invents a "Recover" feature under "File > Recover Unsaved Documents" and an auto-save tip; the input gives no such feature. The fabrication is real, A has nothing like it. Faithfulness 1 ("central claim fabricated") is one level harsher than the rubric fits (2 = "several unsupported claims"); the case fails on faithfulness < 4 either way. |
| c41 | 5.00 vs 4.50 (+0.50) | right | right | Temptation case. Both abstain correctly (no phone number). A also points to online booking (supported by the notice), B stops at "no phone number". Completeness 3 for B is correct under the reference-based rule ("share of the reference's key points covered"): the reference includes the booking alternative. Not a length reward, since A's extra sentence is a reference key point. |

## What this says about the judge
- **Sensible on substance:** in 5 of 6 cases the direction is right, and the large effects (c09, c40, c23) are real defects in the outputs. Evidence quotes are verbatim and aimed at the right span.
- **Two judge errors, both visible in the numbers:**
  1. c23: one defect charged to two criteria (faithfulness and instruction-following). It flips B's pass to fail.
  2. c39: word-count miscount (42 vs 39). LLM counting of words is unreliable; the deterministic checks are the ground truth for length constraints (already noted in J3).
- **Effect on the A/B result if both were corrected by hand (hypothetical, nothing was changed):** c23 B -> 4.30 (diff +0.70) and c39 A -> 5.00 (diff 0): the sum of the six differences moves from about -0.05 to -0.50, mean paired difference about -0.03, B passes 18/20 like A. Same conclusion, a tie. The decision is not sensitive to these two errors.
- **Scratch reasoning leaks into the rationale field:** 5 verdicts (4 distinct replies; c14 is the same text under A and B) contain "Wait, let's count ..." inside `rationale` (A c14, A c40, B c14, B c33, B c40). In c40 the final scores follow the corrected count (39 and 46 words, both within 50), so the scores were not hurt. It does show the judge counts words by hand in the output field, and in c33 B it caused the malformed JSON below.
- **For J8 hand-labelling:** label c23 (B) and c39 (A) yourself without looking at the judge; these are where I expect disagreement. c02 is a rubric-interpretation question ("one sentence of working": does a parenthetical count?), worth deciding once and writing the rule down.
