# Judge validation (judge gemini-3.5-flash-lite, rubric-v1) - EXPLORATORY, N in every row

Pass rule: mitigated = rubric pass rule recomputed from stored scores; naive = score >= 8 of 10. Positive class = pass. Labels: by construction (deterministic) and human.

Caveats:
- By-construction labels for the reference-free suite cases check the case's own constraints only, so they cannot see a faithfulness failure; a judge that flags one will look like a disagreement.
- By-construction labels for reference-based cases also require the reference's key facts (a completeness matter), but the pass rule has no completeness threshold, so a terse-but-faithful answer can be label-fail and judge-pass.
- The 'probes' stratum is a set of designed extremes (6 correct, 30 wrong or defective): kappa there measures separation of constructed corruptions, not agreement on natural answers; the headline is suite_distinct.
- Configs A and B share 6 identical answers, so suite_A and suite_B overlap; suite_distinct removes the duplicates. Strata share items, so they are not independent evidence.
- Perfect agreement collapses the bootstrap interval to a point ('CI degenerate'); use the exact interval on agreement and the N.
- Positives are few (suite: 4 label-fail of 34), so one item moves kappa a lot; every kappa here is exploratory.

## Agreement and kappa vs by-construction labels

| stratum / judge | N | agreement | kappa [95% bootstrap] | false pass (judge pass among label fail) | false fail | INVALID |
|---|---|---|---|---|---|---|
| suite_A / mitigated  (labels: 18 pass, 2 fail) | 20 | 1.00 (20/20; always-pass 0.90) | 1.00 [CI degenerate] | 0/2 | 0/18 | 0 |
| suite_A / naive  (labels: 18 pass, 2 fail) | 13 | 1.00 (13/13; always-pass 0.85) | 1.00 [CI degenerate] | 0/2 | 0/11 | 0 |
| suite_B / mitigated  (labels: 17 pass, 3 fail) | 20 | 0.90 (18/20; always-pass 0.85) | 0.61 [-0.07, 1.00] | 1/3 | 1/17 | 0 |
| suite_B / naive  (labels: 17 pass, 3 fail) | 0 | - | - | - | - | - |
| suite_distinct / mitigated  (labels: 30 pass, 4 fail) | 34 | 0.94 (32/34; always-pass 0.88) | 0.72 [0.00, 1.00] | 1/4 | 1/30 | 0 |
| suite_distinct / naive  (labels: 30 pass, 4 fail) | 13 | 1.00 (13/13; always-pass 0.85) | 1.00 [CI degenerate] | 0/2 | 0/11 | 0 |
| probes / mitigated  (labels: 6 pass, 30 fail) | 36 | 1.00 (36/36; always-pass 0.17) | 1.00 [CI degenerate] | 0/30 | 0/6 | 0 |
| probes / naive  (labels: 6 pass, 30 fail) | 36 | 0.94 (34/36; always-pass 0.17) | 0.77 [0.30, 1.00] | 0/30 | 2/6 | 0 |
| all_distinct / mitigated  (labels: 36 pass, 34 fail) | 70 | 0.97 (68/70; always-pass 0.51) | 0.94 [0.85, 1.00] | 1/34 | 1/36 | 0 |
| all_distinct / naive  (labels: 36 pass, 34 fail) | 49 | 0.96 (47/49; always-pass 0.35) | 0.91 [0.76, 1.00] | 0/32 | 2/17 | 0 |

Negative controls (per stratum, mitigated judge):
- suite_A (N=20)
  - PASS always-pass judge: expected kappa = 0 (no information); observed 0.0
  - PASS always-fail judge: expected kappa = 0 (no information); observed 0.0
  - PASS inverted judge: expected kappa below the real judge's (sign check); observed -0.2195
  - PASS labels shuffled x10000: expected mean kappa within +-0.1 of 0; observed 0.0008 (permutation p of >= observed: 0.0045)
- suite_B (N=20)
  - PASS always-pass judge: expected kappa = 0 (no information); observed 0.0
  - PASS always-fail judge: expected kappa = 0 (no information); observed 0.0
  - PASS inverted judge: expected kappa below the real judge's (sign check); observed -0.2081
  - PASS labels shuffled x10000: expected mean kappa within +-0.1 of 0; observed -0.0003 (permutation p of >= observed: 0.0451)
- suite_distinct (N=34)
  - PASS always-pass judge: expected kappa = 0 (no information); observed 0.0
  - PASS always-fail judge: expected kappa = 0 (no information); observed 0.0
  - PASS inverted judge: expected kappa below the real judge's (sign check); observed -0.1878
  - PASS labels shuffled x10000: expected mean kappa within +-0.1 of 0; observed 0.0013 (permutation p of >= observed: 0.0027)
- probes (N=36)
  - PASS always-pass judge: expected kappa = 0 (no information); observed 0.0
  - PASS always-fail judge: expected kappa = 0 (no information); observed 0.0
  - PASS inverted judge: expected kappa below the real judge's (sign check); observed -0.3846
  - PASS labels shuffled x10000: expected mean kappa within +-0.1 of 0; observed -0.0014 (permutation p of >= observed: 0.0001)
- all_distinct (N=70)
  - PASS always-pass judge: expected kappa = 0 (no information); observed 0.0
  - PASS always-fail judge: expected kappa = 0 (no information); observed 0.0
  - PASS inverted judge: expected kappa below the real judge's (sign check); observed -0.9413
  - PASS labels shuffled x10000: expected mean kappa within +-0.1 of 0; observed -0.0004 (permutation p of >= observed: 0.0001)

## Agreement and kappa vs human labels

N=20 labelled (18 pass / 2 fail; 20 distinct texts; 0 blank skipped; unmatched []).

| pair | N | agreement | kappa [95% bootstrap] | false pass | false fail | INVALID |
|---|---|---|---|---|---|---|
| mitigated judge vs human | 20 | 1.00 (20/20; always-pass 0.90) | 1.00 [CI degenerate] | 0/2 | 0/18 | 0 |
| naive judge vs human | 13 | 1.00 (13/13; always-pass 0.85) | 1.00 [CI degenerate] | 0/2 | 0/11 | 0 |
| by-construction vs human | 20 | 1.00 (20/20; always-pass 0.90) | 1.00 [CI degenerate] | 0/2 | 0/18 | 0 |

- PASS always-pass judge: expected kappa = 0 (no information); observed 0.0
- PASS always-fail judge: expected kappa = 0 (no information); observed 0.0
- PASS inverted judge: expected kappa below the real judge's (sign check); observed -0.2195
- PASS labels shuffled x10000: expected mean kappa within +-0.1 of 0; observed 0.0008 (permutation p of >= observed: 0.0045)
- disagreements (mitigated vs human): []

## Test-retest (cache bypassed by nonce)

N=10 cases x 3 runs (A, retest1, retest2); comparable 10, scored in every run 10; INVALID per run {'A': 0, 'retest1': 0, 'retest2': 0}.

- pass/fail flips: 0/10 = 0.00 (95% exact CI [0.0, 0.3085]); cases []
- overall sd per case (sample (n-1)): mean 0.095, max 0.404; 4/10 cases changed overall at least once
- criterion scores identical across runs: faithfulness 10/10, instruction_following 7/10, completeness 9/10
- run pairs: A~retest1 pass-agreement 1.00 (n=10), mean |d overall| 0.060; A~retest2 pass-agreement 1.00 (n=10), mean |d overall| 0.140; retest1~retest2 pass-agreement 1.00 (n=10), mean |d overall| 0.130

- PASS nonce runs bypass the cache: expected 0 cached calls in every nonce run (a re-run of the same command re-serves its saved draws and shows >0); observed {'retest1': 0, 'retest2': 0}
- PASS nonces distinct and non-empty: expected baseline '' and each rerun its own nonce; observed ['', 'retest1', 'retest2']

## Adversarial summary (N=6 base cases per row, exploratory)

Accepted = mitigated faithfulness >= 4 / naive score >= 8. Lower is better except where noted.

| probe | metric | naive | mitigated |
|---|---|---|---|
| confidently wrong, reference shown | accepted (fooled) | 0/6 | 0/6 |
| confidently wrong, reference hidden | accepted (fooled) | 0/6 | 0/6 |
| verbose wrong vs terse correct | verbose-wrong scored >= terse-correct | 0/6 | 0/6 |
| right answer + fabricated reasoning | accepted (fooled) | 0/6 | 0/6 |
| padded minus terse-correct | mean score delta | mean -6.00 (N=6) | mean 0.06 (N=6) |
| correct vs confidently wrong, pairwise | picked the correct answer (both orders) | 6/6 | 6/6 |

Terse-correct answers accepted (should be high): naive 4/6, mitigated 6/6. Discrimination (correct accepted rate minus fooled rate, reference mode): {'naive': 0.6667, 'mitigated': 1.0}. Trivial baselines: always-accept fooled 1.0 / correct-accepted 1.0, always-reject 0.0 / 0.0.

- PASS naive: terse-correct answers accepted: expected most correct answers accepted (else 'fooled 0' only means 'rejects everything'); observed 4/6
- PASS mitigated: terse-correct answers accepted: expected most correct answers accepted (else 'fooled 0' only means 'rejects everything'); observed 6/6
- stored-vs-recomputed mismatches: none
