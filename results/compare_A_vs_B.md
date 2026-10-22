# Config A vs B

Judge `gemini-3.5-flash-lite` (JSON mode off). N=20 paired cases; exploratory at this size, a CI that spans 0 is not evidence of equality.
Differences are A minus B; win rate is A over B. 95% percentile bootstrap CIs, 10,000 resamples.

**Decision: no significant difference**

- CI of the mean overall difference [-0.362, +0.325] spans 0

| | A | B | A - B (paired, N=20) |
|---|---|---|---|
| mean overall | 4.610 [4.133, 4.947] | 4.612 [4.175, 4.938] | -0.003 [-0.362, 0.325] |
| faithfulness | 4.600 [4.000, 5.000] | 4.500 [3.900, 5.000] | 0.100 [-0.500, 0.700] |
| instruction_following | 4.700 [4.300, 5.000] | 4.750 [4.350, 5.000] | -0.050 [-0.350, 0.250] |
| completeness | 4.500 [3.900, 4.950] | 4.600 [4.100, 4.950] | -0.100 [-0.600, 0.300] |
| pass rate | 18/20 | 17/20 | 0.050 [-0.100, 0.200] |

Win rate of A over B (order-swapped, N=20 pairs): 0.550 [0.450, 0.650]; A 3 / B 1 / tie 16; flip rate 0.0.
Differing answers only (N=14): 0.571 [0.429, 0.714]; A 3 / B 1 / tie 10.

Judge usage: 80 calls (61 network, 19 cached, 0 re-asks), 200400 tokens in / 26185 out; cost $0.00 actual (free tier); $0.0964 at list price (standard tier, $0.3/$2.5 per 1M in/out, https://ai.google.dev/gemini-api/docs/pricing, retrieved 2026-10-23).
