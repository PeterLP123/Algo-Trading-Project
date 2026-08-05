# Strategy 2 improvement experiments

These experiments were run after the original holdout and the frozen-forward observations through 4 August 2026 had been seen. They are deliberately labelled **adaptive retrospective research**: they do not repair, replace, or validate the published Strategy 2 result.

Every search stage was specified in a hash-verified `spec.json` before it was run. Failed and bankrupt variants are retained alongside the eventual cost-conditional candidate. The runners expose data and output paths but no strategy-parameter command-line overrides.

A later estimator audit found that the original engine reflected negative 21-day Abdi-Ranaldo
moments above zero with an absolute value, which is not a correction defined in the paper. Those
legacy net-cost search outputs remain in the ledger as path provenance, but they are not used as
current execution-cost evidence. The selected candidate was held fixed and reevaluated under both
paper corrections and every registered fixed-cost scenario; no search or tuning was reopened.

## Complete search ledger

| Stage | Registered variants | Result over the 263-day combined post-selection window |
|---|---:|---|
| Confirmed-reversal Trial A | 1 | The legacy run reached zero equity after extreme turnover. Its former 15,659 USDT cost estimate is retained for provenance but invalidated by the estimator audit. |
| Entry-rule sweep | 4 new candidates plus references | Confirmation rules generated much more turnover. Legacy net rankings are retained but were not promoted or rerun after the audit. |
| Liquid-basket sweep | 5 candidates | No candidate was gross-positive. ETH/BNB/SOL was least negative at −300 USDT gross; this conclusion does not depend on the cost correction. |
| Fixed-holding sweep | 5 candidates | No candidate was gross-positive. A five-day ETH/BNB/SOL hold was least negative at −82 USDT gross; this conclusion does not depend on the cost correction. |
| Relative-value sweep | 6 candidates | Long-alt/short-BTC variants became gross-positive. The selected seven-day ETH/BNB/SOL version earned +465 USDT gross; its former −325 USDT legacy net result is superseded below. |
| Relative-horizon extension | 4 candidates | Ten- to 28-day holds did not improve on seven days. |
| Fixed-candidate cost sensitivity | 6 cost scenarios, no strategy retuning | Monthly-corrected Abdi-Ranaldo returned +2.67% (+235 USDT); two-day-corrected returned −1.38% (−86 USDT). The registered fixed 20 bps case returned +1.54% (+176 USDT), fixed 40 bps lost 59 USDT, and break-even was 35.0 bps one way. |

The adaptive strategy search screened **24 candidate variants** before the cost sensitivity was run. This count, the full path, and the negative results matter: the final positive number is not an untouched test.

## What was found

The only somewhat useful research lead is a beta-reduced expression of the original economic idea:

- enter after the unchanged accelerating BTC-notional-dominance spike;
- hold an equal-weight ETH/BNB/SOL basket long and BTC short;
- allocate 50% gross to each leg, for 1x total gross exposure;
- hold for seven days;
- charge 10% annualised carry to the BTC short.

It is **cost conditional**, not deployable evidence. At 20 bps per dollar traded it made +175.56 USDT (+1.54%), with a 0.43 descriptive Sharpe and −2.09% maximum drawdown. Monthly-corrected Abdi-Ranaldo implies 14.9 bps one way and +234.96 USDT net; the paper's two-day correction implies 42.3 bps and −85.90 USDT net. The original holdout segment was still negative, only six entries occurred, and the largest winner represented 121% of total net PnL. A seeded seven-day block bootstrap produced a 95% return interval of −3.34% to +7.51%, which includes zero.

See the [candidate validation](strategy2_cost_sensitivity/results/2026-08-04/candidate_validation.md), [machine-readable validation](strategy2_cost_sensitivity/results/2026-08-04/candidate_validation.json), [effective-cost comparison](strategy2_cost_sensitivity/results/2026-08-04/effective_cost_comparison.png), and [cumulative cost-sensitivity figure](strategy2_cost_sensitivity/results/2026-08-04/cumulative_returns.png).

## Reproduce

Populate the frozen forward cache first, then run offline from the repository root:

```bash
python strategy2_variation_search.py --offline
python strategy2_basket_search.py --offline
python strategy2_holding_search.py --offline
python strategy2_relative_value_search.py --offline
python strategy2_relative_horizon_search.py --offline
python strategy2_cost_sensitivity.py --offline
python strategy2_candidate_validation.py
```

Result directories are non-overwritable. To reproduce an existing run, pass a fresh `--output-dir` to each search runner and fresh validation output paths, then compare the artifacts. The published directories were reproduced byte-for-byte in independent runs.

## Claim boundary and next evidence

The frozen Strategy 2 remains the canonical reported result. The relative-value candidate must accumulate at least **20 new entries from 5 August 2026 onward**, with 30 preferred, before its performance claim is reconsidered. Realised fees, slippage, BTC-short funding, margin, and liquidation constraints must be recorded; the candidate ceases to be economically positive above approximately 35 bps one-way cost in this sample.
