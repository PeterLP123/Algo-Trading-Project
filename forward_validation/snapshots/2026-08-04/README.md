# Frozen-strategy evidence snapshot — 4 August 2026

This snapshot runs both submitted specifications unchanged through the latest completed UTC daily candle available when the evidence was refreshed. No parameter was re-estimated, replaced, or selected using these observations.

Two windows are reported separately:

- **Frozen forward:** 137 daily observations from 21 March to 4 August 2026, after the formal 20 March freeze.
- **Continuous post-selection:** 263 daily observations from 15 November 2025 to 4 August 2026, covering the original 126-observation holdout dates and all 137 later observations. This longer view is post-selection evidence, not a new untouched holdout.

The continuous metrics come directly from one stateful run of the hashed frozen evidence pipeline, anchored to 14 November 2025 equity. They are not constructed by adding the rounded April report table to the later forward totals; the original submitted table remains a separate historical result.

## Continuous post-selection results

| Metric | Strategy 1 | Strategy 2 |
|---|---:|---:|
| Net return from pre-window equity | **+10.06%** | **−19.44%** |
| Annualised return | +9.62% | −18.71% |
| Sharpe ratio | **1.192** | **−2.231** |
| Sortino ratio | 1.792 | −1.127 |
| Maximum drawdown | **−4.96%** | **−19.44%** |
| Gross PnL | +4,562.73 USDT | −1,365.44 USDT |
| Transaction costs | −335.64 USDT | −1,781.91 USDT |
| Net PnL | **+4,227.09 USDT** | **−3,147.34 USDT** |
| Total turnover | 90,823.42 USDT | 282,098.57 USDT |
| Mean gross exposure | 9,952.55 USDT | 1,064.64 USDT |
| Activity | 100.0% of days | 7 entries / 14 active days |

Strategy 1 remained profitable while BTC buy-and-hold returned −32.23% and the equal-weight four-asset basket returned −42.29% over the same daily observations. Its average absolute exposure remained concentrated in BTC at 7,718.46 USDT of 9,952.55 USDT mean gross exposure.

Strategy 2 lost money before costs and incurred 1,781.91 USDT of estimated transaction costs. Seven entries and 14 active days remain too sparse for a precise performance estimate, but the observed economic result is negative and is retained without retuning.

## Frozen-forward update

| Metric | Strategy 1 | Strategy 2 |
|---|---:|---:|
| Net return from cutoff equity | **+4.04%** | **−13.57%** |
| Sharpe ratio | **1.167** | **−2.224** |
| Maximum drawdown | **−4.26%** | **−13.57%** |
| Net PnL | **+1,795.73 USDT** | **−2,047.54 USDT** |
| Activity | 100.0% of days | 5 entries / 11 active days |

All annualised statistics use the project's original 252-observation convention for comparability. Crypto trades seven days per week, so 126 daily observations are about 4.1 calendar months and the 263-observation continuous period is about 8.6 calendar months.

## Artifacts

- [`summary.json`](summary.json), [`daily.csv`](daily.csv), and [`cumulative_returns.png`](cumulative_returns.png): Strategy 1 forward provenance, accounting, and benchmarks;
- [`post_selection_daily.csv`](post_selection_daily.csv) and [`post_selection_cumulative_returns.png`](post_selection_cumulative_returns.png): Strategy 1 continuous post-selection accounting and benchmarks;
- [`strategy2_summary.json`](strategy2_summary.json) and [`strategy2_daily.csv`](strategy2_daily.csv): Strategy 2 forward provenance and accounting;
- [`strategy2_post_selection_daily.csv`](strategy2_post_selection_daily.csv): Strategy 2 continuous post-selection accounting.

The original submitted holdout result remains separately visible in the report and root README. Combining later observations improves the evidence base but does not remove selection bias or establish live performance.
