# Frozen forward snapshot — 21 July 2026

This corrected snapshot extends both submitted strategies from the 20 March 2026 cutoff through 123 completed Binance daily candles. No parameter was re-estimated or replaced.

The correction aligns Strategy 1's full-history index with the submitted notebook's 2 January 2020 start. The earlier snapshot began one day too early, shifting the fixed 10-day rebalance phase; it remains available as a superseded audit artifact.

| Metric | Strategy 1 | Strategy 2 |
|---|---:|---:|
| Window | 21 Mar – 21 Jul 2026 | 21 Mar – 21 Jul 2026 |
| Net return from cutoff equity | **+3.52%** | **−10.36%** |
| Sharpe ratio | **1.109** | **−1.850** |
| Maximum drawdown | **−4.26%** | **−10.73%** |
| Gross PnL | +1,804.96 USDT | −450.80 USDT |
| Transaction costs | −240.07 USDT | −1,111.95 USDT |
| Net PnL | **+1,564.88 USDT** | **−1,562.74 USDT** |
| Activity | 100% of days | 4 entries / 9 active days |

Strategy 1 remained profitable despite BTC returning −5.61% and the equal-weight four-asset basket returning −15.34%. Strategy 2's four episodic trades lost money before costs and incurred substantial turnover costs; that negative result is retained without retuning.

Artifacts:

- [`summary.json`](summary.json), [`daily.csv`](daily.csv), and [`cumulative_returns.png`](cumulative_returns.png): Strategy 1 provenance, accounting, and benchmarks;
- [`strategy2_summary.json`](strategy2_summary.json) and [`strategy2_daily.csv`](strategy2_daily.csv): Strategy 2 provenance and accounting.

This is still a short forward window. It strengthens the evidence trail, not a claim of durable live performance.
