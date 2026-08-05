# Strategy 2 candidate validation

**Verdict: CAUTION — cost-conditional descriptive candidate, not a validated or deployable strategy.**

The fixed candidate is 50% long an equal-weight ETH/BNB/SOL basket and 50% short BTC, entered after the original dominance-spike signal, held seven days at 1x total gross exposure. A 10% annual short-carry stress is charged.

## Observed result

At the registered 20 bps one-way execution scenario, the 263-day combined post-selection result is +1.54% (+175.56 USDT net), Sharpe 0.43, max drawdown -2.09%, with 6 entries and 40 active days.

With the paper's monthly correction, the lagged Abdi–Ranaldo estimate gives +234.96 USDT (+2.67%); with the paper's two-day correction it gives -85.90 USDT (-1.38%). The observed break-even fixed one-way charge is 35.0 bps.

## Uncertainty and concentration

The 7-day circular block bootstrap gives a 95% total-return interval of [-3.34%, +7.51%] and P(return > 0) = 70.6%. With 14-day blocks the interval is [-2.99%, +7.01%] and P(return > 0) = 70.5%.

Only 6 trades exist; 50% are positive. The largest winner represents 121.2% of total net PnL, and removing it changes net PnL to -37.13 USDT.

These intervals do not correct the adaptive search. At least 24 candidate variants were screened, and a Deflated Sharpe Ratio is deliberately not reported because six entries do not support a stable correction.

## Fallacy scan

11/11 statistical fallacies checked. Material flags are: regression-to-the-mean risk in the extreme-dominance trigger; survivorship/availability bias in the selected liquid basket; strong look-elsewhere and forking-path effects; temporal heterogeneity (the original holdout is negative while the later frozen-forward segment is positive); and endogeneity of the notional-share proxy. No causal claim is made.

## Evidence gate

Do not upgrade the claim before at least 20 new entries from 5 August 2026 onward (30 preferred), using the fixed candidate and recording realised execution costs. The frozen published Strategy 2 remains unchanged.
