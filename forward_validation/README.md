# Frozen forward validation

These pipelines extend both submitted strategies beyond the original research cutoff without reopening model selection.

## Freeze contract

The specification in `frozen_strategy1.json` was fixed on 20 March 2026 from the walk-forward selection already reported in the notebook:

| Setting | Frozen value |
|---|---:|
| Moving-average window | 140 days |
| Volatility window | 30 days |
| Activation dead-zone | 1.0 |
| Rebalance interval | 10 days |
| Gross exposure cap | 10,000 USDT |
| Covariance estimator | Ledoit-Wolf, 120-day window |
| MVO risk aversion | 1.0 |
| Transaction costs | lagged 21-day monthly-corrected Abdi-Ranaldo half-spread |

The runner verifies a canonical SHA-256 digest of the specification and offers no command-line model parameters. Changing the JSON causes a hard failure. There is no optimisation or fallback-selection path in `forward_validation.py`.

Strategy 2 is independently frozen in `frozen_strategy2.json`: entry percentile 0.80, 126-day lookback, exit z-score 0.0, seven-day maximum hold, three-day dominance change, 0.95 change percentile, no crash filter, and 2.0 gross utilisation. `strategy2_forward_validation.py` likewise exposes no parameter overrides. MATIC remains in the submitted universe but is unavailable after 10 September 2024; it is retained as missing rather than replaced by a new asset.

## Strategy 2 entry-count evidence gate

Strategy 2 is episodic, so inactive daily observations do not create independent trade evidence.
The hash-verified rule in `strategy2_evidence_gate.json` uses qualifying entries in the continuous
post-selection window: 20 entries are the bare interpretive minimum and 30 are preferred. Below 20,
return and Sharpe are reported as descriptive economic outcomes only; the runner does not label
them reliable performance inference. The thresholds are not exposed as command-line overrides.

This rule was registered on 5 August 2026 after the 4 August snapshot had already produced seven
entries. It is therefore prospective from that point, not a claim of pre-registration before the
existing result. The runner also publishes a linear calendar-time planning estimate from the
observed entry rate, explicitly marked as neither a forecast nor a guarantee.

## Data boundary

The canonical Strategy 1 files under `data_final/` and the hashed local Strategy 2 cache must end exactly on 20 March 2026. They are read but never modified. Only later completed Binance daily candles are fetched and stored under the ignored `forward_validation/cache/` directory.

This preserves:

- the original signal history and moving-average state;
- the MVO covariance history;
- the original 10-day rebalance phase;
- position and transaction-cost continuity across the cutoff.

The 4 August 2026 snapshot was restated after a code audit found that negative 21-day spread
moments had previously been reflected above zero with an absolute value. Abdi and Ranaldo specify
flooring negative estimates at zero. The corrected snapshot changes costs only: the frozen signals,
parameters, positions, entries, turnover, and data boundary are unchanged. Each summary records the
resolved `monthly_corrected` method explicitly.

The current UTC day is rejected because its daily candle is incomplete.

## Run

From the repository root:

```bash
python forward_validation.py
python strategy2_forward_validation.py
```

The default end date is the latest completed UTC day. To create a dated, reviewable snapshot:

```bash
python forward_validation.py \
  --end-date 2026-08-04 \
  --output-dir forward_validation/snapshots/2026-08-04

python strategy2_forward_validation.py \
  --end-date 2026-08-04 \
  --output-dir forward_validation/snapshots/2026-08-04
```

After the forward cache has been populated, the same endpoint can be reproduced without network access:

```bash
python forward_validation.py \
  --end-date 2026-08-04 \
  --output-dir forward_validation/results \
  --offline

python strategy2_forward_validation.py \
  --end-date 2026-08-04 \
  --output-dir forward_validation/results/strategy2 \
  --offline
```

Each run writes:

- `summary.json`: frozen specification, specification digest, pre-freeze data hashes, run timestamp, forward metrics, and continuous post-selection metrics;
- `daily.csv`: daily gross PnL, costs, net PnL, turnover, exposures, equity, and cumulative return.
- `cumulative_returns.png`: net strategy performance against BTC and an equal-weight asset basket.
- `post_selection_daily.csv` and `post_selection_cumulative_returns.png`: Strategy 1 accounting and benchmarks from the original holdout start through the requested endpoint;
- `strategy2_summary.json` and `strategy2_daily.csv`: the equivalent frozen Strategy 2 forward metrics and daily accounting;
- `strategy2_post_selection_daily.csv`: Strategy 2 accounting from the original holdout start through the requested endpoint.

Strategy 2 summaries also embed the evidence-gate definition, its SHA-256 digest, the current gate
status, entries remaining to each threshold, and the non-binding planning estimate.

The local cache and default results directory are ignored. Dated snapshots are intentionally eligible for review and version control.

## Interpretation

Forward observations are a new test, not a continuation of tuning. The continuous post-selection view joins the original 126-observation holdout to all later frozen-strategy observations so return, drawdown, Sharpe, costs, and exposure can be assessed over one uninterrupted period. It is explicitly labelled post-selection evidence and does not replace the originally reported holdout.

Results should be reported with the exact endpoint and sample length. All annualised statistics retain the project's 252-observation convention for comparability; because crypto trades every day, 126 daily observations are about 4.1 calendar months. Short or favourable windows do not validate a strategy by themselves, and unsuccessful snapshots should be retained rather than replaced or retuned.

For Strategy 2, calendar length does not override the entry-count gate. Until at least 20 qualifying
entries are observed, risk-adjusted metrics must not be presented as reliable inferential evidence.
