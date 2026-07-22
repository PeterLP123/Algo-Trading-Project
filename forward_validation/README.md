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
| Transaction costs | lagged 21-day Abdi-Ranaldo half-spread |

The runner verifies a canonical SHA-256 digest of the specification and offers no command-line model parameters. Changing the JSON causes a hard failure. There is no optimisation or fallback-selection path in `forward_validation.py`.

Strategy 2 is independently frozen in `frozen_strategy2.json`: entry percentile 0.80, 126-day lookback, exit z-score 0.0, seven-day maximum hold, three-day dominance change, 0.95 change percentile, no crash filter, and 2.0 gross utilisation. `strategy2_forward_validation.py` likewise exposes no parameter overrides. MATIC remains in the submitted universe but is unavailable after 10 September 2024; it is retained as missing rather than replaced by a new asset.

## Data boundary

The canonical Strategy 1 files under `data_final/` and the hashed local Strategy 2 cache must end exactly on 20 March 2026. They are read but never modified. Only later completed Binance daily candles are fetched and stored under the ignored `forward_validation/cache/` directory.

This preserves:

- the original signal history and moving-average state;
- the MVO covariance history;
- the original 10-day rebalance phase;
- position and transaction-cost continuity across the cutoff.

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
  --end-date 2026-07-21 \
  --output-dir forward_validation/snapshots/2026-07-21-corrected

python strategy2_forward_validation.py \
  --end-date 2026-07-21 \
  --output-dir forward_validation/snapshots/2026-07-21-corrected
```

After the forward cache has been populated, the same endpoint can be reproduced without network access:

```bash
python forward_validation.py \
  --end-date 2026-07-21 \
  --output-dir forward_validation/results \
  --offline

python strategy2_forward_validation.py \
  --end-date 2026-07-21 \
  --output-dir forward_validation/results/strategy2 \
  --offline
```

Each run writes:

- `summary.json`: frozen specification, specification digest, pre-freeze data hashes, run timestamp, and headline metrics;
- `daily.csv`: daily gross PnL, costs, net PnL, turnover, exposures, equity, and cumulative return.
- `cumulative_returns.png`: net strategy performance against BTC and an equal-weight asset basket.
- `strategy2_summary.json` and `strategy2_daily.csv`: the equivalent frozen Strategy 2 metrics and daily accounting.

The local cache and default results directory are ignored. Dated snapshots are intentionally eligible for review and version control.

## Interpretation

Forward observations are a new test, not a continuation of tuning. A short or favourable forward window does not validate the strategy by itself. Results should be reported with the exact endpoint and sample length, and unsuccessful snapshots should be retained rather than replaced or retuned.
