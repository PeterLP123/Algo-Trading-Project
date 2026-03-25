# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

Algorithmic trading research project (UCL COMP0051) analyzing cryptocurrency OHLCV data from Binance. The entire codebase lives in a single Jupyter notebook (`strategy.ipynb`) backed by Parquet data files.

## Running the Notebook

```bash
# Install dependencies
pip install pandas numpy matplotlib ccxt pyarrow jupyterlab

# Launch Jupyter
jupyter lab strategy.ipynb
```

Execute cells top-to-bottom. Cell 4 hits the Binance API and re-downloads ~15,760 bars per symbol — skip it if raw Parquet files already exist in `data/`.

## Data Layout

```
data/
  {SYMBOL}_1h.parquet              # raw OHLCV from Binance
  processed/binance/
    {SYMBOL}_1h_flagged.parquet    # raw + outlier flag columns added
```

Symbols: `BTCUSDT`, `ETHUSDT`, `DOGEUSDT`, `SOLUSDT`. Parquet files are written/read via PyArrow directly (not `pd.read_parquet`) to avoid Arrow extension-type registration conflicts.

## Architecture

The notebook implements a linear ETL pipeline:

1. **Fetch** — CCXT `fetch_ohlcv` against Binance, paginating from `SINCE = "2024-06-01"` in batches of 1000 bars with a 0.3 s rate-limit sleep.
2. **Audit** — `audit_ohlcv(df, symbol, stage)` checks duplicates, gaps, nulls, non-positive prices, negative/zero volume, High < Low, and OHLC candle-rule violations.
3. **Flag outliers** — `flag_ohlcv_outliers(df, rolling_window=72, zscore_threshold=5.0)` computes log returns, rolling z-scores, and appends `raw_return`, `return_zscore`, `is_outlier`, `clean_return` columns without modifying original candles.
4. **Save** — Flagged DataFrames written to `processed/binance/`.

Key configuration lives at the top of the fetch cell:
```python
SYMBOLS   = ["BTC/USDT", "ETH/USDT", "DOGE/USDT", "SOL/USDT"]
TIMEFRAME = "1h"
SINCE     = "2024-06-01"
DATA_DIR  = Path("data")
```

## No Test / Lint Infrastructure

There is no pytest, tox, black, flake8, or CI configuration. Validation is done inline via `audit_ohlcv` output.
