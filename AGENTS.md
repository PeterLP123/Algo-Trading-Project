# AGENTS.md

Guidance for coding agents working in this repository.

## Project overview

This UCL COMP0051 research project compares two daily cryptocurrency strategies:

1. multi-asset trend following with walk-forward parameter selection, covariance-aware sizing, and an untouched holdout;
2. BTC-dominance mean reversion across an altcoin basket, tuned with Optuna.

Both strategies use Binance OHLCV data and an Abdi-Ranaldo transaction-cost proxy. The risk-free series is the FRED Effective Federal Funds Rate.

## Canonical files

- `notebooks/strategy_analysis.ipynb` - end-to-end research notebook and saved results.
- `systematic_crypto/data.py` - market-data normalization, storage, and OHLCV audits.
- `systematic_crypto/signals.py` - signal, regime-filter, and target-weight construction.
- `systematic_crypto/execution.py` - portfolio execution and strategy orchestration.
- `systematic_crypto/costs.py` - spread estimation and cost-sensitivity analysis.
- `systematic_crypto/evaluation.py` - diagnostics, performance summaries, sweeps, and walk-forward evaluation.
- `strategy_helpers.py` - backward-compatible facade for the notebook's historical imports.
- `wf_trend_pipeline.py` - Strategy 1 signal, sizing, backtest, and metrics.
- `strategy2_pipeline.py` - Strategy 2 features, execution, and evaluation.
- `recompute_gross_turnover.py` - reproducibility check for IS/OOS PnL and turnover.
- `report/final_report.tex` and `report/final_report.pdf` - final report source and compiled artifact.

## Running the project

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab notebooks/strategy_analysis.ipynb
```

Run Jupyter from the repository root because the notebook imports the root-level pipeline modules and uses root-relative `data_final/` and `output/` paths.

## Data and secrets

Data, caches, Optuna databases, generated output, and `.env` are intentionally ignored. Do not commit them. A live FRED refresh requires `FRED_API_KEY`; use `.env.example` as the template. The notebook is cache-first by default, so a clean clone requires the documented refresh flags or local cached inputs.

## Validation

```bash
python -m pytest -q
python -m compileall -q strategy_helpers.py systematic_crypto strategy2_pipeline.py wf_trend_pipeline.py recompute_gross_turnover.py
cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error final_report.tex
```

Notebook execution is data- and network-dependent. Preserve its final outputs unless intentionally recomputing the full experiment. Never turn a failed or incomplete rerun into the canonical notebook.
