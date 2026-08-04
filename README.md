# Systematic Cryptocurrency Research

[![CI](https://github.com/PeterLP123/systematic-crypto-research/actions/workflows/ci.yml/badge.svg)](https://github.com/PeterLP123/systematic-crypto-research/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-research-F37626?logo=jupyter&logoColor=white)
[![License: MIT](https://img.shields.io/badge/Code%20license-MIT-2ea44f.svg)](LICENSE)

A cost-aware empirical comparison of two systematic cryptocurrency strategies: multi-asset trend following and BTC-dominance mean reversion. The project uses walk-forward model selection, an untouched holdout, realistic turnover accounting, and the Abdi-Ranaldo spread estimator.

[Read the final report](report/final_report.pdf) · [Explore the research notebook](notebooks/strategy_analysis.ipynb)

## Portfolio takeaway

This repository demonstrates an end-to-end quantitative research workflow: audited market data,
leakage-aware model selection, reusable Python backtest pipelines, realistic cost accounting,
deterministic tests, and parameter-frozen forward monitoring. The result is deliberately
asymmetric: trend following held up in the original holdout and first forward window, while
BTC-dominance mean reversion failed; the failed strategy is preserved rather than retuned.

![Original backtest extended with frozen Strategy 1 and Strategy 2 forward validation](report/figures/readme_performance_overview.png)

The original cumulative-return paths are extended through a separately shaded frozen-forward
period. Both strategies retain their submitted specifications; the added observations are not part
of model selection or the original holdout.

## Research question

Can two economically distinct crypto signals retain useful risk-adjusted performance after stable parameter selection, realistic transaction costs, and an untouched out-of-sample test?

- **Strategy 1 - trend following:** volatility-normalised moving-average signals across BTC, ETH, BNB, and ADA, with Ledoit-Wolf covariance estimation and constrained mean-variance sizing.
- **Strategy 2 - BTC-dominance mean reversion:** an equal-weight altcoin basket entered after extreme increases in BTC's share of aggregate Binance notional volume.

The raw sample contains 2,271 daily candles from 1 January 2020 to 20 March 2026; the return-aligned strategy panel begins on 2 January. Strategy parameters are selected before evaluating the final 126 daily observations.

## Main results

| Metric | Strategy 1 IS | Strategy 1 OOS | Strategy 2 IS | Strategy 2 OOS |
|---|---:|---:|---:|---:|
| Sharpe ratio | 0.923 | **1.300** | 0.343 | **-2.912** |
| Annualised return | 18.38% | **12.56%** | 6.30% | **-14.00%** |
| Maximum drawdown | -22.3% | **-3.8%** | -59.9% | **-7.3%** |
| Net PnL | $31,903 | **$2,431** | $6,818 | **-$1,222** |
| Active days | 93.5% | **100.0%** | 9.0% | **2.4%** |

Strategy 1 retained positive holdout performance and shallow drawdown, although its holdout exposure was heavily concentrated in BTC. Strategy 2 failed its holdout: only two out-of-sample trades occurred, so the negative result is both economically important and statistically weak. The repository preserves that result rather than retuning after seeing the holdout.

## Frozen forward validation

Both strategies are now monitored after the original 20 March 2026 cutoff using their selected parameters exactly as reported—no new search, tuning, or fallback selection is allowed. The runners verify hashed specifications and pre-cutoff histories, fetch only later completed Binance daily candles, and preserve signal, position, and transaction-cost state across the boundary.

```bash
python forward_validation.py
python strategy2_forward_validation.py
```

See [`forward_validation/README.md`](forward_validation/README.md) for the freeze contract, dated-snapshot workflow, and artifact schema. New results are explicitly labelled forward validation and do not replace the original holdout.

The corrected snapshot covers 123 completed days through 21 July 2026. Strategy 1 returned **+3.52%** with a **1.109 Sharpe**, **−4.26% maximum drawdown**, and **+1,565 USDT net PnL**. Strategy 2 returned **−10.36%** with a **−1.850 Sharpe**, **−10.73% maximum drawdown**, and **−1,563 USDT net PnL** across four entries. See the [dated snapshot](forward_validation/snapshots/2026-07-21-corrected/README.md) and its machine-readable accounting. The window remains too short to establish durable performance for either strategy.

## Methodology

1. Download daily Binance OHLCV data with `ccxt` and audit candle integrity.
2. Align asset returns with the FRED Effective Federal Funds Rate.
3. Reserve an untouched 126-day holdout before model selection.
4. Select Strategy 1 parameters with expanding walk-forward folds and a stability-penalised Sharpe score.
5. Tune Strategy 2 with Optuna's TPE sampler on the development sample only.
6. Apply lagged Abdi-Ranaldo half-spread estimates to traded notional.
7. Report gross and net PnL, turnover, drawdown, activity, and holdout performance.

## Research engineering

- **22 deterministic tests** cover time alignment, transaction costs, signal construction,
  immutable strategy specifications, cutoff boundaries, and snapshot self-consistency.
- **Continuous integration** compiles every research module and runs the complete offline test
  suite on each push and pull request.
- **Modular research architecture** separates data validation, signal construction, execution,
  transaction costs, and evaluation behind stable public imports.
- **Frozen specifications** are hash-verified, expose no tuning overrides, and reject modified
  pre-cutoff histories or incomplete candles.
- **Immutable evidence** is published as dated, non-overwritable snapshots with daily accounting,
  machine-readable summaries, and reproducible figures.

## Repository structure

```text
.
├── notebooks/
│   └── strategy_analysis.ipynb      # complete research workflow and saved outputs
├── report/
│   ├── build_readme_performance_figure.py # reproducible README overview
│   ├── final_report.pdf             # compiled six-page report
│   ├── final_report.tex             # report source
│   ├── LICENSE.md                    # rights for research materials
│   ├── references.bib
│   └── figures/                     # curated report and README figures
├── systematic_crypto/
│   ├── data.py                       # market-data storage, normalization, and audits
│   ├── signals.py                    # signals, regime filters, and target weights
│   ├── execution.py                  # execution engine and strategy orchestration
│   ├── costs.py                      # spread estimation and cost sensitivity
│   └── evaluation.py                 # diagnostics, metrics, sweeps, and walk-forward tests
├── tests/
│   ├── test_pipelines.py            # deterministic pipeline smoke tests
│   ├── test_forward_validation.py   # freeze-boundary and no-retuning checks
│   └── test_module_boundaries.py    # package ownership and legacy-import compatibility
├── forward_validation/
│   ├── frozen_strategy1.json        # hashed selected specification
│   ├── frozen_strategy2.json        # hashed Strategy 2 specification
│   └── README.md                    # forward-test protocol and commands
├── forward_validation.py            # post-2026-03-20 Strategy 1 runner
├── strategy2_forward_validation.py  # post-2026-03-20 Strategy 2 runner
├── strategy_helpers.py              # compatibility facade for existing notebook imports
├── wf_trend_pipeline.py             # Strategy 1 implementation
├── strategy2_pipeline.py            # Strategy 2 implementation
├── recompute_gross_turnover.py      # independent PnL/turnover reconciliation
├── requirements.txt
└── CITATION.cff
```

Raw data, API credentials, Optuna databases, caches, and generated notebook output are intentionally excluded from Git.

The package boundaries and allowed dependency direction are documented in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Existing imports from
`strategy_helpers` remain supported, while new code should import from the focused
`systematic_crypto` modules.

## Setup

Python 3.12 is recommended.

```bash
git clone https://github.com/PeterLP123/systematic-crypto-research.git
cd systematic-crypto-research

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cp .env.example .env
jupyter lab notebooks/strategy_analysis.ipynb
```

The notebook is cache-first so a presentation rerun does not silently replace the experiment. For a clean-clone refresh:

1. add a [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) to `.env`;
2. set the relevant `REFRESH_S1_DATA`, `REFRESH_FRED_DATA`, and `REFRESH_S2_CACHE` flags in the first code cell;
3. run Jupyter from the repository root so module imports and data paths resolve correctly.

Binance and FRED data are not redistributed in this repository. Availability and access restrictions can vary by location.

## Validation

Run the deterministic checks without downloading market data:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Rebuild the README performance overview and GitHub social preview from the committed historical
figure and frozen snapshot:

```bash
python report/build_readme_performance_figure.py
```

Build the report with a TeX Live installation:

```bash
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error final_report.tex
```

The full notebook is substantially slower and requires either the local caches or live data refreshes described above.

## Limitations

- This is a historical coursework study, not a live trading system.
- Binance notional-volume share is a proxy for BTC dominance, not total-market-cap dominance.
- The final holdout is short, and Strategy 2 produces only two holdout trades.
- The 123-day forward window is also short; Strategy 2 entered only four times.
- Fees, market impact, funding, borrow constraints, taxes, and operational risk are not modelled in full.
- Strategy 1's positive holdout result is concentrated in BTC and should not be interpreted as broad cross-asset validation.

## Report and citation

The final UCL COMP0051 report is available as both [PDF](report/final_report.pdf) and [LaTeX source](report/final_report.tex). Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## Licensing

The software source code is available under the [MIT License](LICENSE). The written report,
notebook narrative, figures, and published research outputs are provided for viewing and citation
but are not covered by the software licence; see [`report/LICENSE.md`](report/LICENSE.md).

## Disclaimer

This repository is for educational and research purposes only. It is not investment advice, and the reported backtests do not represent live performance.
