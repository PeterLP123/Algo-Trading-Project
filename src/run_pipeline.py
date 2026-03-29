"""
run_pipeline.py — Full pipeline orchestrator for COMP0051 Algorithmic Trading.

Mirrors the cell-by-cell execution of strategy_final_v2.ipynb in the same
linear order. All parameters are read from config.py and passed explicitly
to each module function — no hidden global state.

Usage:
    # From the project root (one level above src/):
    python -m src.run_pipeline

    # Or skip data download (if parquet files already exist):
    python -m src.run_pipeline --skip-download

Outputs (written to output/ by default):
    daily_excess_returns.html/.pdf/.png
    trend_signals_vs_price.html/.pdf/.png
    net_vs_gross_cost_turnover.html/.pdf/.png
    wf_grid_search_heatmaps.html/.pdf/.png
    wf_grid_search_ranking.html/.pdf/.png
    oos_performance.html/.pdf/.png
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path when run as a script
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

# ── Project modules ───────────────────────────────────────────────────────────
import src.config as config
from src import (
    asset_panel as asset_panel_mod,
    backtest,
    data_cleaning,
    data_download,
    data_fred,
    excess_returns as excess_returns_mod,
    grid_search,
    grid_search_viz,
    holdout_evaluation,
    oos_plot,
    performance_metrics,
    position_sizing,
    signal_construction,
    signal_visualization,
    walk_forward_split,
)
from src.utils import display_df, section_header


def main(skip_download: bool = False):
    """Run the complete trading strategy pipeline.

    Args:
        skip_download : if True, skip Binance + FRED downloads and load
                        existing parquet files from DATA_DIR instead.
    """
    # ── Cell 1: load environment variables ───────────────────────────────────
    load_dotenv()

    # ── Cell 4: download OHLCV from Binance ──────────────────────────────────
    if not skip_download:
        section_header(1, "Downloading OHLCV data from Binance")
        data_download.download_ohlcv(
            symbols=config.SYMBOLS,
            timeframe=config.TIMEFRAME,
            since=config.SINCE,
            until=config.UNTIL,
            data_dir=config.DATA_DIR,
        )
    else:
        print("Skipping download — using existing parquet files.")

    # ── Cell 6: fetch Fed Funds Rate from FRED ───────────────────────────────
    if not skip_download:
        section_header(2, "Downloading Fed Funds Rate (FRED)")
        dff = data_fred.download_fred_rate(
            since=config.SINCE,
            until=config.UNTIL,
            data_dir=config.DATA_DIR,
        )
    else:
        import pandas as pd
        dff_path = config.DATA_DIR / "DFF_daily.parquet"
        _dff_df  = pd.read_parquet(dff_path)
        _dff_df.index = pd.to_datetime(_dff_df.index, utc=True)
        dff = _dff_df["fed_funds_rate"]
        print(f"Loaded DFF from {dff_path}: {len(dff):,} observations")

    print(dff.tail())

    # ── Cell 8: data quality checks and cleaning ─────────────────────────────
    section_header(3, "Data quality checks and cleaning")
    cleaned_frames, quality_report = data_cleaning.process_all_symbols(
        symbols=config.SYMBOLS,
        timeframe=config.TIMEFRAME,
        data_dir=config.DATA_DIR,
        value_columns=config.VALUE_COLUMNS,
        pandas_freq=config.PANDAS_FREQ,
    )
    display_df(quality_report)

    # ── Cell 11: excess returns ───────────────────────────────────────────────
    section_header(4, "Computing excess returns")
    cleaned_frames, excess_returns_df, rf_daily = excess_returns_mod.compute_excess_returns(
        cleaned_frames=cleaned_frames,
        dff=dff,
        symbols=config.SYMBOLS,
    )
    excess_returns_mod.plot_excess_returns(
        excess_returns_df=excess_returns_df,
        output_dir=config.OUTPUT_DIR,
    )
    display_df(excess_returns_df.describe().T[["mean", "std", "min", "max"]])

    # ── Cell 13: common asset panel ──────────────────────────────────────────
    section_header(5, "Building asset panel")
    asset_panel = asset_panel_mod.build_asset_panel(
        cleaned_frames=cleaned_frames,
        symbols=config.SYMBOLS,
    )
    display_df(asset_panel.head())

    # ── Cell 15: walk-forward validation split ───────────────────────────────
    section_header(6, "Walk-forward validation split")
    splits = walk_forward_split.compute_splits(
        asset_panel=asset_panel,
        holdout_frac=config.HOLDOUT_FRAC,
        initial_train_bars=config.INITIAL_TRAIN_BARS,
        val_bars=config.VAL_BARS,
        step_bars=config.STEP_BARS,
        wf_mode=config.WF_MODE,
        train_bars=config.TRAIN_BARS,
    )
    dev_index     = splits["dev_index"]
    holdout_index = splits["holdout_index"]
    wf_folds      = splits["wf_folds"]

    # ── Cell 18: trend signal construction ───────────────────────────────────
    section_header(7, "Trend signal construction")
    signal_panel = signal_construction.construct_signals(
        asset_panel=asset_panel,
        symbols=config.SYMBOLS,
        ma_window=config.MA_WINDOW,
        vol_window=config.VOL_WINDOW,
        dead_zone=config.DEAD_ZONE,
        signal_clip=config.SIGNAL_CLIP,
    )
    display_df(signal_panel.head())

    # ── Cell 19: signal visualization ────────────────────────────────────────
    print("  → Plotting trend signals vs price...")
    signal_visualization.plot_signals(
        close_px=asset_panel["close"],
        signal_panel=signal_panel,
        symbols=config.SYMBOLS,
        dead_zone=config.DEAD_ZONE,
        output_dir=config.OUTPUT_DIR,
    )

    # ── Cell 21: position sizing ─────────────────────────────────────────────
    section_header(8, "Position sizing (exposure mapping)")
    exposure_result = position_sizing.compute_exposures(
        signal_panel=signal_panel,
        symbols=config.SYMBOLS,
        dead_zone=config.DEAD_ZONE,
        gross_cap=config.GROSS_CAP,
    )
    exposure_panel = exposure_result["exposure_panel"]
    display_df(exposure_panel.tail())

    # ── Cell 23: backtest ────────────────────────────────────────────────────
    section_header(9, "Running backtest (gross + net P&L)")
    bt_result = backtest.run_baseline_backtest(
        asset_panel=asset_panel,
        exposure_panel=exposure_panel,
        cleaned_frames=cleaned_frames,
        symbols=config.SYMBOLS,
        gross_cap=config.GROSS_CAP,
        v0=config.V0,
        backtest_use_excess=config.BACKTEST_USE_EXCESS,
    )
    backtest.plot_backtest_results(
        results=bt_result,
        output_dir=config.OUTPUT_DIR,
    )
    net_portfolio_value = bt_result["net_portfolio_value"]
    theta_exec          = bt_result["theta_exec"]
    net_pnl             = bt_result["net_pnl"]

    # ── Cell 25: performance metrics ─────────────────────────────────────────
    section_header(10, "Performance metrics (IS / OOS)")
    perf_df = performance_metrics.compute_performance(
        net_portfolio_value=net_portfolio_value,
        theta_exec=theta_exec,
        net_pnl=net_pnl,
        dev_index=dev_index,
        holdout_index=holdout_index,
        v0=config.V0,
        trading_days=config.TRADING_DAYS,
    )

    # ── Cell 27: walk-forward grid search ────────────────────────────────────
    section_header(11, "Walk-forward parameter grid search")
    wf_search_df, WF_BEST_MA, WF_BEST_VOL, WF_BEST_DZ, WF_BEST_REBAL = \
        grid_search.run_grid_search(
            asset_panel=asset_panel,
            dev_index=dev_index,
            wf_folds=wf_folds,
            cleaned_frames=cleaned_frames,
            symbols=config.SYMBOLS,
            ma_grid=config.MA_GRID,
            vol_grid=config.VOL_GRID,
            thresh_grid=config.THRESH_GRID,
            rebal_grid=config.REBAL_GRID,
            stability_penalty=config.STABILITY_PENALTY,
            signal_clip_wf=config.SIGNAL_CLIP_WF,
            gross_cap_wf=config.GROSS_CAP,
            v0_wf=config.V0,
            backtest_use_excess=config.BACKTEST_USE_EXCESS,
            trading_days=config.TRADING_DAYS,
        )

    # ── Cell 28: grid search visualization ───────────────────────────────────
    print("  → Plotting grid search results...")
    grid_search_viz.plot_grid_search(
        wf_search_df=wf_search_df,
        thresh_grid=config.THRESH_GRID,
        rebal_grid=config.REBAL_GRID,
        output_dir=config.OUTPUT_DIR,
    )

    # ── Cell 30: final holdout evaluation ────────────────────────────────────
    section_header(12, "Final holdout evaluation (frozen WF parameters)")
    holdout_result = holdout_evaluation.evaluate_holdout(
        asset_panel=asset_panel,
        cleaned_frames=cleaned_frames,
        symbols=config.SYMBOLS,
        holdout_index=holdout_index,
        wf_best_ma=WF_BEST_MA,
        wf_best_vol=WF_BEST_VOL,
        wf_best_dz=WF_BEST_DZ,
        wf_best_rebal=WF_BEST_REBAL,
        signal_clip_wf=config.SIGNAL_CLIP_WF,
        gross_cap_wf=config.GROSS_CAP,
        v0_wf=config.V0,
        backtest_use_excess=config.BACKTEST_USE_EXCESS,
        trading_days=config.TRADING_DAYS,
    )

    # ── Cell 31: OOS performance plot ────────────────────────────────────────
    print("  → Plotting OOS performance...")
    oos_plot.plot_oos_performance(
        net_pv_h=holdout_result["net_pv_h"],
        holdout_index=holdout_index,
        asset_panel=asset_panel,
        v0=config.V0,
        output_dir=config.OUTPUT_DIR,
    )

    print(f"\n{'=' * 60}\nPipeline complete. Output saved to: {config.OUTPUT_DIR.resolve()}\n{'=' * 60}")

    return {
        "cleaned_frames":    cleaned_frames,
        "asset_panel":       asset_panel,
        "splits":            splits,
        "signal_panel":      signal_panel,
        "exposure_panel":    exposure_panel,
        "bt_result":         bt_result,
        "perf_df":           perf_df,
        "wf_search_df":      wf_search_df,
        "wf_best": {
            "MA":    WF_BEST_MA,
            "VOL":   WF_BEST_VOL,
            "DZ":    WF_BEST_DZ,
            "REBAL": WF_BEST_REBAL,
        },
        "holdout_result": holdout_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COMP0051 trading strategy pipeline"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Binance and FRED downloads; use existing parquet files",
    )
    args = parser.parse_args()
    main(skip_download=args.skip_download)
