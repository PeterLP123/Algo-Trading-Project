"""
grid_search.py — Walk-forward parameter grid search (Cell 27).

Function:
  run_grid_search(asset_panel, dev_index, wf_folds, cleaned_frames, symbols,
                  ma_grid, vol_grid, thresh_grid, rebal_grid,
                  stability_penalty, signal_clip_wf, gross_cap_wf, v0_wf,
                  backtest_use_excess, trading_days)
    Runs every parameter combination on the dev-only asset panel,
    evaluates mean/std Sharpe across WF folds, ranks by
    wf_score = mean_val_sharpe - stability_penalty × std_val_sharpe.
    Returns (wf_search_df, WF_BEST_MA, WF_BEST_VOL, WF_BEST_DZ, WF_BEST_REBAL).
"""

from itertools import product

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    display = print

from src.helpers import (
    build_signal_and_theta,
    compute_half_spread_frac,
    metrics_on_window,
    run_net_backtest,
)


def run_grid_search(
    asset_panel,
    dev_index,
    wf_folds,
    cleaned_frames,
    symbols,
    ma_grid,
    vol_grid,
    thresh_grid,
    rebal_grid,
    stability_penalty=0.1,
    signal_clip_wf=5.0,
    gross_cap_wf=50_000.0,
    v0_wf=50_000.0,
    backtest_use_excess=False,
    trading_days=252,
):
    """Walk-forward grid search over signal and sizing parameters.

    All evaluation is restricted to the development (train+CV) period.
    The holdout is never touched.

    Args:
        asset_panel        : full MultiIndex asset panel
        dev_index          : DatetimeIndex for development period
        wf_folds           : DataFrame of fold definitions from compute_splits()
        cleaned_frames     : dict mapping symbol → cleaned OHLCV DataFrame
        symbols            : list of asset strings
        ma_grid            : list of MA window values to try
        vol_grid           : list of vol window values to try
        thresh_grid        : list of dead-zone threshold values to try
        rebal_grid         : list of rebalance_every values to try
        stability_penalty  : penalises std of Sharpe across folds
        signal_clip_wf     : clip value for z in grid search
        gross_cap_wf       : gross notional budget used during search
        v0_wf              : initial capital for search backtests
        backtest_use_excess: use excess returns (default False)
        trading_days       : annualisation factor

    Returns:
        wf_search_df  : DataFrame of all grid results sorted by wf_score
        WF_BEST_MA    : best MA_WINDOW
        WF_BEST_VOL   : best VOL_WINDOW
        WF_BEST_DZ    : best DEAD_ZONE
        WF_BEST_REBAL : best rebalance_every
    """
    asset_panel_dev      = asset_panel.loc[dev_index].copy()
    half_spread_frac_dev = compute_half_spread_frac(asset_panel_dev, cleaned_frames, symbols)

    wf_rows = []
    for ma_w, vol_w, dz, rebal in product(ma_grid, vol_grid, thresh_grid, rebal_grid):
        theta_p = build_signal_and_theta(
            asset_panel_dev,
            symbols,
            ma_window=ma_w,
            vol_window=vol_w,
            dead_zone=dz,
            signal_clip=signal_clip_wf,
            gross_cap=gross_cap_wf,
            rebalance_every=rebal,
        )
        out_p = run_net_backtest(
            asset_panel_dev,
            theta_p,
            symbols,
            half_spread_frac_dev,
            backtest_use_excess=backtest_use_excess,
            v0=v0_wf,
        )
        net_pv = out_p["net_portfolio_value"]
        turn   = out_p["turnover"]

        fold_sharpes, fold_rets, fold_dds, fold_tos = [], [], [], []
        for _, row in wf_folds.iterrows():
            val_ix = dev_index[
                (dev_index >= row["val_start"]) & (dev_index <= row["val_end"])
            ]
            m = metrics_on_window(net_pv, turn, val_ix, trading_days=trading_days)
            fold_sharpes.append(m["sharpe"])
            fold_rets.append(m["total_return"])
            fold_dds.append(m["max_dd"])
            fold_tos.append(m["mean_turnover"])

        fs     = np.array(fold_sharpes, dtype=float)
        mean_s = float(np.nanmean(fs))
        std_s  = float(np.nanstd(fs, ddof=1)) if np.sum(np.isfinite(fs)) > 1 else 0.0
        score  = mean_s - stability_penalty * std_s

        wf_rows.append(
            {
                "MA_WINDOW":         ma_w,
                "VOL_WINDOW":        vol_w,
                "DEAD_ZONE":         dz,
                "rebalance_every":   rebal,
                "mean_val_sharpe":   mean_s,
                "std_val_sharpe":    std_s,
                "wf_score":          score,
                "mean_val_return":   float(np.nanmean(fold_rets)),
                "mean_val_max_dd":   float(np.nanmean(fold_dds)),
                "mean_val_turnover": float(np.nanmean(fold_tos)),
            }
        )

    wf_search_df = (
        pd.DataFrame(wf_rows)
        .sort_values("wf_score", ascending=False)
        .reset_index(drop=True)
    )
    display(wf_search_df.head(15))

    best        = wf_search_df.iloc[0]
    WF_BEST_MA    = int(best["MA_WINDOW"])
    WF_BEST_VOL   = int(best["VOL_WINDOW"])
    WF_BEST_DZ    = float(best["DEAD_ZONE"])
    WF_BEST_REBAL = int(best["rebalance_every"])
    print(
        f"Chosen (WF): MA={WF_BEST_MA}, VOL={WF_BEST_VOL}, DEAD_ZONE={WF_BEST_DZ}, "
        f"rebalance_every={WF_BEST_REBAL} | "
        f"mean val Sharpe={best['mean_val_sharpe']:.4f}, score={best['wf_score']:.4f}"
    )

    return wf_search_df, WF_BEST_MA, WF_BEST_VOL, WF_BEST_DZ, WF_BEST_REBAL
