"""
grid_search.py - Walk-forward parameter grid search (Cell 27).

Function:
  run_grid_search(asset_panel, dev_index, wf_folds, cleaned_frames, symbols,
                  ma_grid, vol_grid, thresh_grid, rebal_grid,
                  stability_penalty, signal_clip_wf, gross_cap_wf, v0_wf,
                  backtest_use_excess, trading_days)
    Runs every parameter combination on the dev-only asset panel,
    evaluates mean/std Sharpe across WF folds, filters robust candidates,
    and selects the winner with a turnover-aware tie break.
"""

from itertools import product

import numpy as np
import pandas as pd

from src.helpers import (
    build_signal_and_theta,
    compute_half_spread_frac,
    metrics_on_window,
    run_net_backtest,
)
from src.utils import display_df


def _within_tie_band(series: pd.Series, best_value: float, tie_band: float) -> pd.Series:
    if not np.isfinite(best_value):
        return pd.Series(False, index=series.index)
    threshold = best_value - abs(best_value) * tie_band
    return series >= threshold


def _nanmean_or_nan(values) -> float:
    arr = np.array(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if len(finite) else np.nan


def _select_best_row(
    wf_search_df: pd.DataFrame,
    *,
    tie_band: float,
    default_candidate: dict[str, float | int],
) -> tuple[pd.Series, str]:
    candidates = wf_search_df[wf_search_df["is_candidate"]].copy()
    if not candidates.empty:
        best_score = float(candidates["wf_score"].max())
        finalists = candidates[_within_tie_band(candidates["wf_score"], best_score, tie_band)].copy()
        finalists["_abs_mean_val_max_dd"] = finalists["mean_val_max_dd"].abs()
        chosen = finalists.sort_values(
            ["mean_val_turnover", "mean_val_return", "_abs_mean_val_max_dd"],
            ascending=[True, False, True],
        ).iloc[0]
        return chosen, "candidate"

    fallback_mask = (
        (wf_search_df["MA_WINDOW"] == int(default_candidate["MA_WINDOW"]))
        & (wf_search_df["VOL_WINDOW"] == int(default_candidate["VOL_WINDOW"]))
        & (wf_search_df["DEAD_ZONE"] == float(default_candidate["DEAD_ZONE"]))
        & (wf_search_df["rebalance_every"] == int(default_candidate["rebalance_every"]))
    )
    if fallback_mask.any():
        return wf_search_df.loc[fallback_mask].iloc[0], "default_fallback"

    fallback = wf_search_df.sort_values(
        ["wf_score", "mean_val_turnover", "mean_val_return", "mean_val_max_dd"],
        ascending=[False, True, False, False],
    ).iloc[0]
    return fallback, "best_available_fallback"


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
    max_mean_val_turnover=4_000.0,
    min_positive_fold_share=0.55,
    wf_tie_band=0.05,
    default_ma=120,
    default_vol=40,
    default_dz=1.25,
    default_rebal=10,
):
    """Walk-forward grid search over signal and sizing parameters."""
    asset_panel_dev = asset_panel.loc[dev_index].copy()
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
        turn = out_p["turnover"]
        gross_pnl = out_p["gross_pnl"]
        cost_t = out_p["cost_t"]
        theta_exec = out_p["theta_exec"]

        fold_sharpes, fold_rets, fold_dds = [], [], []
        fold_tos, fold_cost_ratios, fold_active_days = [], [], []

        for _, row in wf_folds.iterrows():
            val_ix = dev_index[(dev_index >= row["val_start"]) & (dev_index <= row["val_end"])]
            m = metrics_on_window(
                net_pv,
                turn,
                val_ix,
                trading_days=trading_days,
                gross_pnl=gross_pnl,
                cost_t=cost_t,
                theta_exec=theta_exec,
            )
            fold_sharpes.append(m["sharpe"])
            fold_rets.append(m["total_return"])
            fold_dds.append(m["max_dd"])
            fold_tos.append(m["mean_turnover"])
            fold_cost_ratios.append(m["cost_to_gross_ratio"])
            fold_active_days.append(m["active_days_pct"])

        fs = np.array(fold_sharpes, dtype=float)
        finite_fs = fs[np.isfinite(fs)]
        mean_s = float(np.mean(finite_fs)) if len(finite_fs) else np.nan
        std_s = float(np.std(finite_fs, ddof=1)) if len(finite_fs) > 1 else 0.0
        score = mean_s - stability_penalty * std_s if np.isfinite(mean_s) else np.nan
        positive_fold_share = float(np.mean(finite_fs > 0)) if len(finite_fs) else 0.0
        mean_turnover = _nanmean_or_nan(fold_tos)
        is_candidate = bool(
            np.isfinite(mean_s)
            and mean_s > 0
            and positive_fold_share >= min_positive_fold_share
            and np.isfinite(mean_turnover)
            and mean_turnover <= max_mean_val_turnover
        )

        wf_rows.append(
            {
                "MA_WINDOW": ma_w,
                "VOL_WINDOW": vol_w,
                "DEAD_ZONE": dz,
                "rebalance_every": rebal,
                "mean_val_sharpe": mean_s,
                "std_val_sharpe": std_s,
                "wf_score": score,
                "mean_val_return": _nanmean_or_nan(fold_rets),
                "mean_val_max_dd": _nanmean_or_nan(fold_dds),
                "mean_val_turnover": mean_turnover,
                "positive_fold_share": positive_fold_share,
                "cost_to_gross_ratio": _nanmean_or_nan(fold_cost_ratios),
                "active_days_pct": _nanmean_or_nan(fold_active_days),
                "is_candidate": is_candidate,
            }
        )

    wf_search_df = pd.DataFrame(wf_rows)
    default_candidate = {
        "MA_WINDOW": default_ma,
        "VOL_WINDOW": default_vol,
        "DEAD_ZONE": default_dz,
        "rebalance_every": default_rebal,
    }
    best, selection_mode = _select_best_row(
        wf_search_df,
        tie_band=wf_tie_band,
        default_candidate=default_candidate,
    )

    wf_search_df["is_selected"] = False
    wf_search_df.loc[best.name, "is_selected"] = True
    wf_search_df = wf_search_df.sort_values(
        ["is_selected", "is_candidate", "wf_score", "mean_val_turnover", "mean_val_return"],
        ascending=[False, False, False, True, False],
    ).reset_index(drop=True)

    display_df(wf_search_df.head(15))

    wf_best_ma = int(best["MA_WINDOW"])
    wf_best_vol = int(best["VOL_WINDOW"])
    wf_best_dz = float(best["DEAD_ZONE"])
    wf_best_rebal = int(best["rebalance_every"])
    print(
        f"Chosen (WF/{selection_mode}): MA={wf_best_ma}, VOL={wf_best_vol}, "
        f"DEAD_ZONE={wf_best_dz}, rebalance_every={wf_best_rebal} | "
        f"mean val Sharpe={best['mean_val_sharpe']:.4f}, score={best['wf_score']:.4f}, "
        f"turnover={best['mean_val_turnover']:.2f}"
    )

    return wf_search_df, wf_best_ma, wf_best_vol, wf_best_dz, wf_best_rebal
