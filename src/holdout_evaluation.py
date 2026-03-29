"""
holdout_evaluation.py — Final holdout (OOS) evaluation with frozen parameters (Cell 30).

Function:
  evaluate_holdout(asset_panel, cleaned_frames, symbols, holdout_index,
                   wf_best_ma, wf_best_vol, wf_best_dz, wf_best_rebal,
                   signal_clip_wf, gross_cap_wf, v0_wf, backtest_use_excess,
                   trading_days)
    Runs build_signal_and_theta + run_net_backtest on the FULL sample with
    frozen WF-tuned parameters, then computes metrics on holdout_index only.
    Returns a dict with the holdout results and metrics DataFrame.
"""

import pandas as pd

from src.helpers import build_signal_and_theta, compute_half_spread_frac, run_net_backtest
from src.performance_metrics import calmar_ratio, max_dd, sharpe_ratio, sortino_ratio
from src.utils import display_df


def evaluate_holdout(
    asset_panel,
    cleaned_frames,
    symbols,
    holdout_index,
    wf_best_ma,
    wf_best_vol,
    wf_best_dz,
    wf_best_rebal,
    signal_clip_wf=5.0,
    gross_cap_wf=50_000.0,
    v0_wf=50_000.0,
    backtest_use_excess=False,
    trading_days=252,
):
    """Run the strategy on the full sample with frozen WF parameters.

    Parameters are frozen to the walk-forward winner. The strategy is
    re-estimated on the full sample index (so rolling windows are causal
    up to each holdout date), but metrics are reported ONLY on holdout_index.

    Args:
        asset_panel        : full MultiIndex asset panel
        cleaned_frames     : dict mapping symbol → cleaned OHLCV DataFrame
        symbols            : list of asset strings
        holdout_index      : DatetimeIndex for the final test period
        wf_best_ma         : best MA_WINDOW from grid search
        wf_best_vol        : best VOL_WINDOW from grid search
        wf_best_dz         : best DEAD_ZONE from grid search
        wf_best_rebal      : best rebalance_every from grid search
        signal_clip_wf     : clip value for z (same as used in grid search)
        gross_cap_wf       : gross notional budget
        v0_wf              : initial capital
        backtest_use_excess: use excess returns (default False)
        trading_days       : annualisation factor

    Returns:
        dict with keys:
          out_hold          : full run_net_backtest output dict
          net_pv_h          : pd.Series of net portfolio value (full sample)
          turn_h            : pd.Series of turnover (full sample)
          net_pnl_h         : pd.Series of net P&L (full sample)
          eq_oos            : net portfolio value restricted to holdout_index
          r_oos             : daily returns on holdout equity
          holdout_metrics   : dict of scalar metrics
          holdout_metrics_df: pd.DataFrame (1 row × metrics)
    """
    half_spread_frac_full = compute_half_spread_frac(asset_panel, cleaned_frames, symbols)

    theta_holdout_run = build_signal_and_theta(
        asset_panel,
        symbols,
        ma_window=wf_best_ma,
        vol_window=wf_best_vol,
        dead_zone=wf_best_dz,
        signal_clip=signal_clip_wf,
        gross_cap=gross_cap_wf,
        rebalance_every=wf_best_rebal,
    )
    out_hold = run_net_backtest(
        asset_panel,
        theta_holdout_run,
        symbols,
        half_spread_frac_full,
        backtest_use_excess=backtest_use_excess,
        v0=v0_wf,
    )

    net_pv_h  = out_hold["net_portfolio_value"]
    turn_h    = out_hold["turnover"]
    net_pnl_h = out_hold["net_pnl"]

    print(f"Execution diagnostic uses rebalance_every={wf_best_rebal}")
    _ = run_net_backtest(
        asset_panel,
        theta_holdout_run,
        symbols,
        half_spread_frac_full,
        backtest_use_excess=backtest_use_excess,
        v0=v0_wf,
        print_diagnostic=True,
        diagnostic_window=6,
    )

    eq_oos = net_pv_h.reindex(holdout_index).dropna()
    r_oos  = eq_oos.pct_change().dropna()

    holdout_metrics = {
        "n_days":           len(eq_oos),
        "sharpe":           sharpe_ratio(r_oos, trading_days),
        "sortino":          sortino_ratio(r_oos, trading_days=trading_days),
        "calmar":           calmar_ratio(r_oos, eq_oos, trading_days),
        "max_drawdown":     max_dd(eq_oos),
        "mean_turnover":    float(turn_h.reindex(holdout_index).mean()),
        "total_net_pnl":    float(net_pnl_h.reindex(holdout_index).sum()),
        "pct_return_on_V0": float(net_pnl_h.reindex(holdout_index).sum() / v0_wf),
        "final_net_equity": float(eq_oos.iloc[-1]) if len(eq_oos) else float("nan"),
    }

    holdout_metrics_df = pd.DataFrame([holdout_metrics], index=["Holdout (final OOS)"])
    display_df(holdout_metrics_df.T)
    print("Holdout period:", holdout_index.min(), "→", holdout_index.max())

    return {
        "out_hold":           out_hold,
        "net_pv_h":           net_pv_h,
        "turn_h":             turn_h,
        "net_pnl_h":          net_pnl_h,
        "eq_oos":             eq_oos,
        "r_oos":              r_oos,
        "holdout_metrics":    holdout_metrics,
        "holdout_metrics_df": holdout_metrics_df,
    }
