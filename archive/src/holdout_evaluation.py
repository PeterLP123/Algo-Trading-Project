"""
holdout_evaluation.py - Final OOS evaluation with frozen parameters (Cell 30).

Evaluates exactly two models on the untouched final test window:
  1. Historical baseline (40/15/0.5/2)
  2. New walk-forward-selected model
"""

import numpy as np
import pandas as pd

from src.helpers import build_signal_and_theta, compute_half_spread_frac, run_net_backtest
from src.performance_metrics import calmar_ratio, max_dd, sharpe_ratio, sortino_ratio
from src.utils import display_df


def _build_asset_breakdown(out_hold, half_spread_frac_full, eval_index, symbols):
    asset_gross = out_hold["theta_exec"].shift(1).fillna(0.0) * out_hold["r"]
    asset_cost = (
        out_hold["delta_theta_exec"].abs()
        * half_spread_frac_full.reindex(out_hold["r"].index, columns=symbols).fillna(0.0)
    )
    asset_net = asset_gross - asset_cost

    return pd.DataFrame(
        {
            "gross": asset_gross.loc[eval_index].sum(),
            "cost": asset_cost.loc[eval_index].sum(),
            "net": asset_net.loc[eval_index].sum(),
            "avg_abs_pos": out_hold["theta_exec"].loc[eval_index].abs().mean(),
            "avg_turnover": out_hold["delta_theta_exec"].loc[eval_index].abs().mean(),
        }
    )


def _evaluate_model(
    *,
    asset_panel,
    cleaned_frames,
    symbols,
    eval_index,
    ma_window,
    vol_window,
    dead_zone,
    rebalance_every,
    signal_clip_wf,
    gross_cap_wf,
    v0_wf,
    backtest_use_excess,
    trading_days,
    label,
    print_diagnostic=False,
):
    half_spread_frac_full = compute_half_spread_frac(asset_panel, cleaned_frames, symbols)
    theta_run = build_signal_and_theta(
        asset_panel,
        symbols,
        ma_window=ma_window,
        vol_window=vol_window,
        dead_zone=dead_zone,
        signal_clip=signal_clip_wf,
        gross_cap=gross_cap_wf,
        rebalance_every=rebalance_every,
    )
    out_hold = run_net_backtest(
        asset_panel,
        theta_run,
        symbols,
        half_spread_frac_full,
        backtest_use_excess=backtest_use_excess,
        v0=v0_wf,
        print_diagnostic=print_diagnostic,
        diagnostic_window=6,
    )

    eq_oos = out_hold["net_portfolio_value"].reindex(eval_index).dropna()
    r_oos = eq_oos.pct_change().dropna()
    gross_oos = out_hold["gross_pnl"].reindex(eval_index).dropna()
    cost_oos = out_hold["cost_t"].reindex(eval_index).dropna()
    net_oos = out_hold["net_pnl"].reindex(eval_index).dropna()
    total_abs_gross = float(gross_oos.abs().sum())
    total_cost = float(cost_oos.sum())

    metrics = {
        "model": label,
        "n_days": len(eq_oos),
        "sharpe": sharpe_ratio(r_oos, trading_days),
        "sortino": sortino_ratio(r_oos, trading_days=trading_days),
        "calmar": calmar_ratio(r_oos, eq_oos, trading_days),
        "max_drawdown": max_dd(eq_oos),
        "mean_turnover": float(out_hold["turnover"].reindex(eval_index).mean()),
        "total_gross_pnl": float(gross_oos.sum()),
        "total_cost": total_cost,
        "total_net_pnl": float(net_oos.sum()),
        "pct_return_on_V0": float(net_oos.sum() / v0_wf),
        "cost_to_gross_ratio": float(total_cost / total_abs_gross) if total_abs_gross > 0 else float("nan"),
        "final_net_equity": float(eq_oos.iloc[-1]) if len(eq_oos) else float("nan"),
    }
    asset_breakdown = _build_asset_breakdown(out_hold, half_spread_frac_full, eval_index, symbols)

    return {
        "out_hold": out_hold,
        "eq_oos": eq_oos,
        "r_oos": r_oos,
        "metrics": metrics,
        "asset_breakdown": asset_breakdown,
    }


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
    baseline_ma=40,
    baseline_vol=15,
    baseline_dz=0.5,
    baseline_rebal=2,
):
    """Evaluate the selected model against the historical baseline on final OOS."""
    print(f"Execution diagnostic uses rebalance_every={wf_best_rebal}")

    selected = _evaluate_model(
        asset_panel=asset_panel,
        cleaned_frames=cleaned_frames,
        symbols=symbols,
        eval_index=holdout_index,
        ma_window=wf_best_ma,
        vol_window=wf_best_vol,
        dead_zone=wf_best_dz,
        rebalance_every=wf_best_rebal,
        signal_clip_wf=signal_clip_wf,
        gross_cap_wf=gross_cap_wf,
        v0_wf=v0_wf,
        backtest_use_excess=backtest_use_excess,
        trading_days=trading_days,
        label="Selected WF model",
        print_diagnostic=True,
    )
    baseline = _evaluate_model(
        asset_panel=asset_panel,
        cleaned_frames=cleaned_frames,
        symbols=symbols,
        eval_index=holdout_index,
        ma_window=baseline_ma,
        vol_window=baseline_vol,
        dead_zone=baseline_dz,
        rebalance_every=baseline_rebal,
        signal_clip_wf=signal_clip_wf,
        gross_cap_wf=gross_cap_wf,
        v0_wf=v0_wf,
        backtest_use_excess=backtest_use_excess,
        trading_days=trading_days,
        label="Baseline 40/15/0.5/2",
        print_diagnostic=False,
    )

    comparison_df = pd.DataFrame([baseline["metrics"], selected["metrics"]]).set_index("model")
    comparison_df["beats_baseline"] = [
        False,
        bool(
            (selected["metrics"]["sharpe"] > baseline["metrics"]["sharpe"])
            and (selected["metrics"]["total_net_pnl"] > baseline["metrics"]["total_net_pnl"])
            and (selected["metrics"]["cost_to_gross_ratio"] < baseline["metrics"]["cost_to_gross_ratio"])
        ),
    ]

    oos_asset_breakdown = pd.concat(
        {
            "Baseline 40/15/0.5/2": baseline["asset_breakdown"],
            "Selected WF model": selected["asset_breakdown"],
        },
        axis=1,
    )

    display_df(comparison_df.T)
    display_df(oos_asset_breakdown)
    print("Final test period:", holdout_index.min(), "â†’", holdout_index.max())

    return {
        "selected_model": selected,
        "baseline_model": baseline,
        "out_hold": selected["out_hold"],
        "net_pv_h": selected["out_hold"]["net_portfolio_value"],
        "turn_h": selected["out_hold"]["turnover"],
        "net_pnl_h": selected["out_hold"]["net_pnl"],
        "eq_oos": selected["eq_oos"],
        "r_oos": selected["r_oos"],
        "holdout_metrics": selected["metrics"],
        "holdout_metrics_df": comparison_df.loc[["Selected WF model"]],
        "comparison_df": comparison_df,
        "oos_asset_breakdown": oos_asset_breakdown,
    }
