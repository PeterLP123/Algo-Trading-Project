"""
backtest.py — Backtest execution and P&L visualization (Cell 23).

Functions:
  run_baseline_backtest(asset_panel, exposure_panel, cleaned_frames,
                        symbols, gross_cap, v0, backtest_use_excess)
    Computes Abdi-Ranaldo half-spread, attaches rebalance metadata to theta,
    calls run_net_backtest, prints summary totals.
    Returns a results dict.

  plot_backtest_results(results, output_dir)
    2-row figure: (1) cumulative gross vs net PnL,
                  (2) daily turnover and cost.
    Saves net_vs_gross_cost_turnover.pdf and .png.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    display = print

from src.helpers import abdi_ranaldo_spread, run_net_backtest


def run_baseline_backtest(
    asset_panel,
    exposure_panel,
    cleaned_frames,
    symbols,
    gross_cap,
    v0,
    backtest_use_excess=False,
):
    """Run the baseline (daily-rebalanced) backtest with Abdi-Ranaldo costs.

    Args:
        asset_panel        : MultiIndex DataFrame (close, excess_return)
        exposure_panel     : MultiIndex DataFrame from compute_exposures()
        cleaned_frames     : dict mapping symbol → cleaned OHLCV DataFrame
        symbols            : list of asset strings
        gross_cap          : gross notional budget (unused except for V0 check)
        v0                 : initial capital in USDT
        backtest_use_excess: if True, use excess_return instead of simple returns

    Returns:
        dict with keys:
          out_strategy1, net_backtest_summary,
          cumulative_gross_pnl, cumulative_net_pnl,
          net_portfolio_value, theta_exec,
          net_pnl, cost_t, turnover, half_spread_frac
    """
    ar_by_asset = {}
    for s in symbols:
        ar_by_asset[s] = abdi_ranaldo_spread(cleaned_frames[s]).reindex(asset_panel.index)

    ar_spread_wide = pd.concat([ar_by_asset[s] for s in symbols], axis=1, keys=symbols)

    # Half-spread fraction (lagged): cost per USDT traded in that name
    half_spread = 0.5 * ar_spread_wide
    half_spread_frac = half_spread.shift(1).replace([float("inf"), float("-inf")], float("nan"))

    # Rebalance-aware execution: this Strategy 1 section still rebalances daily.
    theta_tgt   = exposure_panel["theta"]
    theta_input = theta_tgt.copy()
    theta_input.attrs["theta_target"]   = theta_tgt.copy()
    theta_input.attrs["rebalance_mask"] = pd.Series(True, index=theta_tgt.index, name="is_rebalance")
    theta_input.attrs["rebalance_every"] = 1

    out_strategy1 = run_net_backtest(
        asset_panel,
        theta_input,
        symbols,
        half_spread_frac,
        backtest_use_excess=backtest_use_excess,
        v0=v0,
    )

    theta_exec          = out_strategy1["theta_exec"]
    gross_pnl           = out_strategy1["gross_pnl"]
    cumulative_gross_pnl = gross_pnl.cumsum()
    turnover            = out_strategy1["turnover"]
    cost_t              = out_strategy1["cost_t"]
    net_pnl             = out_strategy1["net_pnl"]
    cumulative_net_pnl  = net_pnl.cumsum()
    net_portfolio_value = out_strategy1["net_portfolio_value"]

    net_backtest_summary = pd.DataFrame(
        {
            "gross_pnl":            gross_pnl,
            "cost":                 cost_t,
            "net_pnl":              net_pnl,
            "cumulative_net_pnl":   cumulative_net_pnl,
            "net_portfolio_value":  net_portfolio_value,
            "turnover":             turnover,
        }
    )

    print(f"Total gross PnL (USDT): {cumulative_gross_pnl.iloc[-1]:,.2f}")
    print(f"Total costs (USDT):     {cost_t.sum():,.2f}")
    print(f"Total net PnL (USDT):   {cumulative_net_pnl.iloc[-1]:,.2f}")
    display(net_backtest_summary.head())

    return {
        "out_strategy1":        out_strategy1,
        "net_backtest_summary": net_backtest_summary,
        "cumulative_gross_pnl": cumulative_gross_pnl,
        "cumulative_net_pnl":   cumulative_net_pnl,
        "net_portfolio_value":  net_portfolio_value,
        "theta_exec":           theta_exec,
        "net_pnl":              net_pnl,
        "cost_t":               cost_t,
        "turnover":             turnover,
        "half_spread_frac":     half_spread_frac,
    }


def plot_backtest_results(results, output_dir):
    """Plot cumulative gross vs net PnL and daily turnover/cost.

    Args:
        results    : dict returned by run_baseline_backtest()
        output_dir : Path — where to save figures
    """
    net_backtest_summary = results["net_backtest_summary"]
    cumulative_gross_pnl = results["cumulative_gross_pnl"]
    cumulative_net_pnl   = results["cumulative_net_pnl"]
    cost_t               = results["cost_t"]
    turnover             = results["turnover"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(net_backtest_summary.index, cumulative_gross_pnl, color="tab:blue",  linewidth=0.9, label="Cumulative gross PnL")
    axes[0].plot(net_backtest_summary.index, cumulative_net_pnl,   color="tab:green", linewidth=0.9, label="Cumulative net PnL")
    axes[0].axhline(0.0, color="0.4", linewidth=0.7, linestyle="--", alpha=0.7)
    axes[0].set_ylabel("USDT")
    axes[0].legend(loc="upper left", frameon=False)
    axes[0].set_title("Gross vs net cumulative PnL")
    axes[0].grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.45)

    axes[1].fill_between(
        net_backtest_summary.index, 0.0, cost_t,
        step="mid", color="tab:red", alpha=0.35, linewidth=0,
    )
    axes[1].plot(net_backtest_summary.index, turnover, color="0.2", linewidth=0.5, alpha=0.65, label="Turnover (USDT)")
    ax_r = axes[1].twinx()
    ax_r.plot(net_backtest_summary.index, cost_t, color="tab:red", linewidth=0.65, alpha=0.85, label="Cost (USDT)")
    axes[1].set_ylabel("Turnover (USDT)")
    ax_r.set_ylabel("Cost (USDT)")
    axes[1].set_title("Daily rebalancing turnover and transaction cost")
    lines, labels   = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax_r.legend(lines + lines2, labels + labels2, loc="upper left", frameon=False, fontsize=8)
    axes[1].grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.45)
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "net_vs_gross_cost_turnover.png", bbox_inches="tight", dpi=150)
    fig.savefig(out_dir / "net_vs_gross_cost_turnover.pdf", bbox_inches="tight")
    plt.show()
