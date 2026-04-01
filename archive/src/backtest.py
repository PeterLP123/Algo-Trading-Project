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
    Saves net_vs_gross_cost_turnover.html/.pdf/.png.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.helpers import abdi_ranaldo_spread, run_net_backtest
from src import config
from src.utils import display_df, save_plotly_fig


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
    display_df(net_backtest_summary.head())

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

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "Gross vs Net Cumulative PnL",
            "Daily Turnover and Transaction Cost",
        ),
        specs=[[{}], [{"secondary_y": True}]],
    )

    fig.add_trace(
        go.Scatter(
            x=net_backtest_summary.index,
            y=cumulative_gross_pnl,
            mode="lines",
            name="Cumulative gross PnL",
            line={"color": config.COLORS["gross"], "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>Gross PnL: %{y:,.2f} USDT<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=net_backtest_summary.index,
            y=cumulative_net_pnl,
            mode="lines",
            name="Cumulative net PnL",
            line={"color": config.COLORS["net"], "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>Net PnL: %{y:,.2f} USDT<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=0.0,
        line_dash="dash",
        line_color=config.COLORS["zero"],
        line_width=1.1,
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=net_backtest_summary.index,
            y=turnover,
            mode="lines",
            name="Turnover (USDT)",
            line={"color": config.COLORS["turnover"], "width": 1.7},
            fill="tozeroy",
            fillcolor="rgba(69, 90, 100, 0.18)",
            hovertemplate="%{x|%Y-%m-%d}<br>Turnover: %{y:,.2f} USDT<extra></extra>",
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=net_backtest_summary.index,
            y=cost_t,
            mode="lines",
            name="Cost (USDT)",
            line={"color": config.COLORS["cost"], "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>Cost: %{y:,.2f} USDT<extra></extra>",
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    fig.update_layout(
        title="Backtest Results",
        width=1200,
        height=780,
    )
    fig.update_xaxes(tickformat="%Y", row=2, col=1, title_text="Date")
    fig.update_yaxes(title_text="USDT", row=1, col=1)
    fig.update_yaxes(title_text="Turnover (USDT)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Cost (USDT)", row=2, col=1, secondary_y=True)

    save_plotly_fig(fig, "net_vs_gross_cost_turnover", output_dir)
