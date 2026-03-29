"""
oos_plot.py — Out-of-sample performance visualization (Cell 31).

Function:
  plot_oos_performance(net_pv_h, holdout_index, asset_panel, v0, output_dir)
    2-row figure:
      (1) Net portfolio value over OOS period vs V0 baseline.
      (2) Cumulative return (%) of strategy vs equal-weight buy & hold.
    Saves oos_performance.html/.pdf/.png.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import config
from src.utils import save_plotly_fig


def plot_oos_performance(net_pv_h, holdout_index, asset_panel, v0, output_dir):
    """Plot OOS net equity and cumulative return vs equal-weight buy & hold.

    Args:
        net_pv_h      : pd.Series of net portfolio value (full sample)
        holdout_index : DatetimeIndex for the OOS period
        asset_panel   : MultiIndex asset panel (for buy-and-hold benchmark)
        v0            : initial capital (plotted as horizontal baseline)
        output_dir    : Path — where to save figures
    """
    eq_oos_plot = net_pv_h.reindex(holdout_index).dropna()
    if len(eq_oos_plot) < 2:
        print("OOS plot skipped: need at least 2 days with valid net equity on holdout.")
        return

    cum_ret_oos = eq_oos_plot / eq_oos_plot.iloc[0] - 1.0
    r_h         = asset_panel["close"].pct_change().reindex(holdout_index).dropna(how="any")
    ew_daily    = r_h.mean(axis=1).reindex(eq_oos_plot.index).fillna(0.0)
    cum_ew_bh   = (1.0 + ew_daily).cumprod() - 1.0

    fig_oos = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "Out-of-sample Net Portfolio Value",
            "OOS Cumulative Return vs Equal-Weight Buy & Hold",
        ),
    )

    fig_oos.add_trace(
        go.Scatter(
            x=eq_oos_plot.index,
            y=eq_oos_plot,
            mode="lines",
            name="Net equity",
            line={"color": config.COLORS["net"], "width": 2.2},
            hovertemplate="%{x|%Y-%m-%d}<br>Net equity: %{y:,.2f} USDT<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig_oos.add_hline(
        y=v0,
        line_dash="dash",
        line_color=config.COLORS["zero"],
        line_width=1.1,
        annotation_text=f"V0 = {v0:,.0f} USDT",
        annotation_position="top left",
        row=1,
        col=1,
    )

    fig_oos.add_trace(
        go.Scatter(
            x=cum_ret_oos.index,
            y=cum_ret_oos,
            mode="lines",
            name="Strategy (OOS)",
            line={"color": config.COLORS["net"], "width": 2.2},
            hovertemplate="%{x|%Y-%m-%d}<br>Strategy: %{y:.2%}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig_oos.add_trace(
        go.Scatter(
            x=cum_ew_bh.index,
            y=cum_ew_bh,
            mode="lines",
            name="Equal-weight buy & hold",
            line={"color": config.COLORS["benchmark"], "width": 2, "dash": "dash"},
            hovertemplate="%{x|%Y-%m-%d}<br>Benchmark: %{y:.2%}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig_oos.add_hline(
        y=0.0,
        line_dash="dash",
        line_color=config.COLORS["zero"],
        line_width=1.1,
        row=2,
        col=1,
    )

    fig_oos.update_layout(
        title="Out-of-Sample Performance",
        width=1200,
        height=780,
    )
    fig_oos.update_xaxes(tickformat="%Y-%m", title_text="Date", row=2, col=1)
    fig_oos.update_yaxes(title_text="USDT", row=1, col=1)
    fig_oos.update_yaxes(title_text="Cumulative return", tickformat=".0%", row=2, col=1)

    save_plotly_fig(fig_oos, "oos_performance", output_dir)
