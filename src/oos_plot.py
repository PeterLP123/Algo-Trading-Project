"""
oos_plot.py — Out-of-sample performance visualization (Cell 31).

Function:
  plot_oos_performance(net_pv_h, holdout_index, asset_panel, v0, output_dir)
    2-row figure:
      (1) Net portfolio value over OOS period vs V0 baseline.
      (2) Cumulative return (%) of strategy vs equal-weight buy & hold.
    Saves oos_performance.pdf and .png.
"""

import matplotlib.pyplot as plt

from src import config
from src.utils import save_fig


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

    fig_oos, axes_oos = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)

    axes_oos[0].plot(eq_oos_plot.index, eq_oos_plot, color=config.C_NET, linewidth=1.0, label="Net equity")
    axes_oos[0].axhline(
        v0, color="0.5", linestyle="--", linewidth=0.8, alpha=0.7, label=f"V0 = {v0:,.0f} USDT"
    )
    axes_oos[0].set_ylabel("USDT")
    axes_oos[0].set_title("Out-of-sample: net portfolio value (frozen WF parameters)")
    axes_oos[0].legend(loc="best", frameon=False)
    axes_oos[0].grid(True, axis="y", linestyle=":", linewidth=config.GRID_LW, alpha=config.GRID_ALPHA)

    axes_oos[1].plot(
        cum_ret_oos.index, cum_ret_oos * 100.0,
        color=config.C_NET, linewidth=1.0,
        label="Strategy (OOS, from first OOS day)",
    )
    axes_oos[1].plot(
        cum_ew_bh.index, cum_ew_bh * 100.0,
        color=config.C_BH, linewidth=0.9, alpha=0.85,
        label="Equal-weight buy & hold (same assets)",
    )
    axes_oos[1].axhline(0.0, color=config.C_ZERO, linewidth=0.7, linestyle="--", alpha=0.7)
    axes_oos[1].set_ylabel("Cumulative return (%)")
    axes_oos[1].set_xlabel("Date")
    axes_oos[1].set_title("OOS cumulative return (normalized to first OOS day = 0%)")
    axes_oos[1].legend(loc="best", frameon=False)
    axes_oos[1].grid(True, axis="y", linestyle=":", linewidth=config.GRID_LW, alpha=config.GRID_ALPHA)
    import matplotlib.dates as mdates
    axes_oos[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes_oos[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes_oos[1].xaxis.get_majorticklabels(), rotation=25, ha="right")

    save_fig(fig_oos, "oos_performance", output_dir)
