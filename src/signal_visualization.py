"""
signal_visualization.py — Price + trend position overlay plots (Cell 19).

Function:
  plot_signals(close_px, signal_panel, symbols, dead_zone, output_dir)
    2×2 subplot showing close price with green/red shading for
    long/short periods. Saves trend_signals_vs_price.pdf and .png.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


def plot_signals(close_px, signal_panel, symbols, dead_zone, output_dir):
    """Plot price with trend position overlay for each asset.

    Long periods (trend_position == +1) shaded green.
    Short periods (trend_position == -1) shaded red.
    MA plotted in blue; close in near-black.

    Args:
        close_px     : DataFrame of close prices (asset columns)
        signal_panel : MultiIndex DataFrame from construct_signals()
        symbols      : list of asset strings
        dead_zone    : used for title context only
        output_dir   : Path — where to save figures
    """
    # Price with trend position overlay (green = long, red = short, unshaded = flat)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=False)
    axes = axes.ravel()

    for ax, s in zip(axes, symbols):
        close = close_px[s]
        ma_50 = signal_panel["ma_50"][s]
        pos   = signal_panel["trend_position"][s]

        idx  = close.index
        y_lo = float(close.min()) * 0.97
        y_hi = float(close.max()) * 1.03

        p       = pos.to_numpy(dtype=float)
        long_m  = np.isfinite(p) & (p == 1.0)
        short_m = np.isfinite(p) & (p == -1.0)

        ax.fill_between(idx, y_lo, y_hi, where=long_m,  color="#2ca02c", alpha=0.22, linewidth=0, label="Long")
        ax.fill_between(idx, y_lo, y_hi, where=short_m, color="#d62728", alpha=0.22, linewidth=0, label="Short")
        ax.plot(idx, close, color="0.15",    linewidth=0.9,  label="Close")
        ax.plot(idx, ma_50, color="tab:blue", linewidth=0.75, alpha=0.75, label="MA 50")

        ax.set_title(s)
        ax.set_ylabel("Price (USDT)")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle("Strategy 1 — trend signal (discrete position) vs close", y=1.06)
    plt.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "trend_signals_vs_price.png", bbox_inches="tight", dpi=150)
    fig.savefig(out_dir / "trend_signals_vs_price.pdf", bbox_inches="tight")
    plt.show()
