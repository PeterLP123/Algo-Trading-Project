"""
excess_returns.py — Excess return computation and visualization (Cell 11).

Functions:
  compute_excess_returns(cleaned_frames, dff, symbols)
    Convert FRED DFF to daily rf, compute excess_return for each asset,
    mutate cleaned_frames in-place (adds rf_daily and excess_return cols).
    Returns (cleaned_frames, excess_returns_df, rf_daily).

  plot_excess_returns(excess_returns_df, output_dir, mpl_rc_params=None)
    Line plot of daily excess returns for all assets.
    Saves as daily_excess_returns.pdf and .png.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


def compute_excess_returns(cleaned_frames, dff, symbols):
    """Compute daily risk-free rate from FRED DFF and excess returns per asset.

    Args:
        cleaned_frames : dict mapping symbol → cleaned OHLCV DataFrame
                         (mutated in-place: rf_daily and excess_return added)
        dff            : pd.Series of annual % Fed Funds Rate (UTC index)
        symbols        : list of symbol strings

    Returns:
        cleaned_frames   : same dict, now with rf_daily and excess_return cols
        excess_returns_df: wide DataFrame of excess returns (one col per symbol)
        rf_daily         : pd.Series of daily rf decimal values
    """
    # Convert FRED DFF (annual %, simple) to an approximate daily return.
    rf_daily = ((1 + dff / 100.0) ** (1 / 365.0) - 1).rename("rf_daily")
    rf_daily = rf_daily.reindex(
        pd.date_range(dff.index.min(), dff.index.max(), freq="D", tz="UTC")
    ).ffill()

    excess_returns = {}
    for symbol in symbols:
        asset = cleaned_frames[symbol].copy()
        asset_return = asset["close"].pct_change().rename("asset_return")
        aligned_rf = rf_daily.reindex(asset.index).ffill().rename("rf_daily")

        asset["rf_daily"] = aligned_rf
        asset["excess_return"] = asset_return - aligned_rf
        cleaned_frames[symbol] = asset

        excess_returns[symbol] = asset["excess_return"].rename(symbol)

    excess_returns_df = pd.concat(excess_returns.values(), axis=1).dropna(how="all")
    return cleaned_frames, excess_returns_df, rf_daily


def plot_excess_returns(excess_returns_df, output_dir, mpl_rc_params=None):
    """Plot daily excess returns for all assets and save to PDF/PNG.

    Args:
        excess_returns_df : wide DataFrame, one column per symbol
        output_dir        : Path — where to save the figures
        mpl_rc_params     : optional dict passed to plt.rcParams.update()
    """
    if mpl_rc_params is not None:
        plt.rcParams.update(mpl_rc_params)

    colors = plt.get_cmap("tab10").colors

    fig, ax = plt.subplots(figsize=(6.5, 3.4), constrained_layout=True)

    for i, col in enumerate(excess_returns_df.columns):
        c = colors[i % len(colors)]
        ax.plot(
            excess_returns_df.index,
            excess_returns_df[col],
            color=c,
            linewidth=0.65,
            alpha=0.55,
            label=col,
        )

    ax.axhline(0.0, color="0.3", linewidth=0.9, linestyle="--", alpha=0.9)
    ax.set_xlabel("Date")
    ax.set_ylabel("Excess return")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.4)
    ax.grid(False, axis="x")

    ax.legend(title="Asset", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "daily_excess_returns.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "daily_excess_returns.png", bbox_inches="tight")

    plt.show()
