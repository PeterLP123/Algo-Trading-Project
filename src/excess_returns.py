"""
excess_returns.py — Excess return computation and visualization (Cell 11).

Functions:
  compute_excess_returns(cleaned_frames, dff, symbols)
    Convert FRED DFF to daily rf, compute excess_return for each asset,
    mutate cleaned_frames in-place (adds rf_daily and excess_return cols).
    Returns (cleaned_frames, excess_returns_df, rf_daily).

  plot_excess_returns(excess_returns_df, output_dir)
    Interactive Plotly line chart of daily excess returns for all assets.
    Saves as daily_excess_returns.html/.pdf/.png.
"""

import pandas as pd
import plotly.graph_objects as go

from . import config
from .utils import save_plotly_fig


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


def plot_excess_returns(excess_returns_df, output_dir):
    """Plot daily excess returns for all assets and save to HTML/PDF/PNG."""
    fig = go.Figure()

    for i, col in enumerate(excess_returns_df.columns):
        fig.add_trace(
            go.Scatter(
                x=excess_returns_df.index,
                y=excess_returns_df[col],
                mode="lines",
                name=col,
                line={
                    "width": 1.5,
                    "color": config.COLOR_SEQUENCE[i % len(config.COLOR_SEQUENCE)],
                },
                opacity=0.8,
                hovertemplate="%{x|%Y-%m-%d}<br>"
                + f"{col}: "
                + "%{y:.2%}<extra></extra>",
            )
        )

    fig.add_hline(
        y=0.0,
        line_dash="dash",
        line_color=config.COLORS["zero"],
        line_width=1.2,
    )
    fig.update_layout(
        title="Daily Excess Returns by Asset",
        xaxis_title="Date",
        yaxis_title="Excess return",
        width=1000,
        height=460,
        legend={
            "title": {"text": "Asset"},
            "orientation": "v",
            "x": 1.02,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
        },
        margin={"r": 140},
    )
    fig.update_xaxes(dtick="M12", tickformat="%Y")
    fig.update_yaxes(tickformat=".0%")

    save_plotly_fig(fig, "daily_excess_returns", output_dir)
