"""
signal_visualization.py — Price + trend position overlay plots (Cell 19).

Function:
  plot_signals(close_px, signal_panel, symbols, dead_zone, output_dir)
    2×2 subplot showing close price with green/red shading for
    long/short periods. Saves trend_signals_vs_price.html/.pdf/.png.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import config
from src.utils import save_plotly_fig


def _contiguous_segments(mask, index):
    """Return contiguous (start, end) spans where *mask* is True."""
    segments = []
    start = None

    for i, is_active in enumerate(mask):
        if is_active and start is None:
            start = index[i]
        elif not is_active and start is not None:
            segments.append((start, index[i - 1]))
            start = None

    if start is not None:
        segments.append((start, index[-1]))

    return segments


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
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=tuple(symbols),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line={"width": 10, "color": config.COLORS["long"]},
            opacity=0.22,
            name="Long regime",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line={"width": 10, "color": config.COLORS["short"]},
            opacity=0.22,
            name="Short regime",
            hoverinfo="skip",
        )
    )

    for plot_idx, s in enumerate(symbols, start=1):
        row = 1 if plot_idx <= 2 else 2
        col = 1 if plot_idx % 2 == 1 else 2
        close = close_px[s]
        ma_50 = signal_panel["ma_50"][s]
        pos = signal_panel["trend_position"][s]

        idx = close.index
        p = pos.to_numpy(dtype=float)
        long_m = np.isfinite(p) & (p == 1.0)
        short_m = np.isfinite(p) & (p == -1.0)

        for start, end in _contiguous_segments(long_m, idx):
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor=config.COLORS["long"],
                opacity=0.18,
                line_width=0,
                row=row,
                col=col,
            )
        for start, end in _contiguous_segments(short_m, idx):
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor=config.COLORS["short"],
                opacity=0.18,
                line_width=0,
                row=row,
                col=col,
            )

        fig.add_trace(
            go.Scatter(
                x=idx,
                y=close,
                mode="lines",
                name="Close",
                legendgroup="close",
                showlegend=(plot_idx == 1),
                line={"color": config.COLORS["close"], "width": 2},
                hovertemplate="%{x|%Y-%m-%d}<br>Close: %{y:,.2f}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=ma_50,
                mode="lines",
                name="MA 50",
                legendgroup="ma",
                showlegend=(plot_idx == 1),
                line={"color": config.COLORS["ma"], "width": 1.7, "dash": "dash"},
                hovertemplate="%{x|%Y-%m-%d}<br>MA 50: %{y:,.2f}<extra></extra>",
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(
            tickformat="%Y",
            showgrid=False,
            title_text="Date" if row == 2 else None,
            row=row,
            col=col,
        )
        fig.update_yaxes(title_text="Price (USDT)", row=row, col=col)

    fig.update_layout(
        title=f"Strategy 1 Trend Signal vs Close Price (dead zone = {dead_zone:.2f})",
        width=1200,
        height=900,
        legend={
            "orientation": "h",
            "x": 0.0,
            "y": 1.08,
            "xanchor": "left",
            "yanchor": "bottom",
        },
    )

    save_plotly_fig(fig, "trend_signals_vs_price", output_dir)
