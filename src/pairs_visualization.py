"""
pairs_visualization.py — Visualization for the cointegration / pairs-trading
strategy (Strategy 2).

Functions:
  plot_spread_timeseries    — spread with ±2σ/±4σ thresholds, IS/OOS shading
  plot_zscore_timeseries    — z-score with entry/exit markers
  plot_rolling_hedge_ratio  — β_t over time per pair
  plot_coint_pvalue_heatmap — heatmap of p-values across walk-forward windows
  plot_pairs_pnl_comparison — gross/net PnL vs trend-following Strategy 1
  plot_pairs_drawdown       — underwater equity curve
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import config
from src.utils import save_plotly_fig


# ── Helpers ──────────────────────────────────────────────────────────────────

def _best_pair(pairs_result: dict) -> tuple[str, str]:
    """Pick the pair with the highest total |signal| activity (most traded)."""
    signals = pairs_result["pair_signals"]
    best = max(signals, key=lambda k: signals[k].abs().sum())
    return best


def _pair_label(pair: tuple[str, str]) -> str:
    return f"{pair[0]} / {pair[1]}"


# ── Plot a: Spread time series ───────────────────────────────────────────────

def plot_spread_timeseries(
    pairs_result: dict,
    dev_index: pd.DatetimeIndex,
    holdout_index: pd.DatetimeIndex,
    output_dir,
):
    """Spread S_t for the best-performing pair with ±2σ and ±4σ threshold lines.

    Shades in-sample (blue) vs out-of-sample (orange) regions.
    """
    pair = _best_pair(pairs_result)
    spread = pairs_result["pair_spreads"][pair].dropna()
    label = _pair_label(pair)

    mu = spread.rolling(config.ROLLING_HEDGE_WINDOW, min_periods=1).mean()
    sigma = spread.rolling(config.ROLLING_HEDGE_WINDOW, min_periods=1).std(ddof=1).replace(0, np.nan)

    fig = go.Figure()

    # IS/OOS shading
    if len(dev_index) > 0:
        fig.add_vrect(
            x0=dev_index.min(), x1=dev_index.max(),
            fillcolor="rgba(41,98,255,0.06)", line_width=0,
            annotation_text="IS", annotation_position="top left",
        )
    if len(holdout_index) > 0:
        fig.add_vrect(
            x0=holdout_index.min(), x1=holdout_index.max(),
            fillcolor="rgba(255,111,0,0.06)", line_width=0,
            annotation_text="OOS", annotation_position="top left",
        )

    # Spread line
    fig.add_trace(go.Scatter(
        x=spread.index, y=spread, mode="lines", name="Spread",
        line={"color": config.COLORS["spread"], "width": 1.5},
    ))

    # Threshold bands
    for mult, dash in [(2, "dash"), (4, "dot")]:
        fig.add_trace(go.Scatter(
            x=mu.index, y=mu + mult * sigma, mode="lines",
            name=f"+{mult}σ", line={"color": config.COLORS["threshold"], "width": 1, "dash": dash},
            showlegend=(mult == 2),
        ))
        fig.add_trace(go.Scatter(
            x=mu.index, y=mu - mult * sigma, mode="lines",
            name=f"−{mult}σ" if mult == 2 else None, showlegend=(mult == 2),
            line={"color": config.COLORS["threshold"], "width": 1, "dash": dash},
        ))

    # Mean line
    fig.add_trace(go.Scatter(
        x=mu.index, y=mu, mode="lines", name="Rolling mean",
        line={"color": config.COLORS["zero"], "width": 1, "dash": "dashdot"},
    ))

    fig.update_layout(
        title=f"Spread Time Series — {label}",
        xaxis_title="Date", yaxis_title="Spread (log-price units)",
        width=1200, height=500,
    )
    save_plotly_fig(fig, "pairs_spread_timeseries", output_dir)


# ── Plot b: Z-score time series ──────────────────────────────────────────────

def plot_zscore_timeseries(
    pairs_result: dict,
    dev_index: pd.DatetimeIndex,
    holdout_index: pd.DatetimeIndex,
    output_dir,
):
    """Z-score over time with entry/exit threshold lines and trade markers."""
    pair = _best_pair(pairs_result)
    zscore = pairs_result["pair_zscores"][pair].dropna()
    signal = pairs_result["pair_signals"][pair].reindex(zscore.index).fillna(0)
    label = _pair_label(pair)

    fig = go.Figure()

    # IS/OOS shading
    if len(dev_index) > 0:
        fig.add_vrect(
            x0=dev_index.min(), x1=dev_index.max(),
            fillcolor="rgba(41,98,255,0.06)", line_width=0,
        )
    if len(holdout_index) > 0:
        fig.add_vrect(
            x0=holdout_index.min(), x1=holdout_index.max(),
            fillcolor="rgba(255,111,0,0.06)", line_width=0,
        )

    # Z-score line
    fig.add_trace(go.Scatter(
        x=zscore.index, y=zscore, mode="lines", name="Z-score",
        line={"color": config.COLORS["spread"], "width": 1.3},
    ))

    # Threshold lines
    for level, name, color, dash in [
        (config.ZSCORE_ENTRY, f"+{config.ZSCORE_ENTRY} entry", config.COLORS["short"], "dash"),
        (-config.ZSCORE_ENTRY, f"−{config.ZSCORE_ENTRY} entry", config.COLORS["long"], "dash"),
        (config.ZSCORE_EXIT, f"±{config.ZSCORE_EXIT} exit", config.COLORS["zero"], "dot"),
        (-config.ZSCORE_EXIT, None, config.COLORS["zero"], "dot"),
        (config.ZSCORE_STOP, f"±{config.ZSCORE_STOP} stop", config.COLORS["cost"], "dot"),
        (-config.ZSCORE_STOP, None, config.COLORS["cost"], "dot"),
    ]:
        fig.add_hline(
            y=level, line_dash=dash, line_color=color, line_width=0.9,
            annotation_text=name if name else None,
            annotation_position="bottom right" if name else None,
        )

    # Entry/exit markers
    sig_diff = signal.diff().fillna(0)
    entries = sig_diff[sig_diff != 0].index.intersection(zscore.index)
    entry_mask = signal.loc[entries] != 0
    exit_mask = signal.loc[entries] == 0

    entry_dates = entries[entry_mask]
    exit_dates = entries[exit_mask]

    if len(entry_dates) > 0:
        fig.add_trace(go.Scatter(
            x=entry_dates, y=zscore.loc[entry_dates],
            mode="markers", name="Entry",
            marker={"symbol": "triangle-up", "size": 8, "color": config.COLORS["long"]},
        ))
    if len(exit_dates) > 0:
        fig.add_trace(go.Scatter(
            x=exit_dates, y=zscore.loc[exit_dates],
            mode="markers", name="Exit",
            marker={"symbol": "circle", "size": 6, "color": config.COLORS["short"]},
        ))

    fig.update_layout(
        title=f"Z-Score Time Series — {label}",
        xaxis_title="Date", yaxis_title="Z-score",
        width=1200, height=500,
    )
    save_plotly_fig(fig, "pairs_zscore_timeseries", output_dir)


# ── Plot c: Rolling hedge ratio ──────────────────────────────────────────────

def plot_rolling_hedge_ratio(pairs_result: dict, output_dir):
    """Rolling hedge ratio β_t over time for all pairs with signal activity."""
    betas = pairs_result["pair_betas"]
    signals = pairs_result["pair_signals"]

    # Show pairs that had any trading activity
    active_pairs = [k for k in betas if signals[k].abs().sum() > 0]
    if not active_pairs:
        active_pairs = list(betas.keys())[:3]

    n_pairs = len(active_pairs)
    rows = (n_pairs + 1) // 2
    cols = min(n_pairs, 2)

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[_pair_label(p) for p in active_pairs],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for i, pair in enumerate(active_pairs):
        r, c = divmod(i, cols)
        beta = betas[pair].dropna()
        fig.add_trace(
            go.Scatter(
                x=beta.index, y=beta, mode="lines",
                name=_pair_label(pair),
                line={"width": 1.3},
                showlegend=False,
            ),
            row=r + 1, col=c + 1,
        )
        fig.add_hline(
            y=1.0, line_dash="dash", line_color=config.COLORS["zero"],
            line_width=0.8, row=r + 1, col=c + 1,
        )

    fig.update_layout(
        title="Rolling Hedge Ratio (β)",
        width=1200, height=300 * rows,
    )
    fig.update_yaxes(title_text="β")
    save_plotly_fig(fig, "pairs_rolling_hedge_ratio", output_dir)


# ── Plot d: Cointegration p-value heatmap ────────────────────────────────────

def plot_coint_pvalue_heatmap(pairs_result: dict, output_dir):
    """Heatmap of Engle-Granger p-values across all pairs and walk-forward windows."""
    coint_results = pairs_result["coint_results"]
    if not coint_results:
        print("No cointegration results to plot.")
        return

    df = pd.DataFrame(coint_results)

    # Build window labels from OOS start dates
    df["window_label"] = df["oos_start"].dt.strftime("%Y-%m")
    df["pair_label"] = df.apply(lambda r: f"{r['pair_x']} / {r['pair_y']}", axis=1)

    pivot = df.pivot_table(
        values="p_value", index="pair_label", columns="window_label", aggfunc="first",
    )

    # Text annotations: show p-values rounded to 3 decimals
    text = pivot.round(3).astype(str).values

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        text=text,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale=[
            [0.0, "#00C853"],    # green at p=0
            [0.05, "#00C853"],   # green up to threshold
            [0.10, "#FFD600"],   # yellow
            [0.5, "#FF6D00"],    # orange
            [1.0, "#FF1744"],    # red at p=1
        ],
        zmin=0, zmax=1,
        colorbar={"title": "p-value", "tickvals": [0, 0.05, 0.1, 0.5, 1.0]},
    ))

    fig.update_layout(
        title="Cointegration P-Value Heatmap (Engle-Granger, per Walk-Forward Window)",
        xaxis_title="OOS Window Start",
        yaxis_title="Asset Pair",
        width=max(800, 60 * len(pivot.columns)),
        height=max(400, 60 * len(pivot.index)),
    )
    save_plotly_fig(fig, "pairs_coint_pvalue_heatmap", output_dir)


# ── Plot e: Cumulative PnL comparison ────────────────────────────────────────

def plot_pairs_pnl_comparison(
    pairs_result: dict,
    trend_net_portfolio_value: pd.Series,
    v0: float,
    output_dir,
):
    """Gross and net equity curves for pairs strategy, overlaid with trend-following.

    Two-row figure:
      Row 1: Pairs strategy gross and net cumulative PnL
      Row 2: Net equity comparison — pairs vs trend-following (Strategy 1)
    """
    pairs_npv = pairs_result["net_portfolio_value"]
    pairs_gross = pairs_result["gross_pnl"].cumsum()
    pairs_net_cum = pairs_result["net_pnl"].cumsum()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        subplot_titles=(
            "Strategy 2 (Pairs) — Gross vs Net Cumulative PnL",
            "Net Portfolio Value — Pairs vs Trend-Following",
        ),
    )

    # Row 1: pairs gross and net PnL
    fig.add_trace(go.Scatter(
        x=pairs_gross.index, y=pairs_gross, mode="lines",
        name="Pairs gross PnL",
        line={"color": config.COLORS["pairs_gross"], "width": 2},
        hovertemplate="%{x|%Y-%m-%d}<br>Gross PnL: %{y:,.2f} USDT<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=pairs_net_cum.index, y=pairs_net_cum, mode="lines",
        name="Pairs net PnL",
        line={"color": config.COLORS["pairs_net"], "width": 2},
        hovertemplate="%{x|%Y-%m-%d}<br>Net PnL: %{y:,.2f} USDT<extra></extra>",
    ), row=1, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color=config.COLORS["zero"],
                  line_width=1, row=1, col=1)

    # Row 2: net equity comparison
    fig.add_trace(go.Scatter(
        x=pairs_npv.index, y=pairs_npv, mode="lines",
        name="Pairs net equity",
        line={"color": config.COLORS["pairs_net"], "width": 2},
        hovertemplate="%{x|%Y-%m-%d}<br>Pairs: %{y:,.2f} USDT<extra></extra>",
    ), row=2, col=1)

    trend_npv_aligned = trend_net_portfolio_value.reindex(pairs_npv.index)
    fig.add_trace(go.Scatter(
        x=trend_npv_aligned.index, y=trend_npv_aligned, mode="lines",
        name="Trend net equity (S1)",
        line={"color": config.COLORS["net"], "width": 2, "dash": "dash"},
        hovertemplate="%{x|%Y-%m-%d}<br>Trend: %{y:,.2f} USDT<extra></extra>",
    ), row=2, col=1)

    fig.add_hline(y=v0, line_dash="dot", line_color=config.COLORS["zero"],
                  line_width=0.9, row=2, col=1)

    fig.update_layout(title="PnL Comparison", width=1200, height=780)
    fig.update_xaxes(tickformat="%Y", row=2, col=1, title_text="Date")
    fig.update_yaxes(title_text="USDT", row=1, col=1)
    fig.update_yaxes(title_text="USDT", row=2, col=1)

    save_plotly_fig(fig, "pairs_pnl_comparison", output_dir)


# ── Plot f: Drawdown ─────────────────────────────────────────────────────────

def plot_pairs_drawdown(pairs_result: dict, output_dir):
    """Underwater equity curve: drawdown from peak over time."""
    npv = pairs_result["net_portfolio_value"]
    peak = npv.cummax()
    dd = ((npv - peak) / peak.replace(0, np.nan)) * 100  # percentage

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dd.index, y=dd, mode="lines",
        name="Drawdown",
        line={"color": config.COLORS["short"], "width": 1.5},
        fill="tozeroy",
        fillcolor="rgba(255,23,68,0.15)",
        hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>",
    ))

    fig.add_hline(y=0, line_color=config.COLORS["zero"], line_width=0.8)

    fig.update_layout(
        title="Strategy 2 (Pairs) — Drawdown from Peak",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        width=1200, height=450,
    )
    save_plotly_fig(fig, "pairs_drawdown", output_dir)
