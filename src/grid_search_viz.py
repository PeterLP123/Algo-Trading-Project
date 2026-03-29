"""
grid_search_viz.py — Walk-forward grid search visualization (Cell 28).

Function:
  plot_grid_search(wf_search_df, thresh_grid, rebal_grid, output_dir)
    Produces two figures:
      1. 2×2 heatmap grid (DEAD_ZONE × rebalance_every) of wf_score.
      2. Horizontal bar chart ranking all grid points.
    Saves HTML/PDF/PNG outputs for both figures.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import config
from src.utils import save_plotly_fig


def plot_grid_search(wf_search_df, thresh_grid, rebal_grid, output_dir):
    """Plot heatmaps and ranking bar chart for the walk-forward grid search.

    Args:
        wf_search_df : DataFrame returned by run_grid_search()
        thresh_grid  : list of DEAD_ZONE values (used as subplot axes)
        rebal_grid   : list of rebalance_every values (used as subplot axes)
        output_dir   : Path — where to save figures
    """
    vmin = float(wf_search_df["wf_score"].min())
    vmax = float(wf_search_df["wf_score"].max())

    subplot_titles = [
        f"DEAD_ZONE={dz}, rebalance_every={rebal}"
        for dz in thresh_grid
        for rebal in rebal_grid
    ]
    fig_hm = make_subplots(
        rows=len(thresh_grid),
        cols=len(rebal_grid),
        subplot_titles=tuple(subplot_titles),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    plot_idx = 0
    for dz in thresh_grid:
        for rebal in rebal_grid:
            plot_idx += 1
            row = 1 + (plot_idx - 1) // len(rebal_grid)
            col = 1 + (plot_idx - 1) % len(rebal_grid)

        sub = wf_search_df[
            (wf_search_df["DEAD_ZONE"] == dz) & (wf_search_df["rebalance_every"] == rebal)
        ]
        pivot = sub.pivot(index="MA_WINDOW", columns="VOL_WINDOW", values="wf_score")
        pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))

        fig_hm.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=[str(v) for v in pivot.columns],
                y=[str(v) for v in pivot.index],
                colorscale="Viridis",
                zmin=vmin,
                zmax=vmax,
                text=np.round(pivot.values, 3),
                texttemplate="%{text:.3f}",
                textfont={"size": 11},
                hovertemplate="MA_WINDOW: %{y}<br>VOL_WINDOW: %{x}<br>wf_score: %{z:.3f}<extra></extra>",
                colorbar={
                    "title": {"text": "wf_score"},
                    "len": 0.8,
                } if plot_idx == 1 else None,
                showscale=(plot_idx == 1),
            ),
            row=row,
            col=col,
        )
        fig_hm.update_xaxes(title_text="VOL_WINDOW", row=row, col=col)
        fig_hm.update_yaxes(title_text="MA_WINDOW", row=row, col=col)

    fig_hm.update_layout(
        title="Walk-Forward Grid Search: Validation wf_score",
        width=1100,
        height=850,
    )
    save_plotly_fig(fig_hm, "wf_grid_search_heatmaps", output_dir)

    wf_sorted = wf_search_df.sort_values("wf_score", ascending=True)
    labels = [
        f"{int(r['MA_WINDOW'])}|{int(r['VOL_WINDOW'])}|{r['DEAD_ZONE']}|{int(r['rebalance_every'])}"
        for _, r in wf_sorted.iterrows()
    ]
    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(
            x=wf_sorted["wf_score"],
            y=labels,
            orientation="h",
            marker={"color": config.COLORS["net"]},
            hovertemplate="%{y}<br>wf_score: %{x:.3f}<extra></extra>",
            name="wf_score",
        )
    )
    fig_bar.add_vline(
        x=float(wf_sorted["wf_score"].iloc[-1]),
        line_dash="dash",
        line_color=config.COLORS["cost"],
        line_width=1.5,
        annotation_text="Best",
        annotation_position="top",
    )
    fig_bar.update_layout(
        title="All Grid Points (MA | VOL | dead-zone | rebalance days)",
        xaxis_title="wf_score",
        yaxis_title="Parameter set",
        width=1000,
        height=max(500, 28 * len(wf_sorted)),
    )
    save_plotly_fig(fig_bar, "wf_grid_search_ranking", output_dir)
