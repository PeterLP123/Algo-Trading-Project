"""
grid_search_viz.py — Walk-forward grid search visualization (Cell 28).

Function:
  plot_grid_search(wf_search_df, thresh_grid, rebal_grid, output_dir)
    Produces two figures:
      1. 2×2 heatmap grid (DEAD_ZONE × rebalance_every) of wf_score.
      2. Horizontal bar chart ranking all grid points.
    Saves wf_grid_search_heatmaps.pdf/.png and wf_grid_search_ranking.pdf/.png.
"""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def plot_grid_search(wf_search_df, thresh_grid, rebal_grid, output_dir):
    """Plot heatmaps and ranking bar chart for the walk-forward grid search.

    Args:
        wf_search_df : DataFrame returned by run_grid_search()
        thresh_grid  : list of DEAD_ZONE values (used as subplot axes)
        rebal_grid   : list of rebalance_every values (used as subplot axes)
        output_dir   : Path — where to save figures
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vmin = float(wf_search_df["wf_score"].min())
    vmax = float(wf_search_df["wf_score"].max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis

    fig_hm, axes_hm = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    mappable = None
    for ax, (dz, rebal) in zip(axes_hm.flat, product(thresh_grid, rebal_grid)):
        sub = wf_search_df[
            (wf_search_df["DEAD_ZONE"] == dz) & (wf_search_df["rebalance_every"] == rebal)
        ]
        pivot = sub.pivot(index="MA_WINDOW", columns="VOL_WINDOW", values="wf_score")
        pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, norm=norm)
        mappable = im
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("VOL_WINDOW")
        ax.set_ylabel("MA_WINDOW")
        ax.set_title(f"DEAD_ZONE={dz}, rebalance_every={rebal}")
        mid = vmin + 0.55 * (vmax - vmin)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.3f}",
                        ha="center",
                        va="center",
                        color="white" if val < mid else "black",
                        fontsize=9,
                    )
    if mappable is not None:
        fig_hm.colorbar(
            mappable,
            ax=axes_hm,
            shrink=0.85,
            label="wf_score (mean val Sharpe − penalty × std)",
        )
    fig_hm.suptitle("Walk-forward grid search: validation wf_score", y=1.02, fontsize=12)
    fig_hm.savefig(out_dir / "wf_grid_search_heatmaps.png", bbox_inches="tight", dpi=150)
    fig_hm.savefig(out_dir / "wf_grid_search_heatmaps.pdf", bbox_inches="tight")
    plt.show()

    wf_sorted = wf_search_df.sort_values("wf_score", ascending=True)
    labels = [
        f"{int(r['MA_WINDOW'])}|{int(r['VOL_WINDOW'])}|{r['DEAD_ZONE']}|{int(r['rebalance_every'])}"
        for _, r in wf_sorted.iterrows()
    ]
    h = max(6.0, 0.22 * len(wf_sorted))
    fig_bar, ax_bar = plt.subplots(figsize=(8, h), constrained_layout=True)
    ax_bar.barh(np.arange(len(wf_sorted)), wf_sorted["wf_score"], color="steelblue")
    ax_bar.set_yticks(np.arange(len(wf_sorted)))
    ax_bar.set_yticklabels(labels, fontsize=7)
    ax_bar.set_xlabel("wf_score")
    ax_bar.set_title("All grid points (MA | VOL | dead-zone | rebalance days)")
    ax_bar.axvline(
        float(wf_sorted["wf_score"].iloc[-1]),
        color="crimson",
        linestyle="--",
        linewidth=1,
        alpha=0.85,
        label="best",
    )
    ax_bar.legend(loc="lower right", fontsize=8)
    fig_bar.savefig(out_dir / "wf_grid_search_ranking.png", bbox_inches="tight", dpi=150)
    fig_bar.savefig(out_dir / "wf_grid_search_ranking.pdf", bbox_inches="tight")
    plt.show()
