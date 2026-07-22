#!/usr/bin/env python3
"""Build the README performance overview from committed evidence artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY_FIGURE = PROJECT_ROOT / "report" / "figures" / "cumulative_returns.png"
DEFAULT_FORWARD_DAILY = (
    PROJECT_ROOT / "forward_validation" / "snapshots" / "2026-07-21" / "daily.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "report" / "figures" / "readme_performance_overview.png"
FREEZE_CUTOFF = pd.Timestamp("2026-03-20", tz="UTC")


def load_forward_daily(path: Path) -> pd.DataFrame:
    daily = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    required = {
        "cumulative_return",
        "btc_buy_and_hold_cumulative_return",
        "equal_weight_buy_and_hold_cumulative_return",
    }
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Forward daily data is missing columns: {sorted(missing)}")
    if daily.empty or daily.index.min() <= FREEZE_CUTOFF:
        raise ValueError("Forward daily data must contain only observations after the freeze cutoff.")
    return daily


def _with_cutoff_anchor(series: pd.Series) -> pd.Series:
    anchor = pd.Series([0.0], index=pd.DatetimeIndex([FREEZE_CUTOFF]), dtype=float)
    return pd.concat([anchor, series.astype(float)]).sort_index() * 100.0


def build_figure(history_figure: Path, forward_daily: Path, output: Path) -> None:
    history_image = plt.imread(history_figure)
    daily = load_forward_daily(forward_daily)

    strategy = _with_cutoff_anchor(daily["cumulative_return"])
    btc = _with_cutoff_anchor(daily["btc_buy_and_hold_cumulative_return"])
    equal_weight = _with_cutoff_anchor(daily["equal_weight_buy_and_hold_cumulative_return"])

    colors = {
        "strategy": "#1d4ed8",
        "btc": "#d97706",
        "basket": "#64748b",
        "grid": "#e2e8f0",
        "ink": "#0f172a",
    }

    fig = plt.figure(figsize=(16.0, 5.9), constrained_layout=True, facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=(2.05, 1.0), wspace=0.025)

    history_ax = fig.add_subplot(grid[0, 0])
    history_ax.imshow(history_image)
    history_ax.set_title(
        "Original development and holdout",
        color=colors["ink"],
        fontsize=13,
        fontweight="semibold",
        pad=8,
    )
    history_ax.axis("off")

    forward_ax = fig.add_subplot(grid[0, 1])
    forward_ax.set_facecolor("#fffaf3")
    forward_ax.plot(
        strategy.index,
        strategy,
        color=colors["strategy"],
        linewidth=2.4,
        label="Strategy 1",
        zorder=3,
    )
    forward_ax.plot(
        btc.index,
        btc,
        color=colors["btc"],
        linewidth=1.7,
        linestyle=(0, (5, 2)),
        label="BTC buy-and-hold",
        zorder=2,
    )
    forward_ax.plot(
        equal_weight.index,
        equal_weight,
        color=colors["basket"],
        linewidth=1.6,
        linestyle=(0, (2, 2)),
        label="Equal-weight basket",
        zorder=2,
    )
    forward_ax.axhline(0.0, color="#94a3b8", linewidth=0.9, linestyle="--", zorder=1)
    forward_ax.axvline(FREEZE_CUTOFF, color="#94a3b8", linewidth=0.9, zorder=1)

    label_date = strategy.index.max() + pd.Timedelta(days=5)
    label_specs = (
        (strategy, "Strategy 1", colors["strategy"], "bold"),
        (btc, "BTC", colors["btc"], "normal"),
        (equal_weight, "Equal weight", colors["basket"], "normal"),
    )
    for series, label, color, weight in label_specs:
        value = float(series.iloc[-1])
        forward_ax.text(
            label_date,
            value,
            f"{label}  {value:+.1f}%",
            color=color,
            fontsize=9.2,
            fontweight=weight,
            va="center",
            ha="left",
        )

    forward_ax.set_title(
        "Frozen forward validation",
        color=colors["ink"],
        fontsize=13,
        fontweight="semibold",
        pad=19,
    )
    forward_ax.text(
        0.5,
        1.01,
        "Strategy unchanged · 123 completed UTC daily candles",
        transform=forward_ax.transAxes,
        color="#475569",
        fontsize=9.2,
        ha="center",
        va="bottom",
    )
    forward_ax.set_xlabel("2026", color=colors["ink"])
    forward_ax.set_ylabel("Cumulative return from cutoff", color=colors["ink"])
    forward_ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
    forward_ax.xaxis.set_major_locator(mdates.MonthLocator())
    forward_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    forward_ax.set_xlim(FREEZE_CUTOFF - pd.Timedelta(days=3), label_date + pd.Timedelta(days=38))
    plotted_min = min(float(strategy.min()), float(btc.min()), float(equal_weight.min()))
    plotted_max = max(float(strategy.max()), float(btc.max()), float(equal_weight.max()))
    forward_ax.set_ylim(min(-18.0, plotted_min - 2.0), max(7.0, plotted_max + 2.0))
    forward_ax.grid(True, axis="y", color=colors["grid"], linewidth=0.8)
    forward_ax.grid(False, axis="x")
    forward_ax.spines[["top", "right"]].set_visible(False)
    forward_ax.spines[["left", "bottom"]].set_color("#cbd5e1")
    forward_ax.tick_params(colors="#475569")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-figure", type=Path, default=DEFAULT_HISTORY_FIGURE)
    parser.add_argument("--forward-daily", type=Path, default=DEFAULT_FORWARD_DAILY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_figure(args.history_figure, args.forward_daily, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
