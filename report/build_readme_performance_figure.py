#!/usr/bin/env python3
"""Build the README performance figure and GitHub social preview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY_FIGURE = PROJECT_ROOT / "report" / "figures" / "cumulative_returns.png"
DEFAULT_SNAPSHOT = PROJECT_ROOT / "forward_validation" / "snapshots" / "2026-07-21-corrected"
DEFAULT_OUTPUT = PROJECT_ROOT / "report" / "figures" / "readme_performance_overview.png"
DEFAULT_SOCIAL_OUTPUT = PROJECT_ROOT / "report" / "figures" / "github_social_preview.png"
HISTORY_START = pd.Timestamp("2020-01-02")
HOLDOUT_START = pd.Timestamp("2025-11-15")
FREEZE_CUTOFF = pd.Timestamp("2026-03-20")
INITIAL_CAPITAL = 10_000.0

# Pixel calibration for the committed submitted-report figure. Reusing that
# published visual preserves its historical paths exactly; only post-cutoff
# audited PnL is added.
EXPECTED_HISTORY_SIZE = (1667, 737)
PLOT_X_START = 167
PLOT_X_END = 1582
ZERO_RETURN_Y = 563.0
PIXELS_PER_50_PERCENT = 73.0
LINE_COLORS = {"Strategy 1": (31, 119, 180), "Strategy 2": (44, 160, 44)}


def digitise_published_lines(path: Path) -> dict[str, pd.Series]:
    image = Image.open(path).convert("RGB")
    if image.size != EXPECTED_HISTORY_SIZE:
        raise ValueError(
            f"Published figure size changed: expected {EXPECTED_HISTORY_SIZE}, found {image.size}."
        )
    pixels = np.asarray(image)
    x_values = np.arange(PLOT_X_START, PLOT_X_END + 1)
    dates = pd.to_datetime(
        np.linspace(HISTORY_START.value, FREEZE_CUTOFF.value, len(x_values)).astype("int64")
    )
    lines: dict[str, pd.Series] = {}

    for label, rgb in LINE_COLORS.items():
        mask = np.all(pixels == rgb, axis=2)
        y_values = []
        for x_value in x_values:
            matching_y = np.flatnonzero(mask[:, x_value])
            y_values.append(float(np.median(matching_y)) if len(matching_y) else np.nan)
        y_series = pd.Series(y_values, index=dates, dtype=float)
        y_series = y_series.interpolate(limit_area="inside").ffill()
        y_series = y_series.loc[y_series.first_valid_index() :]
        lines[label] = (ZERO_RETURN_Y - y_series) * (50.0 / PIXELS_PER_50_PERCENT)
    return lines


def load_forward_daily(path: Path) -> pd.DataFrame:
    daily = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    required = {"net_pnl", "cumulative_return"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Forward daily data is missing columns: {sorted(missing)}")
    if daily.empty or daily.index.min().tz_convert(None) <= FREEZE_CUTOFF:
        raise ValueError("Forward daily data must contain only observations after the freeze cutoff.")
    daily.index = daily.index.tz_convert(None)
    return daily


def extend_with_forward_pnl(history: pd.Series, daily: pd.DataFrame) -> pd.Series:
    cutoff_value = float(history.iloc[-1])
    extension = cutoff_value + daily["net_pnl"].cumsum() / INITIAL_CAPITAL * 100.0
    anchor = pd.Series([cutoff_value], index=pd.DatetimeIndex([FREEZE_CUTOFF]))
    return pd.concat([history.loc[history.index < FREEZE_CUTOFF], anchor, extension])


def load_extended_series(
    history_figure: Path, snapshot_dir: Path
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    historical = digitise_published_lines(history_figure)
    strategy1_daily = load_forward_daily(snapshot_dir / "daily.csv")
    strategy2_daily = load_forward_daily(snapshot_dir / "strategy2_daily.csv")
    strategy1_summary = json.loads((snapshot_dir / "summary.json").read_text(encoding="utf-8"))
    strategy2_summary = json.loads(
        (snapshot_dir / "strategy2_summary.json").read_text(encoding="utf-8")
    )
    strategy1_daily.attrs["sharpe"] = strategy1_summary["metrics"]["sharpe"]
    strategy2_daily.attrs["sharpe"] = strategy2_summary["metrics"]["sharpe"]
    strategy1 = extend_with_forward_pnl(historical["Strategy 1"], strategy1_daily)
    strategy2 = extend_with_forward_pnl(historical["Strategy 2"], strategy2_daily)
    return strategy1, strategy2, strategy1_daily, strategy2_daily


def build_figure(history_figure: Path, snapshot_dir: Path, output: Path) -> None:
    strategy1, strategy2, strategy1_daily, strategy2_daily = load_extended_series(
        history_figure, snapshot_dir
    )

    colors = {"Strategy 1": "#1f77b4", "Strategy 2": "#2ca02c"}
    shades = {"Development": "#E3ECF7", "Holdout": "#F3E9DC", "Frozen forward": "#ECEAF4"}
    forward_end = max(strategy1.index.max(), strategy2.index.max())

    fig, ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    ax.axvspan(HISTORY_START, HOLDOUT_START, facecolor=shades["Development"], edgecolor="none")
    ax.axvspan(HOLDOUT_START, FREEZE_CUTOFF, facecolor=shades["Holdout"], edgecolor="none")
    ax.axvspan(FREEZE_CUTOFF, forward_end, facecolor=shades["Frozen forward"], edgecolor="none")
    ax.axvline(FREEZE_CUTOFF, color="#6b7280", linewidth=0.9, linestyle=":", zorder=2)

    ax.plot(strategy1.index, strategy1, color=colors["Strategy 1"], linewidth=1.75, zorder=3)
    ax.plot(strategy2.index, strategy2, color=colors["Strategy 2"], linewidth=1.75, zorder=3)
    ax.axhline(0.0, color="0.45", linewidth=0.9, linestyle="--", zorder=1)

    s1_forward_return = float(strategy1_daily["cumulative_return"].iloc[-1])
    s2_forward_return = float(strategy2_daily["cumulative_return"].iloc[-1])
    label_date = forward_end + pd.Timedelta(days=12)
    ax.text(
        label_date,
        float(strategy1.iloc[-1]),
        f"S1 {s1_forward_return:+.1%} forward",
        color=colors["Strategy 1"],
        fontsize=8.8,
        fontweight="bold",
        va="center",
    )
    ax.text(
        label_date,
        float(strategy2.iloc[-1]),
        f"S2 {s2_forward_return:+.1%} forward",
        color=colors["Strategy 2"],
        fontsize=8.8,
        fontweight="bold",
        va="center",
    )

    handles = [
        Line2D([0], [0], color=colors["Strategy 1"], linewidth=1.75, label="Strategy 1"),
        Line2D([0], [0], color=colors["Strategy 2"], linewidth=1.75, label="Strategy 2"),
        Patch(facecolor=shades["Development"], edgecolor="none", label="Development"),
        Patch(facecolor=shades["Holdout"], edgecolor="none", label="Holdout"),
        Patch(facecolor=shades["Frozen forward"], edgecolor="none", label="Frozen forward"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, ncol=3, columnspacing=1.0)
    ax.set_title("Strategy cumulative returns with frozen forward validation", pad=16)
    ax.text(
        0.5,
        1.01,
        "Net of estimated costs · parameters frozen after 20 March 2026 · 123 completed daily candles",
        transform=ax.transAxes,
        color="#475569",
        fontsize=8.8,
        ha="center",
        va="bottom",
    )
    ax.set_ylabel("Cumulative return (%)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(HISTORY_START - pd.Timedelta(days=110), forward_end + pd.Timedelta(days=95))
    ax.set_ylim(-60.0, max(380.0, float(strategy1.max()) + 18.0))
    ax.grid(True, axis="y", alpha=0.28)
    ax.grid(False, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_social_preview(history_figure: Path, snapshot_dir: Path, output: Path) -> None:
    strategy1, strategy2, strategy1_daily, strategy2_daily = load_extended_series(
        history_figure, snapshot_dir
    )
    colors = {"Strategy 1": "#1677B8", "Strategy 2": "#279B37"}
    background = "#F8FAFC"
    text_primary = "#0F172A"
    text_secondary = "#475569"
    forward_end = max(strategy1.index.max(), strategy2.index.max())

    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=background)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(0.94, 1.36),
        left=0.055,
        right=0.965,
        top=0.91,
        bottom=0.14,
        wspace=0.12,
    )

    copy_ax = fig.add_subplot(grid[0, 0])
    copy_ax.axis("off")
    copy_ax.set_xlim(0.0, 1.0)
    copy_ax.set_ylim(0.0, 1.0)
    copy_ax.text(
        0.0,
        0.98,
        "SYSTEMATIC CRYPTO\nRESEARCH",
        color=text_primary,
        fontsize=25,
        fontweight="bold",
        ha="left",
        va="top",
        linespacing=0.92,
    )
    copy_ax.text(
        0.0,
        0.69,
        "Cost-aware backtests with\nfrozen forward validation",
        color=text_secondary,
        fontsize=13.5,
        ha="left",
        va="top",
        linespacing=1.25,
    )
    copy_ax.text(
        0.0,
        0.50,
        "123 DAYS AFTER THE FREEZE",
        color=text_secondary,
        fontsize=9.5,
        fontweight="bold",
        ha="left",
        va="top",
    )

    cards = (
        ("TREND FOLLOWING", strategy1_daily, colors["Strategy 1"]),
        ("BTC-DOMINANCE MEAN REVERSION", strategy2_daily, colors["Strategy 2"]),
    )
    for y, (label, daily, color) in zip((0.36, 0.16), cards, strict=True):
        forward_return = float(daily["cumulative_return"].iloc[-1])
        copy_ax.plot([0.0, 0.0], [y - 0.095, y + 0.055], color=color, linewidth=5)
        copy_ax.text(
            0.045,
            y + 0.045,
            label,
            color=text_secondary,
            fontsize=8.2,
            fontweight="bold",
            ha="left",
            va="top",
        )
        copy_ax.text(
            0.045,
            y - 0.015,
            f"{forward_return:+.1%}",
            color=color,
            fontsize=23,
            fontweight="bold",
            ha="left",
            va="top",
        )
        copy_ax.text(
            0.31,
            y - 0.003,
            f"{float(daily.attrs['sharpe']):.2f} Sharpe",
            color=text_secondary,
            fontsize=10,
            ha="left",
            va="top",
        )

    plot_ax = fig.add_subplot(grid[0, 1])
    plot_ax.axvspan(HISTORY_START, HOLDOUT_START, facecolor="#E3ECF7", edgecolor="none")
    plot_ax.axvspan(HOLDOUT_START, FREEZE_CUTOFF, facecolor="#F3E9DC", edgecolor="none")
    plot_ax.axvspan(FREEZE_CUTOFF, forward_end, facecolor="#ECEAF4", edgecolor="none")
    plot_ax.axvline(FREEZE_CUTOFF, color="#64748B", linewidth=1.0, linestyle=":", zorder=2)
    plot_ax.axhline(0.0, color="#94A3B8", linewidth=0.9, linestyle="--", zorder=1)
    plot_ax.plot(strategy1.index, strategy1, color=colors["Strategy 1"], linewidth=2.1, zorder=3)
    plot_ax.plot(strategy2.index, strategy2, color=colors["Strategy 2"], linewidth=2.1, zorder=3)
    plot_ax.set_title(
        "Net cumulative return",
        color=text_primary,
        fontsize=15,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    plot_ax.text(
        1.0,
        1.025,
        "parameters frozen 20 Mar 2026",
        transform=plot_ax.transAxes,
        color=text_secondary,
        fontsize=8.5,
        ha="right",
        va="bottom",
    )
    plot_ax.set_xlim(HISTORY_START, forward_end)
    plot_ax.set_ylim(-60.0, max(390.0, float(strategy1.max()) + 15.0))
    plot_ax.set_ylabel("Return (%)", color=text_secondary, fontsize=9.5)
    plot_ax.xaxis.set_major_locator(mdates.YearLocator())
    plot_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plot_ax.tick_params(axis="both", labelsize=8.5, colors=text_secondary)
    plot_ax.grid(True, axis="y", alpha=0.18)
    plot_ax.grid(False, axis="x")
    plot_ax.spines[["top", "right"]].set_visible(False)
    plot_ax.spines[["bottom", "left"]].set_color("#CBD5E1")

    fig.text(
        0.055,
        0.055,
        "WALK-FORWARD SELECTION   •   UNTOUCHED HOLDOUT   •   REALISTIC COSTS   •   NO RETUNING",
        color=text_secondary,
        fontsize=9.3,
        fontweight="bold",
        ha="left",
        va="center",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100, facecolor=background)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-figure", type=Path, default=DEFAULT_HISTORY_FIGURE)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--social-output", type=Path, default=DEFAULT_SOCIAL_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_figure(args.history_figure, args.snapshot_dir, args.output)
    build_social_preview(args.history_figure, args.snapshot_dir, args.social_output)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.social_output}")


if __name__ == "__main__":
    main()
