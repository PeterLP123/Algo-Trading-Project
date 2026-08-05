#!/usr/bin/env python3
"""Build the README performance figure and GitHub social preview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY_FIGURE = PROJECT_ROOT / "report" / "figures" / "cumulative_returns.png"
DEFAULT_SNAPSHOT = PROJECT_ROOT / "forward_validation" / "snapshots" / "2026-08-04"
DEFAULT_OUTPUT = PROJECT_ROOT / "report" / "figures" / "readme_performance_overview.png"
DEFAULT_COMBINED_OUTPUT = PROJECT_ROOT / "report" / "figures" / "readme_combined_test_period.png"
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


def load_post_selection_series(
    snapshot_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy1 = pd.read_csv(
        snapshot_dir / "post_selection_daily.csv", parse_dates=["date"]
    ).set_index("date").sort_index()
    strategy2 = pd.read_csv(
        snapshot_dir / "strategy2_post_selection_daily.csv", parse_dates=["date"]
    ).set_index("date").sort_index()
    required_strategy1 = {
        "cumulative_return",
        "btc_buy_and_hold_cumulative_return",
        "equal_weight_buy_and_hold_cumulative_return",
    }
    required_strategy2 = {"cumulative_return"}
    if missing := required_strategy1.difference(strategy1.columns):
        raise ValueError(f"Strategy 1 post-selection data is missing: {sorted(missing)}")
    if missing := required_strategy2.difference(strategy2.columns):
        raise ValueError(f"Strategy 2 post-selection data is missing: {sorted(missing)}")

    for label, daily in (("Strategy 1", strategy1), ("Strategy 2", strategy2)):
        if daily.empty or daily.index.has_duplicates:
            raise ValueError(f"{label} post-selection data must be non-empty and unique by date.")
        daily.index = daily.index.tz_convert(None)
        if daily.index.min() != HOLDOUT_START:
            raise ValueError(
                f"{label} post-selection data must begin on {HOLDOUT_START.date()}."
            )
    if not strategy1.index.equals(strategy2.index):
        raise ValueError("Strategy post-selection snapshots must use the same daily index.")

    strategy1_payload = json.loads(
        (snapshot_dir / "summary.json").read_text(encoding="utf-8")
    )
    strategy2_payload = json.loads(
        (snapshot_dir / "strategy2_summary.json").read_text(encoding="utf-8")
    )
    expected_correction = "monthly_corrected"
    for label, payload in (
        ("Strategy 1", strategy1_payload),
        ("Strategy 2", strategy2_payload),
    ):
        if payload.get("spread_correction") != expected_correction:
            raise ValueError(
                f"{label} snapshot must use the paper's {expected_correction!r} "
                "Abdi-Ranaldo correction."
            )
    strategy1_summary = strategy1_payload["post_selection_metrics"]
    strategy2_summary = strategy2_payload["post_selection_metrics"]
    for daily, metrics in ((strategy1, strategy1_summary), (strategy2, strategy2_summary)):
        if len(daily) != int(metrics["n_days"]):
            raise ValueError("Post-selection daily rows do not match the summary sample size.")
        if not np.isclose(
            float(daily["cumulative_return"].iloc[-1]),
            float(metrics["total_return"]),
        ):
            raise ValueError("Post-selection cumulative return does not match the summary.")
        daily.attrs.update(metrics)
        daily.attrs["spread_correction"] = expected_correction
    strategy2.attrs["evidence_gate"] = strategy2_payload["post_selection_evidence_gate"]
    return strategy1, strategy2


def build_figure(history_figure: Path, snapshot_dir: Path, output: Path) -> None:
    strategy1, strategy2, strategy1_daily, strategy2_daily = load_extended_series(
        history_figure, snapshot_dir
    )

    colors = {"Strategy 1": "#1f77b4", "Strategy 2": "#2ca02c"}
    shades = {"Development": "#E3ECF7", "Holdout": "#F3E9DC", "Frozen forward": "#ECEAF4"}
    forward_end = max(strategy1.index.max(), strategy2.index.max())
    forward_days = len(strategy1_daily)
    if len(strategy2_daily) != forward_days:
        raise ValueError("Strategy forward snapshots must contain the same number of days.")

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
        f"Submitted history · frozen forward restated with monthly-corrected costs · "
        f"{forward_days} completed daily candles",
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


def build_combined_figure(snapshot_dir: Path, output: Path) -> None:
    strategy1, strategy2 = load_post_selection_series(snapshot_dir)
    end_date = strategy1.index.max()
    combined_days = len(strategy1)
    colors = {
        "Strategy 1": "#1677B8",
        "Strategy 2": "#D97706",
        "BTC": "#475569",
        "Equal-weight": "#94A3B8",
    }
    shades = {"Original holdout dates": "#F3E9DC", "Frozen forward": "#ECEAF4"}
    series = {
        "Strategy 1": strategy1["cumulative_return"],
        "Strategy 2": strategy2["cumulative_return"],
        "BTC": strategy1["btc_buy_and_hold_cumulative_return"],
        "Equal-weight": strategy1["equal_weight_buy_and_hold_cumulative_return"],
    }

    fig, ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    ax.set_facecolor("#F8FAFC")
    ax.axvspan(
        HOLDOUT_START,
        FREEZE_CUTOFF,
        facecolor=shades["Original holdout dates"],
        edgecolor="none",
    )
    ax.axvspan(
        FREEZE_CUTOFF,
        end_date,
        facecolor=shades["Frozen forward"],
        edgecolor="none",
    )
    ax.axvline(FREEZE_CUTOFF, color="#64748B", linewidth=1.0, linestyle=":", zorder=2)
    ax.axhline(0.0, color="#64748B", linewidth=0.9, linestyle="--", zorder=1)

    for label, values in series.items():
        is_strategy = label.startswith("Strategy")
        ax.plot(
            values.index,
            values,
            color=colors[label],
            linewidth=2.2 if is_strategy else 1.45,
            linestyle="-" if is_strategy else "--",
            alpha=1.0 if is_strategy else 0.9,
            zorder=4 if is_strategy else 3,
        )

    label_date = end_date + pd.Timedelta(days=8)
    for label, values in series.items():
        ax.text(
            label_date,
            float(values.iloc[-1]),
            f"{label} {float(values.iloc[-1]):+.1%}",
            color=colors[label],
            fontsize=8.7,
            fontweight="bold" if label.startswith("Strategy") else "normal",
            va="center",
        )

    handles = [
        Line2D([0], [0], color=colors["Strategy 1"], linewidth=2.2, label="Strategy 1"),
        Line2D([0], [0], color=colors["Strategy 2"], linewidth=2.2, label="Strategy 2"),
        Line2D([0], [0], color=colors["BTC"], linewidth=1.45, linestyle="--", label="BTC"),
        Line2D(
            [0],
            [0],
            color=colors["Equal-weight"],
            linewidth=1.45,
            linestyle="--",
            label="Equal-weight basket",
        ),
        Patch(
            facecolor=shades["Original holdout dates"],
            edgecolor="none",
            label="Original holdout dates",
        ),
        Patch(facecolor=shades["Frozen forward"], edgecolor="none", label="Frozen forward"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, ncol=3, columnspacing=1.0)
    ax.set_title("Continuous post-selection performance", pad=16)
    ax.text(
        0.5,
        1.01,
        f"{combined_days} completed daily observations · 15 November 2025 to 4 August 2026 · "
        "net of monthly-corrected Abdi-Ranaldo costs",
        transform=ax.transAxes,
        color="#475569",
        fontsize=8.8,
        ha="center",
        va="bottom",
    )
    ax.annotate(
        "parameters frozen",
        xy=(FREEZE_CUTOFF, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(0, -5),
        textcoords="offset points",
        color="#475569",
        fontsize=8.0,
        ha="center",
        va="top",
    )
    ax.set_ylabel("Cumulative return from 14 November 2025 equity")
    ax.set_xlabel("Completed UTC daily candle")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    first_tick = (HOLDOUT_START + pd.offsets.MonthBegin()).normalize()
    ax.set_xticks(pd.date_range(first_tick, end_date, freq="2MS"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_xlim(HOLDOUT_START - pd.Timedelta(days=8), end_date + pd.Timedelta(days=40))
    observed_min = min(float(values.min()) for values in series.values())
    observed_max = max(float(values.max()) for values in series.values())
    ax.set_ylim(min(-0.50, observed_min - 0.03), max(0.15, observed_max + 0.03))
    ax.grid(True, axis="y", color="#CBD5E1", linewidth=0.8, alpha=0.7)
    ax.grid(False, axis="x")
    ax.spines[["top", "right"]].set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_social_preview(snapshot_dir: Path, output: Path) -> None:
    strategy1_daily, strategy2_daily = load_post_selection_series(snapshot_dir)
    colors = {"Strategy 1": "#1677B8", "Strategy 2": "#D97706"}
    background = "#F8FAFC"
    text_primary = "#0F172A"
    text_secondary = "#475569"
    combined_end = strategy1_daily.index.max()
    combined_days = len(strategy1_daily)

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
        "Paper-corrected cost estimates with\ncontinuous post-selection evidence",
        color=text_secondary,
        fontsize=13.5,
        ha="left",
        va="top",
        linespacing=1.25,
    )
    copy_ax.text(
        0.0,
        0.50,
        f"{combined_days} DAYS · 15 NOV 2025 TO 4 AUG 2026",
        color=text_secondary,
        fontsize=9.5,
        fontweight="bold",
        ha="left",
        va="top",
    )

    strategy2_gate = strategy2_daily.attrs["evidence_gate"]
    cards = (
        (
            "TREND FOLLOWING",
            strategy1_daily,
            colors["Strategy 1"],
            f"{float(strategy1_daily.attrs['sharpe']):.2f} Sharpe",
        ),
        (
            "BTC-DOMINANCE MEAN REVERSION",
            strategy2_daily,
            colors["Strategy 2"],
            f"{strategy2_gate['observed_entries']} / "
            f"{strategy2_gate['minimum_entries']} entry gate",
        ),
    )
    for y, (label, daily, color, evidence_label) in zip((0.36, 0.16), cards, strict=True):
        combined_return = float(daily["cumulative_return"].iloc[-1])
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
            f"{combined_return:+.1%}",
            color=color,
            fontsize=23,
            fontweight="bold",
            ha="left",
            va="top",
        )
        copy_ax.text(
            0.31,
            y - 0.003,
            evidence_label,
            color=text_secondary,
            fontsize=10,
            ha="left",
            va="top",
        )

    plot_ax = fig.add_subplot(grid[0, 1])
    plot_ax.axvspan(HOLDOUT_START, FREEZE_CUTOFF, facecolor="#F3E9DC", edgecolor="none")
    plot_ax.axvspan(FREEZE_CUTOFF, combined_end, facecolor="#ECEAF4", edgecolor="none")
    plot_ax.axvline(FREEZE_CUTOFF, color="#64748B", linewidth=1.0, linestyle=":", zorder=2)
    plot_ax.axhline(0.0, color="#94A3B8", linewidth=0.9, linestyle="--", zorder=1)
    plot_ax.plot(
        strategy1_daily.index,
        strategy1_daily["cumulative_return"],
        color=colors["Strategy 1"],
        linewidth=2.1,
        label="Strategy 1",
        zorder=3,
    )
    plot_ax.plot(
        strategy2_daily.index,
        strategy2_daily["cumulative_return"],
        color=colors["Strategy 2"],
        linewidth=2.1,
        label="Strategy 2",
        zorder=3,
    )
    plot_ax.set_title(
        "Post-selection performance",
        color=text_primary,
        fontsize=15,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    plot_ax.text(
        1.0,
        1.025,
        "net of monthly-corrected costs · no retuning",
        transform=plot_ax.transAxes,
        color=text_secondary,
        fontsize=8.5,
        ha="right",
        va="bottom",
    )
    plot_ax.set_xlim(HOLDOUT_START, combined_end)
    plot_ax.set_ylim(-0.23, 0.13)
    plot_ax.set_ylabel("Return", color=text_secondary, fontsize=9.5)
    plot_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    plot_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plot_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    plot_ax.tick_params(axis="both", labelsize=8.5, colors=text_secondary)
    plot_ax.grid(True, axis="y", alpha=0.18)
    plot_ax.grid(False, axis="x")
    plot_ax.spines[["top", "right"]].set_visible(False)
    plot_ax.spines[["bottom", "left"]].set_color("#CBD5E1")
    plot_ax.legend(loc="lower left", frameon=False, fontsize=8.5)

    fig.text(
        0.055,
        0.055,
        "126-DAY ORIGINAL HOLDOUT   •   137-DAY FROZEN FORWARD   •   263 DAYS TOTAL   •   "
        "PAPER-CORRECTED COSTS   •   NO RETUNING",
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
    parser.add_argument("--combined-output", type=Path, default=DEFAULT_COMBINED_OUTPUT)
    parser.add_argument("--social-output", type=Path, default=DEFAULT_SOCIAL_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_figure(args.history_figure, args.snapshot_dir, args.output)
    build_combined_figure(args.snapshot_dir, args.combined_output)
    build_social_preview(args.snapshot_dir, args.social_output)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.combined_output}")
    print(f"Wrote {args.social_output}")


if __name__ == "__main__":
    main()
