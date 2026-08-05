#!/usr/bin/env python3
"""Compare frozen Strategy 2 with a single-change confirmed-reversal candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import pandas as pd

import strategy2_forward_validation as s2fv


PROJECT_ROOT = Path(__file__).resolve().parent
SPEC_PATH = PROJECT_ROOT / "experiments" / "strategy2b_confirmed_reversal" / "spec.json"
EXPECTED_SPEC_DIGEST = "ac439d1ecb99e43f6b1b8122553fecf4679f7a9ed14b69b2defa22a2ca3e2680"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "strategy2b_confirmed_reversal"
    / "results"
    / "2026-08-04"
)
COMPARISON_METRICS = [
    "n_days",
    "total_return",
    "annualised_return",
    "sharpe",
    "sortino",
    "max_drawdown",
    "total_gross_pnl",
    "total_cost",
    "total_net_pnl",
    "total_turnover",
    "mean_gross_exposure",
    "mean_active_gross_exposure",
    "trade_entries",
    "active_days",
]


class Strategy2BExperimentError(RuntimeError):
    """Raised when the prospective candidate contract is not reproducible."""


def canonical_json_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_candidate_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    digest = canonical_json_digest(spec)
    if digest != EXPECTED_SPEC_DIGEST:
        raise Strategy2BExperimentError(
            f"Strategy 2B specification changed: expected {EXPECTED_SPEC_DIGEST}, found {digest}."
        )
    if spec.get("baseline_frozen_spec_digest") != s2fv.EXPECTED_SPEC_DIGEST:
        raise Strategy2BExperimentError("Strategy 2B no longer references the frozen baseline.")
    change = spec.get("single_change", {})
    expected_change = {
        "parameter": "entry_mode",
        "baseline": "accelerating_spike",
        "candidate": "confirmed_reversal",
    }
    if any(change.get(key) != value for key, value in expected_change.items()):
        raise Strategy2BExperimentError("Strategy 2B must change only the entry mode.")
    if pd.Timestamp(spec["evidence_seen_through"], tz="UTC") >= pd.Timestamp(
        spec["first_prospective_candle"], tz="UTC"
    ):
        raise Strategy2BExperimentError(
            "The first prospective candle must follow all evidence used in the design."
        )
    return spec


def validate_output_target(output_dir: Path) -> None:
    protected = [
        output_dir / "summary.json",
        output_dir / "comparison.csv",
        output_dir / "post_selection_daily.csv",
        output_dir / "trade_ledger.csv",
        output_dir / "cumulative_returns.png",
    ]
    if any(path.exists() for path in protected):
        raise FileExistsError(f"Strategy 2B experiment output already exists at {output_dir}.")


def evaluate_windows(
    run: dict[str, Any],
    *,
    end_date: pd.Timestamp,
    trading_days: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame]]:
    windows = {
        "original_holdout": (
            pd.date_range(s2fv.ORIGINAL_HOLDOUT_START, s2fv.FREEZE_CUTOFF, freq="D"),
            s2fv.POST_SELECTION_ANCHOR,
        ),
        "frozen_forward": (
            pd.date_range(s2fv.FREEZE_CUTOFF + pd.Timedelta(days=1), end_date, freq="D"),
            s2fv.FREEZE_CUTOFF,
        ),
        "continuous_post_selection": (
            pd.date_range(s2fv.ORIGINAL_HOLDOUT_START, end_date, freq="D"),
            s2fv.POST_SELECTION_ANCHOR,
        ),
    }
    metrics: dict[str, dict[str, Any]] = {}
    daily: dict[str, pd.DataFrame] = {}
    for label, (evaluation_index, anchor_date) in windows.items():
        metrics[label], daily[label] = s2fv.evaluate_strategy2_window(
            run,
            evaluation_index=evaluation_index,
            anchor_date=anchor_date,
            trading_days=trading_days,
        )
    return metrics, daily


def build_comparison(
    baseline_metrics: dict[str, dict[str, Any]],
    candidate_metrics: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for window in baseline_metrics:
        for strategy, metrics in (
            ("frozen_strategy2", baseline_metrics[window]),
            ("strategy2b_confirmed_reversal", candidate_metrics[window]),
        ):
            row = {"window": window, "strategy": strategy}
            row.update({metric: metrics[metric] for metric in COMPARISON_METRICS})
            rows.append(row)
    return pd.DataFrame(rows)


def build_trade_ledger(daily: pd.DataFrame, *, strategy: str) -> pd.DataFrame:
    active = daily["active"].astype(bool)
    starts = active & ~active.shift(fill_value=False)
    trade_ids = starts.cumsum().where(active)
    rows = []
    for trade_number, trade_id in enumerate(sorted(trade_ids.dropna().unique()), start=1):
        mask = trade_ids.eq(trade_id)
        active_dates = daily.index[mask]
        exit_accounting_date = active_dates.max() + pd.Timedelta(days=1)
        accounting_dates = active_dates.append(
            pd.DatetimeIndex([exit_accounting_date])
        ).intersection(daily.index)
        gross_pnl = float(daily.loc[mask, "gross_pnl"].sum())
        cost = float(daily.loc[accounting_dates, "transaction_cost"].sum())
        rows.append(
            {
                "strategy": strategy,
                "trade_number": trade_number,
                "active_start": str(active_dates.min().date()),
                "active_end": str(active_dates.max().date()),
                "active_days": int(mask.sum()),
                "gross_pnl": gross_pnl,
                "transaction_cost": cost,
                "net_pnl": gross_pnl - cost,
                "turnover": float(daily.loc[accounting_dates, "turnover"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_daily_comparison(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> pd.DataFrame:
    if not baseline.index.equals(candidate.index):
        raise ValueError("Strategy 2B and baseline daily windows must align exactly.")
    daily = pd.DataFrame(
        {
            "baseline_cumulative_return": baseline["cumulative_return"],
            "candidate_cumulative_return": candidate["cumulative_return"],
            "baseline_gross_pnl": baseline["gross_pnl"],
            "candidate_gross_pnl": candidate["gross_pnl"],
            "baseline_transaction_cost": baseline["transaction_cost"],
            "candidate_transaction_cost": candidate["transaction_cost"],
            "baseline_net_pnl": baseline["net_pnl"],
            "candidate_net_pnl": candidate["net_pnl"],
            "baseline_active": baseline["active"],
            "candidate_active": candidate["active"],
        }
    )
    daily.index.name = "date"
    return daily


def build_figure(daily: pd.DataFrame, *, output: Path, end_date: pd.Timestamp) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    ax.set_facecolor("#F8FAFC")
    ax.axvspan(
        s2fv.ORIGINAL_HOLDOUT_START,
        s2fv.FREEZE_CUTOFF,
        facecolor="#F3E9DC",
        edgecolor="none",
        label="Original holdout dates",
    )
    ax.axvspan(
        s2fv.FREEZE_CUTOFF,
        end_date,
        facecolor="#ECEAF4",
        edgecolor="none",
        label="Frozen forward dates",
    )
    ax.axvline(s2fv.FREEZE_CUTOFF, color="#64748B", linewidth=1.0, linestyle=":")
    ax.axhline(0.0, color="#94A3B8", linewidth=0.9, linestyle="--")
    ax.plot(
        daily.index,
        daily["baseline_cumulative_return"],
        color="#D97706",
        linewidth=2.0,
        label="Frozen Strategy 2",
    )
    ax.plot(
        daily.index,
        daily["candidate_cumulative_return"],
        color="#7C3AED",
        linewidth=2.2,
        label="Strategy 2B confirmed reversal",
    )
    ax.set_title("Strategy 2B retrospective entry-rule diagnostic")
    ax.text(
        0.5,
        1.01,
        "Single change: wait for dominance to turn down after an extreme · net of estimated costs",
        transform=ax.transAxes,
        color="#475569",
        fontsize=8.5,
        ha="center",
        va="bottom",
    )
    ax.set_ylabel("Cumulative return from 14 November 2025 equity")
    ax.set_xlabel("Completed UTC daily candle")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.grid(True, axis="y", color="#CBD5E1", linewidth=0.8, alpha=0.7)
    ax.grid(False, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", frameon=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_experiment(
    *,
    history_path: Path,
    cache_dir: Path,
    output_dir: Path,
    end_date: pd.Timestamp,
    offline: bool,
) -> dict[str, Any]:
    validate_output_target(output_dir)
    candidate_spec = load_candidate_spec()
    expected_end = pd.Timestamp(candidate_spec["retrospective_screen"]["end"], tz="UTC")
    if end_date != expected_end:
        raise Strategy2BExperimentError(
            f"The registered retrospective endpoint is {expected_end.date()}, not {end_date.date()}."
        )

    baseline_spec = s2fv.load_frozen_spec()
    history = s2fv.load_frozen_history(history_path, baseline_spec)
    combined, cache_hashes = s2fv.build_combined_history(
        history,
        baseline_spec,
        end_date=end_date,
        cache_dir=cache_dir,
        offline=offline,
    )
    baseline_run = s2fv.build_frozen_strategy2_run(combined, baseline_spec)
    candidate_run = s2fv.build_frozen_strategy2_run(
        combined,
        baseline_spec,
        entry_mode=candidate_spec["single_change"]["candidate"],
    )
    trading_days = int(baseline_spec["execution"]["trading_days_per_year"])
    baseline_metrics, baseline_daily = evaluate_windows(
        baseline_run,
        end_date=end_date,
        trading_days=trading_days,
    )
    candidate_metrics, candidate_daily = evaluate_windows(
        candidate_run,
        end_date=end_date,
        trading_days=trading_days,
    )

    comparison = build_comparison(baseline_metrics, candidate_metrics)
    baseline_post = baseline_daily["continuous_post_selection"]
    candidate_post = candidate_daily["continuous_post_selection"]
    daily_comparison = build_daily_comparison(baseline_post, candidate_post)
    trade_ledger = pd.concat(
        [
            build_trade_ledger(baseline_post, strategy="frozen_strategy2"),
            build_trade_ledger(
                candidate_post,
                strategy="strategy2b_confirmed_reversal",
            ),
        ],
        ignore_index=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "comparison.csv"
    daily_path = output_dir / "post_selection_daily.csv"
    trade_ledger_path = output_dir / "trade_ledger.csv"
    figure_path = output_dir / "cumulative_returns.png"
    summary_path = output_dir / "summary.json"
    comparison.to_csv(comparison_path, index=False, float_format="%.10g")
    daily_comparison.to_csv(daily_path, float_format="%.10g")
    trade_ledger.to_csv(trade_ledger_path, index=False, float_format="%.10g")
    build_figure(daily_comparison, output=figure_path, end_date=end_date)

    baseline_post_metrics = baseline_metrics["continuous_post_selection"]
    candidate_post_metrics = candidate_metrics["continuous_post_selection"]
    screen = {
        "positive_candidate_gross_pnl": candidate_post_metrics["total_gross_pnl"] > 0.0,
        "candidate_net_pnl_above_baseline": (
            candidate_post_metrics["total_net_pnl"]
            > baseline_post_metrics["total_net_pnl"]
        ),
        "positive_candidate_net_pnl": candidate_post_metrics["total_net_pnl"] > 0.0,
        "passed_all_retrospective_checks": False,
    }
    screen["passed_all_retrospective_checks"] = all(
        value for key, value in screen.items() if key != "passed_all_retrospective_checks"
    )
    summary = {
        "status": "retrospective_post_hoc_diagnostic",
        "requested_end_date": str(end_date.date()),
        "candidate_spec_digest": EXPECTED_SPEC_DIGEST,
        "candidate_spec": candidate_spec,
        "baseline_frozen_spec_digest": s2fv.EXPECTED_SPEC_DIGEST,
        "forward_cache_sha256": cache_hashes,
        "results": {
            "frozen_strategy2": baseline_metrics,
            "strategy2b_confirmed_reversal": candidate_metrics,
        },
        "retrospective_screen": screen,
        "interpretation": {
            "untouched_test": False,
            "reason": (
                "The candidate was designed after observing Strategy 2 evidence through "
                "2026-08-04; all reported comparison windows are retrospective diagnostics."
            ),
            "first_prospective_candle": candidate_spec["first_prospective_candle"],
        },
        "artifacts": {
            "comparison": comparison_path.name,
            "post_selection_daily": daily_path.name,
            "trade_ledger": trade_ledger_path.name,
            "figure": figure_path.name,
            "summary": summary_path.name,
        },
    }
    summary = json_safe(summary)
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-date", default="2026-08-04")
    parser.add_argument("--history-path", type=Path, default=Path("s2_binance_cache.pkl"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("forward_validation/cache/strategy2"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--offline", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_experiment(
        history_path=args.history_path,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        end_date=pd.Timestamp(args.end_date, tz="UTC"),
        offline=args.offline,
    )
    baseline = summary["results"]["frozen_strategy2"]["continuous_post_selection"]
    candidate = summary["results"]["strategy2b_confirmed_reversal"][
        "continuous_post_selection"
    ]
    print("Strategy 2B confirmed-reversal retrospective diagnostic")
    print(f"  Baseline:  {baseline['total_return']:+.2%}, {baseline['trade_entries']} entries")
    print(
        f"  Candidate: {candidate['total_return']:+.2%}, "
        f"{candidate['trade_entries']} entries"
    )
    print(
        "  Gross PnL: "
        f"{candidate['total_gross_pnl']:+.2f} USDT; "
        f"costs: {candidate['total_cost']:.2f} USDT"
    )
    print(
        "  Retrospective screen: "
        f"{'PASS' if summary['retrospective_screen']['passed_all_retrospective_checks'] else 'FAIL'}"
    )
    print("  Interpretation: post-hoc diagnostic only; no untouched test remains.")


if __name__ == "__main__":
    main()
