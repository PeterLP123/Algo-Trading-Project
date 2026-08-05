#!/usr/bin/env python3
"""Run a bounded, fully recorded Strategy 2 entry-rule variation search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import pandas as pd

import strategy2_forward_validation as s2fv
from strategy2_pipeline import ENTRY_MODES
from strategy2b_experiment import (
    COMPARISON_METRICS,
    build_trade_ledger,
    canonical_json_digest,
    json_safe,
)


PROJECT_ROOT = Path(__file__).resolve().parent
SPEC_PATH = PROJECT_ROOT / "experiments" / "strategy2_variation_search" / "spec.json"
EXPECTED_SPEC_DIGEST = "7439a32d424e0cee02693c42bdeb9d00a2a7eb838c36d8b666b1a48e821c4263"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "strategy2_variation_search"
    / "results"
    / "2026-08-04"
)
WINDOWS = {
    "original_holdout": (s2fv.ORIGINAL_HOLDOUT_START, s2fv.FREEZE_CUTOFF, s2fv.POST_SELECTION_ANCHOR),
    "frozen_forward": (
        s2fv.FREEZE_CUTOFF + pd.Timedelta(days=1),
        pd.Timestamp("2026-08-04", tz="UTC"),
        s2fv.FREEZE_CUTOFF,
    ),
    "continuous_post_selection": (
        s2fv.ORIGINAL_HOLDOUT_START,
        pd.Timestamp("2026-08-04", tz="UTC"),
        s2fv.POST_SELECTION_ANCHOR,
    ),
}


class VariationSearchError(RuntimeError):
    """Raised when the registered variation search contract is violated."""


def load_search_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    digest = canonical_json_digest(spec)
    if digest != EXPECTED_SPEC_DIGEST:
        raise VariationSearchError(
            f"Strategy 2 search specification changed: expected {EXPECTED_SPEC_DIGEST}, found {digest}."
        )
    if spec.get("baseline_frozen_spec_digest") != s2fv.EXPECTED_SPEC_DIGEST:
        raise VariationSearchError("The variation search no longer references the frozen baseline.")
    variants = spec.get("variants", [])
    ids = [variant.get("id") for variant in variants]
    if len(ids) != len(set(ids)) or not ids:
        raise VariationSearchError("Variation IDs must be non-empty and unique.")
    for variant in variants:
        if variant.get("entry_mode") not in ENTRY_MODES:
            raise VariationSearchError(
                f"Unknown registered entry mode for {variant.get('id')}: "
                f"{variant.get('entry_mode')}."
            )
        gross_util = variant.get("gross_util")
        if not isinstance(gross_util, (int, float)) or gross_util <= 0:
            raise VariationSearchError("Every variation requires positive gross utilisation.")
        if variant.get("role") == "candidate" and float(gross_util) != 1.0:
            raise VariationSearchError("All candidate entry rules must use the common 1x risk.")
    if "accelerating_spike_1x" not in ids:
        raise VariationSearchError("The same-risk 1x baseline is required.")
    return spec


def validate_output_target(output_dir: Path) -> None:
    protected = [
        output_dir / "summary.json",
        output_dir / "comparison.csv",
        output_dir / "trade_ledger.csv",
        output_dir / "post_selection_daily.csv",
        output_dir / "cumulative_returns.png",
    ]
    if any(path.exists() for path in protected):
        raise FileExistsError(f"Strategy 2 variation output already exists at {output_dir}.")


def diagnose_run(run: dict[str, Any]) -> dict[str, Any]:
    equity = run["net_value"]
    non_positive = equity.le(0.0)
    first_non_positive = equity.index[non_positive][0] if non_positive.any() else None
    failure_mask = equity.index <= first_non_positive if first_non_positive is not None else equity.notna()
    return {
        "minimum_equity": float(equity.min()),
        "final_equity": float(equity.iloc[-1]),
        "first_non_positive_date": (
            str(first_non_positive.date()) if first_non_positive is not None else None
        ),
        "gross_pnl_through_failure": float(run["gross_pnl_daily"].loc[failure_mask].sum()),
        "cost_through_failure": float(run["cost_pnl_daily"].loc[failure_mask].sum()),
        "entries_through_failure": int(
            sum(
                entry <= first_non_positive
                for entry in run["trade_entries"]
            )
            if first_non_positive is not None
            else len(run["trade_entries"])
        ),
    }


def evaluate_registered_windows(
    run: dict[str, Any],
    *,
    end_date: pd.Timestamp,
    trading_days: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame]]:
    metrics: dict[str, dict[str, Any]] = {}
    daily: dict[str, pd.DataFrame] = {}
    for label, (start, registered_end, anchor) in WINDOWS.items():
        window_end = end_date if label != "original_holdout" else registered_end
        if label != "original_holdout" and registered_end != end_date:
            raise VariationSearchError("The registered comparison endpoint changed.")
        evaluation_index = pd.date_range(start, window_end, freq="D")
        try:
            window_metrics, window_daily = s2fv.evaluate_strategy2_window(
                run,
                evaluation_index=evaluation_index,
                anchor_date=anchor,
                trading_days=trading_days,
            )
        except ValueError as exc:
            metrics[label] = {
                "status": "unavailable",
                "reason": str(exc),
            }
            continue
        metrics[label] = {"status": "completed", **window_metrics}
        daily[label] = window_daily
    return metrics, daily


def comparison_rows(
    spec: dict[str, Any],
    results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    variants = {variant["id"]: variant for variant in spec["variants"]}
    for variant_id, result in results.items():
        variant = variants[variant_id]
        for window, metrics in result["windows"].items():
            row = {
                "variant": variant_id,
                "role": variant["role"],
                "entry_mode": variant["entry_mode"],
                "gross_util": float(variant["gross_util"]),
                "window": window,
                "status": metrics["status"],
                "failure_reason": metrics.get("reason", ""),
            }
            row.update({metric: metrics.get(metric) for metric in COMPARISON_METRICS})
            rows.append(row)
    return pd.DataFrame(rows)


def apply_usefulness_screen(
    spec: dict[str, Any],
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    screen_spec = spec["retrospective_usefulness_screen"]
    window = screen_spec["window"]
    same_risk_baseline = results[screen_spec["require_net_pnl_above"]]["windows"][window]
    if same_risk_baseline["status"] != "completed":
        raise VariationSearchError("The same-risk baseline could not be evaluated.")

    assessments = {}
    for variant in spec["variants"]:
        if variant["role"] != "candidate":
            continue
        metrics = results[variant["id"]]["windows"][window]
        completed = metrics["status"] == "completed"
        checks = {
            "positive_equity_at_anchor": completed,
            "minimum_entries": completed
            and metrics["trade_entries"] >= int(screen_spec["minimum_entries"]),
            "positive_gross_pnl": completed and metrics["total_gross_pnl"] > 0.0,
            "positive_net_pnl": completed and metrics["total_net_pnl"] > 0.0,
            "net_pnl_above_same_risk_baseline": completed
            and metrics["total_net_pnl"] > same_risk_baseline["total_net_pnl"],
        }
        assessments[variant["id"]] = {
            "passed": all(checks.values()),
            "checks": checks,
        }

    passing = [
        variant_id
        for variant_id, assessment in assessments.items()
        if assessment["passed"]
    ]
    passing.sort(
        key=lambda variant_id: (
            -results[variant_id]["windows"][window]["total_net_pnl"],
            abs(results[variant_id]["windows"][window]["max_drawdown"]),
        )
    )
    return {
        "definition": screen_spec,
        "same_risk_baseline_net_pnl": same_risk_baseline["total_net_pnl"],
        "assessments": assessments,
        "passing_variants": passing,
        "selected_descriptive_candidate": passing[0] if passing else None,
    }


def build_post_selection_daily(
    results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    series = {}
    for variant_id, result in results.items():
        daily = result["daily"].get("continuous_post_selection")
        if daily is not None:
            series[variant_id] = daily["cumulative_return"]
    combined = pd.DataFrame(series)
    combined.index.name = "date"
    return combined


def build_figure(
    daily: pd.DataFrame,
    *,
    selected: str | None,
    output: Path,
) -> None:
    colors = {
        "frozen_strategy2_2x": "#94A3B8",
        "accelerating_spike_1x": "#D97706",
        "confirmed_reversal_1x": "#7C3AED",
        "falling_after_extreme_1x": "#1677B8",
        "relative_momentum_confirmation_1x": "#0F766E",
        "falling_with_relative_momentum_1x": "#BE123C",
    }
    fig, ax = plt.subplots(figsize=(11.2, 5.8), constrained_layout=True)
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
        pd.Timestamp("2026-08-04", tz="UTC"),
        facecolor="#ECEAF4",
        edgecolor="none",
        label="Frozen forward dates",
    )
    ax.axvline(s2fv.FREEZE_CUTOFF, color="#64748B", linewidth=1.0, linestyle=":")
    ax.axhline(0.0, color="#94A3B8", linewidth=0.9, linestyle="--")
    for variant_id in daily.columns:
        if variant_id == "confirmed_reversal_2x":
            continue
        is_selected = variant_id == selected
        is_reference = variant_id == "frozen_strategy2_2x"
        ax.plot(
            daily.index,
            daily[variant_id],
            color=colors.get(variant_id, "#475569"),
            linewidth=2.6 if is_selected else 1.7,
            linestyle="--" if is_reference else "-",
            alpha=1.0 if is_selected else 0.82,
            label=variant_id.replace("_", " "),
        )
    ax.set_title(
        "Bounded Strategy 2 entry-rule search at common 1x candidate risk",
        pad=20,
        fontsize=13,
    )
    ax.text(
        0.5,
        1.01,
        "All variants through 4 August 2026 are retrospective diagnostics · net of estimated costs",
        transform=ax.transAxes,
        color="#475569",
        fontsize=8.7,
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
    ax.legend(loc="best", fontsize=7.6, frameon=True, ncol=2)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_search(
    *,
    history_path: Path,
    cache_dir: Path,
    output_dir: Path,
    end_date: pd.Timestamp,
    offline: bool,
) -> dict[str, Any]:
    validate_output_target(output_dir)
    spec = load_search_spec()
    expected_end = pd.Timestamp(spec["evidence_seen_through"], tz="UTC")
    if end_date != expected_end:
        raise VariationSearchError(
            f"The registered search endpoint is {expected_end.date()}, not {end_date.date()}."
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
    trading_days = int(baseline_spec["execution"]["trading_days_per_year"])
    results: dict[str, dict[str, Any]] = {}
    trade_ledgers = []
    for variant in spec["variants"]:
        run = s2fv.build_frozen_strategy2_run(
            combined,
            baseline_spec,
            entry_mode=variant["entry_mode"],
            gross_util=float(variant["gross_util"]),
        )
        metrics, daily = evaluate_registered_windows(
            run,
            end_date=end_date,
            trading_days=trading_days,
        )
        results[variant["id"]] = {
            "run_diagnostics": diagnose_run(run),
            "windows": metrics,
            "daily": daily,
        }
        post_selection_daily = daily.get("continuous_post_selection")
        if post_selection_daily is not None:
            trade_ledgers.append(
                build_trade_ledger(post_selection_daily, strategy=variant["id"])
            )

    committed_baseline = json.loads(
        (
            PROJECT_ROOT
            / "forward_validation"
            / "snapshots"
            / "2026-08-04"
            / "strategy2_summary.json"
        ).read_text(encoding="utf-8")
    )["post_selection_metrics"]
    reproduced_baseline = results["frozen_strategy2_2x"]["windows"][
        "continuous_post_selection"
    ]
    baseline_reproduced = all(
        reproduced_baseline.get(metric) == committed_baseline.get(metric)
        for metric in (
            "n_days",
            "total_return",
            "total_gross_pnl",
            "total_cost",
            "total_net_pnl",
            "trade_entries",
            "active_days",
        )
    )
    if not baseline_reproduced:
        raise VariationSearchError("The published frozen baseline did not reproduce exactly.")

    usefulness = apply_usefulness_screen(spec, results)
    daily_comparison = build_post_selection_daily(results)
    comparison = comparison_rows(spec, results)
    trade_ledger = (
        pd.concat(trade_ledgers, ignore_index=True)
        if trade_ledgers
        else pd.DataFrame()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "comparison.csv"
    trade_ledger_path = output_dir / "trade_ledger.csv"
    daily_path = output_dir / "post_selection_daily.csv"
    figure_path = output_dir / "cumulative_returns.png"
    summary_path = output_dir / "summary.json"
    comparison.to_csv(comparison_path, index=False, float_format="%.10g")
    trade_ledger.to_csv(trade_ledger_path, index=False, float_format="%.10g")
    daily_comparison.to_csv(daily_path, float_format="%.10g")
    build_figure(
        daily_comparison,
        selected=usefulness["selected_descriptive_candidate"],
        output=figure_path,
    )

    serializable_results = {
        variant_id: {
            "run_diagnostics": result["run_diagnostics"],
            "windows": result["windows"],
        }
        for variant_id, result in results.items()
    }
    summary = json_safe(
        {
            "status": "completed_bounded_retrospective_search",
            "requested_end_date": str(end_date.date()),
            "search_spec_digest": EXPECTED_SPEC_DIGEST,
            "search_spec": spec,
            "baseline_frozen_spec_digest": s2fv.EXPECTED_SPEC_DIGEST,
            "baseline_reproduced_exactly": baseline_reproduced,
            "forward_cache_sha256": cache_hashes,
            "results": serializable_results,
            "usefulness_screen": usefulness,
            "multiplicity": {
                "candidate_variants_screened": sum(
                    variant["role"] == "candidate" for variant in spec["variants"]
                ),
                "selection_bias_warning": (
                    "The selected candidate, if any, was chosen retrospectively from multiple "
                    "rules and requires independent forward evidence."
                ),
            },
            "artifacts": {
                "comparison": comparison_path.name,
                "trade_ledger": trade_ledger_path.name,
                "post_selection_daily": daily_path.name,
                "figure": figure_path.name,
                "summary": summary_path.name,
            },
        }
    )
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
    summary = run_search(
        history_path=args.history_path,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        end_date=pd.Timestamp(args.end_date, tz="UTC"),
        offline=args.offline,
    )
    print("Bounded Strategy 2 entry-rule search")
    print(f"  Baseline reproduced: {summary['baseline_reproduced_exactly']}")
    for variant in summary["search_spec"]["variants"]:
        metrics = summary["results"][variant["id"]]["windows"][
            "continuous_post_selection"
        ]
        if metrics["status"] != "completed":
            print(f"  {variant['id']}: unavailable ({metrics['reason']})")
            continue
        print(
            f"  {variant['id']}: {metrics['total_return']:+.2%}, "
            f"gross {metrics['total_gross_pnl']:+.2f}, net {metrics['total_net_pnl']:+.2f}, "
            f"{metrics['trade_entries']} entries"
        )
    selected = summary["usefulness_screen"]["selected_descriptive_candidate"]
    print(f"  Selected descriptive candidate: {selected or 'none'}")
    print("  Interpretation: retrospective search only; independent evidence required.")


if __name__ == "__main__":
    main()
