#!/usr/bin/env python3
"""Run a bounded Strategy 2 long-alt/short-BTC relative-value search."""

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
from strategy2_pipeline import EXIT_MODES, POSITION_MODES
from strategy2_variation_search import (
    COMPARISON_METRICS,
    build_post_selection_daily,
    diagnose_run,
    evaluate_registered_windows,
)
from strategy2b_experiment import build_trade_ledger, canonical_json_digest, json_safe


PROJECT_ROOT = Path(__file__).resolve().parent
SPEC_PATH = PROJECT_ROOT / "experiments" / "strategy2_relative_value_search" / "spec.json"
EXPECTED_SPEC_DIGEST = "23228febb7e40a8a54f3b39b9fecc7842a557cc96973a0037140566f548e8db1"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "strategy2_relative_value_search" / "results" / "2026-08-04"
)
SHORT_CARRY_RATE_ANNUAL = 0.10


class RelativeValueSearchError(RuntimeError):
    """Raised when the registered relative-value contract is violated."""


def load_search_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    digest = canonical_json_digest(spec)
    if digest != EXPECTED_SPEC_DIGEST:
        raise RelativeValueSearchError(
            f"Strategy 2 relative-value specification changed: expected {EXPECTED_SPEC_DIGEST}, "
            f"found {digest}."
        )
    if spec.get("baseline_frozen_spec_digest") != s2fv.EXPECTED_SPEC_DIGEST:
        raise RelativeValueSearchError("The relative-value search lost its frozen baseline link.")
    frozen_alts = set(s2fv.load_frozen_spec()["symbols"]) - {"BTC"}
    variants = spec.get("variants", [])
    ids = [variant.get("id") for variant in variants]
    if not ids or len(ids) != len(set(ids)):
        raise RelativeValueSearchError("Relative-value IDs must be non-empty and unique.")
    for variant in variants:
        if variant.get("position_mode") not in POSITION_MODES:
            raise RelativeValueSearchError(f"Unknown position mode for {variant.get('id')}.")
        if variant.get("exit_mode") not in EXIT_MODES:
            raise RelativeValueSearchError(f"Unknown exit mode for {variant.get('id')}.")
        if not isinstance(variant.get("max_hold"), int) or variant["max_hold"] < 1:
            raise RelativeValueSearchError(f"Invalid hold for {variant.get('id')}.")
        alts = variant.get("alt_universe", [])
        if not alts or len(alts) != len(set(alts)) or not set(alts).issubset(frozen_alts):
            raise RelativeValueSearchError(f"Invalid basket for {variant.get('id')}.")
        if variant.get("role") not in {
            "original_exit_reference",
            "same_basket_long_only_reference",
            "candidate",
        }:
            raise RelativeValueSearchError(f"Unknown role for {variant.get('id')}.")
    return spec


def validate_output_target(output_dir: Path) -> None:
    names = (
        "summary.json",
        "comparison.csv",
        "trade_ledger.csv",
        "post_selection_daily.csv",
        "cumulative_returns.png",
    )
    if any((output_dir / name).exists() for name in names):
        raise FileExistsError(f"Strategy 2 relative-value output already exists at {output_dir}.")


def comparison_rows(spec: dict[str, Any], results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    variants = {variant["id"]: variant for variant in spec["variants"]}
    for variant_id, result in results.items():
        variant = variants[variant_id]
        for window, metrics in result["windows"].items():
            row = {
                "variant": variant_id,
                "role": variant["role"],
                "alt_universe": " ".join(variant["alt_universe"]),
                "position_mode": variant["position_mode"],
                "exit_mode": variant["exit_mode"],
                "max_hold": variant["max_hold"],
                "short_carry_rate_annual": (
                    SHORT_CARRY_RATE_ANNUAL
                    if variant["position_mode"] == "long_alts_short_btc"
                    else 0.0
                ),
                "window": window,
                "status": metrics["status"],
                "failure_reason": metrics.get("reason", ""),
            }
            row.update({metric: metrics.get(metric) for metric in COMPARISON_METRICS})
            rows.append(row)
    return pd.DataFrame(rows)


def apply_usefulness_screen(spec: dict[str, Any], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    screen_spec = spec["retrospective_usefulness_screen"]
    window = screen_spec["window"]
    baseline_id = screen_spec["require_net_pnl_above"]
    baseline = results[baseline_id]["windows"][window]
    if baseline["status"] != "completed":
        raise RelativeValueSearchError("The long-only comparison could not be evaluated.")
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
            "positive_net_pnl_after_spread_and_carry": completed
            and metrics["total_net_pnl"] > 0.0,
            "net_pnl_above_long_only_reference": completed
            and metrics["total_net_pnl"] > baseline["total_net_pnl"],
        }
        assessments[variant["id"]] = {"passed": all(checks.values()), "checks": checks}
    passing = [key for key, assessment in assessments.items() if assessment["passed"]]
    passing.sort(
        key=lambda key: (
            -results[key]["windows"][window]["total_net_pnl"],
            abs(results[key]["windows"][window]["max_drawdown"]),
        )
    )
    return {
        "definition": screen_spec,
        "long_only_reference_net_pnl": baseline["total_net_pnl"],
        "assessments": assessments,
        "passing_variants": passing,
        "selected_descriptive_candidate": passing[0] if passing else None,
    }


def build_figure(
    daily: pd.DataFrame,
    *,
    selected: str | None,
    output: Path,
    title: str = "Bounded Strategy 2 relative-value search",
    subtitle: str = (
        "50% long alts / 50% short BTC · 10% annual short-carry stress · net of spread costs"
    ),
) -> None:
    palette = ["#94A3B8", "#D97706", "#7C3AED", "#BE123C", "#1677B8", "#0F766E", "#334155", "#9333EA"]
    fig, ax = plt.subplots(figsize=(11.4, 6.0), constrained_layout=True)
    ax.set_facecolor("#F8FAFC")
    ax.axvspan(s2fv.ORIGINAL_HOLDOUT_START, s2fv.FREEZE_CUTOFF, facecolor="#F3E9DC", edgecolor="none", label="Original holdout dates")
    ax.axvspan(s2fv.FREEZE_CUTOFF, pd.Timestamp("2026-08-04", tz="UTC"), facecolor="#ECEAF4", edgecolor="none", label="Frozen forward dates")
    ax.axvline(s2fv.FREEZE_CUTOFF, color="#64748B", linewidth=1.0, linestyle=":")
    ax.axhline(0.0, color="#94A3B8", linewidth=0.9, linestyle="--")
    for color, variant_id in zip(palette, daily.columns):
        is_reference = "long_" in variant_id
        ax.plot(
            daily.index,
            daily[variant_id],
            color=color,
            linewidth=2.8 if variant_id == selected else 1.65,
            linestyle="--" if is_reference else "-",
            alpha=1.0 if variant_id == selected else 0.82,
            label=variant_id.replace("_", " "),
        )
    ax.set_title(title, pad=20, fontsize=13)
    ax.text(
        0.5,
        1.01,
        subtitle,
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
    ax.legend(loc="best", fontsize=6.9, frameon=True, ncol=2)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_search(
    *, history_path: Path, cache_dir: Path, output_dir: Path, end_date: pd.Timestamp, offline: bool
) -> dict[str, Any]:
    validate_output_target(output_dir)
    spec = load_search_spec()
    expected_end = pd.Timestamp(spec["evidence_seen_through"], tz="UTC")
    if end_date != expected_end:
        raise RelativeValueSearchError(
            f"The registered endpoint is {expected_end.date()}, not {end_date.date()}."
        )
    baseline_spec = s2fv.load_frozen_spec()
    history = s2fv.load_frozen_history(history_path, baseline_spec)
    combined, cache_hashes = s2fv.build_combined_history(
        history, baseline_spec, end_date=end_date, cache_dir=cache_dir, offline=offline
    )
    trading_days = int(baseline_spec["execution"]["trading_days_per_year"])
    results: dict[str, dict[str, Any]] = {}
    trade_ledgers = []
    for variant in spec["variants"]:
        is_relative = variant["position_mode"] == "long_alts_short_btc"
        run = s2fv.build_frozen_strategy2_run(
            combined,
            baseline_spec,
            entry_mode="accelerating_spike",
            gross_util=1.0,
            alt_universe=variant["alt_universe"],
            exit_mode=variant["exit_mode"],
            max_hold=int(variant["max_hold"]),
            position_mode=variant["position_mode"],
            short_carry_rate_annual=SHORT_CARRY_RATE_ANNUAL if is_relative else 0.0,
        )
        metrics, daily = evaluate_registered_windows(run, end_date=end_date, trading_days=trading_days)
        results[variant["id"]] = {
            "run_diagnostics": diagnose_run(run),
            "cost_diagnostics": {
                "total_spread_cost": float(run["spread_cost_pnl_daily"].sum()),
                "total_short_carry": float(run["short_carry_pnl_daily"].sum()),
            },
            "windows": metrics,
            "daily": daily,
        }
        post_selection = daily.get("continuous_post_selection")
        if post_selection is not None:
            trade_ledgers.append(build_trade_ledger(post_selection, strategy=variant["id"]))
    usefulness = apply_usefulness_screen(spec, results)
    daily_comparison = build_post_selection_daily(results)
    comparison = comparison_rows(spec, results)
    trade_ledger = pd.concat(trade_ledgers, ignore_index=True) if trade_ledgers else pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "comparison": output_dir / "comparison.csv",
        "trade_ledger": output_dir / "trade_ledger.csv",
        "post_selection_daily": output_dir / "post_selection_daily.csv",
        "figure": output_dir / "cumulative_returns.png",
        "summary": output_dir / "summary.json",
    }
    comparison.to_csv(paths["comparison"], index=False, float_format="%.10g")
    trade_ledger.to_csv(paths["trade_ledger"], index=False, float_format="%.10g")
    daily_comparison.to_csv(paths["post_selection_daily"], float_format="%.10g")
    build_figure(daily_comparison, selected=usefulness["selected_descriptive_candidate"], output=paths["figure"])
    summary = json_safe(
        {
            "status": "completed_bounded_retrospective_search",
            "requested_end_date": str(end_date.date()),
            "search_spec_digest": EXPECTED_SPEC_DIGEST,
            "search_spec": spec,
            "baseline_frozen_spec_digest": s2fv.EXPECTED_SPEC_DIGEST,
            "results": {
                key: {
                    "run_diagnostics": value["run_diagnostics"],
                    "cost_diagnostics": value["cost_diagnostics"],
                    "windows": value["windows"],
                }
                for key, value in results.items()
            },
            "usefulness_screen": usefulness,
            "multiplicity": {
                "candidate_variants_screened": sum(v["role"] == "candidate" for v in spec["variants"]),
                "selection_bias_warning": (
                    "Any selected relative-value rule follows three prior retrospective searches "
                    "and requires independent forward evidence."
                ),
            },
            "forward_cache_sha256": cache_hashes,
            "artifacts": {key: path.name for key, path in paths.items()},
        }
    )
    paths["summary"].write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-date", default="2026-08-04")
    parser.add_argument("--history-path", type=Path, default=Path("s2_binance_cache.pkl"))
    parser.add_argument("--cache-dir", type=Path, default=Path("forward_validation/cache/strategy2"))
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
    print("Bounded Strategy 2 relative-value search")
    for variant in summary["search_spec"]["variants"]:
        metrics = summary["results"][variant["id"]]["windows"]["continuous_post_selection"]
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
