#!/usr/bin/env python3
"""Run a bounded Strategy 2 liquid-basket variation search."""

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
from strategy2_variation_search import (
    COMPARISON_METRICS,
    build_post_selection_daily,
    diagnose_run,
    evaluate_registered_windows,
)
from strategy2b_experiment import (
    build_trade_ledger,
    canonical_json_digest,
    json_safe,
)


PROJECT_ROOT = Path(__file__).resolve().parent
SPEC_PATH = PROJECT_ROOT / "experiments" / "strategy2_basket_search" / "spec.json"
EXPECTED_SPEC_DIGEST = "b9508e679cf9f50a3bfbe4ef1b28b72fed329d324d964c24eeb6d9dde58089f5"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "strategy2_basket_search"
    / "results"
    / "2026-08-04"
)


class BasketSearchError(RuntimeError):
    """Raised when the registered basket-search contract is violated."""


def load_search_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    digest = canonical_json_digest(spec)
    if digest != EXPECTED_SPEC_DIGEST:
        raise BasketSearchError(
            f"Strategy 2 basket specification changed: expected {EXPECTED_SPEC_DIGEST}, "
            f"found {digest}."
        )
    if spec.get("baseline_frozen_spec_digest") != s2fv.EXPECTED_SPEC_DIGEST:
        raise BasketSearchError("The basket search no longer references the frozen baseline.")

    frozen_alts = set(s2fv.load_frozen_spec()["symbols"]) - {"BTC"}
    variants = spec.get("variants", [])
    ids = [variant.get("id") for variant in variants]
    if not ids or len(ids) != len(set(ids)):
        raise BasketSearchError("Basket IDs must be non-empty and unique.")
    if ids[0] != "broad_all_1x" or variants[0].get("role") != "same_risk_baseline":
        raise BasketSearchError("The registered broad 1x basket must be the first baseline.")
    for variant in variants:
        alts = variant.get("alt_universe", [])
        if not alts or len(alts) != len(set(alts)):
            raise BasketSearchError(f"{variant.get('id')} requires a unique non-empty basket.")
        if not set(alts).issubset(frozen_alts):
            raise BasketSearchError(f"{variant.get('id')} includes an unregistered asset.")
        if variant.get("role") not in {"same_risk_baseline", "candidate"}:
            raise BasketSearchError(f"Unknown role for {variant.get('id')}.")
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
        raise FileExistsError(f"Strategy 2 basket output already exists at {output_dir}.")


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
                "alt_universe": " ".join(variant["alt_universe"]),
                "n_assets": len(variant["alt_universe"]),
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
    baseline_id = screen_spec["require_net_pnl_above"]
    baseline = results[baseline_id]["windows"][window]
    if baseline["status"] != "completed":
        raise BasketSearchError("The broad same-risk baseline could not be evaluated.")

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
            "net_pnl_above_broad_baseline": completed
            and metrics["total_net_pnl"] > baseline["total_net_pnl"],
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
        "broad_baseline_net_pnl": baseline["total_net_pnl"],
        "assessments": assessments,
        "passing_variants": passing,
        "selected_descriptive_candidate": passing[0] if passing else None,
    }


def build_figure(daily: pd.DataFrame, *, selected: str | None, output: Path) -> None:
    colors = {
        "broad_all_1x": "#94A3B8",
        "liquid5_1x": "#D97706",
        "core4_1x": "#7C3AED",
        "core3_1x": "#1677B8",
        "eth_bnb_1x": "#0F766E",
        "eth_only_1x": "#BE123C",
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
        is_selected = variant_id == selected
        is_reference = variant_id == "broad_all_1x"
        ax.plot(
            daily.index,
            daily[variant_id],
            color=colors.get(variant_id, "#475569"),
            linewidth=2.7 if is_selected else 1.8,
            linestyle="--" if is_reference else "-",
            alpha=1.0 if is_selected else 0.84,
            label=variant_id.replace("_", " "),
        )
    ax.set_title("Bounded Strategy 2 liquid-basket search at common 1x risk", pad=20, fontsize=13)
    ax.text(
        0.5,
        1.01,
        "Original entry and exit rules fixed · retrospective through 4 August 2026 · net of costs",
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
        raise BasketSearchError(
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
            entry_mode="accelerating_spike",
            gross_util=1.0,
            alt_universe=variant["alt_universe"],
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
        post_selection = daily.get("continuous_post_selection")
        if post_selection is not None:
            trade_ledgers.append(build_trade_ledger(post_selection, strategy=variant["id"]))

    baseline = results["broad_all_1x"]["windows"]["continuous_post_selection"]
    if baseline["status"] != "completed":
        raise BasketSearchError("The registered broad 1x baseline failed.")
    usefulness = apply_usefulness_screen(spec, results)
    daily_comparison = build_post_selection_daily(results)
    comparison = comparison_rows(spec, results)
    trade_ledger = pd.concat(trade_ledgers, ignore_index=True) if trade_ledgers else pd.DataFrame()

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
            "results": serializable_results,
            "usefulness_screen": usefulness,
            "multiplicity": {
                "candidate_variants_screened": sum(
                    variant["role"] == "candidate" for variant in spec["variants"]
                ),
                "selection_bias_warning": (
                    "Any selected basket was chosen retrospectively after the entry-rule search "
                    "failed and requires independent forward evidence."
                ),
            },
            "forward_cache_sha256": cache_hashes,
            "artifacts": {
                "comparison": comparison_path.name,
                "trade_ledger": trade_ledger_path.name,
                "post_selection_daily": daily_path.name,
                "figure": figure_path.name,
                "summary": summary_path.name,
            },
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
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
    print("Bounded Strategy 2 liquid-basket search")
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
