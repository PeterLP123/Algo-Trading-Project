#!/usr/bin/env python3
"""Run a bounded Strategy 2 relative-value horizon extension search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import strategy2_forward_validation as s2fv
import strategy2_relative_value_search as relative
from strategy2_variation_search import (
    build_post_selection_daily,
    diagnose_run,
    evaluate_registered_windows,
)
from strategy2b_experiment import build_trade_ledger, canonical_json_digest, json_safe


PROJECT_ROOT = Path(__file__).resolve().parent
SPEC_PATH = PROJECT_ROOT / "experiments" / "strategy2_relative_horizon_search" / "spec.json"
EXPECTED_SPEC_DIGEST = "78edd099883baa5c8cef3a680b86c7d4f881aa207c974fc6cb2b76c27051dadb"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "strategy2_relative_horizon_search" / "results" / "2026-08-04"
)


class RelativeHorizonSearchError(RuntimeError):
    """Raised when the registered relative-horizon contract is violated."""


def load_search_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    digest = canonical_json_digest(spec)
    if digest != EXPECTED_SPEC_DIGEST:
        raise RelativeHorizonSearchError(
            f"Strategy 2 horizon specification changed: expected {EXPECTED_SPEC_DIGEST}, "
            f"found {digest}."
        )
    if spec.get("baseline_frozen_spec_digest") != s2fv.EXPECTED_SPEC_DIGEST:
        raise RelativeHorizonSearchError("The horizon search lost its frozen baseline link.")
    variants = spec.get("variants", [])
    ids = [variant.get("id") for variant in variants]
    if not ids or len(ids) != len(set(ids)):
        raise RelativeHorizonSearchError("Horizon IDs must be non-empty and unique.")
    for variant in variants:
        if variant.get("position_mode") != "long_alts_short_btc":
            raise RelativeHorizonSearchError("Every horizon variant must use relative-value weights.")
        if variant.get("exit_mode") != "fixed_holding":
            raise RelativeHorizonSearchError("Every horizon variant must use a fixed hold.")
        if variant.get("alt_universe") != ["ETH", "BNB", "SOL"]:
            raise RelativeHorizonSearchError("Every horizon variant must use the registered core basket.")
        if not isinstance(variant.get("max_hold"), int) or variant["max_hold"] < 7:
            raise RelativeHorizonSearchError(f"Invalid horizon for {variant.get('id')}.")
        if variant.get("role") not in {"same_structure_baseline", "candidate"}:
            raise RelativeHorizonSearchError(f"Unknown role for {variant.get('id')}.")
    return spec


def validate_output_target(output_dir: Path) -> None:
    names = ("summary.json", "comparison.csv", "trade_ledger.csv", "post_selection_daily.csv", "cumulative_returns.png")
    if any((output_dir / name).exists() for name in names):
        raise FileExistsError(f"Strategy 2 horizon output already exists at {output_dir}.")


def run_search(
    *, history_path: Path, cache_dir: Path, output_dir: Path, end_date: pd.Timestamp, offline: bool
) -> dict[str, Any]:
    validate_output_target(output_dir)
    spec = load_search_spec()
    expected_end = pd.Timestamp(spec["evidence_seen_through"], tz="UTC")
    if end_date != expected_end:
        raise RelativeHorizonSearchError(
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
        run = s2fv.build_frozen_strategy2_run(
            combined,
            baseline_spec,
            entry_mode="accelerating_spike",
            gross_util=1.0,
            alt_universe=variant["alt_universe"],
            exit_mode=variant["exit_mode"],
            max_hold=int(variant["max_hold"]),
            position_mode=variant["position_mode"],
            short_carry_rate_annual=relative.SHORT_CARRY_RATE_ANNUAL,
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

    usefulness = relative.apply_usefulness_screen(spec, results)
    daily_comparison = build_post_selection_daily(results)
    comparison = relative.comparison_rows(spec, results)
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
    relative.build_figure(
        daily_comparison,
        selected=usefulness["selected_descriptive_candidate"],
        output=paths["figure"],
        title="Bounded Strategy 2 relative-value horizon extension",
        subtitle=(
            "7- to 28-day fixed holds · 10% annual short-carry stress · net of spread costs"
        ),
    )
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
                    "Any selected horizon follows four prior retrospective searches and requires "
                    "independent forward evidence."
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
    print("Bounded Strategy 2 relative-value horizon search")
    for variant in summary["search_spec"]["variants"]:
        metrics = summary["results"][variant["id"]]["windows"]["continuous_post_selection"]
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
