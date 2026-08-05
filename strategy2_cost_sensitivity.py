#!/usr/bin/env python3
"""Evaluate one fixed Strategy 2 candidate under registered cost scenarios."""

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
from strategy2_pipeline import COST_MODES
from strategy2_variation_search import (
    COMPARISON_METRICS,
    build_post_selection_daily,
    diagnose_run,
    evaluate_registered_windows,
)
from strategy2b_experiment import build_trade_ledger, canonical_json_digest, json_safe
from wf_trend_pipeline import ABDI_RANALDO_CORRECTIONS


PROJECT_ROOT = Path(__file__).resolve().parent
SPEC_PATH = PROJECT_ROOT / "experiments" / "strategy2_cost_sensitivity" / "spec.json"
EXPECTED_SPEC_DIGEST = "74691f80c3a4b9c9d24143e6b64bced17fd8ac1a55c7f5cf2d7ce8baf56eb1da"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "strategy2_cost_sensitivity" / "results" / "2026-08-04"
)


class CostSensitivityError(RuntimeError):
    """Raised when the registered cost-sensitivity contract is violated."""


def load_sensitivity_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    digest = canonical_json_digest(spec)
    if digest != EXPECTED_SPEC_DIGEST:
        raise CostSensitivityError(
            f"Strategy 2 cost specification changed: expected {EXPECTED_SPEC_DIGEST}, found {digest}."
        )
    if spec.get("baseline_frozen_spec_digest") != s2fv.EXPECTED_SPEC_DIGEST:
        raise CostSensitivityError("The cost sensitivity lost its frozen baseline link.")
    candidate = spec.get("candidate_contract", {})
    if candidate != {
        "entry_mode": "accelerating_spike",
        "alt_universe": ["ETH", "BNB", "SOL"],
        "position_mode": "long_alts_short_btc",
        "exit_mode": "fixed_holding",
        "max_hold": 7,
        "gross_util": 1.0,
        "short_carry_rate_annual": 0.10,
    }:
        raise CostSensitivityError("The selected candidate contract changed.")
    scenarios = spec.get("cost_scenarios", [])
    ids = [scenario.get("id") for scenario in scenarios]
    if not ids or len(ids) != len(set(ids)):
        raise CostSensitivityError("Cost scenario IDs must be non-empty and unique.")
    for scenario in scenarios:
        if scenario.get("cost_mode") not in COST_MODES:
            raise CostSensitivityError(f"Unknown cost mode for {scenario.get('id')}.")
        correction = scenario.get("spread_correction")
        if scenario.get("cost_mode") == "abdi_ranaldo":
            if correction not in ABDI_RANALDO_CORRECTIONS:
                raise CostSensitivityError(
                    f"Invalid Abdi-Ranaldo correction for {scenario.get('id')}."
                )
        elif correction is not None:
            raise CostSensitivityError(
                f"Fixed-cost scenario {scenario.get('id')} cannot select a spread correction."
            )
        bps = scenario.get("fixed_one_way_cost_bps")
        if not isinstance(bps, (int, float)) or bps < 0:
            raise CostSensitivityError(f"Invalid fixed cost for {scenario.get('id')}.")
    decision = spec["usefulness_definition"]["decision_scenario"]
    if decision not in ids:
        raise CostSensitivityError("The decision scenario is not registered.")
    return spec


def validate_output_target(output_dir: Path) -> None:
    names = (
        "summary.json",
        "comparison.csv",
        "trade_ledger.csv",
        "post_selection_daily.csv",
        "cumulative_returns.png",
        "effective_cost_comparison.png",
    )
    if any((output_dir / name).exists() for name in names):
        raise FileExistsError(f"Strategy 2 cost output already exists at {output_dir}.")


def build_figure(daily: pd.DataFrame, *, output: Path) -> None:
    styles = {
        "abdi_ranaldo_monthly_corrected": ("#1D4ED8", "--", 2.1),
        "abdi_ranaldo_two_day_corrected": ("#C2410C", ":", 2.1),
        "fixed_5bps_one_way": ("#64748B", "-", 1.5),
        "fixed_10bps_one_way": ("#0F766E", "-", 1.7),
        "fixed_20bps_one_way": ("#7C3AED", "-", 2.5),
        "fixed_40bps_one_way": ("#BE123C", "-.", 1.8),
    }
    labels = {
        "abdi_ranaldo_monthly_corrected": "Abdi-Ranaldo monthly-corrected",
        "abdi_ranaldo_two_day_corrected": "Abdi-Ranaldo two-day-corrected",
        "fixed_5bps_one_way": "Fixed 5 bps one way",
        "fixed_10bps_one_way": "Fixed 10 bps one way",
        "fixed_20bps_one_way": "Fixed 20 bps one way",
        "fixed_40bps_one_way": "Fixed 40 bps one way",
    }
    fig, ax = plt.subplots(figsize=(10.8, 5.7), constrained_layout=True)
    ax.set_facecolor("#F8FAFC")
    ax.axvspan(s2fv.ORIGINAL_HOLDOUT_START, s2fv.FREEZE_CUTOFF, facecolor="#F3E9DC", edgecolor="none", label="Original holdout dates")
    ax.axvspan(s2fv.FREEZE_CUTOFF, pd.Timestamp("2026-08-04", tz="UTC"), facecolor="#ECEAF4", edgecolor="none", label="Frozen forward dates")
    ax.axvline(s2fv.FREEZE_CUTOFF, color="#64748B", linewidth=1.0, linestyle=":")
    ax.axhline(0.0, color="#94A3B8", linewidth=0.9, linestyle="--")
    for scenario_id in daily.columns:
        color, linestyle, linewidth = styles[scenario_id]
        ax.plot(
            daily.index,
            daily[scenario_id],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=labels[scenario_id],
        )
    ax.set_title("Strategy 2 relative-value candidate: execution-cost sensitivity", pad=20, fontsize=13)
    ax.text(
        0.5,
        1.01,
        "Same six entries and 10% annual short carry · paper-correct spread variants and fixed charges",
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
    ax.legend(loc="best", fontsize=7.1, frameon=True, ncol=2)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_effective_cost_figure(
    results: dict[str, dict[str, Any]], *, output: Path
) -> None:
    labels = {
        "fixed_5bps_one_way": "Fixed 5 bps",
        "fixed_10bps_one_way": "Fixed 10 bps",
        "abdi_ranaldo_monthly_corrected": "A-R monthly-corrected",
        "fixed_20bps_one_way": "Fixed 20 bps",
        "fixed_40bps_one_way": "Fixed 40 bps",
        "abdi_ranaldo_two_day_corrected": "A-R two-day-corrected",
    }
    rows = []
    for scenario_id, result in results.items():
        diagnostics = result["cost_diagnostics"]
        turnover = float(diagnostics["post_selection_turnover"])
        execution_cost = float(diagnostics["post_selection_execution_cost"])
        rows.append(
            {
                "scenario": scenario_id,
                "label": labels[scenario_id],
                "effective_bps": execution_cost / turnover * 10_000.0,
            }
        )
    costs = pd.DataFrame(rows).sort_values("effective_bps")
    break_even = float(
        results["fixed_20bps_one_way"]["cost_diagnostics"][
            "post_selection_break_even_one_way_cost_bps"
        ]
    )
    colors = [
        "#C2410C"
        if row.scenario.startswith("abdi_ranaldo")
        else "#7C3AED"
        if row.scenario == "fixed_20bps_one_way"
        else "#94A3B8"
        for row in costs.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    ax.set_facecolor("#F8FAFC")
    bars = ax.barh(
        costs["label"],
        costs["effective_bps"],
        color=colors,
        edgecolor="#475569",
        linewidth=0.7,
    )
    ax.axvline(
        break_even,
        color="#0F172A",
        linewidth=1.4,
        linestyle="--",
        label=f"Observed break-even: {break_even:.1f} bps",
    )
    ax.bar_label(bars, labels=[f"{value:.1f}" for value in costs["effective_bps"]], padding=4)
    ax.set_xlim(0.0, max(costs["effective_bps"].max(), break_even) * 1.16)
    ax.invert_yaxis()
    ax.set_title("Effective one-way execution cost by scenario", fontsize=13, pad=22)
    ax.text(
        0.5,
        1.01,
        "Turnover-weighted over 263 days and 117,087 USDT traded · short carry excluded",
        transform=ax.transAxes,
        color="#475569",
        fontsize=8.7,
        ha="center",
        va="bottom",
    )
    ax.set_xlabel("Execution cost (basis points per dollar traded)")
    ax.grid(True, axis="x", color="#CBD5E1", linewidth=0.8, alpha=0.7)
    ax.grid(False, axis="y")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_sensitivity(
    *, history_path: Path, cache_dir: Path, output_dir: Path, end_date: pd.Timestamp, offline: bool
) -> dict[str, Any]:
    validate_output_target(output_dir)
    spec = load_sensitivity_spec()
    expected_end = pd.Timestamp(spec["evidence_seen_through"], tz="UTC")
    if end_date != expected_end:
        raise CostSensitivityError(
            f"The registered endpoint is {expected_end.date()}, not {end_date.date()}."
        )
    baseline_spec = s2fv.load_frozen_spec()
    history = s2fv.load_frozen_history(history_path, baseline_spec)
    combined, cache_hashes = s2fv.build_combined_history(
        history, baseline_spec, end_date=end_date, cache_dir=cache_dir, offline=offline
    )
    candidate = spec["candidate_contract"]
    trading_days = int(baseline_spec["execution"]["trading_days_per_year"])
    evaluation_index = pd.date_range(s2fv.ORIGINAL_HOLDOUT_START, end_date, freq="D")
    results: dict[str, dict[str, Any]] = {}
    trade_ledgers = []
    for scenario in spec["cost_scenarios"]:
        run = s2fv.build_frozen_strategy2_run(
            combined,
            baseline_spec,
            entry_mode=candidate["entry_mode"],
            gross_util=float(candidate["gross_util"]),
            alt_universe=candidate["alt_universe"],
            exit_mode=candidate["exit_mode"],
            max_hold=int(candidate["max_hold"]),
            position_mode=candidate["position_mode"],
            short_carry_rate_annual=float(candidate["short_carry_rate_annual"]),
            cost_mode=scenario["cost_mode"],
            fixed_one_way_cost_bps=float(scenario["fixed_one_way_cost_bps"]),
            spread_correction=scenario.get(
                "spread_correction", "monthly_corrected"
            ),
        )
        metrics, daily = evaluate_registered_windows(run, end_date=end_date, trading_days=trading_days)
        window = daily["continuous_post_selection"]
        window_turnover = float(run["turnover"].reindex(evaluation_index).sum())
        window_spread_cost = float(run["spread_cost_pnl_daily"].reindex(evaluation_index).sum())
        window_short_carry = float(run["short_carry_pnl_daily"].reindex(evaluation_index).sum())
        window_gross = float(run["gross_pnl_daily"].reindex(evaluation_index).sum())
        break_even_bps = (
            (window_gross - window_short_carry) / window_turnover * 10_000.0
            if window_turnover > 0.0
            else None
        )
        results[scenario["id"]] = {
            "run_diagnostics": diagnose_run(run),
            "cost_diagnostics": {
                "post_selection_turnover": window_turnover,
                "post_selection_execution_cost": window_spread_cost,
                "post_selection_short_carry": window_short_carry,
                "post_selection_break_even_one_way_cost_bps": break_even_bps,
            },
            "windows": metrics,
            "daily": daily,
        }
        trade_ledgers.append(build_trade_ledger(window, strategy=scenario["id"]))

    decision_id = spec["usefulness_definition"]["decision_scenario"]
    decision = results[decision_id]["windows"]["continuous_post_selection"]
    usefulness = {
        "definition": spec["usefulness_definition"],
        "checks": {
            "minimum_entries": decision["trade_entries"] >= int(spec["usefulness_definition"]["minimum_entries"]),
            "positive_gross_pnl": decision["total_gross_pnl"] > 0.0,
            "positive_net_pnl_after_execution_and_carry": decision["total_net_pnl"] > 0.0,
        },
    }
    usefulness["passed_cost_conditional_screen"] = all(usefulness["checks"].values())
    daily_comparison = build_post_selection_daily(results)
    rows = []
    scenario_map = {item["id"]: item for item in spec["cost_scenarios"]}
    for scenario_id, result in results.items():
        scenario = scenario_map[scenario_id]
        for window, metrics in result["windows"].items():
            row = {
                "scenario": scenario_id,
                "role": scenario["role"],
                "cost_mode": scenario["cost_mode"],
                "spread_correction": scenario.get("spread_correction"),
                "fixed_one_way_cost_bps": scenario["fixed_one_way_cost_bps"],
                "window": window,
                "status": metrics["status"],
                "failure_reason": metrics.get("reason", ""),
            }
            row.update({metric: metrics.get(metric) for metric in COMPARISON_METRICS})
            rows.append(row)
    comparison = pd.DataFrame(rows)
    trade_ledger = pd.concat(trade_ledgers, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "comparison": output_dir / "comparison.csv",
        "trade_ledger": output_dir / "trade_ledger.csv",
        "post_selection_daily": output_dir / "post_selection_daily.csv",
        "figure": output_dir / "cumulative_returns.png",
        "effective_cost_figure": output_dir / "effective_cost_comparison.png",
        "summary": output_dir / "summary.json",
    }
    comparison.to_csv(paths["comparison"], index=False, float_format="%.10g")
    trade_ledger.to_csv(paths["trade_ledger"], index=False, float_format="%.10g")
    daily_comparison.to_csv(paths["post_selection_daily"], float_format="%.10g")
    build_figure(daily_comparison, output=paths["figure"])
    build_effective_cost_figure(results, output=paths["effective_cost_figure"])
    summary = json_safe(
        {
            "status": "completed_bounded_retrospective_cost_sensitivity",
            "requested_end_date": str(end_date.date()),
            "sensitivity_spec_digest": EXPECTED_SPEC_DIGEST,
            "sensitivity_spec": spec,
            "baseline_frozen_spec_digest": s2fv.EXPECTED_SPEC_DIGEST,
            "results": {
                key: {
                    "run_diagnostics": value["run_diagnostics"],
                    "cost_diagnostics": value["cost_diagnostics"],
                    "windows": value["windows"],
                }
                for key, value in results.items()
            },
            "usefulness": usefulness,
            "interpretation": (
                "This is a cost-conditional retrospective candidate. Both paper-corrected "
                "Abdi-Ranaldo variants and the fixed scenarios are reported without retuning; "
                "achievable live execution remains unmeasured."
            ),
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
    summary = run_sensitivity(
        history_path=args.history_path,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        end_date=pd.Timestamp(args.end_date, tz="UTC"),
        offline=args.offline,
    )
    print("Strategy 2 fixed-candidate cost sensitivity")
    for scenario in summary["sensitivity_spec"]["cost_scenarios"]:
        metrics = summary["results"][scenario["id"]]["windows"]["continuous_post_selection"]
        print(
            f"  {scenario['id']}: {metrics['total_return']:+.2%}, "
            f"gross {metrics['total_gross_pnl']:+.2f}, net {metrics['total_net_pnl']:+.2f}"
        )
    print(f"  Cost-conditional screen passed: {summary['usefulness']['passed_cost_conditional_screen']}")


if __name__ == "__main__":
    main()
