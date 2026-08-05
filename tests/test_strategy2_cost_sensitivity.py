import json
from pathlib import Path

import pandas as pd
import pytest

import strategy2_cost_sensitivity as sensitivity


def test_cost_sensitivity_spec_fixes_candidate_and_decision_scenario() -> None:
    spec = sensitivity.load_sensitivity_spec()

    assert sensitivity.canonical_json_digest(spec) == sensitivity.EXPECTED_SPEC_DIGEST
    assert spec["candidate_id"] == "core3_relative_fixed_7d_1x"
    assert spec["candidate_contract"]["short_carry_rate_annual"] == 0.10
    assert [scenario["fixed_one_way_cost_bps"] for scenario in spec["cost_scenarios"]] == [
        0.0,
        0.0,
        5.0,
        10.0,
        20.0,
        40.0,
    ]
    assert [
        scenario.get("spread_correction")
        for scenario in spec["cost_scenarios"][:2]
    ] == ["monthly_corrected", "two_day_corrected"]
    assert spec["usefulness_definition"]["decision_scenario"] == (
        "fixed_20bps_one_way"
    )


def test_cost_sensitivity_cli_exposes_no_candidate_or_cost_overrides() -> None:
    option_dests = {action.dest for action in sensitivity.build_parser()._actions}

    assert option_dests.isdisjoint(
        {
            "alt_universe",
            "position_mode",
            "exit_mode",
            "max_hold",
            "cost_mode",
            "fixed_one_way_cost_bps",
            "short_carry_rate_annual",
        }
    )


def test_committed_cost_sensitivity_artifacts_are_self_consistent() -> None:
    output = Path("experiments/strategy2_cost_sensitivity/results/2026-08-04")
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    comparison = pd.read_csv(output / "comparison.csv")
    daily = pd.read_csv(output / "post_selection_daily.csv", parse_dates=["date"])
    assert summary["sensitivity_spec_digest"] == sensitivity.EXPECTED_SPEC_DIGEST
    assert len(comparison) == len(summary["sensitivity_spec"]["cost_scenarios"]) * 3
    assert len(daily) == 263
    assert (output / summary["artifacts"]["figure"]).stat().st_size > 0
    assert (output / summary["artifacts"]["effective_cost_figure"]).stat().st_size > 0


def test_cost_conditional_candidate_passes_only_below_break_even() -> None:
    summary = json.loads(
        Path(
            "experiments/strategy2_cost_sensitivity/results/2026-08-04/summary.json"
        ).read_text(encoding="utf-8")
    )
    results = summary["results"]
    window = "continuous_post_selection"

    assert summary["usefulness"]["passed_cost_conditional_screen"] is True
    assert results["fixed_20bps_one_way"]["windows"][window]["total_net_pnl"] > 0
    assert results["fixed_40bps_one_way"]["windows"][window]["total_net_pnl"] < 0
    assert results["abdi_ranaldo_monthly_corrected"]["windows"][window][
        "total_net_pnl"
    ] > 0
    assert results["abdi_ranaldo_two_day_corrected"]["windows"][window][
        "total_net_pnl"
    ] < 0
    diagnostics = results["fixed_20bps_one_way"]["cost_diagnostics"]
    assert diagnostics["post_selection_execution_cost"] == pytest.approx(
        diagnostics["post_selection_turnover"] * 20.0 / 10_000.0
    )
    assert diagnostics["post_selection_break_even_one_way_cost_bps"] == pytest.approx(
        34.9942272723
    )
