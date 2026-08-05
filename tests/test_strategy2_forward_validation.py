import ast
import json
from pathlib import Path

import pandas as pd
import pytest

import strategy2_forward_validation as s2fv


def test_frozen_strategy2_spec_and_parameters_are_immutable() -> None:
    spec = s2fv.load_frozen_spec()

    assert s2fv.canonical_spec_digest(spec) == s2fv.EXPECTED_SPEC_DIGEST
    assert spec["frozen_at"] == "2026-03-20"
    assert spec["parameters"] == {
        "entry_pct": 0.8,
        "lookback": 126,
        "exit_z": 0.0,
        "max_hold": 7,
        "roc_window": 3,
        "roc_pct": 0.95,
        "crash_threshold": None,
        "gross_util": 2.0,
    }
    assert spec["unavailable_forward_symbols"] == ["MATIC"]


def test_strategy2_cli_exposes_no_parameter_overrides() -> None:
    option_dests = {action.dest for action in s2fv.build_parser()._actions}
    forbidden = set(s2fv.load_frozen_spec()["parameters"])

    assert option_dests.isdisjoint(forbidden)
    assert option_dests.isdisjoint({"minimum_entries", "preferred_entries"})


def test_strategy2_evidence_gate_is_fixed_and_prospective() -> None:
    gate = s2fv.load_evidence_gate()

    assert s2fv.canonical_spec_digest(gate) == s2fv.EXPECTED_EVIDENCE_GATE_DIGEST
    assert gate["metric"] == "trade_entries"
    assert gate["scope"] == "continuous_post_selection"
    assert gate["minimum_entries"] == 20
    assert gate["preferred_entries"] == 30
    assert gate["registered_on"] == "2026-08-05"
    assert gate["registration_basis"] == {
        "snapshot_end": "2026-08-04",
        "observed_entries": 7,
    }


@pytest.mark.parametrize(
    ("observed", "expected_status"),
    [(19, "below_minimum"), (20, "minimum_met"), (30, "preferred_met")],
)
def test_entry_count_gate_has_fixed_status_boundaries(
    observed: int,
    expected_status: str,
) -> None:
    assessment = s2fv.assess_entry_count_gate(
        {"trade_entries": observed, "n_days": 500},
        s2fv.load_evidence_gate(),
    )

    assert assessment["status"] == expected_status
    assert assessment["minimum_met"] is (observed >= 20)
    assert assessment["preferred_met"] is (observed >= 30)


def test_entry_count_gate_uses_entries_and_labels_time_as_planning_only() -> None:
    assessment = s2fv.assess_entry_count_gate(
        {"trade_entries": 7, "n_days": 263, "active_days": 14},
        s2fv.load_evidence_gate(),
    )

    assert assessment["status"] == "below_minimum"
    assert assessment["remaining_to_minimum"] == 13
    assert assessment["remaining_to_preferred"] == 23
    assert assessment["planning_estimate"] == {
        "basis_entries": 7,
        "basis_days": 263,
        "observed_days_per_entry": pytest.approx(263 / 7),
        "estimated_total_days_to_minimum": 751,
        "estimated_additional_days_to_minimum": 488,
        "planning_only": True,
    }


def test_strategy2_runner_imports_no_optimisation_library() -> None:
    tree = ast.parse(Path(s2fv.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "optuna" not in imported


def test_committed_strategy2_snapshot_is_self_consistent() -> None:
    snapshot = Path("forward_validation/snapshots/2026-07-21-corrected")
    summary = json.loads((snapshot / "strategy2_summary.json").read_text(encoding="utf-8"))
    daily = pd.read_csv(snapshot / "strategy2_daily.csv", parse_dates=["date"])
    metrics = summary["metrics"]

    assert summary["frozen_spec_digest"] == s2fv.EXPECTED_SPEC_DIGEST
    assert len(daily) == metrics["n_days"] == 123
    assert daily["date"].min() > s2fv.FREEZE_CUTOFF
    assert daily["date"].max() == pd.Timestamp(metrics["end"], tz="UTC")
    assert daily["net_pnl"].sum() == pytest.approx(metrics["total_net_pnl"], rel=1e-9)
    assert daily.iloc[-1]["cumulative_return"] == pytest.approx(metrics["total_return"], rel=1e-9)
    assert metrics["trade_entries"] == 4
    assert metrics["active_days"] == 9


def test_latest_strategy2_snapshot_includes_continuous_post_selection_evidence() -> None:
    snapshot = Path("forward_validation/snapshots/2026-08-04")
    summary = json.loads((snapshot / "strategy2_summary.json").read_text(encoding="utf-8"))
    forward = pd.read_csv(snapshot / "strategy2_daily.csv", parse_dates=["date"])
    combined = pd.read_csv(
        snapshot / "strategy2_post_selection_daily.csv", parse_dates=["date"]
    )
    forward_metrics = summary["metrics"]
    combined_metrics = summary["post_selection_metrics"]

    assert summary["spread_correction"] == s2fv.ABDI_RANALDO_CORRECTION
    assert s2fv.ABDI_RANALDO_CORRECTION == "monthly_corrected"
    assert len(forward) == forward_metrics["n_days"] == 137
    assert len(combined) == combined_metrics["n_days"] == 263
    assert combined["date"].min() == s2fv.ORIGINAL_HOLDOUT_START
    assert combined["date"].max() == pd.Timestamp("2026-08-04", tz="UTC")
    assert combined["net_pnl"].sum() == pytest.approx(
        combined_metrics["total_net_pnl"], rel=1e-9
    )
    assert combined.iloc[-1]["cumulative_return"] == pytest.approx(
        combined_metrics["total_return"], rel=1e-9
    )
    assert combined_metrics["trade_entries"] == 7
    assert combined_metrics["active_days"] == 14
    assert summary["evidence_gate_digest"] == s2fv.EXPECTED_EVIDENCE_GATE_DIGEST
    assert summary["evidence_gate"] == s2fv.load_evidence_gate()
    assert summary["post_selection_evidence_gate"] == s2fv.assess_entry_count_gate(
        combined_metrics,
        summary["evidence_gate"],
    )
