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
