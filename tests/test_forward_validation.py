import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import forward_validation as fv


def make_ohlcv(index: pd.DatetimeIndex, start: float = 100.0) -> pd.DataFrame:
    close = pd.Series(np.arange(len(index), dtype=float) + start, index=index)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000.0,
        },
        index=index,
    )


def test_frozen_spec_digest_and_parameters_are_immutable() -> None:
    spec = fv.load_frozen_spec()

    assert fv.canonical_spec_digest(spec) == fv.EXPECTED_SPEC_DIGEST
    assert spec["frozen_at"] == "2026-03-20"
    assert spec["parameters"] == {
        "ma_window": 140,
        "vol_window": 30,
        "dead_zone": 1.0,
        "signal_clip": 5.0,
        "gross_cap": 10_000.0,
        "rebalance_every": 10,
        "use_mvo": True,
        "cov_window": 120,
        "gamma": 1.0,
    }
    assert set(spec["frozen_history_sha256"]) == {
        "BTCUSDT_1d_cleaned.parquet",
        "ETHUSDT_1d_cleaned.parquet",
        "BNBUSDT_1d_cleaned.parquet",
        "ADAUSDT_1d_cleaned.parquet",
    }


def test_cli_exposes_no_strategy_parameter_overrides() -> None:
    option_dests = {action.dest for action in fv.build_parser()._actions}
    forbidden = {
        "ma_window",
        "vol_window",
        "dead_zone",
        "signal_clip",
        "gross_cap",
        "rebalance_every",
        "use_mvo",
        "cov_window",
        "gamma",
    }

    assert option_dests.isdisjoint(forbidden)


def test_runner_imports_no_optimisation_library() -> None:
    tree = ast.parse(Path(fv.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "optuna" not in imported
    assert "sklearn" not in imported


def test_combined_history_starts_forward_window_strictly_after_cutoff() -> None:
    history_index = pd.date_range(end=fv.FREEZE_CUTOFF, periods=5, freq="D")
    end_date = fv.FREEZE_CUTOFF + pd.Timedelta(days=3)
    forward_index = pd.date_range(
        fv.FREEZE_CUTOFF + pd.Timedelta(days=1),
        end_date,
        freq="D",
    )

    combined = fv.combine_frozen_and_forward(
        make_ohlcv(history_index),
        make_ohlcv(forward_index, start=105.0),
        symbol="BTC/USDT",
        end_date=end_date,
    )

    assert combined.index.max() == end_date
    assert combined.loc[combined.index > fv.FREEZE_CUTOFF].index.equals(forward_index)
    assert combined.loc[combined.index <= fv.FREEZE_CUTOFF].index.equals(history_index)


def test_modified_prefreeze_history_is_rejected() -> None:
    history_index = pd.date_range(end=fv.FREEZE_CUTOFF - pd.Timedelta(days=1), periods=5, freq="D")

    with pytest.raises(fv.FrozenSpecificationError, match="must end exactly"):
        fv.validate_frozen_history(make_ohlcv(history_index), symbol="BTC/USDT")

    with pytest.raises(fv.FrozenSpecificationError, match="content changed"):
        fv.validate_history_hashes(
            {"BTCUSDT_1d_cleaned.parquet": "modified"},
            {"BTCUSDT_1d_cleaned.parquet": "canonical"},
        )


def test_future_or_incomplete_daily_candle_is_rejected() -> None:
    now = pd.Timestamp("2026-07-22 12:00:00", tz="UTC")

    assert fv.resolve_end_date(None, now=now) == pd.Timestamp("2026-07-21", tz="UTC")
    with pytest.raises(ValueError, match="not a completed UTC daily candle"):
        fv.resolve_end_date("2026-07-22", now=now)


def test_forward_metrics_use_cutoff_equity_as_the_return_anchor() -> None:
    index = pd.date_range(fv.FREEZE_CUTOFF, periods=3, freq="D")
    symbols = ["BTC/USDT", "ETH/USDT"]
    close_values = pd.DataFrame(
        {"BTC/USDT": [100.0, 101.0, 102.0], "ETH/USDT": [50.0, 51.0, 52.0]},
        index=index,
    )
    asset_panel = pd.concat({"close": close_values}, axis=1)
    exposure = pd.DataFrame(1_000.0, index=index, columns=symbols)
    asset_returns = close_values.pct_change().fillna(0.0)
    run_state = {
        "net_portfolio_value": pd.Series([10_000.0, 10_100.0, 10_201.0], index=index),
        "gross_pnl": pd.Series([0.0, 100.0, 101.0], index=index),
        "net_pnl": pd.Series([0.0, 100.0, 101.0], index=index),
        "cost_t": pd.Series(0.0, index=index),
        "turnover": pd.Series([0.0, 2_000.0, 0.0], index=index),
        "theta_exec": exposure,
        "delta_theta_exec": pd.DataFrame(
            [[0.0, 0.0], [1_000.0, 1_000.0], [0.0, 0.0]],
            index=index,
            columns=symbols,
        ),
        "r": asset_returns,
    }

    metrics, daily = fv.evaluate_forward_window(
        run_state,
        asset_panel,
        forward_index=index[index > fv.FREEZE_CUTOFF],
    )

    assert metrics["anchor_date"] == "2026-03-20"
    assert metrics["total_return"] == pytest.approx(0.0201)
    assert daily.iloc[0]["cumulative_return"] == pytest.approx(0.01)
    assert daily.index.min() > fv.FREEZE_CUTOFF


def test_dated_snapshots_cannot_be_overwritten(tmp_path: Path) -> None:
    snapshot = tmp_path / "forward_validation" / "snapshots" / "2026-07-21"
    snapshot.mkdir(parents=True)
    (snapshot / "summary.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="immutable"):
        fv.validate_output_target(snapshot)

    fv.validate_output_target(tmp_path / "forward_validation" / "results")


def test_committed_forward_snapshot_is_self_consistent() -> None:
    snapshot = Path("forward_validation/snapshots/2026-07-21")
    summary = json.loads((snapshot / "summary.json").read_text(encoding="utf-8"))
    daily = pd.read_csv(snapshot / "daily.csv", parse_dates=["date"])
    metrics = summary["metrics"]

    assert (snapshot / summary["artifacts"]["figure"]).stat().st_size > 0
    assert summary["frozen_spec_digest"] == fv.EXPECTED_SPEC_DIGEST
    assert summary["frozen_history_sha256"] == summary["frozen_spec"]["frozen_history_sha256"]
    assert len(daily) == metrics["n_days"] == 123
    assert daily["date"].min() > fv.FREEZE_CUTOFF
    assert daily["date"].max() == pd.Timestamp(metrics["end"], tz="UTC")
    assert daily["net_pnl"].sum() == pytest.approx(metrics["total_net_pnl"], rel=1e-9)
    assert daily.iloc[-1]["cumulative_return"] == pytest.approx(metrics["total_return"], rel=1e-9)
