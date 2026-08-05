import numpy as np
import pandas as pd
import pytest

from strategy2_pipeline import build_s2_entry_signal, build_s2_features
from strategy_helpers import ensure_utc_index
from wf_trend_pipeline import abdi_ranaldo_spread, apply_rebalance_theta, make_rebalance_mask


def test_ensure_utc_index_localizes_naive_datetimes() -> None:
    frame = pd.DataFrame({"value": [1.0, 2.0]}, index=pd.date_range("2026-01-01", periods=2))

    normalized = ensure_utc_index(frame)

    assert str(normalized.index.tz) == "UTC"
    assert normalized.equals(frame.set_axis(normalized.index))


def test_rebalance_mask_and_hold_periods_are_consistent() -> None:
    index = pd.date_range("2026-01-01", periods=4, tz="UTC")
    targets = pd.DataFrame({"BTC": [1.0, 2.0, 3.0, 4.0]}, index=index)

    mask = make_rebalance_mask(index, every=2)
    executed = apply_rebalance_theta(targets, every=2, rebalance_mask=mask)

    assert mask.tolist() == [True, False, True, False]
    assert executed["BTC"].tolist() == [1.0, 1.0, 3.0, 3.0]


def test_abdi_ranaldo_spread_is_non_negative_when_available() -> None:
    index = pd.date_range("2026-01-01", periods=50, tz="UTC")
    close = pd.Series(np.linspace(100.0, 120.0, len(index)), index=index)
    frame = pd.DataFrame({"high": close * 1.01, "low": close * 0.99, "close": close})

    spread = abdi_ranaldo_spread(frame, window=21)

    assert spread.name == "abdi_ranaldo_spread"
    assert spread.dropna().ge(0.0).all()
    assert not spread.dropna().empty


def test_abdi_ranaldo_corrections_match_the_paper_definitions() -> None:
    index = pd.date_range("2026-01-01", periods=12, tz="UTC")
    close = pd.Series(
        [100.0, 103.0, 99.0, 104.0, 98.0, 105.0, 97.0, 106.0, 96.0, 107.0, 95.0, 108.0],
        index=index,
    )
    frame = pd.DataFrame(
        {"high": close * 1.02, "low": close * 0.98, "close": close}, index=index
    )
    log_close = np.log(frame["close"])
    midpoint = (np.log(frame["high"]) + np.log(frame["low"])) / 2.0
    squared_spread = 4.0 * (log_close.shift(1) - midpoint.shift(1)) * (
        log_close.shift(1) - midpoint
    )

    monthly = abdi_ranaldo_spread(
        frame, window=3, correction="monthly_corrected"
    )
    two_day = abdi_ranaldo_spread(
        frame, window=3, correction="two_day_corrected"
    )

    expected_monthly = np.sqrt(
        squared_spread.rolling(3, min_periods=3).mean().clip(lower=0.0)
    ).rename("abdi_ranaldo_spread")
    expected_two_day = (
        np.sqrt(squared_spread.clip(lower=0.0))
        .rolling(3, min_periods=3)
        .mean()
        .rename("abdi_ranaldo_spread")
    )
    pd.testing.assert_series_equal(monthly, expected_monthly)
    pd.testing.assert_series_equal(two_day, expected_two_day)


def test_abdi_ranaldo_rejects_the_legacy_absolute_correction() -> None:
    index = pd.date_range("2026-01-01", periods=4, tz="UTC")
    frame = pd.DataFrame(
        {"high": [101.0] * 4, "low": [99.0] * 4, "close": [100.0] * 4},
        index=index,
    )

    with pytest.raises(ValueError, match="Unknown Abdi-Ranaldo correction"):
        abdi_ranaldo_spread(frame, correction="legacy_absolute")


def test_strategy_two_features_build_an_equal_weight_altcoin_basket() -> None:
    index = pd.date_range("2026-01-01", periods=50, tz="UTC")
    close = pd.DataFrame(
        {
            "BTC": np.linspace(100.0, 120.0, len(index)),
            "ETH": np.linspace(50.0, 70.0, len(index)),
            "SOL": np.linspace(25.0, 35.0, len(index)),
        },
        index=index,
    )
    notional = pd.DataFrame({"BTC": 1_000.0, "ETH": 500.0, "SOL": 500.0}, index=index)

    features = build_s2_features(close, notional)

    assert features["alt_universe"] == ["ETH", "SOL"]
    assert np.allclose(features["basket_weights"].to_numpy(), 0.5)
    assert np.allclose(features["btc_dom"].to_numpy(), 0.5)


def test_confirmed_reversal_waits_for_an_extreme_dominance_peak() -> None:
    index = pd.date_range("2026-01-01", periods=70, tz="UTC")
    dominance = pd.Series(0.50 + np.arange(len(index)) * 0.001, index=index)
    dominance.iloc[65] = dominance.iloc[64] + 0.006
    dominance.iloc[66] = dominance.iloc[65] - 0.005
    zscore = pd.Series(0.0, index=index)
    zscore.iloc[65] = 3.0
    zscore.iloc[66] = 2.0

    confirmed = build_s2_entry_signal(
        dominance,
        zscore,
        lookback=126,
        entry_pct=0.8,
        roc_window=3,
        roc_pct=0.95,
        entry_mode="confirmed_reversal",
    )
    accelerating = build_s2_entry_signal(
        dominance,
        zscore,
        lookback=126,
        entry_pct=0.8,
        roc_window=3,
        roc_pct=0.95,
        entry_mode="accelerating_spike",
    )

    assert accelerating["entry_signal"].loc[index[65]] == 1
    assert confirmed["entry_signal"].loc[index[65]] == 0
    assert confirmed["dominance_reversal"].loc[index[66]] == 1
    assert confirmed["entry_signal"].loc[index[66]] == 1


def test_strategy_two_entry_mode_rejects_unknown_variants() -> None:
    index = pd.date_range("2026-01-01", periods=70, tz="UTC")
    values = pd.Series(np.linspace(0.4, 0.5, len(index)), index=index)

    with pytest.raises(ValueError, match="Unknown Strategy 2 entry mode"):
        build_s2_entry_signal(
            values,
            values,
            lookback=126,
            entry_pct=0.8,
            roc_window=3,
            roc_pct=0.95,
            entry_mode="retuned_after_holdout",
        )


def test_confirmation_modes_are_distinct_and_use_only_current_information() -> None:
    index = pd.date_range("2026-01-01", periods=70, tz="UTC")
    dominance = pd.Series(0.50 + np.arange(len(index)) * 0.001, index=index)
    dominance.iloc[65] = dominance.iloc[64] + 0.006
    dominance.iloc[66] = dominance.iloc[65] - 0.005
    dominance.iloc[67] = dominance.iloc[66] - 0.003
    zscore = pd.Series(0.0, index=index)
    zscore.iloc[65:67] = [3.0, 2.0]
    relative_momentum = pd.Series(-0.01, index=index)
    relative_momentum.iloc[66:68] = 0.02

    falling = build_s2_entry_signal(
        dominance,
        zscore,
        lookback=126,
        entry_pct=0.8,
        roc_window=3,
        roc_pct=0.95,
        entry_mode="falling_after_extreme",
        alt_relative_momentum=relative_momentum,
    )
    relative = build_s2_entry_signal(
        dominance,
        zscore,
        lookback=126,
        entry_pct=0.8,
        roc_window=3,
        roc_pct=0.95,
        entry_mode="relative_momentum_confirmation",
        alt_relative_momentum=relative_momentum,
    )
    dual = build_s2_entry_signal(
        dominance,
        zscore,
        lookback=126,
        entry_pct=0.8,
        roc_window=3,
        roc_pct=0.95,
        entry_mode="falling_with_relative_momentum",
        alt_relative_momentum=relative_momentum,
    )

    assert falling["entry_signal"].loc[index[66]] == 1
    assert relative["entry_signal"].loc[index[66]] == 1
    assert dual["entry_signal"].loc[index[66]] == 1
    assert dual["entry_signal"].loc[index[65]] == 0
