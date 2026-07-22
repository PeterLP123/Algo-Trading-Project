import numpy as np
import pandas as pd

from strategy2_pipeline import build_s2_features
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
