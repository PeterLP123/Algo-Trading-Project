"""
signal_construction.py — Trend signal construction (Cell 18).

Function:
  construct_signals(asset_panel, symbols, ma_window, vol_window,
                    dead_zone, signal_clip)
    Computes trend_raw, vol_20, z-score, and discrete trend_position
    for each asset. Returns signal_panel with MultiIndex columns.
"""

import numpy as np
import pandas as pd


def construct_signals(asset_panel, symbols, ma_window, vol_window, dead_zone, signal_clip):
    """Build trend signal panel from close prices.

    Signal logic:
      ma_n       = rolling mean of close over ma_window bars
      trend_raw  = close / ma_n - 1          (distance from MA)
      vol_20     = rolling std of returns over vol_window bars (ddof=0)
      z          = (trend_raw / vol_20).clip(±signal_clip)
      trend_position:
        +1  if z > dead_zone  (long)
        -1  if z < -dead_zone (short)
         0  otherwise         (flat)

    Args:
        asset_panel  : MultiIndex DataFrame with "close" field
        symbols      : list of asset strings
        ma_window    : lookback for moving average
        vol_window   : lookback for rolling volatility
        dead_zone    : activation threshold on z
        signal_clip  : clip z to ±signal_clip

    Returns:
        signal_panel : MultiIndex DataFrame (field × asset) with fields:
                       ma_50, trend_raw, vol_20, z, trend_position
    """
    close_px = asset_panel["close"]

    ma50_list, trend_raw_list, vol_20_list, z_list, trend_position_list = [], [], [], [], []

    for s in symbols:
        c = close_px[s]
        ma_50 = c.rolling(ma_window, min_periods=ma_window).mean()
        trend_raw = c / ma_50 - 1
        ret = c.pct_change()
        vol_20 = ret.rolling(vol_window, min_periods=vol_window).std(ddof=0).replace(0.0, np.nan)
        # z: standardised trend for activation / diagnostics only
        # (single vol division here is not sizing)
        z = (trend_raw / vol_20).clip(-signal_clip, signal_clip)

        tp = np.where(z > dead_zone, 1.0, np.where(z < -dead_zone, -1.0, 0.0))
        trend_position = pd.Series(tp, index=c.index, dtype=float).where(z.notna(), np.nan)

        ma50_list.append(ma_50.rename(s))
        trend_raw_list.append(trend_raw.rename(s))
        vol_20_list.append(vol_20.rename(s))
        z_list.append(z.rename(s))
        trend_position_list.append(trend_position.rename(s))

    mi = pd.MultiIndex.from_product

    def _field_wide(field: str, parts: list) -> pd.DataFrame:
        return pd.concat(parts, axis=1, keys=mi([[field], symbols], names=["field", "asset"]))

    ma_50_wide          = _field_wide("ma_50",          ma50_list)
    trend_raw_wide      = _field_wide("trend_raw",      trend_raw_list)
    vol_20_wide         = _field_wide("vol_20",         vol_20_list)
    z_wide              = _field_wide("z",              z_list)
    trend_position_wide = _field_wide("trend_position", trend_position_list)

    signal_panel = pd.concat(
        [ma_50_wide, trend_raw_wide, vol_20_wide, z_wide, trend_position_wide],
        axis=1,
    ).sort_index(axis=1)

    assert signal_panel.index.equals(asset_panel.index)

    return signal_panel
