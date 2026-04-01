"""
asset_panel.py — Build the common aligned asset panel (Cell 13).

Function:
  build_asset_panel(cleaned_frames, symbols) -> pd.DataFrame
    Concatenates close and excess_return for all symbols into a single
    DataFrame with MultiIndex columns (field, asset).
    Drops rows where any field/asset is NaN (intersection of valid dates).
"""

import pandas as pd


def build_asset_panel(cleaned_frames, symbols):
    """Build a MultiIndex asset panel aligned across all symbols.

    Creates a DataFrame with two-level columns (field, asset):
      - field ∈ {"close", "excess_return"}
      - asset ∈ symbols

    Args:
        cleaned_frames : dict mapping symbol → cleaned DataFrame
                         (must already have excess_return column from
                          excess_returns.compute_excess_returns)
        symbols        : list of symbol strings (determines column order)

    Returns:
        asset_panel : pd.DataFrame with MultiIndex columns, DatetimeIndex rows.
                      Rows where any column is NaN are dropped.
    """
    close_wide = pd.concat(
        [cleaned_frames[s]["close"] for s in symbols],
        axis=1,
        keys=pd.MultiIndex.from_product([["close"], symbols], names=["field", "asset"]),
    )
    excess_wide = pd.concat(
        [cleaned_frames[s]["excess_return"] for s in symbols],
        axis=1,
        keys=pd.MultiIndex.from_product([["excess_return"], symbols], names=["field", "asset"]),
    )
    asset_panel = pd.concat([close_wide, excess_wide], axis=1).sort_index(axis=1)
    # Intersection of valid dates: all assets have close and excess_return
    asset_panel = asset_panel.dropna(how="any")
    return asset_panel
