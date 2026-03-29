"""
data_fred.py — FRED risk-free rate download (Cell 6).

Function:
  download_fred_rate(since, until, data_dir) -> pd.Series
    Fetches the daily Effective Fed Funds Rate (DFF) from FRED and
    saves it as a Parquet file. Reads FRED_API_KEY from environment.
"""

import os
from pathlib import Path

import pandas as pd
from fredapi import Fred


def download_fred_rate(since, until, data_dir):
    """Download the Effective Fed Funds Rate (DFF) from FRED.

    Args:
        since    : start date string "YYYY-MM-DD"
        until    : end date string "YYYY-MM-DD"
        data_dir : Path — directory to write DFF_daily.parquet into

    Returns:
        dff : pd.Series indexed by UTC datetime, values in annual %
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    fred = Fred(api_key=os.environ["FRED_API_KEY"])

    dff = fred.get_series("DFF", observation_start=since, observation_end=until)
    dff = dff.rename("fed_funds_rate").rename_axis("date")
    dff.index = pd.to_datetime(dff.index, utc=True)

    dff_path = data_dir / "DFF_daily.parquet"
    dff.to_frame().to_parquet(dff_path, engine="pyarrow")
    print(f"Fed Funds Rate: {len(dff):,} observations → {dff_path}")

    return dff
