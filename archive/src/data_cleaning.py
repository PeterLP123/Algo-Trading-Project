"""
data_cleaning.py — Data quality checks and OHLCV cleaning (Cell 8).

Functions:
  clean_ohlcv(df)
    Remove duplicates, coerce types, validate OHLC rules, flag outliers.

  quick_audit(df, symbol, stage)
    Return a dict of quality metrics (rows, duplicates, missing bars, etc.).

  process_all_symbols(symbols, timeframe, data_dir, value_columns, pandas_freq)
    Load raw parquet for each symbol → clean → save cleaned parquet.
    Returns (cleaned_frames dict, quality_report DataFrame).
"""

from pathlib import Path

import numpy as np
import pandas as pd


def clean_ohlcv(df: pd.DataFrame, value_columns=None) -> pd.DataFrame:
    """Clean an OHLCV DataFrame: dedup, type coercion, validity filter, outlier flag.

    Outliers (rolling z-score > 5 on returns) are FLAGGED, not removed.

    Args:
        df            : raw OHLCV DataFrame with a DatetimeIndex
        value_columns : list of columns to validate; defaults to OHLCV

    Returns:
        Cleaned DataFrame with additional columns:
          simple_return, return_zscore, is_outlier
    """
    if value_columns is None:
        value_columns = ["open", "high", "low", "close", "volume"]

    frame = df.sort_index().copy()
    frame = frame[~frame.index.duplicated(keep="first")]

    for col in value_columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    valid = (
        frame[value_columns].notna().all(axis=1)
        & (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (frame["volume"] >= 0)
        & (frame["high"] >= frame["low"])
        & (frame["high"] >= frame[["open", "close"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close"]].min(axis=1))
    )
    frame = frame.loc[valid].copy()

    r = frame["close"].pct_change()
    z = (r - r.rolling(30, min_periods=10).mean()) / r.rolling(30, min_periods=10).std(ddof=0).replace(0.0, np.nan)
    frame["simple_return"] = r
    frame["return_zscore"] = z
    frame["is_outlier"] = z.abs().gt(5.0).fillna(False)
    return frame


def quick_audit(
    df: pd.DataFrame,
    symbol: str,
    stage: str,
    pandas_freq: str = "1D",
    value_columns=None,
) -> dict:
    """Return quality metrics for a DataFrame at a given processing stage.

    Args:
        df            : DataFrame to audit (must have DatetimeIndex)
        symbol        : asset name for labelling
        stage         : "raw" or "cleaned"
        pandas_freq   : frequency string for missing-bar detection
        value_columns : OHLCV column names (default: standard OHLCV list)

    Returns:
        dict with keys: symbol, stage, rows, duplicates, missing_bars,
                        null_rows, outliers
    """
    if value_columns is None:
        value_columns = ["open", "high", "low", "close", "volume"]
    idx = pd.DatetimeIndex(df.index)
    expected = pd.date_range(idx.min(), idx.max(), freq=pandas_freq, tz="UTC")
    return {
        "symbol": symbol,
        "stage": stage,
        "rows": len(df),
        "duplicates": int(idx.duplicated().sum()),
        "missing_bars": int(len(expected.difference(idx))),
        "null_rows": int(df[value_columns].isna().any(axis=1).sum()),
        "outliers": int(df["is_outlier"].sum()) if "is_outlier" in df else 0,
    }


def process_all_symbols(symbols, timeframe, data_dir, value_columns=None, pandas_freq="1D"):
    """Load raw parquet, clean, and save cleaned parquet for each symbol.

    Args:
        symbols       : list of symbol strings, e.g. ["BTC/USDT", ...]
        timeframe     : timeframe string used in filenames, e.g. "1d"
        data_dir      : Path to directory containing raw parquet files
        value_columns : OHLCV column names (default: standard OHLCV list)
        pandas_freq   : frequency for missing-bar detection (default "1D")

    Returns:
        cleaned_frames : dict mapping symbol → cleaned DataFrame
        quality_report : DataFrame comparing raw vs cleaned quality metrics
    """
    if value_columns is None:
        value_columns = ["open", "high", "low", "close", "volume"]

    data_dir = Path(data_dir)
    cleaned_frames = {}
    audit_rows = []

    for symbol in symbols:
        key = symbol.replace("/", "")
        raw_path = data_dir / f"{key}_{timeframe}.parquet"

        raw = pd.read_parquet(raw_path)
        raw.index = pd.to_datetime(raw.index, utc=True)
        clean = clean_ohlcv(raw, value_columns=value_columns)

        clean_path = data_dir / f"{key}_{timeframe}_cleaned.parquet"
        clean.to_parquet(clean_path, engine="pyarrow")

        cleaned_frames[symbol] = clean
        audit_rows.append(quick_audit(raw,   symbol, "raw",     pandas_freq=pandas_freq, value_columns=value_columns))
        audit_rows.append(quick_audit(clean, symbol, "cleaned", pandas_freq=pandas_freq, value_columns=value_columns))

    quality_report = pd.DataFrame(audit_rows).sort_values(["symbol", "stage"]).reset_index(drop=True)
    return cleaned_frames, quality_report
