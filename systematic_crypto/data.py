"""Market-data normalization, storage, and OHLCV quality controls."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


VALUE_COLUMNS = ["open", "high", "low", "close", "volume"]


def ensure_utc_index(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    normalized = obj.copy()
    index = pd.DatetimeIndex(normalized.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    normalized.index = index
    return normalized


def write_parquet_safe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(table, path)


def read_parquet_safe(path: Path) -> pd.DataFrame:
    table = pq.read_table(path)
    return table.to_pandas()


def load_hourly_risk_free_rate(path: Path, *, hours_per_year: int) -> pd.Series:
    rf_frame = read_parquet_safe(path)

    if isinstance(rf_frame, pd.Series):
        rf_series = rf_frame.rename("fed_funds_rate")
    elif "fed_funds_rate" in rf_frame.columns:
        rf_series = rf_frame["fed_funds_rate"]
    elif rf_frame.shape[1] == 1:
        rf_series = rf_frame.iloc[:, 0].rename("fed_funds_rate")
    else:
        raise ValueError(f"Could not identify the risk-free column in {path}")

    rf_series = ensure_utc_index(rf_series).sort_index().astype(float)
    return ((1.0 + rf_series.div(100.0)).pow(1.0 / hours_per_year) - 1.0).rename("risk_free_rate")


def audit_ohlcv(df: pd.DataFrame, symbol: str, stage: str, *, timeframe: str) -> dict:
    audit_frame = ensure_utc_index(df.sort_index())
    index = pd.DatetimeIndex(audit_frame.index)
    expected_index = pd.date_range(index.min(), index.max(), freq=timeframe, tz="UTC")
    gap_hours = index.to_series().diff().dropna().div(pd.Timedelta(hours=1))

    high_low_violations = (audit_frame["high"] < audit_frame["low"]).fillna(False)
    candle_rule_violations = (
        (audit_frame["high"] < audit_frame[["open", "close"]].max(axis=1))
        | (audit_frame["low"] > audit_frame[["open", "close"]].min(axis=1))
    ).fillna(False)

    return {
        "symbol": symbol,
        "stage": stage,
        "rows": len(audit_frame),
        "start": index.min(),
        "end": index.max(),
        "duplicate_timestamps": int(index.duplicated().sum()),
        "missing_bars": int(len(expected_index.difference(index))),
        "null_rows": int(audit_frame[VALUE_COLUMNS].isna().any(axis=1).sum()),
        "null_values": int(audit_frame[VALUE_COLUMNS].isna().sum().sum()),
        "non_positive_price_rows": int((audit_frame[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()),
        "negative_volume_rows": int((audit_frame["volume"] < 0).sum()),
        "zero_volume_rows": int((audit_frame["volume"] == 0).sum()),
        "high_below_low_rows": int(high_low_violations.sum()),
        "candle_rule_violations": int(candle_rule_violations.sum()),
        "max_gap_hours": float(gap_hours.max()) if not gap_hours.empty else 0.0,
        "outliers_flagged": int(audit_frame["is_outlier"].sum()) if "is_outlier" in audit_frame else 0,
        "missing_bar_rows": int(audit_frame["missing_bar"].sum()) if "missing_bar" in audit_frame else 0,
    }

def flag_ohlcv_outliers(
    df: pd.DataFrame,
    risk_free_rate: pd.Series,
    *,
    rolling_window: int = 72,
    zscore_threshold: float = 5.0,
) -> pd.DataFrame:
    flagged = ensure_utc_index(df.sort_index())
    flagged.index.name = "timestamp"
    flagged["missing_bar"] = False
    flagged["simple_return"] = flagged["close"].pct_change()

    aligned_rf = risk_free_rate.reindex(flagged.index, method="ffill")
    flagged["risk_free_rate"] = aligned_rf.shift(1)
    flagged["excess_simple_return"] = flagged["simple_return"] - flagged["risk_free_rate"]

    min_periods = max(rolling_window // 3, 12)
    rolling_returns = flagged["excess_simple_return"].rolling(window=rolling_window, min_periods=min_periods)
    rolling_std = rolling_returns.std(ddof=0).replace(0.0, np.nan)
    flagged["return_zscore"] = (flagged["excess_simple_return"] - rolling_returns.mean()) / rolling_std
    flagged["is_outlier"] = flagged["return_zscore"].abs().gt(zscore_threshold).fillna(False)
    flagged["clean_return"] = flagged["excess_simple_return"]
    return flagged


def run_data_quality_pipeline(
    *,
    data_dir: Path,
    processed_dir: Path,
    symbols: list[str],
    timeframe: str,
    risk_free_rate: pd.Series,
) -> dict:
    processed_dir.mkdir(parents=True, exist_ok=True)

    flagged_frames: dict[str, pd.DataFrame] = {}
    quality_rows: list[dict] = []
    flagged_rows: list[pd.DataFrame] = []

    for symbol in symbols:
        raw_path = data_dir / f"{symbol.replace('/', '')}_{timeframe}.parquet"
        raw_df = ensure_utc_index(read_parquet_safe(raw_path))
        quality_rows.append(audit_ohlcv(raw_df, symbol, "raw", timeframe=timeframe))

        flagged_df = flag_ohlcv_outliers(raw_df, risk_free_rate=risk_free_rate)
        flagged_path = processed_dir / f"{symbol.replace('/', '')}_{timeframe}_flagged.parquet"
        write_parquet_safe(flagged_df, flagged_path)

        flagged_frames[symbol] = flagged_df
        quality_rows.append(audit_ohlcv(flagged_df, symbol, "flagged", timeframe=timeframe))

        symbol_flags = flagged_df.loc[
            flagged_df["is_outlier"],
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "simple_return",
                "risk_free_rate",
                "excess_simple_return",
                "return_zscore",
            ],
        ].copy()
        symbol_flags.insert(0, "symbol", symbol)
        flagged_rows.append(symbol_flags.reset_index())

    quality_report = pd.DataFrame(quality_rows).sort_values(["symbol", "stage"]).reset_index(drop=True)
    post_clean_checks = quality_report.loc[
        quality_report["stage"] == "flagged",
        [
            "symbol",
            "missing_bars",
            "null_rows",
            "non_positive_price_rows",
            "negative_volume_rows",
            "high_below_low_rows",
            "candle_rule_violations",
            "outliers_flagged",
        ],
    ]
    flagged_outliers = pd.concat(flagged_rows, ignore_index=True).sort_values(["symbol", "timestamp"])
    outlier_counts = (
        flagged_outliers.groupby("symbol")
        .size()
        .rename("flagged_outlier_count")
        .reset_index()
        if not flagged_outliers.empty
        else pd.DataFrame({"symbol": symbols, "flagged_outlier_count": 0})
    )

    return {
        "flagged_frames": flagged_frames,
        "quality_report": quality_report,
        "post_clean_checks": post_clean_checks,
        "flagged_outliers": flagged_outliers,
        "outlier_counts": outlier_counts,
    }
