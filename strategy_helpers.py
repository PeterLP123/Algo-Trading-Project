from __future__ import annotations

from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from statsmodels.tsa.vector_ar.vecm import coint_johansen


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


def infer_rank_95(test_stats: np.ndarray, critical_values_95: np.ndarray) -> int:
    rejections = list(test_stats > critical_values_95)
    if rejections[0] and not rejections[1]:
        return 1
    if not rejections[0]:
        return 0
    return 2


def interpret_pair_result(trace_rank_95: int, maxeig_rank_95: int) -> str:
    if trace_rank_95 == 1 and maxeig_rank_95 == 1:
        return (
            "Confirmed rank-1 cointegration. Both Johansen tests support one stationary spread, "
            "which is the standard setup for pairs trading."
        )
    if trace_rank_95 == 0 and maxeig_rank_95 == 0:
        return (
            "No cointegration evidence at 95%. The data do not support a mean-reverting "
            "spread for this pair under the current specification."
        )
    if trace_rank_95 != maxeig_rank_95:
        return (
            "Mixed evidence. The trace and max-eigenvalue tests disagree, so this pair "
            "should be treated as inconclusive rather than tradable."
        )
    return (
        "Non-standard outcome. Rank 2 in a two-asset system suggests the prices may be "
        "stationary already or the model specification is not appropriate for pairs trading."
    )


def run_pair_diagnostics(
    flagged_frames: dict[str, pd.DataFrame],
    symbols: list[str],
    *,
    det_order: int = 0,
    k_ar_diff: int = 1,
    verbose: bool = False,
) -> dict:
    pair_results: dict[str, dict] = {}
    summary_rows: list[dict] = []

    for left_symbol, right_symbol in combinations(symbols, 2):
        pair_name = f"{left_symbol} vs {right_symbol}"
        pair_prices = pd.concat(
            [
                flagged_frames[left_symbol]["close"].rename(left_symbol),
                flagged_frames[right_symbol]["close"].rename(right_symbol),
            ],
            axis=1,
        ).dropna()
        log_prices = np.log(pair_prices)
        johansen_result = coint_johansen(log_prices, det_order=det_order, k_ar_diff=k_ar_diff)

        trace_table = pd.DataFrame(
            {
                "rank_null": [f"r <= {i}" for i in range(log_prices.shape[1])],
                "trace_stat": johansen_result.lr1,
                "crit_95": johansen_result.cvt[:, 1],
            }
        )
        trace_table["reject_95"] = trace_table["trace_stat"] > trace_table["crit_95"]

        maxeig_table = pd.DataFrame(
            {
                "rank_null": [f"r = {i}" for i in range(log_prices.shape[1])],
                "maxeig_stat": johansen_result.lr2,
                "crit_95": johansen_result.cvm[:, 1],
            }
        )
        maxeig_table["reject_95"] = maxeig_table["maxeig_stat"] > maxeig_table["crit_95"]

        trace_rank_95 = infer_rank_95(johansen_result.lr1, johansen_result.cvt[:, 1])
        maxeig_rank_95 = infer_rank_95(johansen_result.lr2, johansen_result.cvm[:, 1])
        interpretation = interpret_pair_result(trace_rank_95, maxeig_rank_95)
        cointegration_confirmed_95 = trace_rank_95 == 1 and maxeig_rank_95 == 1

        pair_results[pair_name] = {
            "prices": pair_prices,
            "log_prices": log_prices,
            "trace_table": trace_table,
            "maxeig_table": maxeig_table,
            "trace_rank_95": trace_rank_95,
            "maxeig_rank_95": maxeig_rank_95,
            "cointegration_confirmed_95": cointegration_confirmed_95,
            "interpretation": interpretation,
        }
        summary_rows.append(
            {
                "pair": pair_name,
                "observations": len(log_prices),
                "trace_rank_95": trace_rank_95,
                "maxeig_rank_95": maxeig_rank_95,
                "cointegration_confirmed_95": cointegration_confirmed_95,
                "interpretation": interpretation,
            }
        )

        if verbose:
            print("=" * 100)
            print(pair_name)
            print(f"Assumptions: det_order={det_order}, k_ar_diff={k_ar_diff}")
            print(f"Observations used: {len(log_prices):,}")
            print("Trace test at 95%:")
            print(trace_table)
            print("Max-eigenvalue test at 95%:")
            print(maxeig_table)
            print(f"Inferred rank at 95% -> trace: {trace_rank_95}, max-eigenvalue: {maxeig_rank_95}")
            print(f"Interpretation: {interpretation}\n")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["cointegration_confirmed_95", "pair"],
        ascending=[False, True],
    ).reset_index(drop=True)

    confirmed_pairs = summary_df.loc[summary_df["cointegration_confirmed_95"], "pair"].tolist()
    mixed_pairs = summary_df.loc[summary_df["trace_rank_95"] != summary_df["maxeig_rank_95"], "pair"].tolist()

    return {
        "summary_df": summary_df,
        "pair_results": pair_results,
        "confirmed_pairs": confirmed_pairs,
        "mixed_pairs": mixed_pairs,
        "det_order": det_order,
        "k_ar_diff": k_ar_diff,
    }


def build_cross_sectional_inputs(
    flagged_frames: dict[str, pd.DataFrame],
    asset_columns: list[str],
    *,
    signal_return_column: str,
    pnl_return_column: str,
    beta_window: int,
    signal_lookback: int,
    dispersion_lookback: int,
) -> dict:
    close_df = pd.concat(
        [flagged_frames[symbol]["close"].rename(symbol) for symbol in asset_columns],
        axis=1,
    ).dropna()
    signal_return_df = pd.concat(
        [flagged_frames[symbol][signal_return_column].rename(symbol) for symbol in asset_columns],
        axis=1,
    ).reindex(close_df.index)
    pnl_return_df = pd.concat(
        [flagged_frames[symbol][pnl_return_column].rename(symbol) for symbol in asset_columns],
        axis=1,
    ).reindex(close_df.index)

    combined_mask = signal_return_df.notna().all(axis=1) & pnl_return_df.notna().all(axis=1)
    close_df = close_df.loc[combined_mask]
    signal_return_df = signal_return_df.loc[combined_mask]
    pnl_return_df = pnl_return_df.loc[combined_mask]

    market_return_series = signal_return_df.mean(axis=1).rename("equal_weight_market_return")
    rolling_market_var = market_return_series.rolling(
        beta_window,
        min_periods=max(beta_window // 3, 24),
    ).var(ddof=0).replace(0.0, np.nan)

    beta_df = pd.DataFrame(index=signal_return_df.index, columns=asset_columns, dtype=float)
    for symbol in asset_columns:
        rolling_cov = signal_return_df[symbol].rolling(
            beta_window,
            min_periods=max(beta_window // 3, 24),
        ).cov(market_return_series)
        beta_df[symbol] = rolling_cov.div(rolling_market_var)

    residual_return_df = signal_return_df.sub(beta_df.mul(market_return_series, axis=0), axis=0)
    residual_signal_df = residual_return_df.rolling(
        signal_lookback,
        min_periods=max(signal_lookback // 2, 3),
    ).mean()

    cross_sectional_mean = residual_signal_df.mean(axis=1)
    cross_sectional_std = residual_signal_df.std(axis=1, ddof=0).replace(0.0, np.nan)
    zscore_df = residual_signal_df.sub(cross_sectional_mean, axis=0).div(cross_sectional_std, axis=0)

    cross_sectional_dispersion_series = residual_signal_df.std(axis=1, ddof=0).rename("cross_sectional_dispersion")
    dispersion_floor_series = cross_sectional_dispersion_series.rolling(
        dispersion_lookback,
        min_periods=max(dispersion_lookback // 3, 24),
    ).median().shift(1).rename("dispersion_floor")

    return {
        "close_df": close_df,
        "signal_return_df": signal_return_df,
        "pnl_return_df": pnl_return_df,
        "market_return_series": market_return_series,
        "beta_df": beta_df,
        "residual_return_df": residual_return_df,
        "residual_signal_df": residual_signal_df,
        "zscore_df": zscore_df,
        "cross_sectional_dispersion_series": cross_sectional_dispersion_series,
        "dispersion_floor_series": dispersion_floor_series,
    }


def build_top_bottom_signal_frame(
    lagged_zscore_df: pd.DataFrame,
    lagged_dispersion_series: pd.Series,
    dispersion_floor_series: pd.Series,
    regime_filter_series: pd.Series | None = None,
    *,
    entry_zscore: float,
    exit_zscore: float,
    rebalance_every_bars: int,
    min_hold_bars: int,
    top_k: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    signal_rows: list[pd.Series] = []
    trade_log: list[dict] = []
    rebalance_mask = pd.Series(False, index=lagged_zscore_df.index, name="rebalance_bar")

    current_signal = pd.Series(0.0, index=lagged_zscore_df.columns)
    holding_bars = 0
    entry_timestamp = None

    for step, timestamp in enumerate(lagged_zscore_df.index):
        scores = lagged_zscore_df.loc[timestamp]
        rebalance_mask.loc[timestamp] = step % rebalance_every_bars == 0

        dispersion_value = lagged_dispersion_series.loc[timestamp]
        dispersion_floor = dispersion_floor_series.loc[timestamp]
        dispersion_ok = (
            pd.notna(dispersion_value)
            and pd.notna(dispersion_floor)
            and dispersion_value >= dispersion_floor
        )
        regime_ok = True if regime_filter_series is None else bool(regime_filter_series.loc[timestamp])

        if current_signal.abs().sum() > 0:
            holding_bars += 1

        exited_this_bar = False
        if rebalance_mask.loc[timestamp] and scores.notna().all():
            if current_signal.abs().sum() > 0 and holding_bars >= min_hold_bars:
                active_longs = current_signal[current_signal > 0].index
                active_shorts = current_signal[current_signal < 0].index

                long_exit = len(active_longs) == 0 or scores.loc[active_longs].ge(-exit_zscore).all()
                short_exit = len(active_shorts) == 0 or scores.loc[active_shorts].le(exit_zscore).all()

                if (not dispersion_ok) or (long_exit and short_exit):
                    trade_log.append(
                        {
                            "entry_time": entry_timestamp,
                            "exit_time": timestamp,
                            "holding_bars": holding_bars,
                            "completed": True,
                        }
                    )
                    current_signal = pd.Series(0.0, index=lagged_zscore_df.columns)
                    holding_bars = 0
                    entry_timestamp = None
                    exited_this_bar = True

            if current_signal.abs().sum() == 0 and (not exited_this_bar) and dispersion_ok and regime_ok:
                long_candidates = scores[scores <= -entry_zscore].sort_values().head(top_k).index
                short_candidates = scores[scores >= entry_zscore].sort_values(ascending=False).head(top_k).index

                if len(long_candidates) == top_k and len(short_candidates) == top_k:
                    current_signal = pd.Series(0.0, index=lagged_zscore_df.columns)
                    current_signal.loc[long_candidates] = 1.0
                    current_signal.loc[short_candidates] = -1.0
                    holding_bars = 0
                    entry_timestamp = timestamp

        signal_rows.append(current_signal.copy())

    if entry_timestamp is not None:
        trade_log.append(
            {
                "entry_time": entry_timestamp,
                "exit_time": lagged_zscore_df.index[-1],
                "holding_bars": holding_bars,
                "completed": False,
            }
        )

    signal_df = pd.DataFrame(signal_rows, index=lagged_zscore_df.index)
    signal_df.columns = lagged_zscore_df.columns
    trade_log_df = pd.DataFrame(trade_log)
    return signal_df, trade_log_df, rebalance_mask


def build_tsmom_signal_frame(
    raw_signal_df: pd.DataFrame,
    *,
    rebalance_every_bars: int,
    min_hold_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    signal_rows: list[pd.Series] = []
    trade_log: list[dict] = []
    rebalance_mask = pd.Series(False, index=raw_signal_df.index, name="rebalance_bar")

    current_signal = pd.Series(0.0, index=raw_signal_df.columns)
    bars_since_last_trade = min_hold_bars
    entry_timestamp = None

    for step, timestamp in enumerate(raw_signal_df.index):
        is_rebalance = step % rebalance_every_bars == 0
        rebalance_mask.loc[timestamp] = is_rebalance
        new_signal = raw_signal_df.loc[timestamp]

        if is_rebalance and bars_since_last_trade >= min_hold_bars:
            if new_signal.notna().all() and not new_signal.equals(current_signal):
                if current_signal.abs().sum() > 0 and entry_timestamp is not None:
                    trade_log.append(
                        {
                            "entry_time": entry_timestamp,
                            "exit_time": timestamp,
                            "holding_bars": bars_since_last_trade,
                            "completed": True,
                        }
                    )
                current_signal = new_signal.copy()
                bars_since_last_trade = 0
                entry_timestamp = timestamp

        bars_since_last_trade += 1
        signal_rows.append(current_signal.copy())

    if entry_timestamp is not None:
        trade_log.append(
            {
                "entry_time": entry_timestamp,
                "exit_time": raw_signal_df.index[-1],
                "holding_bars": bars_since_last_trade,
                "completed": False,
            }
        )

    signal_df = pd.DataFrame(signal_rows, index=raw_signal_df.index)
    signal_df.columns = raw_signal_df.columns
    trade_log_df = pd.DataFrame(trade_log)
    return signal_df, trade_log_df, rebalance_mask


def build_mean_reversion_regime_filter(
    market_return_series: pd.Series,
    *,
    trend_lookback_bars: int,
    trend_vol_window: int,
    trend_zscore_cap: float,
) -> dict:
    rolling_trend_mean = market_return_series.rolling(
        trend_lookback_bars,
        min_periods=max(trend_lookback_bars // 3, 24),
    ).mean()
    rolling_trend_vol = market_return_series.rolling(
        trend_vol_window,
        min_periods=max(trend_vol_window // 3, 24),
    ).std(ddof=0)
    annualized_like_scale = rolling_trend_vol.div(np.sqrt(max(trend_lookback_bars, 1)))
    trend_strength_series = rolling_trend_mean.abs().div(annualized_like_scale.replace(0.0, np.nan)).rename(
        "market_trend_strength"
    )
    regime_filter_series = trend_strength_series.le(trend_zscore_cap).fillna(False).rename("mean_reversion_regime_ok")
    return {
        "trend_strength_series": trend_strength_series,
        "regime_filter_series": regime_filter_series,
    }


def build_inverse_vol_weights(signal_row: pd.Series, inv_vol_row: pd.Series) -> pd.Series:
    weight_row = pd.Series(0.0, index=signal_row.index)
    long_assets = signal_row[signal_row > 0].index
    short_assets = signal_row[signal_row < 0].index

    if len(long_assets) == 0 or len(short_assets) == 0:
        return weight_row

    long_inv_vol = inv_vol_row.loc[long_assets].replace([np.inf, -np.inf], np.nan).dropna()
    short_inv_vol = inv_vol_row.loc[short_assets].replace([np.inf, -np.inf], np.nan).dropna()
    if long_inv_vol.empty or short_inv_vol.empty:
        return weight_row

    weight_row.loc[long_inv_vol.index] = 0.5 * long_inv_vol / long_inv_vol.sum()
    weight_row.loc[short_inv_vol.index] = -0.5 * short_inv_vol / short_inv_vol.sum()
    return weight_row


def build_tsmom_inverse_vol_weights(signal_row: pd.Series, inv_vol_row: pd.Series) -> pd.Series:
    weight_row = pd.Series(0.0, index=signal_row.index)
    long_assets = signal_row[signal_row > 0].index
    short_assets = signal_row[signal_row < 0].index
    active_assets = long_assets.append(short_assets)

    if len(active_assets) == 0:
        return weight_row

    active_inv_vol = inv_vol_row.loc[active_assets].replace([np.inf, -np.inf], np.nan).dropna()
    if active_inv_vol.empty:
        return weight_row

    normed = active_inv_vol / active_inv_vol.sum()
    for asset in normed.index:
        weight_row.loc[asset] = normed.loc[asset] * signal_row.loc[asset]
    return weight_row


def build_conviction_scale_series(
    active_strength_series: pd.Series,
    dispersion_ratio_series: pd.Series,
    *,
    scale_floor: float,
    conviction_span: float,
) -> pd.Series:
    strength_component = ((active_strength_series - 1.0) / max(conviction_span, 1e-9)).clip(lower=0.0, upper=1.0)
    dispersion_component = (dispersion_ratio_series - 1.0).clip(lower=0.0, upper=1.0)
    scale = scale_floor + (1.0 - scale_floor) * (0.65 * strength_component + 0.35 * dispersion_component)
    return scale.fillna(0.0).clip(lower=0.0, upper=1.0).rename("conviction_scale")


def build_cross_sectional_conviction_target_weights(
    signal_df: pd.DataFrame,
    inv_vol_df: pd.DataFrame,
    abs_zscore_df: pd.DataFrame,
    conviction_scale_series: pd.Series,
    *,
    entry_zscore: float,
) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for timestamp in signal_df.index:
        signal_row = signal_df.loc[timestamp]
        weight_row = pd.Series(0.0, index=signal_row.index)
        long_assets = signal_row[signal_row > 0].index
        short_assets = signal_row[signal_row < 0].index
        if len(long_assets) == 0 or len(short_assets) == 0:
            rows.append(weight_row)
            continue

        inv_vol_row = inv_vol_df.loc[timestamp].replace([np.inf, -np.inf], np.nan)
        abs_zscore_row = abs_zscore_df.loc[timestamp]
        long_scores = (
            inv_vol_row.loc[long_assets]
            .mul(abs_zscore_row.loc[long_assets].clip(lower=entry_zscore))
            .dropna()
        )
        short_scores = (
            inv_vol_row.loc[short_assets]
            .mul(abs_zscore_row.loc[short_assets].clip(lower=entry_zscore))
            .dropna()
        )
        if long_scores.empty or short_scores.empty:
            rows.append(weight_row)
            continue

        conviction_scale = float(conviction_scale_series.loc[timestamp])
        weight_row.loc[long_scores.index] = 0.5 * conviction_scale * long_scores / long_scores.sum()
        weight_row.loc[short_scores.index] = -0.5 * conviction_scale * short_scores / short_scores.sum()
        rows.append(weight_row)

    target_weight_df = pd.DataFrame(rows, index=signal_df.index)
    target_weight_df.columns = signal_df.columns
    return target_weight_df


def build_inverse_vol_target_weights(
    signal_df: pd.DataFrame,
    inv_vol_df: pd.DataFrame,
    *,
    mode: str,
) -> pd.DataFrame:
    if mode == "cross_sectional":
        builder = build_inverse_vol_weights
    elif mode == "tsmom":
        builder = build_tsmom_inverse_vol_weights
    else:
        raise ValueError(f"Unknown inverse-vol weighting mode: {mode}")

    target_weight_df = pd.DataFrame(
        [builder(signal_df.loc[timestamp], inv_vol_df.loc[timestamp]) for timestamp in signal_df.index],
        index=signal_df.index,
    )
    target_weight_df.columns = signal_df.columns
    return target_weight_df


def run_execution_engine(
    *,
    close_df: pd.DataFrame,
    pnl_return_df: pd.DataFrame,
    target_weight_df: pd.DataFrame,
    rebalance_mask: pd.Series,
    initial_capital_usdt: float,
    gross_exposure_cap_usdt: float,
    max_gross_leverage: float,
    transaction_cost_bps: float,
    max_turnover_fraction: float,
    liquidation_equity_floor_usdt: float,
) -> dict:
    theta_rows: list[pd.Series] = []
    asset_gross_pnl_rows: list[pd.Series] = []
    asset_net_pnl_rows: list[pd.Series] = []
    gross_pnl_values: list[float] = []
    net_pnl_values: list[float] = []
    turnover_values: list[float] = []
    transaction_cost_values: list[float] = []
    gross_exposure_values: list[float] = []
    allowed_gross_values: list[float] = []
    exposure_utilization_values: list[float] = []
    equity_before_mark_values: list[float] = []
    equity_after_mark_values: list[float] = []
    equity_values: list[float] = []
    active_assets_values: list[int] = []

    liquidated = False
    hard_notional_cap_usdt = min(gross_exposure_cap_usdt, 100_000.0)
    previous_theta = pd.Series(0.0, index=close_df.columns, dtype=float)
    previous_returns = pd.Series(0.0, index=close_df.columns, dtype=float)
    equity_prev = initial_capital_usdt

    for step_idx, timestamp in enumerate(close_df.index):
        current_returns = pnl_return_df.loc[timestamp].fillna(0.0)
        equity_before_mark = equity_prev

        if step_idx == 0:
            theta_pretrade = previous_theta.copy()
        else:
            theta_pretrade = previous_theta * (1.0 + previous_returns)

        allowed_gross_exposure = min(hard_notional_cap_usdt, max_gross_leverage * equity_before_mark)

        if liquidated or equity_before_mark <= liquidation_equity_floor_usdt:
            liquidated = True
            target_theta = pd.Series(0.0, index=close_df.columns, dtype=float)
        elif rebalance_mask.loc[timestamp]:
            target_theta_full = target_weight_df.loc[timestamp].fillna(0.0) * allowed_gross_exposure
            turnover_raw = (target_theta_full - theta_pretrade).abs().sum()

            if (
                target_theta_full.abs().sum() > theta_pretrade.abs().sum()
                and turnover_raw > 0
                and max_turnover_fraction < np.inf
            ):
                max_turnover = max_turnover_fraction * max(allowed_gross_exposure, 1.0)
                if turnover_raw > max_turnover > 0:
                    scale = max_turnover / turnover_raw
                    target_theta = theta_pretrade + scale * (target_theta_full - theta_pretrade)
                else:
                    target_theta = target_theta_full
            else:
                target_theta = target_theta_full

            post_trade_gross = target_theta.abs().sum()
            if post_trade_gross > allowed_gross_exposure > 0:
                target_theta = target_theta * (allowed_gross_exposure / post_trade_gross)
        else:
            target_theta = theta_pretrade.copy()

        trade_vector = target_theta - theta_pretrade
        trade_abs = trade_vector.abs()
        turnover = trade_abs.sum()
        transaction_cost = turnover * transaction_cost_bps / 10_000.0

        gross_pnl_asset = target_theta * current_returns
        gross_pnl = gross_pnl_asset.sum()
        equity_after_mark = max(equity_before_mark + gross_pnl, 0.0)
        net_pnl = gross_pnl - transaction_cost
        net_pnl_asset = gross_pnl_asset.copy()
        if net_pnl_asset.abs().sum() > 0:
            allocation = net_pnl_asset.abs() / net_pnl_asset.abs().sum()
            net_pnl_asset = gross_pnl_asset - allocation * transaction_cost
        elif trade_abs.sum() > 0:
            allocation = trade_abs / trade_abs.sum()
            net_pnl_asset = -allocation * transaction_cost

        equity_t = max(equity_before_mark + net_pnl, 0.0)
        if equity_t <= liquidation_equity_floor_usdt:
            liquidated = True

        gross_exposure = target_theta.abs().sum()
        exposure_utilization = gross_exposure / allowed_gross_exposure if allowed_gross_exposure > 0 else 0.0

        theta_rows.append(target_theta)
        asset_gross_pnl_rows.append(gross_pnl_asset)
        asset_net_pnl_rows.append(net_pnl_asset)
        gross_pnl_values.append(gross_pnl)
        net_pnl_values.append(net_pnl)
        turnover_values.append(turnover)
        transaction_cost_values.append(transaction_cost)
        gross_exposure_values.append(gross_exposure)
        allowed_gross_values.append(allowed_gross_exposure)
        exposure_utilization_values.append(exposure_utilization)
        equity_before_mark_values.append(equity_before_mark)
        equity_after_mark_values.append(equity_after_mark)
        equity_values.append(equity_t)
        active_assets_values.append(int(target_theta.abs().gt(0.0).sum()))

        previous_theta = pd.Series(0.0, index=close_df.columns, dtype=float) if liquidated else target_theta
        previous_returns = current_returns
        equity_prev = equity_t

    theta_df = pd.DataFrame(theta_rows, index=close_df.index)
    theta_df.columns = close_df.columns
    asset_gross_pnl_df = pd.DataFrame(asset_gross_pnl_rows, index=close_df.index)
    asset_gross_pnl_df.columns = close_df.columns
    asset_net_pnl_df = pd.DataFrame(asset_net_pnl_rows, index=close_df.index)
    asset_net_pnl_df.columns = close_df.columns

    gross_pnl_series = pd.Series(gross_pnl_values, index=close_df.index, name="gross_pnl")
    portfolio_pnl_series = pd.Series(net_pnl_values, index=close_df.index, name="portfolio_pnl")
    transaction_cost_series = pd.Series(transaction_cost_values, index=close_df.index, name="transaction_cost")
    turnover_series = pd.Series(turnover_values, index=close_df.index, name="turnover")
    gross_exposure_series = pd.Series(gross_exposure_values, index=close_df.index, name="gross_exposure")
    allowed_gross_series = pd.Series(allowed_gross_values, index=close_df.index, name="allowed_gross_exposure")
    exposure_utilization_series = pd.Series(
        exposure_utilization_values,
        index=close_df.index,
        name="exposure_utilization",
    )
    equity_before_mark_series = pd.Series(
        equity_before_mark_values,
        index=close_df.index,
        name="equity_before_mark",
    )
    equity_after_mark_series = pd.Series(
        equity_after_mark_values,
        index=close_df.index,
        name="equity_after_mark",
    )
    equity_curve_series = pd.Series(equity_values, index=close_df.index, name="equity_curve")
    portfolio_return_series = portfolio_pnl_series.div(equity_before_mark_series.replace(0.0, np.nan)).fillna(0.0)
    gross_return_series = gross_pnl_series.div(equity_before_mark_series.replace(0.0, np.nan)).fillna(0.0)
    gross_equity_curve_series = (initial_capital_usdt + gross_pnl_series.cumsum()).rename("gross_equity_curve")
    drawdown_series = equity_curve_series.div(equity_curve_series.cummax()).sub(1.0).rename("drawdown")
    active_assets_series = pd.Series(active_assets_values, index=close_df.index, name="active_assets")

    non_rebalance_mask = ~rebalance_mask.astype(bool)
    non_rebalance_turnover_zero = bool(turnover_series.loc[non_rebalance_mask].abs().le(1e-12).all())
    theta_drift_reference = theta_df.shift(1).mul(1.0 + pnl_return_df.shift(1).fillna(0.0))
    drift_check_mask = non_rebalance_mask & theta_df.index.to_series().ne(theta_df.index[0])
    non_rebalance_drift_consistent = bool(
        (
            theta_df.loc[drift_check_mask]
            .sub(theta_drift_reference.loc[drift_check_mask], fill_value=0.0)
            .abs()
            .le(1e-8)
            .to_numpy()
            .all()
        )
    )

    rebalance_mask_bool = rebalance_mask.astype(bool)
    assert (gross_exposure_series <= hard_notional_cap_usdt + 1e-9).all()
    assert (
        (
            gross_exposure_series.loc[rebalance_mask_bool]
            <= allowed_gross_series.loc[rebalance_mask_bool] + 1e-9
        )
        | (gross_exposure_series.loc[rebalance_mask_bool] == 0.0)
    ).all()
    assert non_rebalance_turnover_zero
    assert non_rebalance_drift_consistent

    sanity_checks = {
        "non_rebalance_turnover_zero": non_rebalance_turnover_zero,
        "non_rebalance_drift_consistent": non_rebalance_drift_consistent,
        "final_equity": float(equity_curve_series.iloc[-1]),
        "total_transaction_costs": float(transaction_cost_series.sum()),
        "average_turnover": float(turnover_series.mean()),
        "average_gross_exposure": float(gross_exposure_series.mean()),
        "max_gross_exposure": float(gross_exposure_series.max()),
    }

    return {
        "theta_df": theta_df,
        "asset_gross_pnl_df": asset_gross_pnl_df,
        "asset_net_pnl_df": asset_net_pnl_df,
        "gross_pnl_series": gross_pnl_series,
        "portfolio_pnl_series": portfolio_pnl_series,
        "transaction_cost_series": transaction_cost_series,
        "turnover_series": turnover_series,
        "gross_exposure_series": gross_exposure_series,
        "allowed_gross_series": allowed_gross_series,
        "exposure_utilization_series": exposure_utilization_series,
        "equity_before_mark_series": equity_before_mark_series,
        "equity_after_mark_series": equity_after_mark_series,
        "equity_curve_series": equity_curve_series,
        "gross_equity_curve_series": gross_equity_curve_series,
        "portfolio_return_series": portfolio_return_series,
        "gross_return_series": gross_return_series,
        "drawdown_series": drawdown_series,
        "active_assets_series": active_assets_series,
        "sanity_checks": sanity_checks,
        "liquidated": liquidated,
    }


def run_cross_sectional_strategy(
    *,
    close_df: pd.DataFrame,
    signal_return_df: pd.DataFrame,
    pnl_return_df: pd.DataFrame,
    zscore_df: pd.DataFrame,
    cross_sectional_dispersion_series: pd.Series,
    dispersion_floor_series: pd.Series,
    market_return_series: pd.Series | None = None,
    entry_zscore: float,
    exit_zscore: float,
    rebalance_every_bars: int,
    min_hold_bars: int,
    top_k: int,
    vol_window: int,
    use_regime_filter: bool = False,
    regime_trend_lookback_bars: int = 168,
    regime_trend_vol_window: int = 168,
    regime_trend_zscore_cap: float = 0.75,
    weighting_mode: str = "inverse_vol",
    conviction_scale_floor: float = 0.0,
    conviction_span: float = 1.0,
    initial_capital_usdt: float,
    gross_exposure_cap_usdt: float,
    max_gross_leverage: float,
    transaction_cost_bps: float,
    max_turnover_fraction: float,
    liquidation_equity_floor_usdt: float,
) -> dict:
    lagged_zscore_df = zscore_df.shift(1)
    lagged_dispersion_series = cross_sectional_dispersion_series.shift(1)
    regime_state = None
    if use_regime_filter:
        if market_return_series is None:
            raise ValueError("market_return_series is required when use_regime_filter=True")
        regime_state = build_mean_reversion_regime_filter(
            market_return_series,
            trend_lookback_bars=regime_trend_lookback_bars,
            trend_vol_window=regime_trend_vol_window,
            trend_zscore_cap=regime_trend_zscore_cap,
        )
        regime_filter_series = regime_state["regime_filter_series"].astype("boolean").shift(1).fillna(False).astype(bool)
    else:
        regime_filter_series = None

    signal_df, trade_log_df, rebalance_mask = build_top_bottom_signal_frame(
        lagged_zscore_df,
        lagged_dispersion_series,
        dispersion_floor_series,
        regime_filter_series,
        entry_zscore=entry_zscore,
        exit_zscore=exit_zscore,
        rebalance_every_bars=rebalance_every_bars,
        min_hold_bars=min_hold_bars,
        top_k=top_k,
    )

    vol_df = pnl_return_df.rolling(
        vol_window,
        min_periods=max(vol_window // 3, 24),
    ).std(ddof=0).replace(0.0, np.nan)
    lagged_inv_vol_df = (1.0 / vol_df).shift(1)
    if weighting_mode == "inverse_vol":
        target_weight_df = build_inverse_vol_target_weights(signal_df, lagged_inv_vol_df, mode="cross_sectional")
    elif weighting_mode == "conviction_scaled":
        dispersion_ratio_series = lagged_dispersion_series.div(dispersion_floor_series.replace(0.0, np.nan))
        active_strength_series = (
            lagged_zscore_df.abs().where(signal_df.abs() > 0).mean(axis=1).div(max(entry_zscore, 1e-9))
        )
        conviction_scale_series = build_conviction_scale_series(
            active_strength_series,
            dispersion_ratio_series,
            scale_floor=conviction_scale_floor,
            conviction_span=conviction_span,
        )
        target_weight_df = build_cross_sectional_conviction_target_weights(
            signal_df,
            lagged_inv_vol_df,
            lagged_zscore_df.abs(),
            conviction_scale_series,
            entry_zscore=entry_zscore,
        )
    else:
        raise ValueError(f"Unknown cross-sectional weighting_mode: {weighting_mode}")

    execution_state = run_execution_engine(
        close_df=close_df,
        pnl_return_df=pnl_return_df,
        target_weight_df=target_weight_df,
        rebalance_mask=rebalance_mask,
        initial_capital_usdt=initial_capital_usdt,
        gross_exposure_cap_usdt=gross_exposure_cap_usdt,
        max_gross_leverage=max_gross_leverage,
        transaction_cost_bps=transaction_cost_bps,
        max_turnover_fraction=max_turnover_fraction,
        liquidation_equity_floor_usdt=liquidation_equity_floor_usdt,
    )

    return {
        "signal_df": signal_df,
        "trade_log_df": trade_log_df,
        "rebalance_mask": rebalance_mask,
        "vol_df": vol_df,
        "target_weight_df": target_weight_df,
        "regime_filter_series": regime_filter_series,
        "trend_strength_series": None if regime_state is None else regime_state["trend_strength_series"],
        "weighting_mode": weighting_mode,
        **execution_state,
    }


def run_tsmom_strategy(
    *,
    close_df: pd.DataFrame,
    pnl_return_df: pd.DataFrame,
    raw_signal_df: pd.DataFrame,
    rebalance_every_bars: int,
    min_hold_bars: int,
    vol_window: int,
    initial_capital_usdt: float,
    gross_exposure_cap_usdt: float,
    max_gross_leverage: float,
    transaction_cost_bps: float,
    max_turnover_fraction: float,
    liquidation_equity_floor_usdt: float,
) -> dict:
    signal_df, trade_log_df, rebalance_mask = build_tsmom_signal_frame(
        raw_signal_df,
        rebalance_every_bars=rebalance_every_bars,
        min_hold_bars=min_hold_bars,
    )

    vol_df = pnl_return_df.rolling(
        vol_window,
        min_periods=max(vol_window // 3, 24),
    ).std(ddof=0).replace(0.0, np.nan)
    lagged_inv_vol_df = (1.0 / vol_df).shift(1)
    target_weight_df = build_inverse_vol_target_weights(signal_df, lagged_inv_vol_df, mode="tsmom")

    execution_state = run_execution_engine(
        close_df=close_df,
        pnl_return_df=pnl_return_df,
        target_weight_df=target_weight_df,
        rebalance_mask=rebalance_mask,
        initial_capital_usdt=initial_capital_usdt,
        gross_exposure_cap_usdt=gross_exposure_cap_usdt,
        max_gross_leverage=max_gross_leverage,
        transaction_cost_bps=transaction_cost_bps,
        max_turnover_fraction=max_turnover_fraction,
        liquidation_equity_floor_usdt=liquidation_equity_floor_usdt,
    )

    return {
        "signal_df": signal_df,
        "trade_log_df": trade_log_df,
        "rebalance_mask": rebalance_mask,
        "vol_df": vol_df,
        "target_weight_df": target_weight_df,
        **execution_state,
    }


def summarize_strategy_state(
    strategy_state: dict,
    *,
    initial_capital_usdt: float,
    hours_per_year: int,
) -> dict:
    gross_pnl_series = strategy_state["gross_pnl_series"]
    portfolio_pnl_series = strategy_state["portfolio_pnl_series"]
    transaction_cost_series = strategy_state["transaction_cost_series"]
    turnover_series = strategy_state["turnover_series"]
    gross_exposure_series = strategy_state["gross_exposure_series"]
    exposure_utilization_series = strategy_state["exposure_utilization_series"]
    equity_curve_series = strategy_state["equity_curve_series"]
    portfolio_return_series = strategy_state["portfolio_return_series"]
    gross_return_series = strategy_state["gross_return_series"]
    drawdown_series = strategy_state["drawdown_series"]
    trade_log_df = strategy_state["trade_log_df"]

    active_mask = gross_exposure_series > 0
    active_net_pnl_series = portfolio_pnl_series[active_mask]
    rebalance_turnover_series = turnover_series[strategy_state["rebalance_mask"]]

    annualized_net_return = portfolio_return_series.mean() * hours_per_year
    annualized_net_vol = portfolio_return_series.std(ddof=0) * np.sqrt(hours_per_year)
    annualized_gross_return = gross_return_series.mean() * hours_per_year
    annualized_gross_vol = gross_return_series.std(ddof=0) * np.sqrt(hours_per_year)
    downside_return_series = portfolio_return_series.clip(upper=0.0)
    annualized_downside_vol = np.sqrt((downside_return_series**2).mean()) * np.sqrt(hours_per_year)
    max_drawdown = abs(drawdown_series.min())

    avg_holding_bars = trade_log_df["holding_bars"].mean() if not trade_log_df.empty else np.nan
    trade_count = len(trade_log_df)

    return {
        "ending_equity": equity_curve_series.iloc[-1],
        "total_gross_pnl": gross_pnl_series.sum(),
        "total_transaction_costs": transaction_cost_series.sum(),
        "total_net_pnl": portfolio_pnl_series.sum(),
        "total_return_pct": 100.0 * (equity_curve_series.iloc[-1] / initial_capital_usdt - 1.0),
        "annualized_gross_return_pct": 100.0 * annualized_gross_return,
        "annualized_gross_vol_pct": 100.0 * annualized_gross_vol,
        "annualized_net_return_pct": 100.0 * annualized_net_return,
        "annualized_net_vol_pct": 100.0 * annualized_net_vol,
        "gross_sharpe": annualized_gross_return / annualized_gross_vol if annualized_gross_vol > 0 else np.nan,
        "net_sharpe": annualized_net_return / annualized_net_vol if annualized_net_vol > 0 else np.nan,
        "net_sortino": annualized_net_return / annualized_downside_vol if annualized_downside_vol > 0 else np.nan,
        "net_calmar": annualized_net_return / max_drawdown if max_drawdown > 0 else np.nan,
        "max_drawdown_pct": 100.0 * drawdown_series.min(),
        "active_bar_win_rate_pct": 100.0 * (active_net_pnl_series.gt(0).mean() if not active_net_pnl_series.empty else np.nan),
        "average_gross_exposure": gross_exposure_series[active_mask].mean() if active_mask.any() else 0.0,
        "average_exposure_utilization": exposure_utilization_series[active_mask].mean() if active_mask.any() else 0.0,
        "average_gross_leverage": (
            gross_exposure_series[active_mask]
            .div(strategy_state["equity_before_mark_series"][active_mask].replace(0.0, np.nan))
            .mean()
            if active_mask.any()
            else 0.0
        ),
        "average_turnover_per_rebalance": rebalance_turnover_series.mean() if not rebalance_turnover_series.empty else 0.0,
        "trade_count": trade_count,
        "average_holding_bars": avg_holding_bars,
        "minimum_equity": equity_curve_series.min(),
        "liquidated": strategy_state["liquidated"],
    }


def build_results_summary_table(
    summary_dict: dict,
    *,
    strategy_name: str | None = None,
    include_gross: bool = True,
) -> pd.DataFrame:
    rows: list[tuple[str, object]] = []
    if strategy_name:
        rows.append(("Strategy", strategy_name))

    rows.extend(
        [
            ("Ending equity (USDT)", summary_dict["ending_equity"]),
            ("Total net PnL (USDT)", summary_dict["total_net_pnl"]),
        ]
    )

    if include_gross:
        rows[1:1] = [
            ("Total gross PnL (USDT)", summary_dict["total_gross_pnl"]),
            ("Total transaction costs (USDT)", summary_dict["total_transaction_costs"]),
        ]

    rows.extend(
        [
            ("Total return (%)", summary_dict["total_return_pct"]),
            ("Annualized net return (%)", summary_dict["annualized_net_return_pct"]),
            ("Annualized net volatility (%)", summary_dict["annualized_net_vol_pct"]),
            ("Net Sharpe ratio", summary_dict["net_sharpe"]),
            ("Net Sortino ratio", summary_dict["net_sortino"]),
            ("Net Calmar ratio", summary_dict["net_calmar"]),
            ("Max drawdown (%)", summary_dict["max_drawdown_pct"]),
            ("Active-bar win rate (%)", summary_dict["active_bar_win_rate_pct"]),
            ("Average gross exposure (USDT)", summary_dict["average_gross_exposure"]),
            ("Average exposure utilization", summary_dict["average_exposure_utilization"]),
            ("Average gross leverage (x)", summary_dict["average_gross_leverage"]),
            ("Average turnover per rebalance (USDT)", summary_dict["average_turnover_per_rebalance"]),
            ("Trade count", summary_dict["trade_count"]),
            ("Average holding period (bars)", summary_dict["average_holding_bars"]),
            ("Minimum equity (USDT)", summary_dict["minimum_equity"]),
            ("Liquidated during backtest", "Yes" if summary_dict["liquidated"] else "No"),
        ]
    )
    if include_gross:
        rows.insert(6, ("Annualized gross return (%)", summary_dict["annualized_gross_return_pct"]))
        rows.insert(7, ("Annualized gross volatility (%)", summary_dict["annualized_gross_vol_pct"]))
        rows.insert(8, ("Gross Sharpe ratio", summary_dict["gross_sharpe"]))

    return pd.DataFrame(rows, columns=["metric", "value"])


def build_diagnostics_tail_table(strategy_state: dict, *, rows: int = 5) -> pd.DataFrame:
    return pd.concat(
        [
            strategy_state["gross_exposure_series"],
            strategy_state["transaction_cost_series"],
            strategy_state["turnover_series"],
            strategy_state["equity_curve_series"],
        ],
        axis=1,
    ).tail(rows)


def build_parameter_table(params: dict, *, name: str = "value") -> pd.DataFrame:
    return pd.Series(params, name=name).to_frame()


def run_cross_sectional_parameter_sweep(
    *,
    close_df: pd.DataFrame,
    signal_return_df: pd.DataFrame,
    pnl_return_df: pd.DataFrame,
    zscore_df: pd.DataFrame,
    cross_sectional_dispersion_series: pd.Series,
    dispersion_floor_series: pd.Series,
    market_return_series: pd.Series | None = None,
    entry_zscores: list[float],
    rebalance_bars_list: list[int],
    min_hold_bars_list: list[int],
    exit_zscore: float,
    top_k: int,
    vol_window: int,
    use_regime_filter: bool = False,
    regime_trend_lookback_bars: int = 168,
    regime_trend_vol_window: int = 168,
    regime_trend_zscore_cap: float = 0.75,
    weighting_mode: str = "inverse_vol",
    conviction_scale_floor: float = 0.0,
    conviction_span: float = 1.0,
    initial_capital_usdt: float,
    gross_exposure_cap_usdt: float,
    max_gross_leverage: float,
    transaction_cost_bps: float,
    max_turnover_fraction: float,
    liquidation_equity_floor_usdt: float,
    hours_per_year: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for rebalance_every_bars, entry_zscore, min_hold_bars in product(
        rebalance_bars_list,
        entry_zscores,
        min_hold_bars_list,
    ):
        sweep_state = run_cross_sectional_strategy(
            close_df=close_df,
            signal_return_df=signal_return_df,
            pnl_return_df=pnl_return_df,
            zscore_df=zscore_df,
            cross_sectional_dispersion_series=cross_sectional_dispersion_series,
            dispersion_floor_series=dispersion_floor_series,
            market_return_series=market_return_series,
            entry_zscore=entry_zscore,
            exit_zscore=exit_zscore,
            rebalance_every_bars=rebalance_every_bars,
            min_hold_bars=min_hold_bars,
            top_k=top_k,
            vol_window=vol_window,
            use_regime_filter=use_regime_filter,
            regime_trend_lookback_bars=regime_trend_lookback_bars,
            regime_trend_vol_window=regime_trend_vol_window,
            regime_trend_zscore_cap=regime_trend_zscore_cap,
            weighting_mode=weighting_mode,
            conviction_scale_floor=conviction_scale_floor,
            conviction_span=conviction_span,
            initial_capital_usdt=initial_capital_usdt,
            gross_exposure_cap_usdt=gross_exposure_cap_usdt,
            max_gross_leverage=max_gross_leverage,
            transaction_cost_bps=transaction_cost_bps,
            max_turnover_fraction=max_turnover_fraction,
            liquidation_equity_floor_usdt=liquidation_equity_floor_usdt,
        )
        sweep_summary = summarize_strategy_state(
            sweep_state,
            initial_capital_usdt=initial_capital_usdt,
            hours_per_year=hours_per_year,
        )
        rows.append(
            {
                "rebalance_every_bars": rebalance_every_bars,
                "entry_zscore": entry_zscore,
                "exit_zscore": exit_zscore,
                "min_hold_bars": min_hold_bars,
                "use_regime_filter": use_regime_filter,
                "regime_trend_zscore_cap": regime_trend_zscore_cap,
                "weighting_mode": weighting_mode,
                "ending_equity": sweep_summary["ending_equity"],
                "total_net_pnl": sweep_summary["total_net_pnl"],
                "total_transaction_costs": sweep_summary["total_transaction_costs"],
                "net_sharpe": sweep_summary["net_sharpe"],
                "max_drawdown_pct": sweep_summary["max_drawdown_pct"],
                "average_turnover_per_rebalance": sweep_summary["average_turnover_per_rebalance"],
                "trade_count": sweep_summary["trade_count"],
                "liquidated": sweep_summary["liquidated"],
            }
        )
    return pd.DataFrame(rows).sort_values(["net_sharpe", "ending_equity"], ascending=[False, False]).reset_index(drop=True)


def run_tsmom_parameter_sweep(
    *,
    close_df: pd.DataFrame,
    pnl_return_df: pd.DataFrame,
    lookbacks: list[int],
    rebalance_bars_list: list[int],
    min_hold_bars_list: list[int],
    vol_window: int,
    initial_capital_usdt: float,
    gross_exposure_cap_usdt: float,
    max_gross_leverage: float,
    transaction_cost_bps: float,
    max_turnover_fraction: float,
    liquidation_equity_floor_usdt: float,
    hours_per_year: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for lookback, rebal_bars, min_hold in product(lookbacks, rebalance_bars_list, min_hold_bars_list):
        momentum_df = close_df.pct_change(lookback).shift(1)
        raw_signal_df = np.sign(momentum_df).fillna(0.0)
        sweep_state = run_tsmom_strategy(
            close_df=close_df,
            pnl_return_df=pnl_return_df,
            raw_signal_df=raw_signal_df,
            rebalance_every_bars=rebal_bars,
            min_hold_bars=min_hold,
            vol_window=vol_window,
            initial_capital_usdt=initial_capital_usdt,
            gross_exposure_cap_usdt=gross_exposure_cap_usdt,
            max_gross_leverage=max_gross_leverage,
            transaction_cost_bps=transaction_cost_bps,
            max_turnover_fraction=max_turnover_fraction,
            liquidation_equity_floor_usdt=liquidation_equity_floor_usdt,
        )
        summary = summarize_strategy_state(
            sweep_state,
            initial_capital_usdt=initial_capital_usdt,
            hours_per_year=hours_per_year,
        )
        rows.append(
            {
                "lookback_bars": lookback,
                "lookback_days": lookback / 24,
                "rebalance_every_bars": rebal_bars,
                "min_hold_bars": min_hold,
                "ending_equity": summary["ending_equity"],
                "total_net_pnl": summary["total_net_pnl"],
                "total_transaction_costs": summary["total_transaction_costs"],
                "net_sharpe": summary["net_sharpe"],
                "max_drawdown_pct": summary["max_drawdown_pct"],
                "trade_count": summary["trade_count"],
                "liquidated": summary["liquidated"],
            }
        )
    return pd.DataFrame(rows).sort_values(["net_sharpe", "ending_equity"], ascending=[False, False]).reset_index(drop=True)


def build_strategy_comparison_table(
    *,
    strategy_summary: dict,
    tsmom_summary: dict,
    strategy_returns: pd.Series,
    tsmom_returns: pd.Series,
    initial_capital_usdt: float,
    hours_per_year: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    combined_returns = 0.5 * strategy_returns + 0.5 * tsmom_returns
    combined_equity = (initial_capital_usdt * (1.0 + combined_returns).cumprod()).rename("combined_equity")
    combined_drawdown = combined_equity.div(combined_equity.cummax()).sub(1.0).rename("combined_drawdown")

    comparison = pd.DataFrame(
        {
            "Metric": [
                "Ending equity (USDT)",
                "Total net PnL (USDT)",
                "Annualized net return (%)",
                "Annualized net vol (%)",
                "Net Sharpe",
                "Net Sortino",
                "Calmar",
                "Max drawdown (%)",
                "Trade count",
                "Avg holding (bars)",
            ],
            "Mean-Reversion": [
                strategy_summary["ending_equity"],
                strategy_summary["total_net_pnl"],
                strategy_summary["annualized_net_return_pct"],
                strategy_summary["annualized_net_vol_pct"],
                strategy_summary["net_sharpe"],
                strategy_summary["net_sortino"],
                strategy_summary["net_calmar"],
                strategy_summary["max_drawdown_pct"],
                strategy_summary["trade_count"],
                strategy_summary["average_holding_bars"],
            ],
            "Trend-Following": [
                tsmom_summary["ending_equity"],
                tsmom_summary["total_net_pnl"],
                tsmom_summary["annualized_net_return_pct"],
                tsmom_summary["annualized_net_vol_pct"],
                tsmom_summary["net_sharpe"],
                tsmom_summary["net_sortino"],
                tsmom_summary["net_calmar"],
                tsmom_summary["max_drawdown_pct"],
                tsmom_summary["trade_count"],
                tsmom_summary["average_holding_bars"],
            ],
            "Combined 50/50": [
                combined_equity.iloc[-1],
                combined_equity.iloc[-1] - initial_capital_usdt,
                100.0 * combined_returns.mean() * hours_per_year,
                100.0 * combined_returns.std(ddof=0) * np.sqrt(hours_per_year),
                (
                    (combined_returns.mean() * hours_per_year)
                    / (combined_returns.std(ddof=0) * np.sqrt(hours_per_year))
                    if combined_returns.std(ddof=0) > 0
                    else np.nan
                ),
                (
                    (combined_returns.mean() * hours_per_year)
                    / (np.sqrt((combined_returns.clip(upper=0.0) ** 2).mean()) * np.sqrt(hours_per_year))
                    if (combined_returns.clip(upper=0.0) ** 2).mean() > 0
                    else np.nan
                ),
                (
                    (combined_returns.mean() * hours_per_year) / abs(combined_drawdown.min())
                    if combined_drawdown.min() < 0
                    else np.nan
                ),
                100.0 * combined_drawdown.min(),
                "-",
                "-",
            ],
        }
    )
    return comparison, combined_equity, combined_drawdown


# ── Section 3: Transaction Costs — Roll Model ──────────────────────────────


def estimate_roll_spread(
    close_series: pd.Series,
    *,
    min_periods: int = 100,
) -> dict:
    """Estimate the effective bid-ask spread using the Roll (1984) model.

    The Roll model derives the spread from the negative serial covariance of
    price changes:

        s = sqrt(-Cov(Δp_t, Δp_{t-1}))

    where s is in the same units as price (USDT here).  Converting to basis
    points relative to the average price gives the cost-per-trade in the same
    units expected by the execution engine.

    If the serial covariance is non-negative (i.e. no mean-reversion in ticks,
    which can happen over longer bars) the estimator is undefined.  We return
    NaN for that asset and note it in the diagnostics.

    Parameters
    ----------
    close_series:
        UTC-indexed hourly close prices for a single asset.
    min_periods:
        Minimum number of valid Δp pairs required to produce an estimate.

    Returns
    -------
    dict with keys:
        roll_spread_usdt   – estimated spread in USDT (s)
        roll_spread_bps    – spread in basis points relative to mean close
        serial_cov         – raw Cov(Δp_t, Δp_{t-1})
        n_obs              – number of Δp pairs used
        estimator_valid    – True if serial_cov < 0
    """
    delta_p = close_series.diff().dropna()
    cov_matrix = np.cov(delta_p.iloc[1:].values, delta_p.iloc[:-1].values, ddof=1)
    serial_cov = float(cov_matrix[0, 1])
    n_obs = len(delta_p) - 1

    if n_obs < min_periods or serial_cov >= 0.0:
        return {
            "roll_spread_usdt": np.nan,
            "roll_spread_bps": np.nan,
            "serial_cov": serial_cov,
            "n_obs": n_obs,
            "estimator_valid": False,
        }

    roll_spread_usdt = np.sqrt(-serial_cov)
    mean_price = float(close_series.mean())
    roll_spread_bps = (roll_spread_usdt / mean_price) * 10_000.0

    return {
        "roll_spread_usdt": roll_spread_usdt,
        "roll_spread_bps": roll_spread_bps,
        "serial_cov": serial_cov,
        "n_obs": n_obs,
        "estimator_valid": True,
    }


def estimate_roll_spread_rolling(
    close_series: pd.Series,
    *,
    window: int = 720,
    min_periods: int = 200,
) -> pd.Series:
    """Compute a time-varying Roll spread estimate using a rolling window.

    Returns a Series of spread estimates in basis points, aligned to the
    close_series index.  Entries where the serial covariance is non-negative
    are set to NaN.
    """
    results = []
    for end in range(len(close_series)):
        if end + 1 < window:
            results.append(np.nan)
            continue
        window_prices = close_series.iloc[end + 1 - window : end + 1]
        delta_p = window_prices.diff().dropna()
        delta_p_lag = delta_p.shift(1).dropna()
        common = delta_p.index.intersection(delta_p_lag.index)
        if len(common) < min_periods:
            results.append(np.nan)
            continue
        cov = np.cov(delta_p.loc[common].values, delta_p_lag.loc[common].values, ddof=1)[0, 1]
        if cov >= 0.0:
            results.append(np.nan)
            continue
        mean_price = window_prices.mean()
        results.append((np.sqrt(-cov) / mean_price) * 10_000.0 if mean_price > 0 else np.nan)
    return pd.Series(results, index=close_series.index, name=f"roll_spread_bps_{close_series.name}")


def run_roll_model(
    close_df: pd.DataFrame,
    symbols: list[str],
) -> dict:
    """Run the Roll model across all symbols and return a summary table.

    Parameters
    ----------
    close_df:
        Wide DataFrame of close prices, one column per symbol.
    symbols:
        List of symbol strings matching close_df columns.

    Returns
    -------
    dict with keys:
        summary_df          – per-symbol Roll estimates (DataFrame)
        portfolio_spread_bps – inverse-vol-weighted average spread across assets
        per_symbol          – dict of per-symbol result dicts
        portfolio_cost_bps  – recommended flat cost assumption in bps
    """
    per_symbol: dict[str, dict] = {}
    summary_rows: list[dict] = []

    for symbol in symbols:
        result = estimate_roll_spread(close_df[symbol])
        per_symbol[symbol] = result
        summary_rows.append(
            {
                "symbol": symbol,
                "n_obs": result["n_obs"],
                "serial_cov": result["serial_cov"],
                "roll_spread_usdt": result["roll_spread_usdt"],
                "roll_spread_bps": result["roll_spread_bps"],
                "estimator_valid": result["estimator_valid"],
            }
        )

    summary_df = pd.DataFrame(summary_rows).set_index("symbol")

    valid_bps = summary_df.loc[summary_df["estimator_valid"], "roll_spread_bps"]
    if valid_bps.empty:
        portfolio_spread_bps = np.nan
        portfolio_cost_bps = np.nan
    else:
        portfolio_spread_bps = float(valid_bps.mean())
        portfolio_cost_bps = float(valid_bps.max())  # conservative: use worst-case

    return {
        "summary_df": summary_df,
        "per_symbol": per_symbol,
        "portfolio_spread_bps": portfolio_spread_bps,
        "portfolio_cost_bps": portfolio_cost_bps,
    }


def run_cost_sensitivity_analysis(
    *,
    close_df: pd.DataFrame,
    signal_return_df: pd.DataFrame,
    pnl_return_df: pd.DataFrame,
    zscore_df: pd.DataFrame,
    cross_sectional_dispersion_series: pd.Series,
    dispersion_floor_series: pd.Series,
    market_return_series: pd.Series | None,
    raw_tsmom_signal_df: pd.DataFrame,
    roll_cost_bps: float,
    strategy1_kwargs: dict,
    strategy2_kwargs: dict,
    hours_per_year: int,
    initial_capital_usdt: float,
) -> pd.DataFrame:
    """Re-run both strategies at 0.5×, 1×, and 2× the Roll spread estimate.

    Returns a DataFrame with one row per (strategy, cost_multiplier) combination,
    containing the key performance metrics.
    """
    multipliers = [0.5, 1.0, 2.0]
    rows: list[dict] = []

    for mult in multipliers:
        cost = roll_cost_bps * mult

        # Strategy 1
        s1 = run_cross_sectional_strategy(
            close_df=close_df,
            signal_return_df=signal_return_df,
            pnl_return_df=pnl_return_df,
            zscore_df=zscore_df,
            cross_sectional_dispersion_series=cross_sectional_dispersion_series,
            dispersion_floor_series=dispersion_floor_series,
            market_return_series=market_return_series,
            transaction_cost_bps=cost,
            **strategy1_kwargs,
        )
        s1_sum = summarize_strategy_state(
            s1, initial_capital_usdt=initial_capital_usdt, hours_per_year=hours_per_year
        )

        # Strategy 2
        s2 = run_tsmom_strategy(
            close_df=close_df,
            pnl_return_df=pnl_return_df,
            raw_signal_df=raw_tsmom_signal_df,
            transaction_cost_bps=cost,
            **strategy2_kwargs,
        )
        s2_sum = summarize_strategy_state(
            s2, initial_capital_usdt=initial_capital_usdt, hours_per_year=hours_per_year
        )

        for strategy_name, summary in [
            ("Mean-Reversion", s1_sum),
            ("Trend-Following", s2_sum),
        ]:
            rows.append(
                {
                    "strategy": strategy_name,
                    "cost_multiplier": f"{mult:.1f}×",
                    "cost_bps": round(cost, 2),
                    "total_net_pnl": round(summary["total_net_pnl"], 2),
                    "total_transaction_costs": round(summary["total_transaction_costs"], 2),
                    "annualized_net_return_pct": round(summary["annualized_net_return_pct"], 3),
                    "net_sharpe": round(summary["net_sharpe"], 3),
                    "max_drawdown_pct": round(summary["max_drawdown_pct"], 3),
                    "liquidated": summary["liquidated"],
                }
            )

    return pd.DataFrame(rows)


def slice_time_window(
    obj: pd.DataFrame | pd.Series,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame | pd.Series:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    return obj.loc[start_ts:end_ts].copy()


def summarize_parameter_stability(results_df: pd.DataFrame, parameter_columns: list[str]) -> pd.DataFrame:
    return (
        results_df.groupby(parameter_columns, dropna=False)
        .agg(
            windows_selected=("window", "count"),
            avg_test_net_sharpe=("test_net_sharpe", "mean"),
            avg_test_net_sortino=("test_net_sortino", "mean"),
            avg_test_net_calmar=("test_net_calmar", "mean"),
            avg_test_total_net_pnl=("test_total_net_pnl", "mean"),
        )
        .sort_values(["windows_selected", "avg_test_net_sharpe"], ascending=[False, False])
    )


def run_mean_reversion_walk_forward(
    *,
    windows: list[dict[str, str]],
    close_df: pd.DataFrame,
    signal_return_df: pd.DataFrame,
    pnl_return_df: pd.DataFrame,
    zscore_df: pd.DataFrame,
    cross_sectional_dispersion_series: pd.Series,
    dispersion_floor_series: pd.Series,
    market_return_series: pd.Series | None = None,
    entry_zscores: list[float],
    rebalance_bars_list: list[int],
    min_hold_bars_list: list[int],
    exit_zscore: float,
    top_k: int,
    vol_window: int,
    use_regime_filter: bool = False,
    regime_trend_lookback_bars: int = 168,
    regime_trend_vol_window: int = 168,
    regime_trend_zscore_cap: float = 0.75,
    weighting_mode: str = "inverse_vol",
    conviction_scale_floor: float = 0.0,
    conviction_span: float = 1.0,
    initial_capital_usdt: float,
    gross_exposure_cap_usdt: float,
    max_gross_leverage: float,
    transaction_cost_bps: float,
    max_turnover_fraction: float,
    liquidation_equity_floor_usdt: float,
    hours_per_year: int,
) -> dict:
    rows: list[dict] = []
    curves: list[pd.Series] = []

    for window in windows:
        train_sweep = run_cross_sectional_parameter_sweep(
            close_df=slice_time_window(close_df, window["train_start"], window["train_end"]),
            signal_return_df=slice_time_window(signal_return_df, window["train_start"], window["train_end"]),
            pnl_return_df=slice_time_window(pnl_return_df, window["train_start"], window["train_end"]),
            zscore_df=slice_time_window(zscore_df, window["train_start"], window["train_end"]),
            cross_sectional_dispersion_series=slice_time_window(
                cross_sectional_dispersion_series, window["train_start"], window["train_end"]
            ),
            dispersion_floor_series=slice_time_window(dispersion_floor_series, window["train_start"], window["train_end"]),
            market_return_series=None
            if market_return_series is None
            else slice_time_window(market_return_series, window["train_start"], window["train_end"]),
            entry_zscores=entry_zscores,
            rebalance_bars_list=rebalance_bars_list,
            min_hold_bars_list=min_hold_bars_list,
            exit_zscore=exit_zscore,
            top_k=top_k,
            vol_window=vol_window,
            use_regime_filter=use_regime_filter,
            regime_trend_lookback_bars=regime_trend_lookback_bars,
            regime_trend_vol_window=regime_trend_vol_window,
            regime_trend_zscore_cap=regime_trend_zscore_cap,
            weighting_mode=weighting_mode,
            conviction_scale_floor=conviction_scale_floor,
            conviction_span=conviction_span,
            initial_capital_usdt=initial_capital_usdt,
            gross_exposure_cap_usdt=gross_exposure_cap_usdt,
            max_gross_leverage=max_gross_leverage,
            transaction_cost_bps=transaction_cost_bps,
            max_turnover_fraction=max_turnover_fraction,
            liquidation_equity_floor_usdt=liquidation_equity_floor_usdt,
            hours_per_year=hours_per_year,
        )
        best = train_sweep.iloc[0]
        test_state = run_cross_sectional_strategy(
            close_df=slice_time_window(close_df, window["test_start"], window["test_end"]),
            signal_return_df=slice_time_window(signal_return_df, window["test_start"], window["test_end"]),
            pnl_return_df=slice_time_window(pnl_return_df, window["test_start"], window["test_end"]),
            zscore_df=slice_time_window(zscore_df, window["test_start"], window["test_end"]),
            cross_sectional_dispersion_series=slice_time_window(
                cross_sectional_dispersion_series, window["test_start"], window["test_end"]
            ),
            dispersion_floor_series=slice_time_window(dispersion_floor_series, window["test_start"], window["test_end"]),
            market_return_series=None
            if market_return_series is None
            else slice_time_window(market_return_series, window["test_start"], window["test_end"]),
            entry_zscore=float(best["entry_zscore"]),
            exit_zscore=float(best["exit_zscore"]),
            rebalance_every_bars=int(best["rebalance_every_bars"]),
            min_hold_bars=int(best["min_hold_bars"]),
            top_k=top_k,
            vol_window=vol_window,
            use_regime_filter=use_regime_filter,
            regime_trend_lookback_bars=regime_trend_lookback_bars,
            regime_trend_vol_window=regime_trend_vol_window,
            regime_trend_zscore_cap=regime_trend_zscore_cap,
            weighting_mode=weighting_mode,
            conviction_scale_floor=conviction_scale_floor,
            conviction_span=conviction_span,
            initial_capital_usdt=initial_capital_usdt,
            gross_exposure_cap_usdt=gross_exposure_cap_usdt,
            max_gross_leverage=max_gross_leverage,
            transaction_cost_bps=transaction_cost_bps,
            max_turnover_fraction=max_turnover_fraction,
            liquidation_equity_floor_usdt=liquidation_equity_floor_usdt,
        )
        test_summary = summarize_strategy_state(
            test_state,
            initial_capital_usdt=initial_capital_usdt,
            hours_per_year=hours_per_year,
        )
        rows.append(
            {
                "window": window["label"],
                "train_start": window["train_start"],
                "train_end": window["train_end"],
                "test_start": window["test_start"],
                "test_end": window["test_end"],
                "selected_entry_zscore": float(best["entry_zscore"]),
                "selected_rebalance_every_bars": int(best["rebalance_every_bars"]),
                "selected_min_hold_bars": int(best["min_hold_bars"]),
                "train_best_net_sharpe": float(best["net_sharpe"]),
                "test_ending_equity": test_summary["ending_equity"],
                "test_total_net_pnl": test_summary["total_net_pnl"],
                "test_net_sharpe": test_summary["net_sharpe"],
                "test_net_sortino": test_summary["net_sortino"],
                "test_net_calmar": test_summary["net_calmar"],
                "test_max_drawdown_pct": test_summary["max_drawdown_pct"],
                "test_trade_count": test_summary["trade_count"],
                "test_liquidated": test_summary["liquidated"],
            }
        )
        curves.append(test_state["equity_curve_series"].rename(window["label"]))

    results_df = pd.DataFrame(rows)
    return {
        "results_df": results_df,
        "equity_curves_df": pd.concat(curves, axis=1),
        "parameter_stability_df": summarize_parameter_stability(
            results_df,
            ["selected_entry_zscore", "selected_rebalance_every_bars", "selected_min_hold_bars"],
        ),
    }


def run_tsmom_walk_forward(
    *,
    windows: list[dict[str, str]],
    close_df: pd.DataFrame,
    pnl_return_df: pd.DataFrame,
    lookbacks: list[int],
    rebalance_bars_list: list[int],
    min_hold_bars_list: list[int],
    vol_window: int,
    initial_capital_usdt: float,
    gross_exposure_cap_usdt: float,
    max_gross_leverage: float,
    transaction_cost_bps: float,
    max_turnover_fraction: float,
    liquidation_equity_floor_usdt: float,
    hours_per_year: int,
) -> dict:
    rows: list[dict] = []
    curves: list[pd.Series] = []

    for window in windows:
        train_sweep = run_tsmom_parameter_sweep(
            close_df=slice_time_window(close_df, window["train_start"], window["train_end"]),
            pnl_return_df=slice_time_window(pnl_return_df, window["train_start"], window["train_end"]),
            lookbacks=lookbacks,
            rebalance_bars_list=rebalance_bars_list,
            min_hold_bars_list=min_hold_bars_list,
            vol_window=vol_window,
            initial_capital_usdt=initial_capital_usdt,
            gross_exposure_cap_usdt=gross_exposure_cap_usdt,
            max_gross_leverage=max_gross_leverage,
            transaction_cost_bps=transaction_cost_bps,
            max_turnover_fraction=max_turnover_fraction,
            liquidation_equity_floor_usdt=liquidation_equity_floor_usdt,
            hours_per_year=hours_per_year,
        )
        best = train_sweep.iloc[0]
        raw_signal_df = np.sign(close_df.pct_change(int(best["lookback_bars"])).shift(1)).fillna(0.0)
        test_state = run_tsmom_strategy(
            close_df=slice_time_window(close_df, window["test_start"], window["test_end"]),
            pnl_return_df=slice_time_window(pnl_return_df, window["test_start"], window["test_end"]),
            raw_signal_df=slice_time_window(raw_signal_df, window["test_start"], window["test_end"]),
            rebalance_every_bars=int(best["rebalance_every_bars"]),
            min_hold_bars=int(best["min_hold_bars"]),
            vol_window=vol_window,
            initial_capital_usdt=initial_capital_usdt,
            gross_exposure_cap_usdt=gross_exposure_cap_usdt,
            max_gross_leverage=max_gross_leverage,
            transaction_cost_bps=transaction_cost_bps,
            max_turnover_fraction=max_turnover_fraction,
            liquidation_equity_floor_usdt=liquidation_equity_floor_usdt,
        )
        test_summary = summarize_strategy_state(
            test_state,
            initial_capital_usdt=initial_capital_usdt,
            hours_per_year=hours_per_year,
        )
        rows.append(
            {
                "window": window["label"],
                "train_start": window["train_start"],
                "train_end": window["train_end"],
                "test_start": window["test_start"],
                "test_end": window["test_end"],
                "selected_lookback_bars": int(best["lookback_bars"]),
                "selected_rebalance_every_bars": int(best["rebalance_every_bars"]),
                "selected_min_hold_bars": int(best["min_hold_bars"]),
                "train_best_net_sharpe": float(best["net_sharpe"]),
                "test_ending_equity": test_summary["ending_equity"],
                "test_total_net_pnl": test_summary["total_net_pnl"],
                "test_net_sharpe": test_summary["net_sharpe"],
                "test_net_sortino": test_summary["net_sortino"],
                "test_net_calmar": test_summary["net_calmar"],
                "test_max_drawdown_pct": test_summary["max_drawdown_pct"],
                "test_trade_count": test_summary["trade_count"],
                "test_liquidated": test_summary["liquidated"],
            }
        )
        curves.append(test_state["equity_curve_series"].rename(window["label"]))

    results_df = pd.DataFrame(rows)
    return {
        "results_df": results_df,
        "equity_curves_df": pd.concat(curves, axis=1),
        "parameter_stability_df": summarize_parameter_stability(
            results_df,
            ["selected_lookback_bars", "selected_rebalance_every_bars", "selected_min_hold_bars"],
        ),
    }
