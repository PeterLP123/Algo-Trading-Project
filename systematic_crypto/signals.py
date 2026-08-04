"""Signal construction and portfolio-weight generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


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
