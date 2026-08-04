"""Portfolio execution and strategy orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .signals import (
    build_conviction_scale_series,
    build_cross_sectional_conviction_target_weights,
    build_inverse_vol_target_weights,
    build_mean_reversion_regime_filter,
    build_top_bottom_signal_frame,
    build_tsmom_signal_frame,
)


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
