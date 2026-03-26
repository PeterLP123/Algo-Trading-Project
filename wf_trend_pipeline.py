"""Trend strategy pipeline for walk-forward parameter search (strategy_final_v2)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def abdi_ranaldo_spread(frame: pd.DataFrame, window: int = 21) -> pd.Series:
    cols = {col.lower(): col for col in frame.columns}
    required = {"high", "low", "close"}
    missing = required.difference(cols)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    high = pd.to_numeric(frame[cols["high"]], errors="coerce")
    low = pd.to_numeric(frame[cols["low"]], errors="coerce")
    close = pd.to_numeric(frame[cols["close"]], errors="coerce")

    log_high = np.log(high.where(high > 0))
    log_low = np.log(low.where(low > 0))
    log_close = np.log(close.where(close > 0))

    midpoint = (log_high + log_low) / 2.0
    s2 = 4.0 * (log_close.shift(1) - midpoint.shift(1)) * (log_close.shift(1) - midpoint)
    spread = np.sqrt(s2.rolling(window=window, min_periods=window).mean().abs())
    return spread.replace([np.inf, -np.inf], np.nan).rename("abdi_ranaldo_spread")


def compute_half_spread_frac(
    asset_panel_slice: pd.DataFrame,
    cleaned_frames: dict[str, pd.DataFrame],
    symbols: list[str],
) -> pd.DataFrame:
    ar_cols = []
    for s in symbols:
        ar_cols.append(abdi_ranaldo_spread(cleaned_frames[s]).reindex(asset_panel_slice.index))
    ar_wide = pd.concat(ar_cols, axis=1, keys=symbols)
    close_px = asset_panel_slice["close"]
    half_spread = 0.5 * ar_wide
    return half_spread.shift(1).divide(close_px.shift(1)).replace([np.inf, -np.inf], np.nan)


def apply_rebalance_theta(theta: pd.DataFrame, every: int) -> pd.DataFrame:
    if every <= 1:
        return theta
    pos = np.arange(len(theta))
    hold = (pos % every) != 0
    # Explicit writable copy: theta.copy() can leave read-only blocks (CoW / Arrow).
    arr = np.array(theta.to_numpy(), copy=True, dtype=float)
    arr[hold, :] = np.nan
    out = pd.DataFrame(arr, index=theta.index, columns=theta.columns)
    return out.ffill()


def build_signal_and_theta(
    asset_panel_slice: pd.DataFrame,
    symbols: list[str],
    *,
    ma_window: int,
    vol_window: int,
    dead_zone: float,
    signal_clip: float,
    gross_cap: float,
    rebalance_every: int,
) -> pd.DataFrame:
    close_px = asset_panel_slice["close"]
    ma50_list, trend_raw_list, vol_20_list, signal_strength_list, trend_position_list = [], [], [], [], []

    for s in symbols:
        c = close_px[s]
        ma_n = c.rolling(ma_window, min_periods=ma_window).mean()
        trend_raw = c / ma_n - 1
        ret = c.pct_change()
        vol_20 = ret.rolling(vol_window, min_periods=vol_window).std(ddof=0).replace(0.0, np.nan)
        signal_strength = (trend_raw / vol_20).clip(-signal_clip, signal_clip)
        tp = np.where(signal_strength > dead_zone, 1.0, np.where(signal_strength < -dead_zone, -1.0, 0.0))
        trend_position = pd.Series(tp, index=c.index, dtype=float).where(signal_strength.notna(), np.nan)
        ma50_list.append(ma_n.rename(s))
        trend_raw_list.append(trend_raw.rename(s))
        vol_20_list.append(vol_20.rename(s))
        signal_strength_list.append(signal_strength.rename(s))
        trend_position_list.append(trend_position.rename(s))

    mi = pd.MultiIndex.from_product

    def _field_wide(field: str, parts: list[pd.Series]) -> pd.DataFrame:
        return pd.concat(parts, axis=1, keys=mi([[field], symbols], names=["field", "asset"]))

    signal_panel = pd.concat(
        [
            _field_wide("ma", ma50_list),
            _field_wide("trend_raw", trend_raw_list),
            _field_wide("vol_20", vol_20_list),
            _field_wide("signal_strength", signal_strength_list),
            _field_wide("trend_position", trend_position_list),
        ],
        axis=1,
    ).sort_index(axis=1)

    ss = signal_panel["signal_strength"]
    pos = signal_panel["trend_position"]
    vol = signal_panel["vol_20"]
    active = (pos != 0) & pos.notna()
    w_raw = ss.where(active, 0.0)
    sigma = vol.replace(0.0, np.nan)
    w_risk = w_raw / sigma
    denom = w_risk.abs().sum(axis=1)
    denom_safe = denom.replace(0.0, np.nan)
    w_tilde = w_risk.div(denom_safe, axis=0).fillna(0.0)
    theta = gross_cap * w_tilde
    theta = apply_rebalance_theta(theta, rebalance_every)
    return theta


def run_net_backtest(
    asset_panel_slice: pd.DataFrame,
    theta: pd.DataFrame,
    symbols: list[str],
    half_spread_frac: pd.DataFrame,
    *,
    backtest_use_excess: bool = False,
    v0: float = 10_000.0,
) -> dict[str, pd.Series]:
    if backtest_use_excess:
        r = asset_panel_slice["excess_return"]
    else:
        r = asset_panel_slice["close"].pct_change()

    theta_exec = theta.shift(1)
    gross_pnl = (theta_exec * r).sum(axis=1).fillna(0.0)

    theta_prev = theta.shift(1)
    r_lag = r.shift(1)
    carried = theta_prev * (1.0 + r_lag)
    delta_theta = theta - carried
    turnover = delta_theta.abs().sum(axis=1)
    cost_t = (delta_theta.abs() * half_spread_frac).sum(axis=1).fillna(0.0)

    net_pnl = gross_pnl - cost_t
    cumulative_net_pnl = net_pnl.cumsum()
    net_portfolio_value = v0 + cumulative_net_pnl

    return {
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "net_portfolio_value": net_portfolio_value,
        "turnover": turnover,
        "cost_t": cost_t,
        "r": r,
    }


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.dropna()
    if eq.empty:
        return np.nan
    peak = eq.cummax()
    dd = (eq - peak) / peak.replace(0.0, np.nan)
    return float(dd.min())


def sharpe_ratio(returns: pd.Series, trading_days: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return float(np.sqrt(trading_days) * r.mean() / r.std(ddof=1))


def metrics_on_window(
    net_portfolio_value: pd.Series,
    turnover: pd.Series,
    window_index: pd.DatetimeIndex,
    trading_days: int = 252,
) -> dict[str, float]:
    eq = net_portfolio_value.reindex(window_index).dropna()
    turn = turnover.reindex(window_index).dropna()
    if len(eq) < 2:
        return {
            "sharpe": np.nan,
            "total_return": np.nan,
            "max_dd": np.nan,
            "mean_turnover": np.nan,
        }
    r = eq.pct_change().dropna()
    return {
        "sharpe": sharpe_ratio(r, trading_days),
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        "max_dd": max_drawdown(eq),
        "mean_turnover": float(turn.mean()) if len(turn) else np.nan,
    }
