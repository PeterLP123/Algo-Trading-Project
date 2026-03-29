"""
helpers.py — Core helper functions extracted from Cell 2 of
strategy_final_v2.ipynb.

Functions:
  _field_wide                  — shared MultiIndex DataFrame builder
  abdi_ranaldo_spread          — AR (2017) spread estimator from OHLC
  compute_half_spread_frac     — 0.5 × AR, reindexed and lagged
  make_rebalance_mask          — boolean Series marking rebalance dates
  apply_rebalance_theta        — forward-fill theta between rebalance dates
  build_signal_and_theta       — trend signal + dollar exposure builder
  _print_execution_diagnostics — debug helper for run_net_backtest
  run_net_backtest             — core P&L engine (gross + net)
  _wf_max_drawdown             — alias for performance_metrics.max_dd
  _wf_sharpe_ratio             — alias for performance_metrics.sharpe_ratio
  metrics_on_window            — Sharpe/return/DD/turnover on a date window
"""

import numpy as np
import pandas as pd

# Canonical metric implementations live in performance_metrics.py; alias here
# so internal callers (_wf_max_drawdown, _wf_sharpe_ratio) continue to work.
from .performance_metrics import max_dd as _wf_max_drawdown, sharpe_ratio as _wf_sharpe_ratio


# ── Shared MultiIndex helper ───────────────────────────────────────────────

def _field_wide(field: str, parts: list, symbols: list) -> pd.DataFrame:
    """Wrap a list of per-asset Series into a (field × asset) MultiIndex DataFrame."""
    return pd.concat(
        parts,
        axis=1,
        keys=pd.MultiIndex.from_product([[field], symbols], names=["field", "asset"]),
    )


# ── Helper functions (inlined from wf_trend_pipeline.py) ──────────────────


def abdi_ranaldo_spread(frame, window=21):
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


def compute_half_spread_frac(asset_panel_slice, cleaned_frames, symbols):
    ar_cols = []
    for s in symbols:
        ar_cols.append(abdi_ranaldo_spread(cleaned_frames[s]).reindex(asset_panel_slice.index).rename(s))
    ar_wide = pd.concat(ar_cols, axis=1)
    half_spread = 0.5 * ar_wide
    return half_spread.shift(1).replace([np.inf, -np.inf], np.nan)


def make_rebalance_mask(index, every):
    if every <= 1:
        return pd.Series(True, index=index, name="is_rebalance")
    pos = np.arange(len(index))
    return pd.Series((pos % every) == 0, index=index, name="is_rebalance")


def apply_rebalance_theta(theta, every, rebalance_mask=None):
    if every <= 1:
        return theta
    if rebalance_mask is None:
        rebalance_mask = make_rebalance_mask(theta.index, every)
    hold = ~rebalance_mask.to_numpy(dtype=bool)
    arr = np.array(theta.to_numpy(), copy=True, dtype=float)
    arr[hold, :] = np.nan
    out = pd.DataFrame(arr, index=theta.index, columns=theta.columns)
    return out.ffill()


def build_signal_and_theta(
    asset_panel_slice,
    symbols,
    *,
    ma_window,
    vol_window,
    dead_zone,
    signal_clip,
    gross_cap,
    rebalance_every,
    return_details=False,
):
    close_px = asset_panel_slice["close"]
    ma_list, trend_raw_list, vol_20_list, z_list, trend_position_list = [], [], [], [], []
    for s in symbols:
        c = close_px[s]
        ma_n = c.rolling(ma_window, min_periods=ma_window).mean()
        trend_raw = c / ma_n - 1
        ret = c.pct_change()
        vol_20 = ret.rolling(vol_window, min_periods=vol_window).std(ddof=0).replace(0.0, np.nan)
        z = (trend_raw / vol_20).clip(-signal_clip, signal_clip)
        tp = np.where(z > dead_zone, 1.0, np.where(z < -dead_zone, -1.0, 0.0))
        trend_position = pd.Series(tp, index=c.index, dtype=float).where(z.notna(), np.nan)
        ma_list.append(ma_n.rename(s))
        trend_raw_list.append(trend_raw.rename(s))
        vol_20_list.append(vol_20.rename(s))
        z_list.append(z.rename(s))
        trend_position_list.append(trend_position.rename(s))
    signal_panel = pd.concat(
        [
            _field_wide("ma",             ma_list,             symbols),
            _field_wide("trend_raw",      trend_raw_list,      symbols),
            _field_wide("vol_20",         vol_20_list,         symbols),
            _field_wide("z",              z_list,              symbols),
            _field_wide("trend_position", trend_position_list, symbols),
        ],
        axis=1,
    ).sort_index(axis=1)
    z_sp = signal_panel["z"]
    tr = signal_panel["trend_raw"]
    vol = signal_panel["vol_20"]
    active = (z_sp.abs() > dead_zone) & z_sp.notna()
    w_raw = tr.where(active, 0.0)
    sigma = vol.replace(0.0, np.nan)
    w_risk = w_raw / sigma
    denom = w_risk.abs().sum(axis=1)
    denom_safe = denom.replace(0.0, np.nan)
    w_tilde = w_risk.div(denom_safe, axis=0).fillna(0.0)
    theta_target = gross_cap * w_tilde
    rebalance_mask = make_rebalance_mask(theta_target.index, rebalance_every)
    theta = apply_rebalance_theta(theta_target, rebalance_every, rebalance_mask=rebalance_mask)
    theta.attrs["theta_target"] = theta_target.copy()
    theta.attrs["rebalance_mask"] = rebalance_mask.copy()
    theta.attrs["rebalance_every"] = rebalance_every
    if return_details:
        return {"theta_target": theta_target, "theta": theta, "rebalance_mask": rebalance_mask}
    return theta


def _print_execution_diagnostics(
    theta_target, theta_exec, carried, delta_theta_exec, turnover, cost_t,
    rebalance_mask, diagnostic_rows,
):
    window = max(int(diagnostic_rows), 2)
    end = min(len(theta_target), window)
    diag = pd.concat(
        {
            "theta_target": theta_target.iloc[:end],
            "theta_exec": theta_exec.iloc[:end],
            "carried": carried.iloc[:end],
            "delta_theta_exec": delta_theta_exec.iloc[:end],
        },
        axis=1,
    )
    diag["is_rebalance"] = rebalance_mask.iloc[:end]
    diag["turnover"] = turnover.iloc[:end]
    diag["cost_t"] = cost_t.iloc[:end]
    print("Execution diagnostics sample:")
    print(diag.round(6).to_string())


def run_net_backtest(
    asset_panel_slice,
    theta,
    symbols,
    half_spread_frac,
    *,
    rebalance_every=1,
    print_diagnostics=False,
    diagnostic_rows=6,
    print_diagnostic=None,
    diagnostic_window=None,
    backtest_use_excess=False,
    v0=10_000.0,
):
    if print_diagnostic is not None:
        print_diagnostics = print_diagnostic
    if diagnostic_window is not None:
        diagnostic_rows = diagnostic_window
    if backtest_use_excess:
        r = asset_panel_slice["excess_return"]
    else:
        r = asset_panel_slice["close"].pct_change()
    r = r.reindex(columns=symbols).astype(float).fillna(0.0)
    half_spread_frac = (
        half_spread_frac.reindex(index=r.index, columns=symbols).astype(float).fillna(0.0)
    )
    if isinstance(theta, dict):
        theta_target = theta.get("theta_target")
        theta_ffill = theta.get("theta")
        rebalance_mask = theta.get("rebalance_mask")
        if theta_target is None:
            if theta_ffill is None:
                raise KeyError("theta bundle must include at least one of 'theta_target' or 'theta'")
            theta_target = theta_ffill
        if rebalance_mask is None:
            rebalance_mask = make_rebalance_mask(theta_target.index, rebalance_every)
    else:
        theta_target = theta.attrs.get("theta_target", theta)
        theta_ffill = theta
        rebalance_mask = theta.attrs.get("rebalance_mask")
        rebalance_every = int(theta.attrs.get("rebalance_every", rebalance_every))
        if rebalance_mask is None:
            rebalance_mask = make_rebalance_mask(theta.index, rebalance_every)
    theta_target = theta_target.reindex(index=r.index, columns=symbols).astype(float).fillna(0.0)
    if theta_ffill is None:
        theta_ffill = apply_rebalance_theta(theta_target, rebalance_every, rebalance_mask=rebalance_mask)
    theta_ffill = theta_ffill.reindex(index=r.index, columns=symbols).astype(float).ffill().fillna(0.0)
    rebalance_mask = rebalance_mask.reindex(r.index).fillna(False).astype(bool)
    theta_exec = pd.DataFrame(0.0, index=r.index, columns=symbols)
    carried = pd.DataFrame(0.0, index=r.index, columns=symbols)
    delta_theta_exec = pd.DataFrame(0.0, index=r.index, columns=symbols)
    zero_row = pd.Series(0.0, index=symbols, dtype=float)
    equity_running = v0
    liquidated = False
    liquidated_at = None
    for i in range(len(r.index)):
        if i == 0:
            carried_t = zero_row
        else:
            prev_exec = theta_exec.iloc[i - 1]
            prev_return = r.iloc[i]
            carried_t = prev_exec * (1.0 + prev_return)
        carried.iloc[i] = carried_t.to_numpy(dtype=float)
        if liquidated:
            exec_t = zero_row
            delta_t = zero_row - carried_t
        elif rebalance_mask.iat[i]:
            target_t = theta_target.iloc[i].fillna(0.0)
            delta_t = target_t - carried_t
            exec_t = target_t
        else:
            delta_t = zero_row
            exec_t = carried_t
        delta_theta_exec.iloc[i] = delta_t.to_numpy(dtype=float)
        theta_exec.iloc[i] = exec_t.to_numpy(dtype=float)
        if i > 0:
            bar_gross = float((theta_exec.iloc[i - 1] * r.iloc[i]).sum())
        else:
            bar_gross = 0.0
        bar_cost = float((delta_theta_exec.iloc[i].abs() * half_spread_frac.iloc[i]).sum())
        equity_running += bar_gross - bar_cost
        if not liquidated and equity_running <= 0:
            liquidated = True
            liquidated_at = r.index[i]
    gross_pnl = (theta_exec.shift(1).fillna(0.0) * r).sum(axis=1).fillna(0.0)
    turnover = delta_theta_exec.abs().sum(axis=1)
    cost_t = (delta_theta_exec.abs() * half_spread_frac).sum(axis=1).fillna(0.0)
    net_pnl = gross_pnl - cost_t
    cumulative_net_pnl = net_pnl.cumsum()
    net_portfolio_value = v0 + cumulative_net_pnl
    if print_diagnostics:
        _print_execution_diagnostics(
            theta_target=theta_target,
            theta_exec=theta_exec,
            carried=carried,
            delta_theta_exec=delta_theta_exec,
            turnover=turnover,
            cost_t=cost_t,
            rebalance_mask=rebalance_mask,
            diagnostic_rows=diagnostic_rows,
        )
    return {
        "theta_target": theta_target,
        "theta": theta_ffill,
        "theta_exec": theta_exec,
        "carried": carried,
        "delta_theta_exec": delta_theta_exec,
        "rebalance_mask": rebalance_mask,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "net_portfolio_value": net_portfolio_value,
        "turnover": turnover,
        "cost_t": cost_t,
        "r": r,
        "liquidated_at": liquidated_at,
    }


def metrics_on_window(net_portfolio_value, turnover, window_index, trading_days=252):
    eq = net_portfolio_value.reindex(window_index).dropna()
    turn = turnover.reindex(window_index).dropna()
    if len(eq) < 2:
        return {"sharpe": np.nan, "total_return": np.nan, "max_dd": np.nan, "mean_turnover": np.nan}
    r = eq.pct_change().dropna()
    return {
        "sharpe": _wf_sharpe_ratio(r, trading_days),
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        "max_dd": _wf_max_drawdown(eq),
        "mean_turnover": float(turn.mean()) if len(turn) else np.nan,
    }
