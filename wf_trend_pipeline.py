"""Trend strategy pipeline for the walk-forward research notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf


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
    half_spread = 0.5 * ar_wide
    return half_spread.shift(1).replace([np.inf, -np.inf], np.nan)


def make_rebalance_mask(index: pd.Index, every: int) -> pd.Series:
    if every <= 1:
        return pd.Series(True, index=index, name="is_rebalance")
    pos = np.arange(len(index))
    return pd.Series((pos % every) == 0, index=index, name="is_rebalance")


def apply_rebalance_theta(
    theta: pd.DataFrame,
    every: int,
    rebalance_mask: pd.Series | None = None,
) -> pd.DataFrame:
    if every <= 1:
        return theta
    if rebalance_mask is None:
        rebalance_mask = make_rebalance_mask(theta.index, every)
    hold = ~rebalance_mask.to_numpy(dtype=bool)
    # Explicit writable copy: theta.copy() can leave read-only blocks (CoW / Arrow).
    arr = np.array(theta.to_numpy(), copy=True, dtype=float)
    arr[hold, :] = np.nan
    out = pd.DataFrame(arr, index=theta.index, columns=theta.columns)
    return out.ffill()


def mvo_weights(
    mu: np.ndarray,
    Sigma: np.ndarray,
    gamma: float,
    gross_limit: float = 1.0,
) -> np.ndarray:
    """Solve: max μᵀw − (γ/2) wᵀΣw  s.t. ‖w‖₁ ≤ gross_limit.

    Returns np.zeros(n) on solver failure or non-optimal status.
    """
    n = len(mu)
    w = cp.Variable(n)
    objective = cp.Maximize(mu @ w - (gamma / 2.0) * cp.quad_form(w, Sigma))
    constraints = [cp.norm1(w) <= gross_limit]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.CLARABEL, warm_start=True)
        if problem.status != "optimal" or w.value is None:
            return np.zeros(n)
        return np.array(w.value, dtype=float)
    except Exception:
        return np.zeros(n)


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
    use_mvo: bool = False,
    cov_window: int = 120,
    gamma: float = 1.0,
    feature_cache: dict[tuple[int, int, float], dict[str, pd.DataFrame]] | None = None,
    search_fast_mode: bool = False,
    return_details: bool = False,
) -> pd.DataFrame | dict[str, pd.DataFrame | pd.Series]:
    close_px = asset_panel_slice["close"]
    cache_key = (int(ma_window), int(vol_window), float(signal_clip))
    cached_features = feature_cache.get(cache_key) if feature_cache is not None else None
    if cached_features is None:
        ma50_list, trend_raw_list, vol_20_list, z_list = [], [], [], []
        for s in symbols:
            c = close_px[s]
            ma_n = c.rolling(ma_window, min_periods=ma_window).mean()
            trend_raw = c / ma_n - 1
            ret = c.pct_change()
            vol_20 = ret.rolling(vol_window, min_periods=vol_window).std(ddof=0).replace(0.0, np.nan)
            # z: activation / diagnostics only; not used as dollar-sizing input
            z = (trend_raw / vol_20).clip(-signal_clip, signal_clip)
            ma50_list.append(ma_n.rename(s))
            trend_raw_list.append(trend_raw.rename(s))
            vol_20_list.append(vol_20.rename(s))
            z_list.append(z.rename(s))

        mi = pd.MultiIndex.from_product

        def _field_wide(field: str, parts: list[pd.Series]) -> pd.DataFrame:
            return pd.concat(parts, axis=1, keys=mi([[field], symbols], names=["field", "asset"]))

        signal_panel = pd.concat(
            [
                _field_wide("ma", ma50_list),
                _field_wide("trend_raw", trend_raw_list),
                _field_wide("vol_20", vol_20_list),
                _field_wide("z", z_list),
            ],
            axis=1,
        ).sort_index(axis=1)
        if feature_cache is not None:
            feature_cache[cache_key] = {
                "signal_panel": signal_panel,
                "returns_panel": close_px.reindex(columns=symbols).pct_change(),
            }
    else:
        signal_panel = cached_features["signal_panel"]

    z = signal_panel["z"]
    tr = signal_panel["trend_raw"]
    vol = signal_panel["vol_20"]
    active = (z.abs() > dead_zone) & z.notna()

    if use_mvo:
        if feature_cache is not None and cache_key in feature_cache:
            returns_panel = feature_cache[cache_key]["returns_panel"]
        else:
            # Build a (n_dates × n_assets) daily-returns panel for covariance estimation.
            returns_panel = pd.concat(
                [close_px[s].pct_change().rename(s) for s in symbols], axis=1
            )
        n_assets = len(symbols)
        min_hist = n_assets + 10  # guard: need at least this many clean rows
        w_tilde_arr = np.zeros((len(close_px), n_assets), dtype=float)
        fallback_arr = np.zeros((len(close_px), n_assets), dtype=float)

        for t_idx in range(len(close_px)):
            z_t = z.iloc[t_idx].to_numpy(dtype=float)
            active_t = active.iloc[t_idx].to_numpy(dtype=bool)
            n_active = int(active_t.sum())
            fallback_arr[t_idx] = (
                np.where(active_t, np.sign(z_t) / n_active, 0.0)
                if n_active > 0
                else np.zeros(n_assets)
            )

        if search_fast_mode:
            w_tilde = pd.DataFrame(fallback_arr, index=close_px.index, columns=symbols)
        else:
            for t_idx in range(len(close_px)):
                z_t = z.iloc[t_idx].to_numpy(dtype=float)
                active_t = active.iloc[t_idx].to_numpy(dtype=bool)
                # Zero out inactive assets to honour the dead-zone threshold.
                mu_t = np.where(active_t, z_t, 0.0)
                n_active = int(active_t.sum())

                if t_idx < cov_window:
                    # Not enough history yet: equal-weight fallback.
                    w_row = (
                        np.where(active_t, np.sign(z_t) / n_active, 0.0)
                        if n_active > 0 else np.zeros(n_assets)
                    )
                else:
                    cov_slice = returns_panel.iloc[t_idx - cov_window : t_idx].dropna()
                    if len(cov_slice) < min_hist:
                        # Too many NaN rows in window: equal-weight fallback.
                        w_row = (
                            np.where(active_t, np.sign(z_t) / n_active, 0.0)
                            if n_active > 0 else np.zeros(n_assets)
                        )
                    else:
                        Sigma = LedoitWolf().fit(cov_slice.values).covariance_
                        w_row = mvo_weights(mu_t, Sigma, gamma=gamma)

                w_tilde_arr[t_idx] = w_row

            w_tilde = pd.DataFrame(w_tilde_arr, index=close_px.index, columns=symbols)

    else:
        # Original inverse-volatility path (unchanged).
        # z dead-zone only; w_risk = w_raw / vol_20 is the single vol-adjusted sizing step
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
        return {
            "theta_target": theta_target,
            "theta": theta,
            "rebalance_mask": rebalance_mask,
        }
    return theta


def _print_execution_diagnostics(
    theta_target: pd.DataFrame,
    theta_exec: pd.DataFrame,
    carried: pd.DataFrame,
    delta_theta_exec: pd.DataFrame,
    turnover: pd.Series,
    cost_t: pd.Series,
    rebalance_mask: pd.Series,
    diagnostic_rows: int,
) -> None:
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
    asset_panel_slice: pd.DataFrame,
    theta: pd.DataFrame | dict[str, pd.DataFrame | pd.Series],
    symbols: list[str],
    half_spread_frac: pd.DataFrame,
    *,
    rebalance_every: int = 1,
    print_diagnostics: bool = False,
    diagnostic_rows: int = 6,
    print_diagnostic: bool | None = None,
    diagnostic_window: int | None = None,
    backtest_use_excess: bool = False,
    v0: float = 10_000.0,
) -> dict[str, pd.DataFrame | pd.Series]:
    if print_diagnostic is not None:
        print_diagnostics = print_diagnostic
    if diagnostic_window is not None:
        diagnostic_rows = diagnostic_window

    if backtest_use_excess:
        r = asset_panel_slice["excess_return"]
    else:
        r = asset_panel_slice["close"].pct_change()

    r = r.reindex(columns=symbols).astype(float).fillna(0.0)
    half_spread_frac = half_spread_frac.reindex(index=r.index, columns=symbols).astype(float).fillna(0.0)

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

    # --- Circuit breaker: liquidate if equity hits zero ---
    equity_running = v0
    liquidated = False
    liquidated_at = None

    for i in range(len(r.index)):
        if i == 0:
            carried_t = zero_row
        else:
            prev_exec = theta_exec.iloc[i - 1]
            prev_return = r.iloc[i - 1]
            carried_t = prev_exec * (1.0 + prev_return)

        carried.iloc[i] = carried_t.to_numpy(dtype=float)

        # Circuit breaker: force flat if equity was wiped out
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

        # Update running equity and check circuit breaker
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


def sortino_ratio(returns: pd.Series, mar: float = 0.0, trading_days: int = 252) -> float:
    r = returns.dropna()
    downside = r[r < mar] - mar
    if len(r) < 2 or len(downside) == 0:
        return np.nan
    downside_std = downside.std(ddof=0)
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return float(np.sqrt(trading_days) * (r.mean() - mar) / downside_std)


def calmar_ratio(returns: pd.Series, equity: pd.Series, trading_days: int = 252) -> float:
    r = returns.dropna()
    eq = equity.dropna()
    if len(r) < 2 or len(eq) < 2:
        return np.nan
    if eq.iloc[0] <= 0 or eq.iloc[-1] <= 0:
        return np.nan
    ann_return = (eq.iloc[-1] / eq.iloc[0]) ** (trading_days / len(r)) - 1.0
    mdd = max_drawdown(eq)
    if not np.isfinite(mdd) or mdd == 0:
        return np.nan
    return float(ann_return / abs(mdd))


def mean_holding_horizon_per_asset(theta_exec: pd.DataFrame) -> float:
    durations: list[int] = []
    for col in theta_exec.columns:
        sign = np.sign(theta_exec[col].fillna(0.0))
        flips = (sign != sign.shift()).cumsum()
        for _, grp in sign.groupby(flips):
            if grp.iloc[0] != 0:
                durations.append(len(grp))
    return float(np.mean(durations)) if durations else np.nan


def metrics_on_window(
    net_portfolio_value: pd.Series,
    turnover: pd.Series,
    window_index: pd.DatetimeIndex,
    trading_days: int = 252,
    gross_pnl: pd.Series | None = None,
    cost_t: pd.Series | None = None,
    theta_exec: pd.DataFrame | None = None,
) -> dict[str, float]:
    eq = net_portfolio_value.reindex(window_index).dropna()
    turn = turnover.reindex(window_index).dropna()
    if len(eq) < 2:
        return {
            "sharpe": np.nan,
            "total_return": np.nan,
            "max_dd": np.nan,
            "mean_turnover": np.nan,
            "cost_to_gross_ratio": np.nan,
            "active_days_pct": np.nan,
        }
    r = eq.pct_change().dropna()
    ratio = np.nan
    if gross_pnl is not None and cost_t is not None:
        gross_slice = gross_pnl.reindex(window_index).dropna()
        cost_slice = cost_t.reindex(window_index).dropna()
        abs_gross = float(gross_slice.abs().sum())
        total_cost = float(cost_slice.sum())
        ratio = float(total_cost / abs_gross) if abs_gross > 0 else np.nan
    active_days_pct = np.nan
    if theta_exec is not None:
        gross_exposure = theta_exec.reindex(window_index).abs().sum(axis=1)
        if len(gross_exposure):
            active_days_pct = float((gross_exposure > 0).mean())
    return {
        "sharpe": sharpe_ratio(r, trading_days),
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        "max_dd": max_drawdown(eq),
        "mean_turnover": float(turn.mean()) if len(turn) else np.nan,
        "cost_to_gross_ratio": ratio,
        "active_days_pct": active_days_pct,
    }


def summarize_holdout_model(
    label: str,
    backtest_out: dict[str, pd.DataFrame | pd.Series],
    holdout_index: pd.Index,
    trading_days: int = 252,
    initial_capital: float = 10_000.0,
) -> dict[str, float | str | bool | None]:
    net_portfolio_value = backtest_out["net_portfolio_value"]
    gross_pnl = backtest_out["gross_pnl"]
    net_pnl = backtest_out["net_pnl"]
    cost_t = backtest_out["cost_t"]
    turnover = backtest_out["turnover"]
    theta_exec = backtest_out["theta_exec"]
    liquidated_at = backtest_out.get("liquidated_at")

    eq_eval = net_portfolio_value.reindex(holdout_index).dropna()
    r_eval = eq_eval.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    gross_eval = gross_pnl.reindex(holdout_index).dropna()
    net_eval = net_pnl.reindex(holdout_index).dropna()
    cost_eval = cost_t.reindex(holdout_index).dropna()

    holdout_start = holdout_index.min()
    holdout_start_equity = (
        float(net_portfolio_value.reindex([holdout_start]).iloc[0])
        if holdout_start in net_portfolio_value.index
        else np.nan
    )
    liquidated_before_holdout = bool(
        liquidated_at is not None and pd.Timestamp(liquidated_at) < pd.Timestamp(holdout_start)
    )

    total_abs_gross = float(gross_eval.abs().sum()) if len(gross_eval) else 0.0
    total_cost = float(cost_eval.sum()) if len(cost_eval) else 0.0

    return {
        "model": label,
        "n_days": int(len(eq_eval)),
        "holdout_start_equity": holdout_start_equity,
        "liquidated_at": None if liquidated_at is None else str(pd.Timestamp(liquidated_at)),
        "liquidated_before_holdout": liquidated_before_holdout,
        "sharpe": sharpe_ratio(r_eval, trading_days),
        "sortino": sortino_ratio(r_eval, trading_days=trading_days),
        "calmar": calmar_ratio(r_eval, eq_eval, trading_days),
        "max_drawdown": max_drawdown(eq_eval),
        "mean_turnover": float(turnover.reindex(holdout_index).mean()),
        "total_gross_pnl": float(gross_eval.sum()) if len(gross_eval) else np.nan,
        "total_cost": total_cost,
        "total_net_pnl": float(net_eval.sum()) if len(net_eval) else np.nan,
        "pct_return_on_v0": (
            float(net_eval.sum() / initial_capital) if len(net_eval) else np.nan
        ),
        "cost_to_gross_ratio": (
            float(total_cost / total_abs_gross) if total_abs_gross > 0 else np.nan
        ),
        "final_net_equity": float(eq_eval.iloc[-1]) if len(eq_eval) else np.nan,
        "active_days_pct": (
            float(theta_exec.reindex(holdout_index).abs().sum(axis=1).gt(0).mean())
            if len(eq_eval)
            else np.nan
        ),
    }
