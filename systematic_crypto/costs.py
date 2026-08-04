"""Transaction-cost estimation and strategy cost-sensitivity analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import summarize_strategy_state
from .execution import run_cross_sectional_strategy, run_tsmom_strategy


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
