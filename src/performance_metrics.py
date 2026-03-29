"""
performance_metrics.py — Strategy performance metrics + IS/OOS reporting (Cell 25).

Functions (all extracted verbatim from Cell 25):
  max_dd(equity)
  sharpe_ratio(returns, trading_days)
  sortino_ratio(returns, mar)
  calmar_ratio(returns, equity)
  mean_holding_horizon_per_asset(theta_exec)
  compute_performance(net_portfolio_value, theta_exec, net_pnl,
                      dev_index, holdout_index, v0, trading_days)
    → returns perf_df (3 rows: full, IS, OOS)
"""

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    display = print


def max_dd(equity: pd.Series) -> float:
    """Maximum drawdown: min((equity - peak) / peak)."""
    eq = equity.dropna()
    if eq.empty:
        return float("nan")
    peak = eq.cummax()
    dd = (eq - peak) / peak.replace(0.0, float("nan"))
    return float(dd.min())


def sharpe_ratio(returns: pd.Series, trading_days: int = 252) -> float:
    """Annualised Sharpe ratio (√trading_days × mean / std)."""
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(np.sqrt(trading_days) * r.mean() / r.std(ddof=1))


def sortino_ratio(returns: pd.Series, mar: float = 0.0, trading_days: int = 252) -> float:
    """Annualised Sortino ratio (downside deviation vs mar)."""
    r = returns.dropna()
    downside = r[r < mar] - mar
    if len(r) < 2 or len(downside) == 0:
        return float("nan")
    ds = downside.std(ddof=0)
    if ds == 0 or np.isnan(ds):
        return float("nan")
    return float(np.sqrt(trading_days) * (r.mean() - mar) / ds)


def calmar_ratio(returns: pd.Series, equity: pd.Series, trading_days: int = 252) -> float:
    """Calmar ratio: annualised return / |max drawdown|."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    if equity.iloc[-1] <= 0 or equity.iloc[0] <= 0:
        return float("nan")
    ann = (equity.iloc[-1] / equity.iloc[0]) ** (trading_days / len(r)) - 1.0
    mdd = max_dd(equity)
    if mdd is None or mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(ann / abs(mdd))


def mean_holding_horizon_per_asset(theta_exec: pd.DataFrame) -> float:
    """Mean length of contiguous directional positions across all assets."""
    durations = []
    for col in theta_exec.columns:
        sign  = np.sign(theta_exec[col].fillna(0))
        flips = (sign != sign.shift()).cumsum()
        for _, grp in sign.groupby(flips):
            if grp.iloc[0] != 0:
                durations.append(len(grp))
    return float(np.mean(durations)) if durations else float("nan")


def compute_performance(
    net_portfolio_value,
    theta_exec,
    net_pnl,
    dev_index,
    holdout_index,
    v0,
    trading_days=252,
):
    """Compute Sharpe, Sortino, Calmar, max DD, holding horizon on full/IS/OOS.

    Args:
        net_portfolio_value : pd.Series of equity curve
        theta_exec          : DataFrame of executed exposures
        net_pnl             : pd.Series of daily net P&L
        dev_index           : DatetimeIndex for in-sample period
        holdout_index       : DatetimeIndex for out-of-sample period
        v0                  : initial capital
        trading_days        : annualisation factor (default 252)

    Returns:
        perf_df : pd.DataFrame with 3 rows (full, IS, OOS) × metric columns
    """
    r_net_daily = net_portfolio_value.pct_change().replace(
        [float("inf"), float("-inf")], float("nan")
    )

    def perf_block(mask: pd.Index, label: str) -> dict:
        m  = net_portfolio_value.index.isin(mask)
        eq = net_portfolio_value[m]
        r  = r_net_daily[m]
        total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) > 1 else float("nan")
        ann_ret = (
            float((eq.iloc[-1] / eq.iloc[0]) ** (trading_days / len(r.dropna())) - 1.0)
            if (len(r.dropna()) > 1 and eq.iloc[0] > 0 and eq.iloc[-1] > 0)
            else float("nan")
        )
        return {
            "sample":       label,
            "n_days":       int(m.sum()),
            "total_return": total_ret,
            "ann_return":   ann_ret,
            "sharpe":       sharpe_ratio(r, trading_days),
            "sortino":      sortino_ratio(r, trading_days=trading_days),
            "calmar":       calmar_ratio(r, eq, trading_days),
            "max_drawdown": max_dd(eq),
            "total_net_pnl": float(net_pnl.loc[m].sum()) if m.any() else float("nan"),
        }

    horizon_days = mean_holding_horizon_per_asset(theta_exec)

    perf_full = {
        "sample":       "full",
        "n_days":       len(net_portfolio_value),
        "total_return": float(net_portfolio_value.iloc[-1] / v0 - 1.0),
        "ann_return":   float(
            (net_portfolio_value.iloc[-1] / v0) ** (trading_days / len(r_net_daily.dropna())) - 1.0
        ) if net_portfolio_value.iloc[-1] > 0 else float("nan"),
        "sharpe":       sharpe_ratio(r_net_daily, trading_days),
        "sortino":      sortino_ratio(r_net_daily, trading_days=trading_days),
        "calmar":       calmar_ratio(r_net_daily, net_portfolio_value, trading_days),
        "max_drawdown": max_dd(net_portfolio_value),
        "total_net_pnl": float(net_pnl.sum()),
        "pct_return_on_V0": float(net_pnl.sum() / v0),
        "mean_holding_horizon_days": horizon_days,
    }

    perf_is  = perf_block(dev_index,     "IS (development)")
    perf_oos = perf_block(holdout_index, "OOS (holdout)")

    perf_df = pd.DataFrame([perf_full, perf_is, perf_oos])

    display(perf_df.T)
    print(f"Mean holding horizon (days, per-asset directional position): {horizon_days:.2f}")

    return perf_df
