"""
pairs_backtest.py — Backtest wrapper for the cointegration / pairs-trading
strategy (Strategy 2).

Wires the cointegration signal engine (cointegration.py) into the shared
backtest engine (helpers.run_net_backtest) and the Abdi-Ranaldo cost model.

Functions:
  run_pairs_backtest(asset_panel, cleaned_frames, symbols, ...)
    → dict with backtest outputs + cointegration diagnostics
"""

import pandas as pd

from .cointegration import select_pairs_and_build_theta
from .helpers import compute_half_spread_frac, run_net_backtest


def run_pairs_backtest(
    asset_panel: pd.DataFrame,
    cleaned_frames: dict,
    symbols: list[str],
    *,
    coint_test_window: int = 252,
    rolling_hedge_window: int = 60,
    coint_pvalue_threshold: float = 0.05,
    zscore_entry: float = 2.0,
    zscore_exit: float = 0.5,
    zscore_stop: float = 4.0,
    gross_cap: float = 100_000.0,
    oos_window: int = 63,
    v0: float = 10_000.0,
    backtest_use_excess: bool = False,
) -> dict:
    """Run the full pairs/cointegration backtest.

    1. Walk-forward pair selection and theta construction
    2. Abdi-Ranaldo transaction cost estimation
    3. Shared P&L engine (run_net_backtest)

    Args:
        asset_panel            : MultiIndex DataFrame (close, excess_return)
        cleaned_frames         : dict mapping symbol → cleaned OHLCV DataFrame
        symbols                : list of asset strings
        coint_test_window      : bars for in-sample cointegration test
        rolling_hedge_window   : look-back for hedge ratio and z-score
        coint_pvalue_threshold : Engle-Granger p-value cutoff
        zscore_entry           : |z| threshold to open a position
        zscore_exit            : |z| threshold for mean-reversion exit
        zscore_stop            : |z| threshold for stop-loss exit
        gross_cap              : maximum gross notional exposure
        oos_window             : out-of-sample trading window length (bars)
        v0                     : initial capital (USDT)
        backtest_use_excess    : if True, use excess_return instead of simple returns

    Returns:
        dict with keys from run_net_backtest plus cointegration diagnostics:
          net_portfolio_value, theta_exec, net_pnl, gross_pnl,
          cost_t, turnover, half_spread_frac, liquidated_at,
          pair_signals, pair_spreads, pair_zscores, pair_betas,
          coint_results, warm_up_end, window_selections
    """
    # Step 1: build theta from cointegration signals
    coint_out = select_pairs_and_build_theta(
        asset_panel,
        symbols,
        coint_test_window=coint_test_window,
        rolling_hedge_window=rolling_hedge_window,
        coint_pvalue_threshold=coint_pvalue_threshold,
        zscore_entry=zscore_entry,
        zscore_exit=zscore_exit,
        zscore_stop=zscore_stop,
        gross_cap=gross_cap,
        oos_window=oos_window,
    )

    theta_target = coint_out["theta_target"]
    rebalance_mask = coint_out["rebalance_mask"]

    # Step 2: transaction cost model (same as Strategy 1)
    half_spread_frac = compute_half_spread_frac(asset_panel, cleaned_frames, symbols)

    # Step 3: package theta for the shared backtest engine
    theta_input = theta_target.copy()
    theta_input.attrs["theta_target"] = theta_target.copy()
    theta_input.attrs["rebalance_mask"] = rebalance_mask
    theta_input.attrs["rebalance_every"] = 1

    backtest_out = run_net_backtest(
        asset_panel,
        theta_input,
        symbols,
        half_spread_frac,
        backtest_use_excess=backtest_use_excess,
        v0=v0,
    )

    # Print summary
    cum_gross = backtest_out["gross_pnl"].cumsum()
    cum_net = backtest_out["net_pnl"].cumsum()
    print(f"Strategy 2 (Pairs) - Total gross PnL (USDT): {cum_gross.iloc[-1]:,.2f}")
    print(f"Strategy 2 (Pairs) - Total costs (USDT):     {backtest_out['cost_t'].sum():,.2f}")
    print(f"Strategy 2 (Pairs) - Total net PnL (USDT):   {cum_net.iloc[-1]:,.2f}")

    n_windows = len(coint_out["window_selections"])
    n_active = sum(1 for w in coint_out["window_selections"] if w["n_selected"] > 0)
    print(f"Walk-forward windows: {n_windows} total, {n_active} with >=1 cointegrated pair")

    # Combine outputs
    return {
        # Backtest outputs
        "backtest_out": backtest_out,
        "net_portfolio_value": backtest_out["net_portfolio_value"],
        "theta_exec": backtest_out["theta_exec"],
        "net_pnl": backtest_out["net_pnl"],
        "gross_pnl": backtest_out["gross_pnl"],
        "cost_t": backtest_out["cost_t"],
        "turnover": backtest_out["turnover"],
        "half_spread_frac": half_spread_frac,
        "liquidated_at": backtest_out["liquidated_at"],
        # Cointegration diagnostics
        "pair_signals": coint_out["pair_signals"],
        "pair_spreads": coint_out["pair_spreads"],
        "pair_zscores": coint_out["pair_zscores"],
        "pair_betas": coint_out["pair_betas"],
        "coint_results": coint_out["coint_results"],
        "warm_up_end": coint_out["warm_up_end"],
        "window_selections": coint_out["window_selections"],
    }
