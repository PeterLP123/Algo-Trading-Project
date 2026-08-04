"""Research diagnostics, performance summaries, sweeps, and walk-forward evaluation."""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from .execution import run_cross_sectional_strategy, run_tsmom_strategy


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
