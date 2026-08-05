from __future__ import annotations

import numpy as np
import pandas as pd

from systematic_crypto.data import ensure_utc_index
from wf_trend_pipeline import calmar_ratio, max_drawdown, sharpe_ratio, sortino_ratio

GROSS_CAP = 100_000.0
ENTRY_MODES = {
    "accelerating_spike",
    "confirmed_reversal",
    "falling_after_extreme",
    "relative_momentum_confirmation",
    "falling_with_relative_momentum",
}
EXIT_MODES = {"z_or_max_hold", "fixed_holding"}
POSITION_MODES = {"long_alts", "long_alts_short_btc"}
COST_MODES = {"abdi_ranaldo", "fixed_bps"}


def build_s2_features(cg_close: pd.DataFrame, cg_notional: pd.DataFrame) -> dict:
    close = ensure_utc_index(cg_close).sort_index()
    notional = ensure_utc_index(cg_notional).sort_index()
    if list(close.columns) != list(notional.columns):
        raise ValueError("Strategy 2 close and notional frames must have the same columns.")
    if "BTC" not in close.columns:
        raise ValueError("Strategy 2 data must include a BTC column.")

    alt_universe = [col for col in close.columns if col != "BTC"]
    btc_dom = (notional["BTC"] / notional.sum(axis=1)).rename("btc_dominance")
    dom_change = btc_dom.diff().rename("dom_change")
    dom_zscore = (
        (btc_dom - btc_dom.rolling(20).mean()) / btc_dom.rolling(20).std(ddof=1)
    ).rename("dom_level_zscore_20")
    padded_close = close.ffill()
    alt_returns = padded_close[alt_universe].pct_change(fill_method=None).replace(
        [np.inf, -np.inf], np.nan
    )
    btc_30d_ret = padded_close["BTC"].pct_change(30, fill_method=None).rename("btc_30d_ret")
    dom_roc = btc_dom.diff(3).rename("dom_roc_3")

    rel_ret_3d = padded_close[alt_universe].pct_change(3, fill_method=None).subtract(
        padded_close["BTC"].pct_change(3, fill_method=None), axis=0
    )
    underperf_score = (-rel_ret_3d).clip(lower=0.0)
    vol_20 = padded_close[alt_universe].pct_change(fill_method=None).rolling(20).std()
    inv_vol = (1.0 / vol_20).replace([np.inf, -np.inf], np.nan)
    raw_basket = underperf_score.multiply(inv_vol)
    tilted_basket_weights = raw_basket.div(raw_basket.sum(axis=1), axis=0).fillna(0.0)

    eq_w = pd.DataFrame(
        1.0 / len(alt_universe),
        index=tilted_basket_weights.index,
        columns=tilted_basket_weights.columns,
    )
    # A simple equal-weight alt basket has been materially more robust than the
    # underperformance-tilted basket after BTC dominance spikes.
    basket_weights = eq_w.copy()

    return {
        "index": close.index,
        "cg_close": close,
        "cg_notional": notional,
        "alt_universe": alt_universe,
        "btc_dom": btc_dom,
        "dom_change": dom_change,
        "dom_zscore": dom_zscore,
        "btc_30d_ret": btc_30d_ret,
        "alt_returns": alt_returns,
        "dom_roc_default": dom_roc,
        "rel_ret_3d": rel_ret_3d,
        "underperf_score": underperf_score,
        "vol_20": vol_20,
        "tilted_basket_weights": tilted_basket_weights,
        "basket_weights": basket_weights,
    }


def make_s2_params(
    entry_pct: float,
    lookback: int,
    exit_z: float,
    max_hold: int,
    roc_window: int,
    roc_pct: float,
    crash_threshold: float | None,
    gross_util: float = 1.0,
) -> dict:
    return {
        "entry_pct": float(entry_pct),
        "lookback": int(lookback),
        "exit_z": float(exit_z),
        "max_hold": int(max_hold),
        "roc_window": int(roc_window),
        "roc_pct": float(roc_pct),
        "crash_threshold": None if crash_threshold is None else float(crash_threshold),
        "gross_util": float(gross_util),
    }


def build_s2_entry_signal(
    btc_dom: pd.Series,
    dom_zscore: pd.Series,
    *,
    lookback: int,
    entry_pct: float,
    roc_window: int,
    roc_pct: float,
    entry_mode: str = "accelerating_spike",
    alt_relative_momentum: pd.Series | None = None,
) -> dict[str, pd.Series]:
    if entry_mode not in ENTRY_MODES:
        raise ValueError(
            f"Unknown Strategy 2 entry mode {entry_mode!r}; expected one of "
            f"{sorted(ENTRY_MODES)}."
        )

    min_periods = max(63, lookback // 4)
    rolling_entry_threshold = dom_zscore.rolling(
        lookback,
        min_periods=min_periods,
    ).quantile(entry_pct)
    signal_raw = (dom_zscore > rolling_entry_threshold).astype(int)

    dom_roc = btc_dom.diff(roc_window).rename(f"dom_roc_{roc_window}")
    rolling_roc_threshold = dom_roc.rolling(
        lookback,
        min_periods=min_periods,
    ).quantile(roc_pct)
    roc_filter = (dom_roc > rolling_roc_threshold).astype(int)

    one_day_change = btc_dom.diff().rename("dom_change_1d")
    dominance_reversal = (
        one_day_change.lt(0.0) & one_day_change.shift(1).ge(0.0)
    ).rename("dominance_reversal")
    prior_extreme = signal_raw.shift(1, fill_value=0).astype(bool)
    if alt_relative_momentum is None:
        relative_momentum_positive = pd.Series(False, index=btc_dom.index)
    else:
        relative_momentum_positive = (
            alt_relative_momentum.reindex(btc_dom.index).gt(0.0).fillna(False)
        )

    if entry_mode == "accelerating_spike":
        entry_signal = signal_raw.astype(bool) & roc_filter.astype(bool)
    elif entry_mode == "confirmed_reversal":
        entry_signal = prior_extreme & dominance_reversal
    elif entry_mode == "falling_after_extreme":
        entry_signal = prior_extreme & one_day_change.lt(0.0)
    elif entry_mode == "relative_momentum_confirmation":
        entry_signal = prior_extreme & relative_momentum_positive
    else:
        entry_signal = (
            prior_extreme & one_day_change.lt(0.0) & relative_momentum_positive
        )

    return {
        "rolling_entry_threshold": rolling_entry_threshold,
        "signal_raw": signal_raw,
        "dom_roc": dom_roc,
        "rolling_roc_threshold": rolling_roc_threshold,
        "roc_filter": roc_filter,
        "dominance_reversal": dominance_reversal.astype(int),
        "relative_momentum_positive": relative_momentum_positive.astype(int),
        "entry_signal": entry_signal.astype(int).rename("entry_signal"),
    }


def run_s2_strategy(
    features: dict,
    params: dict,
    alt_universe: list[str],
    initial_capital: float,
    half_spread_frac: pd.DataFrame,
    gross_util: float = 1.0,
) -> dict:
    entry_pct = float(params["entry_pct"])
    lookback = int(params["lookback"])
    exit_z = float(params["exit_z"])
    max_hold = int(params["max_hold"])
    roc_window = int(params["roc_window"])
    roc_pct = float(params["roc_pct"])
    crash_threshold = params.get("crash_threshold")
    entry_mode = str(params.get("entry_mode", "accelerating_spike"))
    exit_mode = str(params.get("exit_mode", "z_or_max_hold"))
    if exit_mode not in EXIT_MODES:
        raise ValueError(
            f"Unknown Strategy 2 exit mode {exit_mode!r}; expected one of "
            f"{sorted(EXIT_MODES)}."
        )
    position_mode = str(params.get("position_mode", "long_alts"))
    if position_mode not in POSITION_MODES:
        raise ValueError(
            f"Unknown Strategy 2 position mode {position_mode!r}; expected one of "
            f"{sorted(POSITION_MODES)}."
        )
    short_carry_rate_annual = float(params.get("short_carry_rate_annual", 0.0))
    if short_carry_rate_annual < 0.0:
        raise ValueError("Strategy 2 annual short carry must be non-negative.")
    cost_mode = str(params.get("cost_mode", "abdi_ranaldo"))
    if cost_mode not in COST_MODES:
        raise ValueError(
            f"Unknown Strategy 2 cost mode {cost_mode!r}; expected one of "
            f"{sorted(COST_MODES)}."
        )
    fixed_one_way_cost_bps = float(params.get("fixed_one_way_cost_bps", 0.0))
    if fixed_one_way_cost_bps < 0.0:
        raise ValueError("Strategy 2 fixed one-way cost must be non-negative.")

    btc_dom = features["btc_dom"]
    dom_zscore = features["dom_zscore"]
    btc_30d_ret = features["btc_30d_ret"]
    basket_weights = features["basket_weights"].reindex(columns=alt_universe)
    alt_returns = features["alt_returns"].reindex(columns=alt_universe)
    alt_relative_momentum = features["rel_ret_3d"].reindex(
        columns=alt_universe
    ).mean(axis=1)
    index = alt_returns.index

    signal_state = build_s2_entry_signal(
        btc_dom,
        dom_zscore,
        lookback=lookback,
        entry_pct=entry_pct,
        roc_window=roc_window,
        roc_pct=roc_pct,
        entry_mode=entry_mode,
        alt_relative_momentum=alt_relative_momentum,
    )
    rolling_entry_threshold = signal_state["rolling_entry_threshold"]
    signal_raw = signal_state["signal_raw"]
    dom_roc = signal_state["dom_roc"]
    rolling_roc_threshold = signal_state["rolling_roc_threshold"]
    roc_filter = signal_state["roc_filter"]
    dominance_reversal = signal_state["dominance_reversal"]
    relative_momentum_positive = signal_state["relative_momentum_positive"]
    entry_signal = signal_state["entry_signal"]

    position_mask = pd.Series(0.0, index=index, name="position_mask")
    in_position = False
    entry_bar = 0
    trade_entries: list[pd.Timestamp] = []
    trade_exits: list[pd.Timestamp] = []
    trade_exit_reasons: list[str] = []
    crash_blocks: list[pd.Timestamp] = []

    for i, dt in enumerate(index):
        sig = int(entry_signal.get(dt, 0))
        z_val = dom_zscore.get(dt, np.nan)
        crash = btc_30d_ret.get(dt, np.nan)
        is_crash = (
            crash_threshold is not None
            and not np.isnan(crash)
            and crash < float(crash_threshold)
        )

        if in_position:
            entry_bar += 1
            if is_crash:
                in_position = False
                trade_exits.append(dt)
                trade_exit_reasons.append("crash exit")
            elif (
                exit_mode == "z_or_max_hold"
                and not np.isnan(z_val)
                and z_val <= exit_z
            ):
                in_position = False
                trade_exits.append(dt)
                trade_exit_reasons.append("z-score exit")
            elif entry_bar >= max_hold:
                in_position = False
                trade_exits.append(dt)
                trade_exit_reasons.append("max hold")
            else:
                position_mask.iloc[i] = 1.0
        elif sig == 1:
            if is_crash:
                crash_blocks.append(dt)
            else:
                in_position = True
                entry_bar = 0
                position_mask.iloc[i] = 1.0
                trade_entries.append(dt)

    if in_position and len(trade_entries) > len(trade_exits):
        trade_exits.append(index[-1])
        trade_exit_reasons.append("end of sample")

    available_alt_mask = alt_returns.notna()
    available_alt_count = available_alt_mask.sum(axis=1)
    alt_exec_weights = basket_weights.multiply(position_mask, axis=0)
    alt_exec_weights = alt_exec_weights.shift(1).fillna(0.0)
    alt_exec_weights = alt_exec_weights.where(available_alt_mask, 0.0)
    weight_sums = alt_exec_weights.sum(axis=1).replace(0.0, np.nan)
    alt_exec_weights = alt_exec_weights.div(weight_sums, axis=0).fillna(0.0)
    if position_mode == "long_alts_short_btc":
        btc_returns = (
            features["cg_close"]["BTC"]
            .pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
            .rename("BTC")
        )
        portfolio_returns = pd.concat([btc_returns, alt_returns], axis=1)
        exec_weights = alt_exec_weights * 0.5
        btc_weight = -0.5 * position_mask.shift(1).fillna(0.0)
        exec_weights.insert(0, "BTC", btc_weight.where(btc_returns.notna(), 0.0))
    else:
        portfolio_returns = alt_returns
        exec_weights = alt_exec_weights
    trade_universe = list(exec_weights.columns)
    half_spread_frac = (
        ensure_utc_index(half_spread_frac)
        .sort_index()
        .reindex(index=index, columns=trade_universe)
        .astype(float)
        .fillna(0.0)
    )

    theta_rows: list[pd.Series] = []
    carried_rows: list[pd.Series] = []
    delta_rows: list[pd.Series] = []
    gross_pnl_asset_rows: list[pd.Series] = []
    gross_pnl_values: list[float] = []
    cost_pnl_values: list[float] = []
    spread_cost_pnl_values: list[float] = []
    short_carry_pnl_values: list[float] = []
    turnover_values: list[float] = []
    gross_equity_values: list[float] = []
    net_equity_values: list[float] = []
    equity_before_trade_values: list[float] = []
    weight_turnover_values: list[float] = []

    zero_row = pd.Series(0.0, index=trade_universe, dtype=float)
    prev_theta = zero_row.copy()
    prev_returns = zero_row.copy()
    equity_prev = float(initial_capital)
    gross_equity_prev = float(initial_capital)

    portfolio_returns_filled = portfolio_returns.fillna(0.0)
    weight_turnover = exec_weights.diff().abs().sum(axis=1).fillna(0.0).rename("s2_weight_turnover")

    for dt in index:
        returns_t = portfolio_returns_filled.loc[dt]
        target_weights_t = exec_weights.loc[dt].fillna(0.0)
        half_spread_t = half_spread_frac.loc[dt].fillna(0.0)

        carried_theta_t = prev_theta * (1.0 + prev_returns)
        equity_before_trade = float(equity_prev)
        gross_target = min(initial_capital * gross_util, GROSS_CAP, equity_before_trade * 10.0)
        target_theta_t = target_weights_t * gross_target
        delta_theta_t = target_theta_t - carried_theta_t

        turnover_t = float(delta_theta_t.abs().sum())
        if cost_mode == "fixed_bps":
            spread_cost_pnl_t = turnover_t * fixed_one_way_cost_bps / 10_000.0
        else:
            spread_cost_pnl_t = float((delta_theta_t.abs() * half_spread_t).sum())
        short_carry_pnl_t = float(
            target_theta_t.clip(upper=0.0).abs().sum()
            * short_carry_rate_annual
            / 365.0
        )
        cost_pnl_t = spread_cost_pnl_t + short_carry_pnl_t
        gross_pnl_asset_t = target_theta_t * returns_t
        gross_pnl_t = float(gross_pnl_asset_t.sum())

        gross_equity_t = max(gross_equity_prev + gross_pnl_t, 0.0)
        net_equity_t = max(equity_before_trade + gross_pnl_t - cost_pnl_t, 0.0)

        carried_rows.append(carried_theta_t)
        theta_rows.append(target_theta_t)
        delta_rows.append(delta_theta_t)
        gross_pnl_asset_rows.append(gross_pnl_asset_t)
        gross_pnl_values.append(gross_pnl_t)
        cost_pnl_values.append(cost_pnl_t)
        spread_cost_pnl_values.append(spread_cost_pnl_t)
        short_carry_pnl_values.append(short_carry_pnl_t)
        turnover_values.append(turnover_t)
        gross_equity_values.append(gross_equity_t)
        net_equity_values.append(net_equity_t)
        equity_before_trade_values.append(equity_before_trade)
        weight_turnover_values.append(float(weight_turnover.loc[dt]))

        prev_theta = target_theta_t
        prev_returns = returns_t
        equity_prev = net_equity_t
        gross_equity_prev = gross_equity_t

    theta = pd.DataFrame(theta_rows, index=index, columns=trade_universe)
    carried_theta = pd.DataFrame(carried_rows, index=index, columns=trade_universe)
    delta_theta = pd.DataFrame(delta_rows, index=index, columns=trade_universe)
    gross_pnl_asset = pd.DataFrame(gross_pnl_asset_rows, index=index, columns=trade_universe)

    equity_before_trade = pd.Series(equity_before_trade_values, index=index, name="s2_equity_before_trade")
    gross_pnl_daily = pd.Series(gross_pnl_values, index=index, name="s2_gross_pnl_daily")
    cost_pnl_daily = pd.Series(cost_pnl_values, index=index, name="s2_cost_pnl_daily")
    spread_cost_pnl_daily = pd.Series(
        spread_cost_pnl_values, index=index, name="s2_spread_cost_pnl_daily"
    )
    short_carry_pnl_daily = pd.Series(
        short_carry_pnl_values, index=index, name="s2_short_carry_pnl_daily"
    )
    turnover = pd.Series(turnover_values, index=index, name="s2_turnover")
    weight_turnover = pd.Series(weight_turnover_values, index=index, name="s2_weight_turnover")
    gross_value = pd.Series(gross_equity_values, index=index, name="s2_gross_value")
    net_value = pd.Series(net_equity_values, index=index, name="s2_net_value")

    equity_base = equity_before_trade.replace(0.0, np.nan)
    gross_daily = gross_pnl_daily.div(equity_base).fillna(0.0).rename("s2_gross_daily")
    cost_daily = cost_pnl_daily.div(equity_base).fillna(0.0).rename("s2_cost_daily")
    net_daily = (gross_daily - cost_daily).rename("s2_net_daily")
    net_returns = net_value.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).rename("s2_r_net_daily")

    pos_days = exec_weights.abs().sum(axis=1)
    trade_log_rows = []
    for trade_num, (entry, exit_, reason) in enumerate(
        zip(trade_entries, trade_exits, trade_exit_reasons), start=1
    ):
        mask = (gross_daily.index > entry) & (gross_daily.index <= exit_)
        trade_log_rows.append(
            {
                "trade_#": trade_num,
                "entry_date": entry.strftime("%Y-%m-%d"),
                "exit_date": exit_.strftime("%Y-%m-%d"),
                "holding_bars": int(pos_days.loc[mask].gt(0).sum()),
                "exit_reason": reason,
                "gross_return": float((1.0 + gross_daily.loc[mask].fillna(0.0)).prod() - 1.0),
            }
        )

    return {
        "params": dict(params),
        "features": features,
        "weights": exec_weights,
        "position_mask": position_mask,
        "rolling_entry_threshold": rolling_entry_threshold,
        "signal_raw": signal_raw,
        "dom_roc": dom_roc,
        "rolling_roc_threshold": rolling_roc_threshold,
        "roc_filter": roc_filter,
        "entry_mode": entry_mode,
        "exit_mode": exit_mode,
        "position_mode": position_mode,
        "cost_mode": cost_mode,
        "fixed_one_way_cost_bps": fixed_one_way_cost_bps,
        "dominance_reversal": dominance_reversal,
        "relative_momentum_positive": relative_momentum_positive,
        "entry_signal": entry_signal,
        "available_alt_count": available_alt_count,
        "trade_entries": trade_entries,
        "trade_exits": trade_exits,
        "trade_exit_reasons": trade_exit_reasons,
        "crash_blocks": crash_blocks,
        "theta": theta,
        "carried_theta": carried_theta,
        "delta_theta": delta_theta,
        "gross_pnl_asset": gross_pnl_asset,
        "equity_before_trade": equity_before_trade,
        "gross_pnl_daily": gross_pnl_daily,
        "cost_pnl_daily": cost_pnl_daily,
        "spread_cost_pnl_daily": spread_cost_pnl_daily,
        "short_carry_pnl_daily": short_carry_pnl_daily,
        "gross_daily": gross_daily,
        "turnover": turnover,
        "weight_turnover": weight_turnover,
        "cost_daily": cost_daily,
        "net_daily": net_daily,
        "gross_value": gross_value,
        "net_value": net_value,
        "net_returns": net_returns,
        "trade_log": pd.DataFrame(trade_log_rows),
    }


def summarize_s2_run(run_state: dict, mask: pd.Index, label: str, trading_days: int = 252) -> dict:
    m = run_state["net_value"].index.isin(mask)
    eq = run_state["net_value"][m].dropna()
    r = run_state["net_returns"][m].dropna()
    trade_entries = int(pd.Index(run_state["trade_entries"]).isin(mask).sum())
    active_days = int(run_state["weights"].abs().sum(axis=1)[m].gt(0).sum())
    cost_drag = float(run_state["cost_daily"][m].sum()) if m.any() else np.nan
    mean_turnover = float(run_state["turnover"][m].mean()) if m.any() else np.nan
    if len(eq) < 2:
        return {
            "sample": label,
            "n_days": int(m.sum()),
            "trade_entries": trade_entries,
            "active_days": active_days,
            "total_return": np.nan,
            "ann_return": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "calmar": np.nan,
            "max_drawdown": np.nan,
            "total_net_pnl": np.nan,
            "cost_drag": cost_drag,
            "mean_turnover": mean_turnover,
        }
    start_eq = float(eq.iloc[0])
    end_eq = float(eq.iloc[-1])
    n_td = len(r)
    ann_return = (
        float((end_eq / start_eq) ** (trading_days / n_td) - 1.0)
        if n_td > 1 and start_eq > 0 and end_eq > 0
        else np.nan
    )
    total_return = float(end_eq / start_eq - 1.0) if start_eq > 0 else np.nan
    return {
        "sample": label,
        "n_days": int(m.sum()),
        "trade_entries": trade_entries,
        "active_days": active_days,
        "total_return": total_return,
        "ann_return": ann_return,
        "sharpe": sharpe_ratio(r, trading_days),
        "sortino": sortino_ratio(r, trading_days=trading_days),
        "calmar": calmar_ratio(r, eq, trading_days),
        "max_drawdown": max_drawdown(eq),
        "total_net_pnl": float(end_eq - start_eq),
        "cost_drag": cost_drag,
        "mean_turnover": mean_turnover,
    }


def score_s2_trial(run_state: dict, wf_folds: pd.DataFrame, dev_index: pd.Index) -> dict:
    fold_rows = []
    for _, row in wf_folds.iterrows():
        val_ix = dev_index[(dev_index >= row["val_start"]) & (dev_index <= row["val_end"])]
        summary = summarize_s2_run(run_state, val_ix, f"Fold {int(row['fold'])}")
        summary["fold"] = int(row["fold"])
        fold_rows.append(summary)

    fold_df = pd.DataFrame(fold_rows)
    dev_summary = summarize_s2_run(run_state, dev_index, "development")
    finite_sharpes = fold_df["sharpe"].dropna()
    mean_val_sharpe = float(finite_sharpes.mean()) if len(finite_sharpes) else np.nan
    std_val_sharpe = float(finite_sharpes.std(ddof=1)) if len(finite_sharpes) > 1 else 0.0
    traded_fold_share = float((fold_df["trade_entries"] > 0).mean()) if len(fold_df) else 0.0
    active_fold_count = int((fold_df["trade_entries"] > 0).sum())
    mean_active_days_per_fold = float(fold_df["active_days"].mean()) if len(fold_df) else np.nan

    inactivity_penalty = 0.0
    penalty_reasons = []
    if dev_summary["trade_entries"] < 8:
        inactivity_penalty += 0.50
        penalty_reasons.append("total_dev_trades < 8")
    if traded_fold_share < 0.35:
        inactivity_penalty += 0.50
        penalty_reasons.append("traded_fold_share < 0.35")

    invalid_reasons = []
    if active_fold_count < 4:
        invalid_reasons.append("fewer than 4 traded folds")
    if not np.isfinite(mean_val_sharpe):
        invalid_reasons.append("no finite fold sharpe")

    score = (
        -1e6
        if invalid_reasons
        else float(mean_val_sharpe - 0.15 * std_val_sharpe - inactivity_penalty)
    )
    return {
        "score": score,
        "fold_df": fold_df,
        "dev_summary": dev_summary,
        "mean_val_sharpe": mean_val_sharpe,
        "std_val_sharpe": std_val_sharpe,
        "mean_val_return": (
            float(fold_df["total_return"].dropna().mean())
            if fold_df["total_return"].notna().any()
            else np.nan
        ),
        "mean_val_max_dd": (
            float(fold_df["max_drawdown"].dropna().mean())
            if fold_df["max_drawdown"].notna().any()
            else np.nan
        ),
        "total_dev_trades": int(dev_summary["trade_entries"]),
        "traded_fold_share": traded_fold_share,
        "mean_active_days_per_fold": mean_active_days_per_fold,
        "active_fold_count": active_fold_count,
        "inactivity_penalty": inactivity_penalty,
        "penalty_reasons": "; ".join(penalty_reasons) if penalty_reasons else "",
        "invalid_reason": "; ".join(invalid_reasons) if invalid_reasons else "",
    }
