from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_helpers import ensure_utc_index
from wf_trend_pipeline import calmar_ratio, max_drawdown, sharpe_ratio, sortino_ratio


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
        (dom_change - dom_change.rolling(10).mean()) / dom_change.rolling(10).std(ddof=1)
    ).rename("dom_zscore_10")
    alt_returns = close[alt_universe].pct_change().replace([np.inf, -np.inf], np.nan)
    btc_30d_ret = close["BTC"].pct_change(30).rename("btc_30d_ret")
    dom_roc = btc_dom.diff(3).rename("dom_roc_3")

    rel_ret_3d = close[alt_universe].pct_change(3).subtract(close["BTC"].pct_change(3), axis=0)
    underperf_score = (-rel_ret_3d).clip(lower=0.0)
    vol_20 = close[alt_universe].pct_change().rolling(20).std()
    inv_vol = (1.0 / vol_20).replace([np.inf, -np.inf], np.nan)
    raw_basket = underperf_score.multiply(inv_vol)
    basket_weights = raw_basket.div(raw_basket.sum(axis=1), axis=0).fillna(0.0)

    eq_w = pd.DataFrame(
        1.0 / len(alt_universe),
        index=basket_weights.index,
        columns=basket_weights.columns,
    )
    basket_weights = basket_weights.where(raw_basket.sum(axis=1).fillna(0.0) > 0.0, eq_w)

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
) -> dict:
    return {
        "entry_pct": float(entry_pct),
        "lookback": int(lookback),
        "exit_z": float(exit_z),
        "max_hold": int(max_hold),
        "roc_window": int(roc_window),
        "roc_pct": float(roc_pct),
        "crash_threshold": None if crash_threshold is None else float(crash_threshold),
    }


def run_s2_strategy(
    features: dict,
    params: dict,
    alt_universe: list[str],
    initial_capital: float,
    cost_bps: float,
) -> dict:
    entry_pct = float(params["entry_pct"])
    lookback = int(params["lookback"])
    exit_z = float(params["exit_z"])
    max_hold = int(params["max_hold"])
    roc_window = int(params["roc_window"])
    roc_pct = float(params["roc_pct"])
    crash_threshold = params.get("crash_threshold")

    btc_dom = features["btc_dom"]
    dom_zscore = features["dom_zscore"]
    btc_30d_ret = features["btc_30d_ret"]
    basket_weights = features["basket_weights"].reindex(columns=alt_universe)
    alt_returns = features["alt_returns"].reindex(columns=alt_universe)
    index = alt_returns.index

    rolling_entry_threshold = dom_zscore.rolling(lookback, min_periods=max(63, lookback // 4)).quantile(entry_pct)
    signal_raw = (dom_zscore > rolling_entry_threshold).astype(int)

    dom_roc = btc_dom.diff(roc_window).rename(f"dom_roc_{roc_window}")
    rolling_roc_threshold = dom_roc.rolling(lookback, min_periods=max(63, lookback // 4)).quantile(roc_pct)
    roc_filter = (dom_roc > rolling_roc_threshold).astype(int)
    entry_signal = (signal_raw & roc_filter).astype(int)

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
            elif not np.isnan(z_val) and z_val <= exit_z:
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

    available_mask = alt_returns.notna()
    available_alt_count = available_mask.sum(axis=1)
    exec_weights = basket_weights.multiply(position_mask, axis=0)
    exec_weights = exec_weights.shift(1).fillna(0.0)
    exec_weights = exec_weights.where(available_mask, 0.0)
    weight_sums = exec_weights.sum(axis=1).replace(0.0, np.nan)
    exec_weights = exec_weights.div(weight_sums, axis=0).fillna(0.0)

    gross_daily = (exec_weights * alt_returns.fillna(0.0)).sum(axis=1).rename("s2_gross_daily")
    turnover = exec_weights.diff().abs().sum(axis=1).fillna(0.0).rename("s2_turnover")
    cost_daily = ((cost_bps / 10000.0) * turnover).rename("s2_cost_daily")
    net_daily = (gross_daily - cost_daily).rename("s2_net_daily")
    gross_value = (initial_capital * (1.0 + gross_daily.fillna(0.0)).cumprod()).rename("s2_gross_value")
    net_value = (initial_capital * (1.0 + net_daily.fillna(0.0)).cumprod()).rename("s2_net_value")
    net_returns = net_value.pct_change().replace([np.inf, -np.inf], np.nan).rename("s2_r_net_daily")

    pos_days = exec_weights.sum(axis=1)
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
        "entry_signal": entry_signal,
        "available_alt_count": available_alt_count,
        "trade_entries": trade_entries,
        "trade_exits": trade_exits,
        "trade_exit_reasons": trade_exit_reasons,
        "crash_blocks": crash_blocks,
        "gross_daily": gross_daily,
        "turnover": turnover,
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
    active_days = int(run_state["weights"].sum(axis=1)[m].gt(0).sum())
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
    n_td = len(r)
    ann_return = (
        float((eq.iloc[-1] / eq.iloc[0]) ** (trading_days / n_td) - 1.0)
        if n_td > 1 and eq.iloc[0] > 0 and eq.iloc[-1] > 0
        else np.nan
    )
    return {
        "sample": label,
        "n_days": int(m.sum()),
        "trade_entries": trade_entries,
        "active_days": active_days,
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        "ann_return": ann_return,
        "sharpe": sharpe_ratio(r, trading_days),
        "sortino": sortino_ratio(r, trading_days=trading_days),
        "calmar": calmar_ratio(r, eq, trading_days),
        "max_drawdown": max_drawdown(eq),
        "total_net_pnl": float(eq.iloc[-1] - eq.iloc[0]),
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
