#!/usr/bin/env python3
"""Recompute IS/OOS gross PnL and turnover from the canonical strategy pipeline."""

from __future__ import annotations

import json
import os
import pickle
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_a, **_k):
        return False

from strategy2_pipeline import build_s2_features, make_s2_params, run_s2_strategy
from strategy_helpers import audit_ohlcv, ensure_utc_index, read_parquet_safe, write_parquet_safe
from wf_trend_pipeline import (
    build_signal_and_theta,
    compute_half_spread_frac,
    metrics_on_window,
    run_net_backtest,
)

# --- mirror notebook cell 1 constants ---
load_dotenv()
REFRESH_S1_DATA = False
REFRESH_FRED_DATA = False
REFRESH_S2_CACHE = False
DATA_DIR = Path("data_final")
OUT_DIR = Path("output")
S1_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT"]
S1_TIMEFRAME = "1d"
S1_SINCE = "2020-01-01"
S1_UNTIL = "2026-03-20"
TRADING_DAYS = 252
V0 = 10_000.0
GROSS_CAP = 100_000.0
COV_WINDOW = 120
GAMMA = 1.0
USE_MVO = True
SIGNAL_CLIP_WF = 5.0
STABILITY_PENALTY = 0.10
FAST_WF_SEARCH = True
FAST_WF_FOLD_STRIDE = 2
WF_SEARCH_USE_MVO = USE_MVO and (not FAST_WF_SEARCH)
WF_SEARCH_FAST_MODE = FAST_WF_SEARCH
MA_GRID = [100, 120, 140]
VOL_GRID = [30, 40]
THRESH_GRID = [1.0, 1.25]
REBAL_GRID = [10]
UTIL_GRID = [0.10, 0.20]
GAMMA_GRID = [0.5, 1.0, 2.0] if WF_SEARCH_USE_MVO else [GAMMA]
MAX_MEAN_VAL_TURNOVER = 4_000.0
MIN_POSITIVE_FOLD_SHARE = 0.55
WF_TIE_BAND = 0.05
DEFAULT_WF = {
    "MA_WINDOW": 120,
    "VOL_WINDOW": 40,
    "DEAD_ZONE": 1.25,
    "rebalance_every": 10,
    "gross_util": 0.10,
}
INCLUDE_CANDIDATES = [
    {
        "MA_WINDOW": 120,
        "VOL_WINDOW": 40,
        "DEAD_ZONE": 1.25,
        "rebalance_every": 10,
        "gross_util": 0.10,
    },
]
HOLDOUT_FRAC_CONTAM = 0.225
FINAL_TEST_BARS = 126
INITIAL_TRAIN_BARS = 600
VAL_BARS = 100
STEP_BARS = 100
WF_MODE = "expanding"
TRAIN_BARS = 600

# Strategy 2 (notebook)
S2_BINANCE = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "BNB": "BNB/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "ADA": "ADA/USDT",
    "AVAX": "AVAX/USDT",
    "DOT": "DOT/USDT",
    "LINK": "LINK/USDT",
    "LTC": "LTC/USDT",
    "MATIC": "MATIC/USDT",
}
S2_EXPECTED_COLUMNS = list(S2_BINANCE)
S2_CACHE = Path("s2_binance_cache.pkl")
S2_CACHE_FIELDS = ["cg_open", "cg_high", "cg_low", "cg_close", "cg_volume", "cg_notional"]


def _load_cached_series_frame(path: Path) -> pd.DataFrame:
    frame = read_parquet_safe(path)
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    return ensure_utc_index(frame).sort_index()


def load_or_refresh_symbol(symbol: str) -> tuple[pd.DataFrame, str]:
    key = symbol.replace("/", "")
    raw_path = DATA_DIR / f"{key}_{S1_TIMEFRAME}.parquet"
    if raw_path.exists() and not REFRESH_S1_DATA:
        raw = _load_cached_series_frame(raw_path)
        return raw, "cache"
    raise FileNotFoundError(f"Missing {raw_path}. Set REFRESH_S1_DATA=True to download.")


def load_or_refresh_fred() -> tuple[pd.Series, str]:
    from fredapi import Fred

    fred_path = DATA_DIR / "DFF_daily.parquet"
    if fred_path.exists() and not REFRESH_FRED_DATA:
        frame = _load_cached_series_frame(fred_path)
    else:
        raise FileNotFoundError(f"Missing {fred_path}.")

    if "fed_funds_rate" in frame.columns:
        series = frame["fed_funds_rate"]
    elif frame.shape[1] == 1:
        series = frame.iloc[:, 0].rename("fed_funds_rate")
    else:
        raise ValueError("Could not identify fed_funds_rate column.")
    return series.astype(float), "cache"


VALUE_COLUMNS = ["open", "high", "low", "close", "volume"]


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    frame = ensure_utc_index(df).sort_index().copy()
    frame = frame[~frame.index.duplicated(keep="first")]
    for col in VALUE_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    valid = (
        frame[VALUE_COLUMNS].notna().all(axis=1)
        & (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (frame["volume"] >= 0)
        & (frame["high"] >= frame["low"])
        & (frame["high"] >= frame[["open", "close"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close"]].min(axis=1))
    )
    frame = frame.loc[valid].copy()
    simple_return = frame["close"].pct_change()
    rolling = simple_return.rolling(24, min_periods=8)
    return_zscore = (simple_return - rolling.mean()) / rolling.std(ddof=0).replace(0.0, np.nan)
    frame["simple_return"] = simple_return
    frame["return_zscore"] = return_zscore
    frame["is_outlier"] = return_zscore.abs().gt(5.0).fillna(False)
    return frame


def _nanmean_or_nan(values):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if len(finite) else np.nan


def _within_tie_band(series: pd.Series, best_value: float, tie_band: float) -> pd.Series:
    if not np.isfinite(best_value):
        return pd.Series(False, index=series.index)
    threshold = best_value - abs(best_value) * tie_band
    return series >= threshold


def s1_window_sums(out: dict, window: pd.Index) -> tuple[float, float, float, float]:
    g = out["gross_pnl"].reindex(window)
    n = out["net_pnl"].reindex(window)
    c = out["cost_t"].reindex(window)
    t = out["turnover"].reindex(window)
    abs_gross = float(g.abs().sum())
    cost_sum = float(c.sum())
    cost_to_abs_gross_pct = 100.0 * cost_sum / abs_gross if abs_gross > 0 else float("nan")
    return (float(g.sum()), float(n.sum()), float(t.sum()), float(cost_to_abs_gross_pct))


def s2_window_sums(run: dict, window: pd.Index) -> tuple[float, float, float]:
    g = run["gross_pnl_daily"].reindex(window)
    net = g - run["cost_pnl_daily"].reindex(window)
    to = run["turnover"].reindex(window)
    return float(g.sum()), float(net.sum()), float(to.sum())


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)

    raw_frames: dict[str, pd.DataFrame] = {}
    for symbol in S1_SYMBOLS:
        raw_frames[symbol], _ = load_or_refresh_symbol(symbol)
        raw_frames[symbol].index.name = "timestamp"

    dff, _ = load_or_refresh_fred()
    dff.index = pd.to_datetime(dff.index, utc=True)

    cleaned_frames: dict[str, pd.DataFrame] = {}
    for symbol in S1_SYMBOLS:
        key = symbol.replace("/", "")
        clean_path = DATA_DIR / f"{key}_{S1_TIMEFRAME}_cleaned.parquet"
        raw = raw_frames[symbol]
        if clean_path.exists() and not REFRESH_S1_DATA:
            clean = _load_cached_series_frame(clean_path)
            if not {"simple_return", "return_zscore", "is_outlier"}.issubset(clean.columns):
                clean = clean_ohlcv(raw)
        else:
            clean = clean_ohlcv(raw)
        cleaned_frames[symbol] = clean

    rf_daily = ((1.0 + dff / 100.0) ** (1.0 / 365.0) - 1.0).rename("rf_daily")
    rf_daily = rf_daily.reindex(pd.date_range(dff.index.min(), dff.index.max(), freq="D", tz="UTC")).ffill()

    for symbol in S1_SYMBOLS:
        asset = cleaned_frames[symbol].copy()
        asset["rf_daily"] = rf_daily.reindex(asset.index).ffill()
        asset["excess_return"] = asset["close"].pct_change() - asset["rf_daily"]
        cleaned_frames[symbol] = asset

    close_wide = pd.concat(
        [cleaned_frames[s]["close"] for s in S1_SYMBOLS],
        axis=1,
        keys=pd.MultiIndex.from_product([["close"], S1_SYMBOLS], names=["field", "asset"]),
    )
    excess_wide = pd.concat(
        [cleaned_frames[s]["excess_return"] for s in S1_SYMBOLS],
        axis=1,
        keys=pd.MultiIndex.from_product([["excess_return"], S1_SYMBOLS], names=["field", "asset"]),
    )
    asset_panel = pd.concat([close_wide, excess_wide], axis=1).sort_index(axis=1).dropna(how="any")

    idx = asset_panel.index.sort_values()
    n = len(idx)
    holdout_index = idx[-FINAL_TEST_BARS:]
    dev_index = idx[:-FINAL_TEST_BARS]

    fold_rows = []
    val_start_pos = INITIAL_TRAIN_BARS
    fold_id = 0
    while val_start_pos + VAL_BARS <= len(dev_index):
        fold_id += 1
        val_ix = dev_index[val_start_pos : val_start_pos + VAL_BARS]
        if WF_MODE == "expanding":
            train_ix = dev_index[:val_start_pos]
        else:
            train_ix = dev_index[val_start_pos - TRAIN_BARS : val_start_pos]
        fold_rows.append(
            {
                "fold": fold_id,
                "train_start": train_ix.min(),
                "train_end": train_ix.max(),
                "train_bars": len(train_ix),
                "val_start": val_ix.min(),
                "val_end": val_ix.max(),
                "val_bars": len(val_ix),
            }
        )
        val_start_pos += STEP_BARS
    wf_folds = pd.DataFrame(fold_rows)

    asset_panel_dev = asset_panel.loc[dev_index].copy()
    half_spread_frac_dev = compute_half_spread_frac(asset_panel_dev, cleaned_frames, S1_SYMBOLS)
    half_spread_frac_full = compute_half_spread_frac(asset_panel, cleaned_frames, S1_SYMBOLS)

    candidate_tuples = list(product(MA_GRID, VOL_GRID, THRESH_GRID, REBAL_GRID, UTIL_GRID, GAMMA_GRID))
    for inc in INCLUDE_CANDIDATES:
        for gamma_wf in GAMMA_GRID:
            candidate_tuples.append(
                (
                    int(inc["MA_WINDOW"]),
                    int(inc["VOL_WINDOW"]),
                    float(inc["DEAD_ZONE"]),
                    int(inc["rebalance_every"]),
                    float(inc["gross_util"]),
                    float(gamma_wf),
                )
            )
    all_candidates = list(dict.fromkeys(candidate_tuples))

    wf_folds_eval = wf_folds
    if FAST_WF_SEARCH:
        wf_folds_eval = wf_folds.iloc[:: max(int(FAST_WF_FOLD_STRIDE), 1)].copy()
        if wf_folds_eval.empty:
            wf_folds_eval = wf_folds.tail(1).copy()

    feature_cache: dict = {}
    wf_rows = []
    for ma_w, vol_w, dz, rebal, util, gamma_wf in all_candidates:
        theta_p = build_signal_and_theta(
            asset_panel_dev,
            S1_SYMBOLS,
            ma_window=ma_w,
            vol_window=vol_w,
            dead_zone=dz,
            signal_clip=SIGNAL_CLIP_WF,
            gross_cap=GROSS_CAP * util,
            rebalance_every=rebal,
            use_mvo=WF_SEARCH_USE_MVO,
            cov_window=COV_WINDOW,
            gamma=gamma_wf,
            feature_cache=feature_cache,
            search_fast_mode=WF_SEARCH_FAST_MODE,
        )
        out_p = run_net_backtest(
            asset_panel_dev,
            theta_p,
            S1_SYMBOLS,
            half_spread_frac_dev,
            backtest_use_excess=False,
            v0=V0,
        )
        fold_sharpes, fold_rets, fold_dds, fold_tos, fold_cost_ratios, fold_active_days = [], [], [], [], [], []
        for _, row in wf_folds_eval.iterrows():
            val_ix = dev_index[(dev_index >= row["val_start"]) & (dev_index <= row["val_end"])]
            metrics = metrics_on_window(
                out_p["net_portfolio_value"],
                out_p["turnover"],
                val_ix,
                trading_days=TRADING_DAYS,
                gross_pnl=out_p["gross_pnl"],
                cost_t=out_p["cost_t"],
                theta_exec=out_p["theta_exec"],
            )
            fold_sharpes.append(metrics["sharpe"])
            fold_rets.append(metrics["total_return"])
            fold_dds.append(metrics["max_dd"])
            fold_tos.append(metrics["mean_turnover"])
            fold_cost_ratios.append(metrics["cost_to_gross_ratio"])
            fold_active_days.append(metrics["active_days_pct"])
        fs = np.asarray(fold_sharpes, dtype=float)
        finite_fs = fs[np.isfinite(fs)]
        mean_s = float(np.mean(finite_fs)) if len(finite_fs) else np.nan
        std_s = float(np.std(finite_fs, ddof=1)) if len(finite_fs) > 1 else 0.0
        wf_score = mean_s - STABILITY_PENALTY * std_s if np.isfinite(mean_s) else np.nan
        positive_fold_share = float(np.mean(np.isfinite(fs) & (fs > 0))) if len(fs) else 0.0
        traded_fold_share = float(np.mean(np.asarray(fold_active_days, dtype=float) > 0)) if len(fold_active_days) else 0.0
        mean_turnover = _nanmean_or_nan(fold_tos)
        is_candidate = bool(
            np.isfinite(mean_s)
            and mean_s > 0
            and positive_fold_share >= MIN_POSITIVE_FOLD_SHARE
            and np.isfinite(mean_turnover)
            and mean_turnover <= MAX_MEAN_VAL_TURNOVER
        )
        wf_rows.append(
            {
                "MA_WINDOW": ma_w,
                "VOL_WINDOW": vol_w,
                "DEAD_ZONE": dz,
                "rebalance_every": rebal,
                "gross_util": util,
                "mean_val_sharpe": mean_s,
                "std_val_sharpe": std_s,
                "wf_score": wf_score,
                "mean_val_return": _nanmean_or_nan(fold_rets),
                "mean_val_max_dd": _nanmean_or_nan(fold_dds),
                "mean_val_turnover": mean_turnover,
                "mean_val_cost_to_gross": _nanmean_or_nan(fold_cost_ratios),
                "positive_fold_share": positive_fold_share,
                "traded_fold_share": traded_fold_share,
                "gamma": gamma_wf,
                "active_days_pct": _nanmean_or_nan(fold_active_days),
                "is_candidate": is_candidate,
            }
        )

    wf_search_df = pd.DataFrame(wf_rows)
    candidates = wf_search_df[wf_search_df["is_candidate"]].copy()
    if not candidates.empty:
        best_score = float(candidates["wf_score"].max())
        finalists = candidates[_within_tie_band(candidates["wf_score"], best_score, WF_TIE_BAND)].copy()
        finalists["abs_mean_val_max_dd"] = finalists["mean_val_max_dd"].abs()
        best = finalists.sort_values(
            ["mean_val_turnover", "mean_val_return", "abs_mean_val_max_dd"],
            ascending=[True, False, True],
        ).iloc[0]
        selection_mode = "candidate"
    else:
        fallback_mask = (
            (wf_search_df["MA_WINDOW"] == DEFAULT_WF["MA_WINDOW"])
            & (wf_search_df["VOL_WINDOW"] == DEFAULT_WF["VOL_WINDOW"])
            & (wf_search_df["DEAD_ZONE"] == DEFAULT_WF["DEAD_ZONE"])
            & (wf_search_df["rebalance_every"] == DEFAULT_WF["rebalance_every"])
            & (wf_search_df["gross_util"] == DEFAULT_WF["gross_util"])
        )
        if fallback_mask.any():
            best = wf_search_df.loc[fallback_mask].iloc[0]
            selection_mode = "default_fallback"
        else:
            best = wf_search_df.sort_values(
                ["wf_score", "mean_val_turnover", "mean_val_return", "mean_val_max_dd"],
                ascending=[False, True, False, False],
            ).iloc[0]
            selection_mode = "best_available_fallback"

    strategy1_selected_params = {
        "MA_WINDOW": int(best["MA_WINDOW"]),
        "VOL_WINDOW": int(best["VOL_WINDOW"]),
        "DEAD_ZONE": float(best["DEAD_ZONE"]),
        "rebalance_every": int(best["rebalance_every"]),
        "gross_util": float(best["gross_util"]),
        "gamma": float(best["gamma"]),
    }

    theta = build_signal_and_theta(
        asset_panel,
        S1_SYMBOLS,
        ma_window=int(strategy1_selected_params["MA_WINDOW"]),
        vol_window=int(strategy1_selected_params["VOL_WINDOW"]),
        dead_zone=float(strategy1_selected_params["DEAD_ZONE"]),
        signal_clip=SIGNAL_CLIP_WF,
        gross_cap=GROSS_CAP * float(strategy1_selected_params["gross_util"]),
        rebalance_every=int(strategy1_selected_params["rebalance_every"]),
        use_mvo=USE_MVO,
        cov_window=COV_WINDOW,
        gamma=float(strategy1_selected_params.get("gamma", GAMMA)),
    )
    s1_out = run_net_backtest(
        asset_panel,
        theta,
        S1_SYMBOLS,
        half_spread_frac_full,
        backtest_use_excess=False,
        v0=V0,
    )

    s1_is_gross, s1_is_net, s1_is_to, s1_is_c2g = s1_window_sums(s1_out, dev_index)
    s1_oos_gross, s1_oos_net, s1_oos_to, s1_oos_c2g = s1_window_sums(s1_out, holdout_index)

    # Strategy 2
    s1_index = asset_panel.index.sort_values()
    _cache_key = (
        len(s1_index),
        str(s1_index.min()),
        str(s1_index.max()),
        "binance_v3_abdi_ranaldo",
        tuple(S2_BINANCE.items()),
    )

    def _align_s2_frame(frame: pd.DataFrame, target_index: pd.Index) -> pd.DataFrame:
        return ensure_utc_index(frame).sort_index().reindex(columns=S2_EXPECTED_COLUMNS).reindex(target_index)

    def _align_s2_payload(payload: dict, target_index: pd.Index) -> dict | None:
        aligned = {}
        for name in S2_CACHE_FIELDS:
            frame = payload.get(name)
            if not isinstance(frame, pd.DataFrame):
                return None
            aligned[name] = _align_s2_frame(frame, target_index)
        return aligned

    def _load_s2_cache():
        if not S2_CACHE.exists():
            return None
        with S2_CACHE.open("rb") as fh:
            payload = pickle.load(fh)
        if payload.get("cache_key") != _cache_key:
            return None
        return _align_s2_payload(payload, s1_index)

    cached = _load_s2_cache()
    if cached is None:
        raise FileNotFoundError("s2_binance_cache.pkl missing or incompatible.")
    s2_frames = cached
    cg_close = s2_frames["cg_close"]
    cg_notional = s2_frames["cg_notional"]
    cg_open = s2_frames["cg_open"]
    cg_high = s2_frames["cg_high"]
    cg_low = s2_frames["cg_low"]
    cg_volume = s2_frames["cg_volume"]
    s2_cleaned_frames = {
        ticker: pd.DataFrame(
            {
                "open": cg_open[ticker],
                "high": cg_high[ticker],
                "low": cg_low[ticker],
                "close": cg_close[ticker],
                "volume": cg_volume[ticker],
            }
        )
        for ticker in S2_EXPECTED_COLUMNS
    }
    s2_half_spread_frac = compute_half_spread_frac(cg_close, s2_cleaned_frames, S2_EXPECTED_COLUMNS)

    with open(OUT_DIR / "s2_optuna_best_params.json") as fh:
        s2_json = json.load(fh)
    s2_best = s2_json["params"]
    s2_params = make_s2_params(
        entry_pct=s2_best["entry_pct"],
        lookback=int(s2_best["lookback"]),
        exit_z=s2_best["exit_z"],
        max_hold=int(s2_best["max_hold"]),
        roc_window=int(s2_best["roc_window"]),
        roc_pct=s2_best["roc_pct"],
        crash_threshold=s2_best.get("crash_threshold"),
        gross_util=float(s2_best.get("gross_util", 1.0)),
    )
    S2_ALT_COINS = [c for c in cg_close.columns if c != "BTC"]
    s2_features = build_s2_features(cg_close, cg_notional)
    s2_run = run_s2_strategy(
        s2_features,
        s2_params,
        S2_ALT_COINS,
        V0,
        s2_half_spread_frac,
        gross_util=float(s2_params.get("gross_util", 1.0)),
    )

    s2_is_gross, s2_is_net, s2_is_to = s2_window_sums(s2_run, dev_index)
    s2_oos_gross, s2_oos_net, s2_oos_to = s2_window_sums(s2_run, holdout_index)

    results = {
        "wf_selection": selection_mode,
        "strategy1_selected_params": strategy1_selected_params,
        "n_dev": int(len(dev_index)),
        "n_holdout": int(len(holdout_index)),
        "strategy_1": {
            "IS": {
                "gross_pnl_usd": s1_is_gross,
                "net_pnl_usd": s1_is_net,
                "total_turnover_usd": s1_is_to,
                "cost_to_abs_gross_pct": s1_is_c2g,
            },
            "OOS": {
                "gross_pnl_usd": s1_oos_gross,
                "net_pnl_usd": s1_oos_net,
                "total_turnover_usd": s1_oos_to,
                "cost_to_abs_gross_pct": s1_oos_c2g,
            },
        },
        "strategy_2": {
            "IS": {
                "gross_pnl_usd": s2_is_gross,
                "net_pnl_usd": s2_is_net,
                "total_turnover_usd": s2_is_to,
            },
            "OOS": {
                "gross_pnl_usd": s2_oos_gross,
                "net_pnl_usd": s2_oos_net,
                "total_turnover_usd": s2_oos_to,
            },
        },
    }

    out_path = OUT_DIR / "computed_gross_turnover_is_oos.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
