#!/usr/bin/env python3
"""Run the frozen Strategy 2 specification on completed post-cutoff candles."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forward_validation import (
    FREEZE_CUTOFF,
    ORIGINAL_HOLDOUT_START,
    POST_SELECTION_ANCHOR,
    file_sha256,
    load_or_fetch_forward,
    resolve_end_date,
)
from strategy2_pipeline import build_s2_features, make_s2_params, run_s2_strategy
from wf_trend_pipeline import compute_half_spread_frac, max_drawdown, sharpe_ratio, sortino_ratio


PROJECT_ROOT = Path(__file__).resolve().parent
SPEC_PATH = PROJECT_ROOT / "forward_validation" / "frozen_strategy2.json"
EXPECTED_SPEC_DIGEST = "ebb7be6005a54cc97bbc6a159d3e3f1bcf0dfc6a6d284b4b1ecd8e2dbd1ce879"
HISTORY_FIELDS = ["cg_open", "cg_high", "cg_low", "cg_close", "cg_volume", "cg_notional"]
OHLCV_TO_HISTORY = {
    "open": "cg_open",
    "high": "cg_high",
    "low": "cg_low",
    "close": "cg_close",
    "volume": "cg_volume",
}


class FrozenStrategy2Error(RuntimeError):
    """Raised when the frozen Strategy 2 inputs no longer match the contract."""


def canonical_spec_digest(spec: dict[str, Any]) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_frozen_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    digest = canonical_spec_digest(spec)
    if digest != EXPECTED_SPEC_DIGEST:
        raise FrozenStrategy2Error(
            f"Frozen Strategy 2 specification changed: expected {EXPECTED_SPEC_DIGEST}, found {digest}."
        )
    if pd.Timestamp(spec["frozen_at"], tz="UTC") != FREEZE_CUTOFF:
        raise FrozenStrategy2Error("Strategy 2 freeze boundary no longer matches Strategy 1.")
    return spec


def load_frozen_history(path: Path, spec: dict[str, Any]) -> dict[str, pd.DataFrame]:
    expected_hash = spec["frozen_history"]["sha256"]
    actual_hash = file_sha256(path)
    if actual_hash != expected_hash:
        raise FrozenStrategy2Error(
            f"Frozen Strategy 2 history changed: expected {expected_hash}, found {actual_hash}."
        )

    # The hash is checked before deserialisation because pickle may execute code.
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    tickers = list(spec["symbols"])
    expected_start = pd.Timestamp(spec["frozen_history"]["start"], tz="UTC")
    expected_end = pd.Timestamp(spec["frozen_history"]["end"], tz="UTC")
    history: dict[str, pd.DataFrame] = {}
    for field in HISTORY_FIELDS:
        frame = payload.get(field)
        if not isinstance(frame, pd.DataFrame):
            raise FrozenStrategy2Error(f"Frozen Strategy 2 history is missing {field}.")
        frame = frame.sort_index().reindex(columns=tickers)
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        else:
            frame.index = frame.index.tz_convert("UTC")
        if frame.index.min() != expected_start or frame.index.max() != expected_end:
            raise FrozenStrategy2Error(
                f"{field} must span {expected_start.date()} to {expected_end.date()}."
            )
        history[field] = frame
    return history


def build_combined_history(
    history: dict[str, pd.DataFrame],
    spec: dict[str, Any],
    *,
    end_date: pd.Timestamp,
    cache_dir: Path,
    offline: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    import ccxt

    symbol_map = dict(spec["symbols"])
    unavailable = set(spec["unavailable_forward_symbols"])
    forward_index = pd.date_range(FREEZE_CUTOFF + pd.Timedelta(days=1), end_date, freq="D")
    forward = {
        field: pd.DataFrame(np.nan, index=forward_index, columns=symbol_map, dtype=float)
        for field in HISTORY_FIELDS
    }
    exchange = None if offline else ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    cache_hashes: dict[str, str] = {}

    for ticker, symbol in symbol_map.items():
        if ticker in unavailable:
            continue
        candles = load_or_fetch_forward(
            symbol=symbol,
            end_date=end_date,
            cache_dir=cache_dir,
            exchange=exchange,
            offline=offline,
        ).reindex(forward_index)
        for ohlcv_field, history_field in OHLCV_TO_HISTORY.items():
            forward[history_field][ticker] = candles[ohlcv_field]
        forward["cg_notional"][ticker] = candles["close"] * candles["volume"]
        cache_path = cache_dir / f"{symbol.replace('/', '')}_1d_forward.parquet"
        cache_hashes[cache_path.name] = file_sha256(cache_path)

    combined = {
        field: pd.concat([history[field], forward[field]]).sort_index()
        for field in HISTORY_FIELDS
    }
    return combined, cache_hashes


def build_frozen_strategy2_run(
    combined: dict[str, pd.DataFrame],
    spec: dict[str, Any],
) -> dict[str, Any]:
    tickers = list(spec["symbols"])
    alt_tickers = [ticker for ticker in tickers if ticker != "BTC"]
    cleaned = {
        ticker: pd.DataFrame(
            {
                "open": combined["cg_open"][ticker],
                "high": combined["cg_high"][ticker],
                "low": combined["cg_low"][ticker],
                "close": combined["cg_close"][ticker],
                "volume": combined["cg_volume"][ticker],
            }
        )
        for ticker in tickers
    }
    features = build_s2_features(combined["cg_close"], combined["cg_notional"])
    half_spread = compute_half_spread_frac(combined["cg_close"], cleaned, tickers)
    parameters = spec["parameters"]
    params = make_s2_params(
        parameters["entry_pct"],
        parameters["lookback"],
        parameters["exit_z"],
        parameters["max_hold"],
        parameters["roc_window"],
        parameters["roc_pct"],
        parameters["crash_threshold"],
        gross_util=parameters["gross_util"],
    )
    return run_s2_strategy(
        features,
        params,
        alt_tickers,
        spec["execution"]["initial_capital"],
        half_spread,
        gross_util=parameters["gross_util"],
    )


def evaluate_strategy2_window(
    run: dict[str, Any],
    *,
    evaluation_index: pd.DatetimeIndex,
    anchor_date: pd.Timestamp,
    trading_days: int = 252,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if evaluation_index.empty or not (evaluation_index > anchor_date).all():
        raise ValueError("Evaluation index must contain only dates after its equity anchor.")
    expected_index = pd.date_range(evaluation_index.min(), evaluation_index.max(), freq="D")
    if not evaluation_index.equals(expected_index):
        raise ValueError("Strategy 2 evaluation index must be an uninterrupted daily window.")

    end_date = evaluation_index.max()
    anchor_equity = float(run["net_value"].loc[anchor_date])
    if not np.isfinite(anchor_equity) or anchor_equity <= 0:
        raise ValueError("Strategy 2 equity was non-positive at the evaluation boundary.")
    forward_equity = run["net_value"].reindex(evaluation_index)
    if forward_equity.isna().any():
        raise ValueError("Strategy 2 run is missing equity observations in the evaluation window.")
    anchored_equity = pd.concat(
        [pd.Series([anchor_equity], index=pd.DatetimeIndex([anchor_date])), forward_equity]
    )
    returns = anchored_equity.pct_change().dropna()
    gross = run["gross_pnl_daily"].reindex(evaluation_index).fillna(0.0)
    costs = run["cost_pnl_daily"].reindex(evaluation_index).fillna(0.0)
    net = gross - costs
    weights = run["weights"].reindex(evaluation_index).fillna(0.0)
    turnover = run["turnover"].reindex(evaluation_index).fillna(0.0)
    gross_exposure = run["theta"].reindex(evaluation_index).abs().sum(axis=1)
    active = weights.abs().sum(axis=1).gt(0)
    entries = [entry for entry in run["trade_entries"] if anchor_date < entry <= end_date]
    n_days = len(evaluation_index)
    total_return = float(forward_equity.iloc[-1] / anchor_equity - 1.0)
    annualised_return = (
        float((1.0 + total_return) ** (trading_days / n_days) - 1.0)
        if total_return > -1.0
        else np.nan
    )

    daily = pd.DataFrame(
        {
            "gross_pnl": gross,
            "transaction_cost": costs,
            "net_pnl": net,
            "portfolio_value": forward_equity,
            "cumulative_return": forward_equity / anchor_equity - 1.0,
            "turnover": turnover,
            "gross_exposure": gross_exposure,
            "active": active,
        }
    )
    daily.index.name = "date"
    metrics = {
        "start": str(evaluation_index.min().date()),
        "end": str(end_date.date()),
        "n_days": n_days,
        "anchor_date": str(anchor_date.date()),
        "anchor_equity": anchor_equity,
        "final_equity": float(forward_equity.iloc[-1]),
        "total_return": total_return,
        "annualised_return": annualised_return,
        "sharpe": float(sharpe_ratio(returns, trading_days)),
        "sortino": float(sortino_ratio(returns, trading_days=trading_days)),
        "max_drawdown": float(max_drawdown(anchored_equity)),
        "total_gross_pnl": float(gross.sum()),
        "total_cost": float(costs.sum()),
        "total_net_pnl": float(net.sum()),
        "total_turnover": float(turnover.sum()),
        "mean_daily_turnover": float(turnover.mean()),
        "mean_gross_exposure": float(gross_exposure.mean()),
        "mean_active_gross_exposure": (
            float(gross_exposure.loc[active].mean()) if active.any() else 0.0
        ),
        "trade_entries": len(entries),
        "entry_dates": [str(entry.date()) for entry in entries],
        "active_days": int(daily["active"].sum()),
        "active_days_pct": float(daily["active"].mean()),
    }
    return metrics, daily


def evaluate_strategy2_forward(
    combined: dict[str, pd.DataFrame],
    spec: dict[str, Any],
    *,
    end_date: pd.Timestamp,
) -> tuple[dict[str, Any], pd.DataFrame]:
    run = build_frozen_strategy2_run(combined, spec)
    forward_index = pd.date_range(FREEZE_CUTOFF + pd.Timedelta(days=1), end_date, freq="D")
    return evaluate_strategy2_window(
        run,
        evaluation_index=forward_index,
        anchor_date=FREEZE_CUTOFF,
        trading_days=int(spec["execution"]["trading_days_per_year"]),
    )


def validate_output_target(output_dir: Path) -> None:
    if "snapshots" not in output_dir.parts:
        return
    protected = [
        output_dir / "strategy2_summary.json",
        output_dir / "strategy2_daily.csv",
        output_dir / "strategy2_post_selection_daily.csv",
    ]
    if any(path.exists() for path in protected):
        raise FileExistsError(f"Strategy 2 snapshot already exists at {output_dir}.")


def run_forward_validation(
    *,
    history_path: Path,
    cache_dir: Path,
    output_dir: Path,
    end_date: pd.Timestamp,
    offline: bool,
) -> dict[str, Any]:
    validate_output_target(output_dir)
    spec = load_frozen_spec()
    history = load_frozen_history(history_path, spec)
    combined, cache_hashes = build_combined_history(
        history,
        spec,
        end_date=end_date,
        cache_dir=cache_dir,
        offline=offline,
    )
    run = build_frozen_strategy2_run(combined, spec)
    trading_days = int(spec["execution"]["trading_days_per_year"])
    forward_index = pd.date_range(FREEZE_CUTOFF + pd.Timedelta(days=1), end_date, freq="D")
    metrics, daily = evaluate_strategy2_window(
        run,
        evaluation_index=forward_index,
        anchor_date=FREEZE_CUTOFF,
        trading_days=trading_days,
    )
    post_selection_index = pd.date_range(ORIGINAL_HOLDOUT_START, end_date, freq="D")
    post_selection_metrics, post_selection_daily = evaluate_strategy2_window(
        run,
        evaluation_index=post_selection_index,
        anchor_date=POST_SELECTION_ANCHOR,
        trading_days=trading_days,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = output_dir / "strategy2_daily.csv"
    post_selection_daily_path = output_dir / "strategy2_post_selection_daily.csv"
    summary_path = output_dir / "strategy2_summary.json"
    daily.to_csv(daily_path, float_format="%.10g")
    post_selection_daily.to_csv(post_selection_daily_path, float_format="%.10g")
    summary = {
        "status": "frozen_strategy2_forward_validation",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "data_source": "Binance daily OHLCV via ccxt",
        "freeze_cutoff": str(FREEZE_CUTOFF.date()),
        "requested_end_date": str(end_date.date()),
        "frozen_spec_digest": EXPECTED_SPEC_DIGEST,
        "frozen_spec": spec,
        "forward_cache_sha256": cache_hashes,
        "metrics": metrics,
        "post_selection_metrics": post_selection_metrics,
        "artifacts": {
            "daily": daily_path.name,
            "post_selection_daily": post_selection_daily_path.name,
            "summary": summary_path.name,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-date", help="Last completed UTC candle (YYYY-MM-DD). Defaults to yesterday.")
    parser.add_argument("--history-path", type=Path, default=Path("s2_binance_cache.pkl"))
    parser.add_argument("--cache-dir", type=Path, default=Path("forward_validation/cache/strategy2"))
    parser.add_argument("--output-dir", type=Path, default=Path("forward_validation/results/strategy2"))
    parser.add_argument("--offline", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_forward_validation(
        history_path=args.history_path,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        end_date=resolve_end_date(args.end_date),
        offline=args.offline,
    )
    metrics = summary["metrics"]
    print("Frozen Strategy 2 forward validation")
    print(f"  Window: {metrics['start']} to {metrics['end']} ({metrics['n_days']} days)")
    print(f"  Net return: {metrics['total_return']:.2%}")
    print(f"  Sharpe: {metrics['sharpe']:.3f}")
    print(f"  Max drawdown: {metrics['max_drawdown']:.2%}")
    print(f"  Net PnL: {metrics['total_net_pnl']:.2f} USDT")
    post_selection = summary["post_selection_metrics"]
    print("Continuous post-selection evidence")
    print(
        f"  Window: {post_selection['start']} to {post_selection['end']} "
        f"({post_selection['n_days']} days)"
    )
    print(f"  Net return: {post_selection['total_return']:.2%}")
    print(f"  Sharpe: {post_selection['sharpe']:.3f}")
    print(f"  Max drawdown: {post_selection['max_drawdown']:.2%}")
    print(f"  Net PnL: {post_selection['total_net_pnl']:.2f} USDT")


if __name__ == "__main__":
    main()
