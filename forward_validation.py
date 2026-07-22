#!/usr/bin/env python3
"""Run Strategy 1 on post-freeze data without parameter re-selection.

The strategy specification was frozen on 2026-03-20. This runner reads the
canonical pre-freeze OHLCV caches, fetches only later completed Binance daily
candles, applies the frozen strategy unchanged, and writes auditable forward
validation artifacts. It deliberately contains no optimisation path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy_helpers import ensure_utc_index, read_parquet_safe, write_parquet_safe
from wf_trend_pipeline import (
    build_signal_and_theta,
    compute_half_spread_frac,
    max_drawdown,
    run_net_backtest,
    sharpe_ratio,
    sortino_ratio,
)


PROJECT_ROOT = Path(__file__).resolve().parent
SPEC_PATH = PROJECT_ROOT / "forward_validation" / "frozen_strategy1.json"
EXPECTED_SPEC_DIGEST = "769a3b2d040c0977f5dd51fc143dd24dcaa2f1202707554f79a137d3d3650b2a"
FREEZE_CUTOFF = pd.Timestamp("2026-03-20", tz="UTC")
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class FrozenSpecificationError(RuntimeError):
    """Raised when the frozen strategy or its data boundary has changed."""


def canonical_spec_digest(spec: dict[str, Any]) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_frozen_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    digest = canonical_spec_digest(spec)
    if digest != EXPECTED_SPEC_DIGEST:
        raise FrozenSpecificationError(
            "Frozen Strategy 1 specification changed. "
            f"Expected {EXPECTED_SPEC_DIGEST}, found {digest}."
        )
    frozen_at = pd.Timestamp(spec["frozen_at"], tz="UTC")
    if frozen_at != FREEZE_CUTOFF:
        raise FrozenSpecificationError(
            f"Expected freeze boundary {FREEZE_CUTOFF.date()}, found {frozen_at.date()}."
        )
    return spec


def parse_utc_date(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.normalize()


def latest_completed_utc_day(now: pd.Timestamp | None = None) -> pd.Timestamp:
    current = pd.Timestamp.now(tz="UTC") if now is None else now
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    else:
        current = current.tz_convert("UTC")
    return current.normalize() - pd.Timedelta(days=1)


def resolve_end_date(value: str | None, now: pd.Timestamp | None = None) -> pd.Timestamp:
    latest_completed = latest_completed_utc_day(now)
    end_date = latest_completed if value is None else parse_utc_date(value)
    if end_date <= FREEZE_CUTOFF:
        raise ValueError(f"End date must be after {FREEZE_CUTOFF.date()}.")
    if end_date > latest_completed:
        raise ValueError(
            f"End date {end_date.date()} is not a completed UTC daily candle; "
            f"latest allowed is {latest_completed.date()}."
        )
    return end_date


def symbol_key(symbol: str) -> str:
    return symbol.replace("/", "")


def _normalise_ohlcv(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    normalized = ensure_utc_index(frame).sort_index().copy()
    normalized = normalized[~normalized.index.duplicated(keep="first")]
    missing = set(OHLCV_COLUMNS).difference(normalized.columns)
    if missing:
        raise ValueError(f"{label} is missing OHLCV columns: {sorted(missing)}")
    normalized = normalized[OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce")
    invalid = (
        normalized.isna().any(axis=1)
        | (normalized[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (normalized["volume"] < 0)
        | (normalized["high"] < normalized[["open", "close"]].max(axis=1))
        | (normalized["low"] > normalized[["open", "close"]].min(axis=1))
    )
    if invalid.any():
        bad_dates = ", ".join(str(ts.date()) for ts in normalized.index[invalid][:5])
        raise ValueError(f"{label} contains invalid candles at: {bad_dates}")
    return normalized


def validate_frozen_history(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    history = _normalise_ohlcv(frame, label=f"{symbol} frozen history")
    if history.empty:
        raise FrozenSpecificationError(f"{symbol} frozen history is empty.")
    if history.index.max() != FREEZE_CUTOFF:
        raise FrozenSpecificationError(
            f"{symbol} frozen history must end exactly on {FREEZE_CUTOFF.date()}, "
            f"found {history.index.max().date()}."
        )
    expected = pd.date_range(history.index.min(), FREEZE_CUTOFF, freq="D", tz="UTC")
    gaps = expected.difference(history.index)
    if len(gaps):
        raise FrozenSpecificationError(
            f"{symbol} frozen history has {len(gaps)} missing daily candles."
        )
    return history


def validate_forward_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    forward = _normalise_ohlcv(frame, label=f"{symbol} forward data")
    forward = forward.loc[(forward.index > FREEZE_CUTOFF) & (forward.index <= end_date)]
    if forward.empty:
        raise ValueError(f"{symbol} has no observations after the freeze boundary.")
    expected = pd.date_range(FREEZE_CUTOFF + pd.Timedelta(days=1), end_date, freq="D", tz="UTC")
    gaps = expected.difference(forward.index)
    if len(gaps):
        preview = ", ".join(str(ts.date()) for ts in gaps[:5])
        raise ValueError(f"{symbol} forward data is missing {len(gaps)} candles: {preview}")
    return forward


def combine_frozen_and_forward(
    history: pd.DataFrame,
    forward: pd.DataFrame,
    *,
    symbol: str,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    frozen = validate_frozen_history(history, symbol=symbol)
    future = validate_forward_frame(forward, symbol=symbol, end_date=end_date)
    if not frozen.index.intersection(future.index).empty:
        raise FrozenSpecificationError(f"{symbol} forward data overlaps the frozen history.")
    combined = pd.concat([frozen, future]).sort_index()
    if combined.index.max() != end_date:
        raise ValueError(f"{symbol} combined data does not reach {end_date.date()}.")
    return combined


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_history_hashes(actual: dict[str, str], expected: dict[str, str]) -> None:
    if set(actual) != set(expected):
        missing = sorted(set(expected).difference(actual))
        unexpected = sorted(set(actual).difference(expected))
        raise FrozenSpecificationError(
            f"Frozen history file set changed; missing={missing}, unexpected={unexpected}."
        )
    mismatches = [name for name in expected if actual[name] != expected[name]]
    if mismatches:
        raise FrozenSpecificationError(
            "Frozen history content changed for: " + ", ".join(sorted(mismatches))
        )


def load_frozen_histories(
    data_dir: Path,
    symbols: list[str],
    expected_hashes: dict[str, str],
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    histories: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    for symbol in symbols:
        path = data_dir / f"{symbol_key(symbol)}_1d_cleaned.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing canonical frozen history {path}. "
                "The forward runner never recreates or modifies pre-freeze data."
            )
        histories[symbol] = validate_frozen_history(read_parquet_safe(path), symbol=symbol)
        hashes[path.name] = file_sha256(path)
    validate_history_hashes(hashes, expected_hashes)
    return histories, hashes


def _rows_to_frame(rows: list[list[float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLUMNS])
    if frame.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS, index=pd.DatetimeIndex([], tz="UTC"))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    return frame.set_index("timestamp")[OHLCV_COLUMNS].sort_index()


def fetch_forward_candles(
    exchange: Any,
    symbol: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    if start_date > end_date:
        return pd.DataFrame(columns=OHLCV_COLUMNS, index=pd.DatetimeIndex([], tz="UTC"))
    since_ms = int(start_date.timestamp() * 1000)
    end_ms = int((end_date + pd.Timedelta(days=1)).timestamp() * 1000) - 1
    rows: list[list[float]] = []

    while since_ms <= end_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe="1d", since=since_ms, limit=1000)
        if not batch:
            break
        for row in batch:
            if since_ms <= int(row[0]) <= end_ms:
                rows.append(row[:6])
        last_ms = int(batch[-1][0])
        if last_ms >= end_ms or len(batch) < 1000:
            break
        next_ms = last_ms + 86_400_000
        if next_ms <= since_ms:
            raise RuntimeError(f"Binance pagination stalled for {symbol}.")
        since_ms = next_ms

    return _rows_to_frame(rows)


def load_or_fetch_forward(
    *,
    symbol: str,
    end_date: pd.Timestamp,
    cache_dir: Path,
    exchange: Any | None,
    offline: bool,
) -> pd.DataFrame:
    cache_path = cache_dir / f"{symbol_key(symbol)}_1d_forward.parquet"
    cached = (
        _normalise_ohlcv(read_parquet_safe(cache_path), label=f"{symbol} forward cache")
        if cache_path.exists()
        else pd.DataFrame(columns=OHLCV_COLUMNS, index=pd.DatetimeIndex([], tz="UTC"))
    )
    cached = cached.loc[cached.index > FREEZE_CUTOFF]
    last_cached = cached.index.max() if not cached.empty else FREEZE_CUTOFF

    if last_cached < end_date:
        if offline:
            raise FileNotFoundError(
                f"Offline cache for {symbol} ends at {last_cached.date()}, before {end_date.date()}."
            )
        if exchange is None:
            raise RuntimeError("A live exchange client is required to refresh forward data.")
        fetched = fetch_forward_candles(
            exchange,
            symbol,
            last_cached + pd.Timedelta(days=1),
            end_date,
        )
        if cached.empty:
            cached = fetched.copy()
        elif not fetched.empty:
            cached = pd.concat([cached, fetched]).sort_index()
        cached = cached[~cached.index.duplicated(keep="last")]
        write_parquet_safe(cached, cache_path)

    return validate_forward_frame(cached, symbol=symbol, end_date=end_date)


def build_asset_panel(frames: dict[str, pd.DataFrame], symbols: list[str]) -> pd.DataFrame:
    close = pd.concat(
        [frames[symbol]["close"].rename(symbol) for symbol in symbols],
        axis=1,
        join="inner",
    ).dropna(how="any")
    close.columns = pd.MultiIndex.from_product([["close"], symbols], names=["field", "asset"])
    return close


def _finite_or_none(value: Any) -> float | int | str | None:
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def evaluate_forward_window(
    run_state: dict[str, Any],
    asset_panel: pd.DataFrame,
    *,
    forward_index: pd.DatetimeIndex,
    cutoff: pd.Timestamp = FREEZE_CUTOFF,
    trading_days: int = 252,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if forward_index.empty or not (forward_index > cutoff).all():
        raise ValueError("Forward evaluation index must contain only dates after the freeze cutoff.")

    equity = run_state["net_portfolio_value"]
    anchor_candidates = equity.index[equity.index <= cutoff]
    if anchor_candidates.empty:
        raise ValueError("No pre-forward equity anchor is available.")
    anchor_date = anchor_candidates.max()
    anchor_equity = float(equity.loc[anchor_date])
    if not np.isfinite(anchor_equity) or anchor_equity <= 0:
        raise ValueError("Strategy equity was non-positive at the forward-validation boundary.")

    forward_equity = equity.reindex(forward_index).dropna()
    anchored_equity = pd.concat(
        [pd.Series([anchor_equity], index=pd.DatetimeIndex([anchor_date])), forward_equity]
    )
    forward_returns = anchored_equity.pct_change().dropna()
    gross = run_state["gross_pnl"].reindex(forward_index).fillna(0.0)
    costs = run_state["cost_t"].reindex(forward_index).fillna(0.0)
    net = run_state["net_pnl"].reindex(forward_index).fillna(0.0)
    turnover = run_state["turnover"].reindex(forward_index).fillna(0.0)
    exposure = run_state["theta_exec"].reindex(forward_index).fillna(0.0)
    delta_exposure = run_state["delta_theta_exec"].reindex(forward_index).fillna(0.0)
    asset_gross = (
        run_state["theta_exec"].shift(1).fillna(0.0) * run_state["r"]
    ).reindex(forward_index).fillna(0.0)

    close = asset_panel["close"]
    asset_returns = close.pct_change().reindex(forward_index).fillna(0.0)
    btc_return = float((1.0 + asset_returns["BTC/USDT"]).prod() - 1.0)
    equal_weight_return = float((1.0 + asset_returns.mean(axis=1)).prod() - 1.0)
    btc_cumulative = (1.0 + asset_returns["BTC/USDT"]).cumprod() - 1.0
    equal_weight_cumulative = (1.0 + asset_returns.mean(axis=1)).cumprod() - 1.0
    total_abs_gross = float(gross.abs().sum())
    n_days = len(forward_equity)
    total_return = float(forward_equity.iloc[-1] / anchor_equity - 1.0)
    annualised_return = (
        float((1.0 + total_return) ** (trading_days / n_days) - 1.0)
        if n_days > 0 and total_return > -1.0
        else np.nan
    )

    metrics = {
        "start": str(forward_index.min().date()),
        "end": str(forward_index.max().date()),
        "n_days": int(n_days),
        "anchor_date": str(anchor_date.date()),
        "anchor_equity": anchor_equity,
        "final_equity": float(forward_equity.iloc[-1]),
        "total_return": total_return,
        "annualised_return": annualised_return,
        "sharpe": sharpe_ratio(forward_returns, trading_days),
        "sortino": sortino_ratio(forward_returns, trading_days=trading_days),
        "max_drawdown": max_drawdown(anchored_equity),
        "total_gross_pnl": float(gross.sum()),
        "total_cost": float(costs.sum()),
        "total_net_pnl": float(net.sum()),
        "mean_daily_turnover": float(turnover.mean()),
        "cost_to_absolute_gross_ratio": (
            float(costs.sum() / total_abs_gross) if total_abs_gross > 0 else np.nan
        ),
        "active_days_pct": float(exposure.abs().sum(axis=1).gt(0).mean()),
        "btc_buy_and_hold_return": btc_return,
        "equal_weight_buy_and_hold_return": equal_weight_return,
        "asset_diagnostics": {
            symbol_key(symbol): {
                "total_gross_pnl": _finite_or_none(asset_gross[symbol].sum()),
                "mean_absolute_exposure": _finite_or_none(exposure[symbol].abs().mean()),
                "total_turnover": _finite_or_none(delta_exposure[symbol].abs().sum()),
            }
            for symbol in exposure.columns
        },
    }

    daily = pd.DataFrame(
        {
            "gross_pnl": gross,
            "transaction_cost": costs,
            "net_pnl": net,
            "portfolio_value": forward_equity,
            "cumulative_return": forward_equity / anchor_equity - 1.0,
            "turnover": turnover,
            "gross_exposure": exposure.abs().sum(axis=1),
            "btc_buy_and_hold_cumulative_return": btc_cumulative,
            "equal_weight_buy_and_hold_cumulative_return": equal_weight_cumulative,
        }
    )
    for symbol in exposure.columns:
        daily[f"exposure_{symbol_key(symbol)}"] = exposure[symbol]
        daily[f"gross_pnl_{symbol_key(symbol)}"] = asset_gross[symbol]
        daily[f"turnover_{symbol_key(symbol)}"] = delta_exposure[symbol].abs()
    daily.index.name = "date"
    return {
        key: value if isinstance(value, dict) else _finite_or_none(value)
        for key, value in metrics.items()
    }, daily


def validate_output_target(output_dir: Path) -> None:
    """Protect dated evidence snapshots from accidental replacement."""
    is_snapshot = "snapshots" in output_dir.parts
    existing = [
        output_dir / "summary.json",
        output_dir / "daily.csv",
        output_dir / "cumulative_returns.png",
    ]
    if is_snapshot and any(path.exists() for path in existing):
        raise FileExistsError(
            f"Dated snapshot {output_dir} already exists and is immutable. "
            "Choose a new endpoint directory."
        )


def write_performance_plot(daily: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(figsize=(9.5, 4.4), constrained_layout=True)
    ax.set_facecolor("#f8fafc")
    ax.plot(
        daily.index,
        daily["cumulative_return"],
        color="#1d4ed8",
        linewidth=2.1,
        label="Frozen Strategy 1 (net)",
    )
    ax.plot(
        daily.index,
        daily["btc_buy_and_hold_cumulative_return"],
        color="#d97706",
        linewidth=1.5,
        label="BTC buy-and-hold",
    )
    ax.plot(
        daily.index,
        daily["equal_weight_buy_and_hold_cumulative_return"],
        color="#64748b",
        linewidth=1.4,
        label="Equal-weight basket",
    )
    ax.axhline(0.0, color="#94a3b8", linestyle="--", linewidth=0.9)
    ax.set_title("Frozen Strategy 1 forward validation")
    ax.set_xlabel("Completed UTC daily candle")
    ax.set_ylabel("Cumulative return from 20 March 2026 close")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, axis="y", color="#e2e8f0", linewidth=0.8)
    ax.grid(False, axis="x")
    ax.legend(frameon=False, loc="best")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_forward_validation(
    *,
    data_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    end_date: pd.Timestamp,
    offline: bool = False,
) -> dict[str, Any]:
    validate_output_target(output_dir)
    spec = load_frozen_spec()
    symbols = list(spec["symbols"])
    histories, history_hashes = load_frozen_histories(
        data_dir,
        symbols,
        expected_hashes=spec["frozen_history_sha256"],
    )

    exchange = None
    if not offline:
        import ccxt

        exchange = ccxt.binance({"enableRateLimit": True})

    combined_frames: dict[str, pd.DataFrame] = {}
    forward_hashes: dict[str, str] = {}
    for symbol in symbols:
        forward = load_or_fetch_forward(
            symbol=symbol,
            end_date=end_date,
            cache_dir=cache_dir,
            exchange=exchange,
            offline=offline,
        )
        combined_frames[symbol] = combine_frozen_and_forward(
            histories[symbol],
            forward,
            symbol=symbol,
            end_date=end_date,
        )
        cache_path = cache_dir / f"{symbol_key(symbol)}_1d_forward.parquet"
        forward_hashes[cache_path.name] = file_sha256(cache_path)

    asset_panel = build_asset_panel(combined_frames, symbols)
    common_end = asset_panel.index.max()
    if common_end != end_date:
        raise ValueError(f"Common asset panel ends at {common_end.date()}, not {end_date.date()}.")

    parameters = spec["parameters"]
    theta = build_signal_and_theta(
        asset_panel,
        symbols,
        ma_window=int(parameters["ma_window"]),
        vol_window=int(parameters["vol_window"]),
        dead_zone=float(parameters["dead_zone"]),
        signal_clip=float(parameters["signal_clip"]),
        gross_cap=float(parameters["gross_cap"]),
        rebalance_every=int(parameters["rebalance_every"]),
        use_mvo=bool(parameters["use_mvo"]),
        cov_window=int(parameters["cov_window"]),
        gamma=float(parameters["gamma"]),
    )
    half_spread = compute_half_spread_frac(asset_panel, combined_frames, symbols)
    run_state = run_net_backtest(
        asset_panel,
        theta,
        symbols,
        half_spread,
        rebalance_every=int(parameters["rebalance_every"]),
        backtest_use_excess=False,
        v0=float(spec["execution"]["initial_capital"]),
    )
    forward_index = asset_panel.index[
        (asset_panel.index > FREEZE_CUTOFF) & (asset_panel.index <= end_date)
    ]
    metrics, daily = evaluate_forward_window(
        run_state,
        asset_panel,
        forward_index=forward_index,
        trading_days=int(spec["execution"]["trading_days_per_year"]),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = output_dir / "daily.csv"
    summary_path = output_dir / "summary.json"
    figure_path = output_dir / "cumulative_returns.png"
    daily.to_csv(daily_path, float_format="%.10g")
    write_performance_plot(daily, figure_path)
    summary = {
        "status": "frozen_forward_validation",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "data_source": "Binance daily OHLCV via ccxt",
        "freeze_cutoff": str(FREEZE_CUTOFF.date()),
        "requested_end_date": str(end_date.date()),
        "frozen_spec_digest": EXPECTED_SPEC_DIGEST,
        "frozen_spec": spec,
        "frozen_history_sha256": history_hashes,
        "forward_cache_sha256": forward_hashes,
        "metrics": metrics,
        "artifacts": {
            "daily": daily_path.name,
            "summary": summary_path.name,
            "figure": figure_path.name,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen Strategy 1 specification on post-2026-03-20 data."
    )
    parser.add_argument("--end-date", help="Last completed UTC candle (YYYY-MM-DD). Defaults to yesterday.")
    parser.add_argument("--data-dir", type=Path, default=Path("data_final"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("forward_validation/cache"),
        help="Ignored local cache for post-freeze Binance candles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("forward_validation/results"),
        help="Output directory for summary.json and daily.csv.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the forward cache only and fail if it does not cover the requested end date.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    end_date = resolve_end_date(args.end_date)
    summary = run_forward_validation(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        end_date=end_date,
        offline=args.offline,
    )
    metrics = summary["metrics"]
    print("Frozen Strategy 1 forward validation")
    print(f"  Window: {metrics['start']} to {metrics['end']} ({metrics['n_days']} days)")
    print(f"  Net return: {metrics['total_return']:.2%}")
    print(f"  Sharpe: {metrics['sharpe']:.3f}")
    print(f"  Max drawdown: {metrics['max_drawdown']:.2%}")
    print(f"  Net PnL: {metrics['total_net_pnl']:.2f} USDT")


if __name__ == "__main__":
    main()
