"""
data_download.py — Binance OHLCV download via CCXT (Cell 4).

Function:
  download_ohlcv(symbols, timeframe, since, until, data_dir)
    Downloads 1-day bars for each symbol from Binance, paginating in
    batches of 1000, and saves each as a Parquet file in data_dir.
"""

import time
from pathlib import Path

import ccxt
import pandas as pd


def download_ohlcv(symbols, timeframe, since, until, data_dir):
    """Download OHLCV bars from Binance for each symbol and save to Parquet.

    Args:
        symbols   : list of CCXT-style symbols, e.g. ["BTC/USDT", "ETH/USDT"]
        timeframe : CCXT timeframe string, e.g. "1d"
        since     : start date string "YYYY-MM-DD"
        until     : end date string "YYYY-MM-DD"
        data_dir  : Path — directory to write parquet files into
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    exchange = ccxt.binance()
    since_ms = exchange.parse8601(f"{since}T00:00:00Z")
    until_ms = exchange.parse8601(f"{until}T23:59:59Z")

    for symbol in symbols:
        print(f"Downloading {symbol}...", end="", flush=True)

        all_bars = []
        since_cursor = since_ms

        while True:
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_cursor, limit=1000)
            if not bars:
                break

            bars = [bar for bar in bars if bar[0] <= until_ms]
            if not bars:
                break

            all_bars.extend(bars)
            since_cursor = bars[-1][0] + 1
            print(".", end="", flush=True)
            time.sleep(0.3)

            if bars[-1][0] >= until_ms:
                break

        df = pd.DataFrame(
            all_bars,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()

        fname = data_dir / f"{symbol.replace('/', '')}_{timeframe}.parquet"
        df.to_parquet(fname, engine="pyarrow")
        print(f" {len(df):,} bars -> {fname}")

    print("Done.")
