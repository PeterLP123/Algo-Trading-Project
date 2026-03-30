"""
config.py — Centralized configuration for the COMP0051 trading strategy.

All constants and parameters extracted from strategy_final_v2.ipynb.
The run_pipeline.py orchestrator reads from here and passes values as
explicit arguments to each module function.

Cell-to-config mapping:
  Cell 1  → Plot export/theme settings
  Cell 4  → SYMBOLS, TIMEFRAME, SINCE, UNTIL, DATA_DIR
  Cell 8  → VALUE_COLUMNS, PANDAS_FREQ
  Cell 11 → COLORS, COLOR_SEQUENCE, PLOTLY_TEMPLATE, EXPORT_FORMATS
  Cell 15 → HOLDOUT_FRAC, INITIAL_TRAIN_BARS, VAL_BARS, STEP_BARS, WF_MODE, TRAIN_BARS
  Cell 18 → MA_WINDOW, VOL_WINDOW, DEAD_ZONE, SIGNAL_CLIP
  Cell 21 → GROSS_CAP
  Cell 23 → BACKTEST_USE_EXCESS, V0
  Cell 25 → TRADING_DAYS
  Cell 27 → STABILITY_PENALTY, SIGNAL_CLIP_WF, MA_GRID, VOL_GRID, THRESH_GRID, REBAL_GRID
"""

from pathlib import Path

import plotly.graph_objects as go

# ── Data download (Cell 4) ────────────────────────────────────────────────────
SYMBOLS    = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT"]
TIMEFRAME  = "1d"
SINCE      = "2020-01-01"
UNTIL      = "2026-03-20"
DATA_DIR   = Path("data_final")
OUTPUT_DIR = Path("output")

# ── Data cleaning (Cell 8) ───────────────────────────────────────────────────
VALUE_COLUMNS = ["open", "high", "low", "close", "volume"]
PANDAS_FREQ   = str(TIMEFRAME).upper()  # "1D"

# ── Signal parameters (Cell 18, initial values; tuned in walk-forward) ───────
MA_WINDOW  = 100
VOL_WINDOW = 40
DEAD_ZONE  = 0.25   # threshold on z for long/short/flat activation
SIGNAL_CLIP = 5.0

# ── Position sizing (Cell 21) ────────────────────────────────────────────────
GROSS_CAP = 100_000.0  # fixed gross notional budget (sum |theta_i| = GROSS_CAP when active)

# ── Backtest (Cell 23) ───────────────────────────────────────────────────────
BACKTEST_USE_EXCESS = False   # True → use excess_return instead of simple returns
V0 = 10_000.0                 # initial capital ($10,000 USDT as per brief; 10x leverage)

# ── Walk-forward split (Cell 15) ─────────────────────────────────────────────
HOLDOUT_FRAC        = 0.225   # ~22.5% final test period; reserved and untouched
INITIAL_TRAIN_BARS  = 756     # ~3 years business days before first validation fold
VAL_BARS            = 126     # ~6 months per validation block
STEP_BARS           = 126     # non-overlapping validation windows
WF_MODE             = "expanding"  # "expanding" | "rolling"
TRAIN_BARS          = 756     # rolling mode only: fixed training window length

# ── Performance metrics (Cell 25) ────────────────────────────────────────────
TRADING_DAYS = 252

# ── Walk-forward grid search (Cell 27) ───────────────────────────────────────
STABILITY_PENALTY = 0.1
SIGNAL_CLIP_WF    = 5.0
MA_GRID           = [50, 75, 100]
VOL_GRID          = [20, 30]
THRESH_GRID       = [0.25, 0.5]
REBAL_GRID        = [1, 3]

# ── Plot theme (shared by all visualization modules) ─────────────────────────
COLORS = {
    "long": "#00C853",
    "short": "#FF1744",
    "net": "#2962FF",
    "gross": "#6200EA",
    "cost": "#FF6D00",
    "benchmark": "#78909C",
    "close": "#263238",
    "ma": "#00ACC1",
    "zero": "#9E9E9E",
    "turnover": "#455A64",
}

COLOR_SEQUENCE = [
    COLORS["net"],
    COLORS["gross"],
    COLORS["ma"],
    COLORS["long"],
    COLORS["cost"],
    COLORS["benchmark"],
]

EXPORT_FORMATS = ["html", "png", "pdf"]

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        colorway=COLOR_SEQUENCE,
        font={"family": "Open Sans, Arial, sans-serif", "size": 13, "color": COLORS["close"]},
        paper_bgcolor="white",
        plot_bgcolor="white",
        title={"x": 0.02, "xanchor": "left", "font": {"size": 20}},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
            "bgcolor": "rgba(255,255,255,0.85)",
        },
        margin={"l": 72, "r": 36, "t": 72, "b": 56},
        hovermode="x unified",
        xaxis={
            "showline": False,
            "showgrid": False,
            "zeroline": False,
            "ticks": "outside",
            "tickcolor": "#B0BEC5",
        },
        yaxis={
            "showline": False,
            "showgrid": True,
            "gridcolor": "#CFD8DC",
            "griddash": "dot",
            "gridwidth": 1,
            "zeroline": False,
            "ticks": "outside",
            "tickcolor": "#B0BEC5",
        },
    )
)
