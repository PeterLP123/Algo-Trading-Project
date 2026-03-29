"""
config.py — Centralized configuration for the COMP0051 trading strategy.

All constants and parameters extracted from strategy_final_v2.ipynb.
The run_pipeline.py orchestrator reads from here and passes values as
explicit arguments to each module function.

Cell-to-config mapping:
  Cell 1  → MPL_RC_PARAMS (plt.rcParams block from Cell 11, applied at startup)
  Cell 4  → SYMBOLS, TIMEFRAME, SINCE, UNTIL, DATA_DIR
  Cell 8  → VALUE_COLUMNS, PANDAS_FREQ
  Cell 11 → MPL_RC_PARAMS
  Cell 15 → HOLDOUT_FRAC, INITIAL_TRAIN_BARS, VAL_BARS, STEP_BARS, WF_MODE, TRAIN_BARS
  Cell 18 → MA_WINDOW, VOL_WINDOW, DEAD_ZONE, SIGNAL_CLIP
  Cell 21 → GROSS_CAP
  Cell 23 → BACKTEST_USE_EXCESS, V0
  Cell 25 → TRADING_DAYS
  Cell 27 → STABILITY_PENALTY, SIGNAL_CLIP_WF, MA_GRID, VOL_GRID, THRESH_GRID, REBAL_GRID
"""

from pathlib import Path

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
GROSS_CAP = 50_000.0  # fixed gross notional budget (sum |theta_i| = GROSS_CAP when active)

# ── Backtest (Cell 23) ───────────────────────────────────────────────────────
BACKTEST_USE_EXCESS = False   # True → use excess_return instead of simple returns
V0 = GROSS_CAP               # initial capital (aligned with gross notional budget)

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

# ── Matplotlib style (Cell 11) ───────────────────────────────────────────────
MPL_RC_PARAMS = {
    "figure.dpi":             500,
    "savefig.dpi":            500,
    "font.family":            "serif",
    "font.size":              10,
    "axes.titlesize":         10,
    "axes.labelsize":         10,
    "xtick.labelsize":        9,
    "ytick.labelsize":        9,
    "legend.fontsize":        9,
    "legend.title_fontsize":  9,
    "axes.spines.top":        False,
    "axes.spines.right":      False,
}
