"""
cointegration.py — Cointegration testing, hedge-ratio estimation, spread
construction, z-score signals, and walk-forward pair selection for Strategy 2.

Functions:
  enumerate_pairs(symbols)           — all C(n,2) ordered asset pairs
  test_cointegration(log_y, log_x)   — Engle-Granger two-step test via statsmodels
  rolling_hedge_ratio(log_y, log_x)  — rolling OLS β from cov/var of log prices
  compute_spread(log_y, log_x, beta) — S_t = log(P_y) − β_t · log(P_x)
  compute_zscore(spread, window)     — rolling z-score of the spread
  generate_pair_signal(zscore, ...)   — state-machine trading signal {−1, 0, +1}
  select_pairs_and_build_theta(...)  — walk-forward orchestrator → theta DataFrame
"""

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


# ── Pair enumeration ─────────────────────────────────────────────────────────

def enumerate_pairs(symbols: list[str]) -> list[tuple[str, str]]:
    """Return all C(n,2) pairs as (x, y) tuples sorted alphabetically.

    Convention: x is the independent variable (regressor), y is dependent.
    """
    return sorted(combinations(sorted(symbols), 2))


# ── Cointegration test ───────────────────────────────────────────────────────

def test_cointegration(
    log_y: pd.Series,
    log_x: pd.Series,
    max_lag: int = 1,
) -> tuple[float, float, np.ndarray]:
    """Engle-Granger cointegration test.

    Wraps statsmodels.tsa.stattools.coint with safe handling of short series.

    Args:
        log_y   : log prices of the dependent asset
        log_x   : log prices of the independent asset
        max_lag : maximum lag for the ADF inner test

    Returns:
        (t_stat, p_value, critical_values)
    """
    y = log_y.dropna()
    x = log_x.dropna()
    common = y.index.intersection(x.index)
    if len(common) < 30:
        return (np.nan, 1.0, np.array([np.nan, np.nan, np.nan]))
    t_stat, p_value, crit = coint(y.loc[common], x.loc[common], maxlag=max_lag)
    return (float(t_stat), float(p_value), crit)


# ── Rolling hedge ratio ─────────────────────────────────────────────────────

def rolling_hedge_ratio(
    log_y: pd.Series,
    log_x: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Rolling OLS hedge ratio β_t = cov(log_y, log_x) / var(log_x).

    Uses pandas rolling windows for vectorised computation.  The first
    *window − 1* values are NaN (insufficient history).

    Args:
        log_y  : log prices of Y (dependent)
        log_x  : log prices of X (independent)
        window : look-back window in trading days

    Returns:
        pd.Series of β_t with the same index as the inputs.
    """
    cov = log_y.rolling(window, min_periods=window).cov(log_x)
    var = log_x.rolling(window, min_periods=window).var(ddof=1)
    beta = (cov / var.replace(0.0, np.nan)).rename("beta")
    return beta


# ── Spread and z-score ───────────────────────────────────────────────────────

def compute_spread(
    log_y: pd.Series,
    log_x: pd.Series,
    beta: pd.Series,
) -> pd.Series:
    """Spread S_t = log(P_y,t) − β_t · log(P_x,t)."""
    return (log_y - beta * log_x).rename("spread")


def compute_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """Rolling z-score of the spread.

    z_t = (S_t − μ_t) / σ_t  where μ and σ are computed over *window* bars.
    Zero standard deviation is replaced with NaN to avoid division by zero.
    """
    mu = spread.rolling(window, min_periods=window).mean()
    sigma = spread.rolling(window, min_periods=window).std(ddof=1).replace(0.0, np.nan)
    return ((spread - mu) / sigma).rename("zscore")


# ── Signal state machine ────────────────────────────────────────────────────

def generate_pair_signal(
    zscore: pd.Series,
    entry: float = 2.0,
    exit_thresh: float = 0.5,
    stop: float = 4.0,
) -> pd.Series:
    """State-machine trading signal for a single pair.

    States:
      +1  long spread  (buy Y, sell X) — entered when z < −entry
      −1  short spread (sell Y, buy X) — entered when z > +entry
       0  flat

    Exit triggers:
      |z| < exit_thresh  →  mean-reversion achieved, go flat
      |z| > stop         →  stop-loss, go flat (no reversal)

    Args:
        zscore      : z-score Series
        entry       : z threshold to enter a trade (absolute value)
        exit_thresh : z threshold to exit (mean-reversion)
        stop        : z threshold for stop-loss exit

    Returns:
        pd.Series of {−1, 0, +1} with same index as zscore.
    """
    z_vals = zscore.to_numpy(dtype=float, na_value=np.nan)
    n = len(z_vals)
    signal = np.zeros(n, dtype=float)
    state = 0.0

    for i in range(n):
        z = z_vals[i]
        if np.isnan(z):
            signal[i] = state
            continue

        if state == 0.0:
            if z < -entry:
                state = 1.0      # long spread (z negative → spread cheap)
            elif z > entry:
                state = -1.0     # short spread (z positive → spread rich)
        elif state == 1.0:       # currently long spread
            if abs(z) < exit_thresh:
                state = 0.0      # mean-reversion exit
            elif z < -stop:
                state = 0.0      # stop-loss exit
        elif state == -1.0:      # currently short spread
            if abs(z) < exit_thresh:
                state = 0.0
            elif z > stop:
                state = 0.0

        signal[i] = state

    return pd.Series(signal, index=zscore.index, name="pair_signal")


# ── Walk-forward orchestrator ────────────────────────────────────────────────

def select_pairs_and_build_theta(
    asset_panel: pd.DataFrame,
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
) -> dict:
    """Walk-forward pair selection and theta construction.

    At each rebalancing boundary the trailing *coint_test_window* bars are
    used to test all C(4,2) = 6 pairs for cointegration.  Pairs whose
    Engle-Granger p-value < *coint_pvalue_threshold* are traded in the
    subsequent *oos_window*-bar out-of-sample period.  If no pair passes,
    the strategy holds cash (θ = 0).

    For each selected pair the rolling hedge ratio, spread, z-score, and
    signal are computed causally (using only data available at each date).
    Dollar positions are allocated equally across selected pairs with a
    post-hoc rescaling to enforce Σ|θ_i| ≤ gross_cap.

    Args:
        asset_panel            : MultiIndex DataFrame with ("close", asset) columns
        symbols                : list of asset symbol strings
        coint_test_window      : bars for in-sample cointegration test
        rolling_hedge_window   : look-back for hedge ratio and z-score
        coint_pvalue_threshold : Engle-Granger p-value cutoff
        zscore_entry           : |z| threshold to open a position
        zscore_exit            : |z| threshold for mean-reversion exit
        zscore_stop            : |z| threshold for stop-loss exit
        gross_cap              : maximum gross notional exposure
        oos_window             : out-of-sample trading window length

    Returns:
        dict with keys:
          theta_target           — DataFrame (index=dates, cols=symbols) of dollar positions
          rebalance_mask         — boolean Series (all True, daily rebalance)
          pair_signals           — {pair_key: Series} of signals per pair
          pair_spreads           — {pair_key: Series} of spreads
          pair_zscores           — {pair_key: Series} of z-scores
          pair_betas             — {pair_key: Series} of hedge ratios
          coint_results          — list[dict] of per-window cointegration test results
          warm_up_end            — Timestamp of first date with possible non-zero theta
          window_selections      — list[dict] with OOS start/end and selected pairs per window
    """
    index = asset_panel.index.sort_values()
    log_close = np.log(asset_panel["close"].reindex(columns=symbols))
    all_pairs = enumerate_pairs(symbols)

    theta_target = pd.DataFrame(0.0, index=index, columns=symbols)
    coint_results: list[dict] = []
    window_selections: list[dict] = []

    # Diagnostics: accumulate full-sample signals/spreads/zscores/betas per pair
    pair_signals: dict[tuple[str, str], pd.Series] = {}
    pair_spreads: dict[tuple[str, str], pd.Series] = {}
    pair_zscores: dict[tuple[str, str], pd.Series] = {}
    pair_betas: dict[tuple[str, str], pd.Series] = {}

    # First OOS bar is at position coint_test_window (need full IS window before it)
    warm_up_end_pos = coint_test_window - 1
    warm_up_end = index[warm_up_end_pos] if warm_up_end_pos < len(index) else index[-1]

    # Pre-compute full-sample rolling hedge ratios, spreads, z-scores for each pair
    # (causally valid — rolling uses only past data).  We slice into OOS windows later.
    _full_betas: dict[tuple[str, str], pd.Series] = {}
    _full_spreads: dict[tuple[str, str], pd.Series] = {}
    _full_zscores: dict[tuple[str, str], pd.Series] = {}

    for (x_sym, y_sym) in all_pairs:
        lp_x = log_close[x_sym]
        lp_y = log_close[y_sym]
        beta = rolling_hedge_ratio(lp_y, lp_x, window=rolling_hedge_window)
        spread = compute_spread(lp_y, lp_x, beta)
        zscore = compute_zscore(spread, window=rolling_hedge_window)
        _full_betas[(x_sym, y_sym)] = beta
        _full_spreads[(x_sym, y_sym)] = spread
        _full_zscores[(x_sym, y_sym)] = zscore

    # Walk-forward loop
    oos_start_pos = coint_test_window
    while oos_start_pos < len(index):
        # Estimation (in-sample) window
        est_start_pos = max(0, oos_start_pos - coint_test_window)
        est_dates = index[est_start_pos:oos_start_pos]

        # OOS window
        oos_end_pos = min(oos_start_pos + oos_window, len(index))
        oos_dates = index[oos_start_pos:oos_end_pos]

        # Test all 6 pairs for cointegration on the estimation window
        window_coint: dict[tuple[str, str], float] = {}
        for (x_sym, y_sym) in all_pairs:
            lp_x = log_close[x_sym].loc[est_dates]
            lp_y = log_close[y_sym].loc[est_dates]
            t_stat, p_val, crits = test_cointegration(lp_y, lp_x)
            window_coint[(x_sym, y_sym)] = p_val
            coint_results.append({
                "window_start": est_dates[0],
                "window_end": est_dates[-1],
                "oos_start": oos_dates[0],
                "oos_end": oos_dates[-1],
                "pair_x": x_sym,
                "pair_y": y_sym,
                "p_value": p_val,
                "t_stat": t_stat,
            })

        # Select cointegrated pairs
        selected = [
            (x, y) for (x, y), pv in window_coint.items()
            if pv < coint_pvalue_threshold
        ]

        window_selections.append({
            "oos_start": oos_dates[0],
            "oos_end": oos_dates[-1],
            "n_selected": len(selected),
            "selected_pairs": selected,
        })

        if not selected:
            oos_start_pos += oos_window
            continue

        # Capital budget per pair (equal-weight, two legs per pair)
        pair_notional = gross_cap / (2.0 * len(selected))

        for (x_sym, y_sym) in selected:
            # Generate signal on OOS dates using pre-computed z-scores.
            # The signal state machine runs fresh (from flat) for each OOS window.
            zscore_oos = _full_zscores[(x_sym, y_sym)].reindex(oos_dates)
            signal_oos = generate_pair_signal(
                zscore_oos, entry=zscore_entry, exit_thresh=zscore_exit, stop=zscore_stop,
            )
            beta_oos = _full_betas[(x_sym, y_sym)].reindex(oos_dates)

            # Accumulate positions: long spread → +Y, −β·X
            for date in oos_dates:
                sig = signal_oos.get(date, 0.0)
                b = beta_oos.get(date, np.nan)
                if np.isnan(sig) or np.isnan(b) or sig == 0.0:
                    continue
                theta_target.loc[date, y_sym] += sig * pair_notional
                theta_target.loc[date, x_sym] += -sig * b * pair_notional

        oos_start_pos += oos_window

    # Post-hoc gross-cap enforcement: scale down if aggregate exposure exceeds budget
    gross_exposure = theta_target.abs().sum(axis=1)
    scale = (gross_cap / gross_exposure).clip(upper=1.0)
    scale = scale.replace([np.inf, -np.inf], 1.0).fillna(1.0)
    theta_target = theta_target.mul(scale, axis=0)

    # Collect full-sample diagnostics for plotting
    for (x_sym, y_sym) in all_pairs:
        pair_key = (x_sym, y_sym)
        pair_betas[pair_key] = _full_betas[pair_key]
        pair_spreads[pair_key] = _full_spreads[pair_key]
        pair_zscores[pair_key] = _full_zscores[pair_key]

        # Build full signal by concatenating per-window signals (gaps stay 0)
        full_signal = pd.Series(0.0, index=index, name="pair_signal")
        # Re-run signal on the full z-score series for diagnostic purposes
        full_signal = generate_pair_signal(
            _full_zscores[pair_key],
            entry=zscore_entry, exit_thresh=zscore_exit, stop=zscore_stop,
        )
        pair_signals[pair_key] = full_signal

    rebalance_mask = pd.Series(True, index=index, name="is_rebalance")

    return {
        "theta_target": theta_target,
        "rebalance_mask": rebalance_mask,
        "pair_signals": pair_signals,
        "pair_spreads": pair_spreads,
        "pair_zscores": pair_zscores,
        "pair_betas": pair_betas,
        "coint_results": coint_results,
        "warm_up_end": warm_up_end,
        "window_selections": window_selections,
    }
