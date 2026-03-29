"""
position_sizing.py — Dollar exposure mapping (Cell 21).

Function:
  compute_exposures(signal_panel, symbols, dead_zone, gross_cap)
    Maps trend_raw → single vol step (w_risk) → L1 weights → dollar theta.
    z is for activation / dead-zone only; w_risk is the only vol-adjusted
    sizing step.
    Returns dict with exposure_panel, theta, w_tilde, w_risk, w_raw, denom.
"""

import numpy as np
import pandas as pd

from .helpers import _field_wide
from .utils import display_df


def compute_exposures(signal_panel, symbols, dead_zone, gross_cap):
    """Compute dollar exposures from the trend signal panel.

    Sizing pipeline:
      active   = z.abs() > dead_zone  (dead-zone activation)
      w_raw    = trend_raw on active assets, else 0
      w_risk   = w_raw / vol_20       (single vol scaling step)
      w_tilde  = w_risk / sum|w_risk| (L1 cross-sectional normalisation)
      theta    = gross_cap * w_tilde  (sum|theta| = gross_cap when active)

    Args:
        signal_panel : MultiIndex DataFrame from construct_signals()
        symbols      : list of asset strings
        dead_zone    : dead-zone threshold on z (same value used in signals)
        gross_cap    : fixed gross notional budget in USDT

    Returns:
        dict with keys:
          exposure_panel : MultiIndex DataFrame (field × asset) for
                           w_raw, w_risk, w_tilde, theta
          theta          : DataFrame of dollar exposures
          w_tilde        : normalised weights
          w_risk         : vol-scaled weights
          w_raw          : raw (pre-vol) weights
          denom          : cross-sectional L1 norm of w_risk
    """
    tr  = signal_panel["trend_raw"]
    z   = signal_panel["z"]
    pos = signal_panel["trend_position"]
    vol = signal_panel["vol_20"]

    active = (z.abs() > dead_zone) & z.notna()
    w_raw  = tr.where(active, 0.0)

    sigma      = vol.replace(0.0, np.nan)
    w_risk     = w_raw / sigma

    denom      = w_risk.abs().sum(axis=1)
    denom_safe = denom.replace(0.0, np.nan)
    w_tilde    = w_risk.div(denom_safe, axis=0).fillna(0.0)

    theta = gross_cap * w_tilde

    exposure_panel = pd.concat(
        [
            _field_wide("w_raw",   [w_raw[s]   for s in symbols], symbols),
            _field_wide("w_risk",  [w_risk[s]  for s in symbols], symbols),
            _field_wide("w_tilde", [w_tilde[s] for s in symbols], symbols),
            _field_wide("theta",   [theta[s]   for s in symbols], symbols),
        ],
        axis=1,
    ).sort_index(axis=1)

    # Sanity checks
    gross_abs   = exposure_panel["theta"].abs().sum(axis=1)
    active_rows = denom > 0
    print(
        f"Gross |theta| sum: min={gross_abs.min():,.2f}, max={gross_abs.max():,.2f}, "
        f"mean={gross_abs.mean():,.2f}"
    )
    print(
        f"  (on active rows where denom>0: min={gross_abs[active_rows].min():,.2f}, "
        f"max={gross_abs[active_rows].max():,.2f}, matches GROSS_CAP={gross_cap:,.0f})"
    )

    # Sizing chain for last 3 dates
    _last3 = signal_panel.index[-3:]
    _sanity = (
        pd.concat(
            {
                "trend_raw":      tr.loc[_last3],
                "vol_20":         vol.loc[_last3],
                "z":              z.loc[_last3],
                "trend_position": pos.loc[_last3],
                "w_raw":          w_raw.loc[_last3],
                "w_risk":         w_risk.loc[_last3],
                "w_norm":         w_tilde.loc[_last3],
                "theta":          theta.loc[_last3],
            },
            axis=1,
        )
        .stack(level="asset", future_stack=True)
        .reset_index()
        .rename(columns={"level_0": "timestamp"})
    )
    print("Sizing sanity check (last 3 dates × assets):")
    display_df(_sanity)

    return {
        "exposure_panel": exposure_panel,
        "theta":          theta,
        "w_tilde":        w_tilde,
        "w_risk":         w_risk,
        "w_raw":          w_raw,
        "denom":          denom,
    }
