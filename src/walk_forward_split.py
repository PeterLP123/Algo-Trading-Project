"""
walk_forward_split.py — Holdout / walk-forward split (Cell 15).

Function:
  compute_splits(asset_panel, holdout_frac, initial_train_bars, val_bars,
                 step_bars, wf_mode, train_bars)
    Reserves the last holdout_frac of dates as a final test set, then
    generates expanding (or rolling) walk-forward folds on the rest.
    Prints summary and returns a dict with idx, dev_index, holdout_index,
    wf_folds.
"""

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    display = print


def compute_splits(
    asset_panel,
    holdout_frac=0.225,
    initial_train_bars=756,
    val_bars=126,
    step_bars=126,
    wf_mode="expanding",
    train_bars=756,
):
    """Create holdout/dev split and walk-forward fold schedule.

    Args:
        asset_panel        : pd.DataFrame with DatetimeIndex (rows = trading days)
        holdout_frac       : fraction of dates reserved as final holdout
        initial_train_bars : minimum training bars before first validation fold
        val_bars           : number of bars per validation window
        step_bars          : stride between consecutive fold starts
        wf_mode            : "expanding" | "rolling"
        train_bars         : rolling-mode only — fixed training window length

    Returns:
        dict with keys:
          idx           : full sorted DatetimeIndex
          n             : total number of dates
          dev_index     : DatetimeIndex for development (train+CV) period
          holdout_index : DatetimeIndex for final holdout (test) period
          wf_folds      : pd.DataFrame with one row per fold
    """
    # --- Outer holdout (20–25% tail) + walk-forward on development sample ---
    idx = asset_panel.index.sort_values()
    n = len(idx)

    holdout_n = max(1, int(round(holdout_frac * n)))
    holdout_index = idx[-holdout_n:]
    dev_index = idx[:-holdout_n]

    n_dev = len(dev_index)
    if initial_train_bars + val_bars > n_dev:
        raise ValueError(
            f"Development sample too short for one fold: n_dev={n_dev}, "
            f"need INITIAL_TRAIN_BARS + VAL_BARS = {initial_train_bars + val_bars}"
        )
    if wf_mode not in ("expanding", "rolling"):
        raise ValueError('WF_MODE must be "expanding" or "rolling"')
    if wf_mode == "rolling" and train_bars > initial_train_bars:
        raise ValueError(
            "For rolling folds, set TRAIN_BARS <= INITIAL_TRAIN_BARS so fold 1 has enough history"
        )

    fold_rows = []
    val_start_pos = initial_train_bars
    fold_id = 0
    while val_start_pos + val_bars <= n_dev:
        fold_id += 1
        val_ix = dev_index[val_start_pos : val_start_pos + val_bars]
        if wf_mode == "expanding":
            train_ix = dev_index[:val_start_pos]
        else:
            t0 = val_start_pos - train_bars
            train_ix = dev_index[t0:val_start_pos]

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
        val_start_pos += step_bars

    wf_folds = pd.DataFrame(fold_rows)
    if wf_folds.empty:
        raise ValueError(
            "No walk-forward folds generated; relax INITIAL_TRAIN_BARS, VAL_BARS, or HOLDOUT_FRAC"
        )

    # --- Printed date ranges ---
    print(f"Full sample: {idx.min().date()} → {idx.max().date()} (n={n})")
    print()
    print(
        f"Development (train/CV): {dev_index.min().date()} → {dev_index.max().date()} "
        f"(n={len(dev_index)})"
    )
    print(
        f"Holdout (final test):     {holdout_index.min().date()} → {holdout_index.max().date()} "
        f"(n={len(holdout_index)}, HOLDOUT_FRAC={holdout_frac})"
    )
    print()
    rolling_note = f", TRAIN_BARS={train_bars}" if wf_mode == "rolling" else ""
    print(
        f"Walk-forward: WF_MODE={wf_mode!r}, INITIAL_TRAIN_BARS={initial_train_bars}, "
        f"VAL_BARS={val_bars}, STEP_BARS={step_bars}{rolling_note}"
    )
    print()
    display(wf_folds)
    print()
    assert dev_index.max() < holdout_index.min(), "dev and holdout ranges should not overlap"
    print("Sanity check: dev_index.max() < holdout_index.min() — OK")

    return {
        "idx": idx,
        "n": n,
        "dev_index": dev_index,
        "holdout_index": holdout_index,
        "wf_folds": wf_folds,
    }
