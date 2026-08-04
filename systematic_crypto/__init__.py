"""Reusable components for the systematic cryptocurrency research workflow.

The package is organized by research responsibility. Import from the focused
modules for new code; the package root and strategy_helpers retain the original
public API for existing notebooks and scripts.
"""

from .data import (
    VALUE_COLUMNS,
    audit_ohlcv,
    ensure_utc_index,
    flag_ohlcv_outliers,
    load_hourly_risk_free_rate,
    read_parquet_safe,
    run_data_quality_pipeline,
    write_parquet_safe,
)
from .signals import (
    build_conviction_scale_series,
    build_cross_sectional_conviction_target_weights,
    build_cross_sectional_inputs,
    build_inverse_vol_target_weights,
    build_inverse_vol_weights,
    build_mean_reversion_regime_filter,
    build_top_bottom_signal_frame,
    build_tsmom_inverse_vol_weights,
    build_tsmom_signal_frame,
)
from .execution import (
    run_cross_sectional_strategy,
    run_execution_engine,
    run_tsmom_strategy,
)
from .evaluation import (
    build_diagnostics_tail_table,
    build_parameter_table,
    build_results_summary_table,
    build_strategy_comparison_table,
    infer_rank_95,
    interpret_pair_result,
    run_cross_sectional_parameter_sweep,
    run_mean_reversion_walk_forward,
    run_pair_diagnostics,
    run_tsmom_parameter_sweep,
    run_tsmom_walk_forward,
    slice_time_window,
    summarize_parameter_stability,
    summarize_strategy_state,
)
from .costs import (
    estimate_roll_spread,
    estimate_roll_spread_rolling,
    run_cost_sensitivity_analysis,
    run_roll_model,
)

__all__ = [
    "VALUE_COLUMNS",
    "audit_ohlcv",
    "build_conviction_scale_series",
    "build_cross_sectional_conviction_target_weights",
    "build_cross_sectional_inputs",
    "build_diagnostics_tail_table",
    "build_inverse_vol_target_weights",
    "build_inverse_vol_weights",
    "build_mean_reversion_regime_filter",
    "build_parameter_table",
    "build_results_summary_table",
    "build_strategy_comparison_table",
    "build_top_bottom_signal_frame",
    "build_tsmom_inverse_vol_weights",
    "build_tsmom_signal_frame",
    "ensure_utc_index",
    "estimate_roll_spread",
    "estimate_roll_spread_rolling",
    "flag_ohlcv_outliers",
    "infer_rank_95",
    "interpret_pair_result",
    "load_hourly_risk_free_rate",
    "read_parquet_safe",
    "run_cost_sensitivity_analysis",
    "run_cross_sectional_parameter_sweep",
    "run_cross_sectional_strategy",
    "run_data_quality_pipeline",
    "run_execution_engine",
    "run_mean_reversion_walk_forward",
    "run_pair_diagnostics",
    "run_roll_model",
    "run_tsmom_parameter_sweep",
    "run_tsmom_strategy",
    "run_tsmom_walk_forward",
    "slice_time_window",
    "summarize_parameter_stability",
    "summarize_strategy_state",
    "write_parquet_safe",
]
