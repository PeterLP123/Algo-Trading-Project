import strategy_helpers
import systematic_crypto
from systematic_crypto import costs, data, evaluation, execution, signals


MODULE_EXPORTS = {
    data: {
        "VALUE_COLUMNS",
        "audit_ohlcv",
        "ensure_utc_index",
        "flag_ohlcv_outliers",
        "load_hourly_risk_free_rate",
        "read_parquet_safe",
        "run_data_quality_pipeline",
        "write_parquet_safe",
    },
    signals: {
        "build_conviction_scale_series",
        "build_cross_sectional_conviction_target_weights",
        "build_cross_sectional_inputs",
        "build_inverse_vol_target_weights",
        "build_inverse_vol_weights",
        "build_mean_reversion_regime_filter",
        "build_top_bottom_signal_frame",
        "build_tsmom_inverse_vol_weights",
        "build_tsmom_signal_frame",
    },
    execution: {
        "run_cross_sectional_strategy",
        "run_execution_engine",
        "run_tsmom_strategy",
    },
    evaluation: {
        "build_diagnostics_tail_table",
        "build_parameter_table",
        "build_results_summary_table",
        "build_strategy_comparison_table",
        "infer_rank_95",
        "interpret_pair_result",
        "run_cross_sectional_parameter_sweep",
        "run_mean_reversion_walk_forward",
        "run_pair_diagnostics",
        "run_tsmom_parameter_sweep",
        "run_tsmom_walk_forward",
        "slice_time_window",
        "summarize_parameter_stability",
        "summarize_strategy_state",
    },
    costs: {
        "estimate_roll_spread",
        "estimate_roll_spread_rolling",
        "run_cost_sensitivity_analysis",
        "run_roll_model",
    },
}


def test_package_exports_are_owned_by_one_focused_module() -> None:
    expected_exports = set().union(*MODULE_EXPORTS.values())
    assert set(systematic_crypto.__all__) == expected_exports

    for module, names in MODULE_EXPORTS.items():
        for name in names:
            package_value = getattr(systematic_crypto, name)
            module_value = getattr(module, name)
            assert package_value is module_value
            if callable(module_value):
                assert module_value.__module__ == module.__name__


def test_strategy_helpers_preserves_the_complete_public_api() -> None:
    assert strategy_helpers.__all__ == systematic_crypto.__all__
    for name in systematic_crypto.__all__:
        assert getattr(strategy_helpers, name) is getattr(systematic_crypto, name)
