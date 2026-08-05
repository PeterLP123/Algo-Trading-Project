import json
from pathlib import Path

import pandas as pd

import strategy2_relative_horizon_search as search


def test_relative_horizon_spec_is_fixed_and_bounded() -> None:
    spec = search.load_search_spec()

    assert search.canonical_json_digest(spec) == search.EXPECTED_SPEC_DIGEST
    assert spec["evidence_seen_through"] == "2026-08-04"
    assert spec["prior_search"]["least_negative_candidate"] == (
        "core3_relative_fixed_7d_1x"
    )
    candidates = [variant for variant in spec["variants"] if variant["role"] == "candidate"]
    assert [variant["max_hold"] for variant in candidates] == [10, 14, 21, 28]
    assert {variant["position_mode"] for variant in candidates} == {
        "long_alts_short_btc"
    }
    assert spec["retrospective_usefulness_screen"]["minimum_entries"] == 5


def test_relative_horizon_cli_exposes_no_strategy_overrides() -> None:
    option_dests = {action.dest for action in search.build_parser()._actions}

    assert option_dests.isdisjoint(
        {
            "alt_universe",
            "position_mode",
            "exit_mode",
            "max_hold",
            "short_carry_rate_annual",
            "gross_util",
        }
    )


def test_committed_relative_horizon_artifacts_are_self_consistent() -> None:
    output = Path("experiments/strategy2_relative_horizon_search/results/2026-08-04")
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    comparison = pd.read_csv(output / "comparison.csv")
    daily = pd.read_csv(output / "post_selection_daily.csv", parse_dates=["date"])
    assert summary["search_spec_digest"] == search.EXPECTED_SPEC_DIGEST
    assert len(comparison) == len(summary["search_spec"]["variants"]) * 3
    assert len(daily) == 263
    assert (output / summary["artifacts"]["figure"]).stat().st_size > 0
