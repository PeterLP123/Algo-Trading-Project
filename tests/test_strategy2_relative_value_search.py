import json
from pathlib import Path

import pandas as pd

import strategy2_relative_value_search as search


def test_relative_value_spec_is_fixed_and_bounded() -> None:
    spec = search.load_search_spec()

    assert search.canonical_json_digest(spec) == search.EXPECTED_SPEC_DIGEST
    assert spec["evidence_seen_through"] == "2026-08-04"
    assert spec["prior_searches"][-1]["least_negative_candidate"] == (
        "core3_fixed_5d_1x"
    )
    candidates = [variant for variant in spec["variants"] if variant["role"] == "candidate"]
    assert len(candidates) == 6
    assert {variant["position_mode"] for variant in candidates} == {
        "long_alts_short_btc"
    }
    assert search.SHORT_CARRY_RATE_ANNUAL == 0.10
    assert spec["retrospective_usefulness_screen"]["minimum_entries"] == 5
    assert spec["prospective_gate"]["minimum_new_entries"] == 20


def test_relative_value_cli_exposes_no_strategy_overrides() -> None:
    option_dests = {action.dest for action in search.build_parser()._actions}

    assert option_dests.isdisjoint(
        {
            "alt_universe",
            "entry_mode",
            "position_mode",
            "exit_mode",
            "max_hold",
            "short_carry_rate_annual",
            "gross_util",
        }
    )


def test_committed_relative_value_artifacts_are_self_consistent() -> None:
    output = Path("experiments/strategy2_relative_value_search/results/2026-08-04")
    summary_path = output / "summary.json"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    comparison = pd.read_csv(output / "comparison.csv")
    daily = pd.read_csv(output / "post_selection_daily.csv", parse_dates=["date"])
    trade_ledger = pd.read_csv(output / "trade_ledger.csv")

    assert summary["search_spec_digest"] == search.EXPECTED_SPEC_DIGEST
    assert len(comparison) == len(summary["search_spec"]["variants"]) * 3
    assert len(daily) == 263
    assert not trade_ledger.empty
    assert (output / summary["artifacts"]["figure"]).stat().st_size > 0
