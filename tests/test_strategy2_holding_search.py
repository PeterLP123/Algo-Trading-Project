import json
from pathlib import Path

import pandas as pd

import strategy2_holding_search as search


def test_holding_search_spec_is_fixed_and_bounded() -> None:
    spec = search.load_search_spec()

    assert search.canonical_json_digest(spec) == search.EXPECTED_SPEC_DIGEST
    assert spec["evidence_seen_through"] == "2026-08-04"
    assert spec["prior_searches"][-1]["least_negative_candidate"] == "core3_1x"
    candidates = [variant for variant in spec["variants"] if variant["role"] == "candidate"]
    assert [variant["max_hold"] for variant in candidates] == [1, 2, 3, 5, 7]
    assert {variant["exit_mode"] for variant in candidates} == {"fixed_holding"}
    assert {tuple(variant["alt_universe"]) for variant in candidates} == {
        ("ETH", "BNB", "SOL")
    }
    assert spec["retrospective_usefulness_screen"]["minimum_entries"] == 5
    assert spec["prospective_gate"]["minimum_new_entries"] == 20


def test_holding_search_cli_exposes_no_strategy_overrides() -> None:
    option_dests = {action.dest for action in search.build_parser()._actions}

    assert option_dests.isdisjoint(
        {
            "alt_universe",
            "entry_mode",
            "exit_mode",
            "entry_pct",
            "lookback",
            "exit_z",
            "max_hold",
            "gross_util",
        }
    )


def test_committed_holding_search_artifacts_are_self_consistent() -> None:
    output = Path("experiments/strategy2_holding_search/results/2026-08-04")
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
