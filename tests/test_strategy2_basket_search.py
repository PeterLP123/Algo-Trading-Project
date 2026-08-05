import json
from pathlib import Path

import pandas as pd

import strategy2_basket_search as search


def test_basket_search_spec_is_fixed_and_bounded() -> None:
    spec = search.load_search_spec()

    assert search.canonical_json_digest(spec) == search.EXPECTED_SPEC_DIGEST
    assert spec["evidence_seen_through"] == "2026-08-04"
    assert spec["prior_search"]["result"] == (
        "no_entry_confirmation_candidate_survived_to_the_evaluation_anchor"
    )
    assert spec["variants"][0] == {
        "id": "broad_all_1x",
        "alt_universe": [
            "ETH",
            "BNB",
            "SOL",
            "XRP",
            "ADA",
            "AVAX",
            "DOT",
            "LINK",
            "LTC",
            "MATIC",
        ],
        "role": "same_risk_baseline",
    }
    assert len([v for v in spec["variants"] if v["role"] == "candidate"]) == 5
    assert spec["retrospective_usefulness_screen"]["minimum_entries"] == 5
    assert spec["prospective_gate"]["minimum_new_entries"] == 20


def test_basket_search_cli_exposes_no_strategy_or_basket_overrides() -> None:
    option_dests = {action.dest for action in search.build_parser()._actions}

    assert option_dests.isdisjoint(
        {
            "alt_universe",
            "entry_mode",
            "entry_pct",
            "lookback",
            "exit_z",
            "max_hold",
            "gross_util",
        }
    )


def test_committed_basket_search_artifacts_are_self_consistent() -> None:
    output = Path("experiments/strategy2_basket_search/results/2026-08-04")
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
