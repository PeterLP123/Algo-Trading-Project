import json
from pathlib import Path

import pandas as pd

import strategy2_variation_search as search


def test_variation_search_spec_is_fixed_and_bounded() -> None:
    spec = search.load_search_spec()

    assert search.canonical_json_digest(spec) == search.EXPECTED_SPEC_DIGEST
    assert spec["evidence_seen_through"] == "2026-08-04"
    assert spec["known_prior_trial"] == {
        "id": "confirmed_reversal_2x",
        "result": "non_positive_equity_before_post_selection_anchor",
    }
    candidates = [variant for variant in spec["variants"] if variant["role"] == "candidate"]
    assert len(candidates) == 4
    assert {variant["gross_util"] for variant in candidates} == {1.0}
    assert spec["retrospective_usefulness_screen"]["minimum_entries"] == 5
    assert spec["prospective_gate"]["minimum_new_entries"] == 20


def test_variation_search_cli_exposes_no_strategy_overrides() -> None:
    option_dests = {action.dest for action in search.build_parser()._actions}

    assert option_dests.isdisjoint(
        {
            "entry_mode",
            "entry_pct",
            "lookback",
            "exit_z",
            "max_hold",
            "roc_window",
            "roc_pct",
            "crash_threshold",
            "gross_util",
        }
    )


def test_committed_variation_search_artifacts_are_self_consistent() -> None:
    output = Path("experiments/strategy2_variation_search/results/2026-08-04")
    summary_path = output / "summary.json"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    comparison = pd.read_csv(output / "comparison.csv")
    daily = pd.read_csv(output / "post_selection_daily.csv", parse_dates=["date"])
    trade_ledger = pd.read_csv(output / "trade_ledger.csv")

    assert summary["search_spec_digest"] == search.EXPECTED_SPEC_DIGEST
    assert summary["baseline_reproduced_exactly"] is True
    assert len(comparison) == len(summary["search_spec"]["variants"]) * 3
    assert len(daily) == 263
    assert not trade_ledger.empty
    assert (output / summary["artifacts"]["figure"]).stat().st_size > 0


def test_confirmed_reversal_bankruptcy_is_preserved() -> None:
    failure = json.loads(
        Path(
            "experiments/strategy2b_confirmed_reversal/results/2026-08-04/failure.json"
        ).read_text(encoding="utf-8")
    )

    assert failure["status"] == "failed_before_evaluation"
    assert failure["diagnostics"]["first_non_positive_date"] == "2021-05-19"
    assert failure["diagnostics"]["final_equity"] == 0.0
    assert failure["diagnostics"]["cost_through_failure"] > failure["diagnostics"][
        "gross_pnl_through_failure"
    ]
