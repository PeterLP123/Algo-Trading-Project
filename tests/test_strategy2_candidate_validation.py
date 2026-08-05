import json

import pytest

import strategy2_candidate_validation as validation


def test_candidate_validation_is_seeded_and_cautious() -> None:
    result = validation.build_validation()

    assert result["confidence"] == "CAUTION"
    assert result["decision"] == {
        "descriptive_candidate": True,
        "validated_strategy": False,
        "deployment_ready": False,
        "reason": (
            "Positive at the registered 20 bps and monthly-corrected spread scenarios, but "
            "negative under the two-day-corrected spread and fixed 40 bps scenarios, "
            "concentrated in two trades, selected adaptively, and below the 20-entry "
            "prospective gate."
        ),
    }
    assert result["fallacy_scan"]["coverage"] == "11/11 checked"
    assert len(result["fallacy_scan"]["items"]) == 11
    assert result["multiplicity"]["adaptive_candidate_variants_screened"] == 24
    assert result["observed_metrics"]["trade_entries"] == 6
    assert result["cost_break_even_one_way_bps"] == pytest.approx(34.9942272723)


def test_candidate_validation_artifacts_match_recomputation() -> None:
    committed = json.loads(validation.DEFAULT_JSON_OUTPUT.read_text(encoding="utf-8"))
    assert committed == validation.build_validation()
    markdown = validation.DEFAULT_MARKDOWN_OUTPUT.read_text(encoding="utf-8")
    assert "cost-conditional descriptive candidate" in markdown
    assert "11/11 statistical fallacies checked" in markdown
