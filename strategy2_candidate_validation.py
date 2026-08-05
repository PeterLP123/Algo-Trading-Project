#!/usr/bin/env python3
"""Validate the selected Strategy 2 cost-conditional candidate without retuning it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_DIR = PROJECT_ROOT / "experiments" / "strategy2_cost_sensitivity" / "results" / "2026-08-04"
SUMMARY_PATH = RESULT_DIR / "summary.json"
DAILY_PATH = RESULT_DIR / "post_selection_daily.csv"
TRADE_PATH = RESULT_DIR / "trade_ledger.csv"
DEFAULT_JSON_OUTPUT = RESULT_DIR / "candidate_validation.json"
DEFAULT_MARKDOWN_OUTPUT = RESULT_DIR / "candidate_validation.md"
DECISION_SCENARIO = "fixed_20bps_one_way"
PAPER_SPREAD_SCENARIOS = (
    "abdi_ranaldo_monthly_corrected",
    "abdi_ranaldo_two_day_corrected",
)
BOOTSTRAP_SEED = 20_260_805
BOOTSTRAP_REPLICATIONS = 10_000
TRADE_BOOTSTRAP_REPLICATIONS = 50_000


class CandidateValidationError(RuntimeError):
    """Raised when validation inputs do not match the frozen candidate."""


def load_inputs() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if summary["sensitivity_spec"]["usefulness_definition"]["decision_scenario"] != DECISION_SCENARIO:
        raise CandidateValidationError("The registered decision scenario changed.")
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"]).set_index("date")
    trades = pd.read_csv(TRADE_PATH)
    trades = trades.loc[trades["strategy"].eq(DECISION_SCENARIO)].reset_index(drop=True)
    expected_entries = summary["results"][DECISION_SCENARIO]["windows"][
        "continuous_post_selection"
    ]["trade_entries"]
    if len(trades) != expected_entries:
        raise CandidateValidationError("Trade ledger does not match the decision scenario.")
    return summary, daily, trades


def daily_returns_from_cumulative(cumulative: pd.Series) -> np.ndarray:
    wealth = 1.0 + cumulative.astype(float)
    daily_returns = wealth.pct_change()
    daily_returns.iloc[0] = wealth.iloc[0] - 1.0
    return daily_returns.to_numpy(dtype=float)


def circular_block_bootstrap(
    returns: np.ndarray,
    *,
    block_length: int,
    replications: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_days = len(returns)
    n_blocks = int(np.ceil(n_days / block_length))
    starts = rng.integers(0, n_days, size=(replications, n_blocks))
    offsets = np.arange(block_length)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n_days
    samples = returns[indices.reshape(replications, -1)[:, :n_days]]
    wealth = np.cumprod(1.0 + samples, axis=1)
    total_returns = wealth[:, -1] - 1.0
    sample_std = samples.std(axis=1, ddof=1)
    sharpes = np.divide(
        samples.mean(axis=1) * np.sqrt(252.0),
        sample_std,
        out=np.zeros(replications),
        where=sample_std > 0.0,
    )
    running_max = np.maximum.accumulate(np.column_stack([np.ones(replications), wealth]), axis=1)
    wealth_with_anchor = np.column_stack([np.ones(replications), wealth])
    max_drawdowns = (wealth_with_anchor / running_max - 1.0).min(axis=1)
    return {
        "block_length_days": block_length,
        "replications": replications,
        "seed": seed,
        "total_return_ci_95": [
            float(np.quantile(total_returns, 0.025)),
            float(np.quantile(total_returns, 0.975)),
        ],
        "sharpe_ci_95": [
            float(np.quantile(sharpes, 0.025)),
            float(np.quantile(sharpes, 0.975)),
        ],
        "max_drawdown_ci_95": [
            float(np.quantile(max_drawdowns, 0.025)),
            float(np.quantile(max_drawdowns, 0.975)),
        ],
        "probability_total_return_above_zero": float(np.mean(total_returns > 0.0)),
    }


def trade_bootstrap(trades: pd.DataFrame) -> dict[str, Any]:
    values = trades["net_pnl"].to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    samples = rng.choice(values, size=(TRADE_BOOTSTRAP_REPLICATIONS, len(values)), replace=True)
    totals = samples.sum(axis=1)
    return {
        "observed_trades": len(values),
        "replications": TRADE_BOOTSTRAP_REPLICATIONS,
        "seed": BOOTSTRAP_SEED + 1,
        "total_net_pnl_ci_95": [
            float(np.quantile(totals, 0.025)),
            float(np.quantile(totals, 0.975)),
        ],
        "probability_total_net_pnl_above_zero": float(np.mean(totals > 0.0)),
        "positive_trade_share": float(np.mean(values > 0.0)),
        "largest_winner_share_of_total_net_pnl": float(values.max() / values.sum()),
        "net_pnl_without_largest_winner": float(values.sum() - values.max()),
    }


def fallacy_scan(summary: dict[str, Any]) -> list[dict[str, str]]:
    decision_windows = summary["results"][DECISION_SCENARIO]["windows"]
    return [
        {
            "id": "simpsons_paradox",
            "status": "caution_temporal_heterogeneity",
            "finding": (
                f"The aggregate net result is positive, but the original holdout is "
                f"{decision_windows['original_holdout']['total_net_pnl']:+.2f} USDT while the "
                f"later frozen-forward segment is {decision_windows['frozen_forward']['total_net_pnl']:+.2f} USDT."
            ),
        },
        {
            "id": "ecological_fallacy",
            "status": "not_detected",
            "finding": "Claims remain at portfolio/trade level; no individual-asset inference is made from aggregates.",
        },
        {
            "id": "berksons_paradox",
            "status": "not_detected",
            "finding": "No association is inferred from conditioning on a common outcome, though asset selection is disclosed separately.",
        },
        {
            "id": "collider_bias",
            "status": "not_detected",
            "finding": "No regression control set or conditioned collider is used.",
        },
        {
            "id": "base_rate_neglect",
            "status": "addressed_with_gate",
            "finding": "The base rate is explicit: six entries in 263 days, below the registered 20-entry interpretation gate.",
        },
        {
            "id": "regression_to_the_mean",
            "status": "red_flag_for_mechanism_claim",
            "finding": "Entries are selected at extreme dominance readings, so subsequent normalisation is expected mechanically; only traded relative PnL can support usefulness.",
        },
        {
            "id": "survivorship_bias",
            "status": "caution",
            "finding": "The retrospectively selected ETH/BNB/SOL basket consists of continuously available liquid survivors; delisted or unavailable assets are not represented.",
        },
        {
            "id": "look_elsewhere_effect",
            "status": "red_flag_for_confirmatory_claim",
            "finding": "At least 24 adaptive candidate variants were screened across entry, basket, holding, relative-value and horizon stages.",
        },
        {
            "id": "garden_of_forking_paths",
            "status": "caution_but_audited",
            "finding": "The search was adaptive and retrospective, but every registered path and failure is retained in the experiment ledger.",
        },
        {
            "id": "correlation_not_causation",
            "status": "addressed_by_claim_scope",
            "finding": "Results are described as associations and cost-conditional performance, not as BTC dominance causing altcoin returns.",
        },
        {
            "id": "reverse_causality",
            "status": "caution",
            "finding": "The notional-share dominance proxy is endogenous to prices and volumes, so directional economic interpretation remains uncertain.",
        },
    ]


def build_validation() -> dict[str, Any]:
    summary, daily, trades = load_inputs()
    returns = daily_returns_from_cumulative(daily[DECISION_SCENARIO])
    metrics = summary["results"][DECISION_SCENARIO]["windows"]["continuous_post_selection"]
    paper_cost_references = {
        scenario: summary["results"][scenario]["windows"][
            "continuous_post_selection"
        ]
        for scenario in PAPER_SPREAD_SCENARIOS
    }
    cost_diagnostics = summary["results"][DECISION_SCENARIO]["cost_diagnostics"]
    scan = fallacy_scan(summary)
    return {
        "status": "caution_cost_conditional_candidate",
        "confidence": "CAUTION",
        "candidate": summary["sensitivity_spec"]["candidate_contract"],
        "decision_scenario": DECISION_SCENARIO,
        "observed_metrics": metrics,
        "paper_cost_references": paper_cost_references,
        "cost_break_even_one_way_bps": cost_diagnostics[
            "post_selection_break_even_one_way_cost_bps"
        ],
        "bootstrap": {
            "method": "seeded circular moving-block bootstrap of post-selection daily returns",
            "seven_day_blocks": circular_block_bootstrap(
                returns,
                block_length=7,
                replications=BOOTSTRAP_REPLICATIONS,
                seed=BOOTSTRAP_SEED,
            ),
            "fourteen_day_blocks": circular_block_bootstrap(
                returns,
                block_length=14,
                replications=BOOTSTRAP_REPLICATIONS,
                seed=BOOTSTRAP_SEED,
            ),
            "interpretation_limit": (
                "Intervals describe resampling uncertainty for this selected path only; they do "
                "not correct adaptive model selection or the six-event evidence limit."
            ),
        },
        "trade_concentration": trade_bootstrap(trades),
        "multiplicity": {
            "adaptive_candidate_variants_screened": 24,
            "correction": "No confirmatory multiplicity correction is claimed because the searches are dependent and adaptive.",
            "deflated_sharpe_ratio": "Not reported: six entries and adaptive path selection make a precise DSR misleading.",
        },
        "fallacy_scan": {
            "coverage": "11/11 checked",
            "items": scan,
        },
        "decision": {
            "descriptive_candidate": True,
            "validated_strategy": False,
            "deployment_ready": False,
            "reason": (
                "Positive at the registered 20 bps and monthly-corrected spread scenarios, but "
                "negative under the two-day-corrected spread and fixed 40 bps scenarios, "
                "concentrated in two trades, selected adaptively, and below the 20-entry "
                "prospective gate."
            ),
        },
    }


def render_markdown(validation: dict[str, Any]) -> str:
    metrics = validation["observed_metrics"]
    monthly = validation["paper_cost_references"][
        "abdi_ranaldo_monthly_corrected"
    ]
    two_day = validation["paper_cost_references"][
        "abdi_ranaldo_two_day_corrected"
    ]
    seven = validation["bootstrap"]["seven_day_blocks"]
    fourteen = validation["bootstrap"]["fourteen_day_blocks"]
    trades = validation["trade_concentration"]
    lines = [
        "# Strategy 2 candidate validation",
        "",
        "**Verdict: CAUTION — cost-conditional descriptive candidate, not a validated or deployable strategy.**",
        "",
        "The fixed candidate is 50% long an equal-weight ETH/BNB/SOL basket and 50% short BTC, entered after the original dominance-spike signal, held seven days at 1x total gross exposure. A 10% annual short-carry stress is charged.",
        "",
        "## Observed result",
        "",
        f"At the registered 20 bps one-way execution scenario, the 263-day combined post-selection result is {metrics['total_return']:+.2%} ({metrics['total_net_pnl']:+.2f} USDT net), Sharpe {metrics['sharpe']:.2f}, max drawdown {metrics['max_drawdown']:.2%}, with {metrics['trade_entries']} entries and {metrics['active_days']} active days.",
        "",
        f"With the paper's monthly correction, the lagged Abdi–Ranaldo estimate gives {monthly['total_net_pnl']:+.2f} USDT ({monthly['total_return']:+.2%}); with the paper's two-day correction it gives {two_day['total_net_pnl']:+.2f} USDT ({two_day['total_return']:+.2%}). The observed break-even fixed one-way charge is {validation['cost_break_even_one_way_bps']:.1f} bps.",
        "",
        "## Uncertainty and concentration",
        "",
        f"The 7-day circular block bootstrap gives a 95% total-return interval of [{seven['total_return_ci_95'][0]:+.2%}, {seven['total_return_ci_95'][1]:+.2%}] and P(return > 0) = {seven['probability_total_return_above_zero']:.1%}. With 14-day blocks the interval is [{fourteen['total_return_ci_95'][0]:+.2%}, {fourteen['total_return_ci_95'][1]:+.2%}] and P(return > 0) = {fourteen['probability_total_return_above_zero']:.1%}.",
        "",
        f"Only {trades['observed_trades']} trades exist; {trades['positive_trade_share']:.0%} are positive. The largest winner represents {trades['largest_winner_share_of_total_net_pnl']:.1%} of total net PnL, and removing it changes net PnL to {trades['net_pnl_without_largest_winner']:+.2f} USDT.",
        "",
        "These intervals do not correct the adaptive search. At least 24 candidate variants were screened, and a Deflated Sharpe Ratio is deliberately not reported because six entries do not support a stable correction.",
        "",
        "## Fallacy scan",
        "",
        "11/11 statistical fallacies checked. Material flags are: regression-to-the-mean risk in the extreme-dominance trigger; survivorship/availability bias in the selected liquid basket; strong look-elsewhere and forking-path effects; temporal heterogeneity (the original holdout is negative while the later frozen-forward segment is positive); and endogeneity of the notional-share proxy. No causal claim is made.",
        "",
        "## Evidence gate",
        "",
        "Do not upgrade the claim before at least 20 new entries from 5 August 2026 onward (30 preferred), using the fixed candidate and recording realised execution costs. The frozen published Strategy 2 remains unchanged.",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.json_output.exists() or args.markdown_output.exists():
        raise FileExistsError("Candidate validation output already exists.")
    validation = build_validation()
    args.json_output.write_text(
        json.dumps(validation, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(render_markdown(validation), encoding="utf-8")
    print("Strategy 2 candidate validation")
    print(f"  Confidence: {validation['confidence']}")
    print(f"  Fallacy scan: {validation['fallacy_scan']['coverage']}")
    print(f"  Validated strategy: {validation['decision']['validated_strategy']}")


if __name__ == "__main__":
    main()
