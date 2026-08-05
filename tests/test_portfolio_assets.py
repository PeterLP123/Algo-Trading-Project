from pathlib import Path

from PIL import Image
import pytest

from report.build_readme_performance_figure import load_post_selection_series


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_github_social_preview_meets_github_image_contract() -> None:
    preview = PROJECT_ROOT / "report" / "figures" / "github_social_preview.png"

    assert preview.stat().st_size < 1_000_000
    with Image.open(preview) as image:
        assert image.format == "PNG"
        assert image.size == (1280, 640)
        assert image.mode == "RGBA"


def test_combined_readme_plot_is_a_reviewable_png() -> None:
    plot = PROJECT_ROOT / "report" / "figures" / "readme_combined_test_period.png"

    assert plot.stat().st_size < 1_000_000
    with Image.open(plot) as image:
        assert image.format == "PNG"
        assert image.width >= 1_600
        assert image.height >= 700


def test_combined_readme_plot_uses_the_post_selection_snapshot() -> None:
    snapshot = PROJECT_ROOT / "forward_validation" / "snapshots" / "2026-08-04"

    strategy1, strategy2 = load_post_selection_series(snapshot)

    assert len(strategy1) == len(strategy2) == 263
    assert strategy1.index.equals(strategy2.index)
    assert strategy1["cumulative_return"].iloc[-1] == pytest.approx(0.0954468178)
    assert strategy2["cumulative_return"].iloc[-1] == pytest.approx(-0.0696493925)
    assert strategy1.attrs["sharpe"] == pytest.approx(1.2514892736)
    assert strategy2.attrs["sharpe"] == pytest.approx(-1.5207033119)
    assert strategy1.attrs["spread_correction"] == "monthly_corrected"
    assert strategy2.attrs["spread_correction"] == "monthly_corrected"
    assert strategy2.attrs["evidence_gate"]["status"] == "below_minimum"
    assert strategy2.attrs["evidence_gate"]["observed_entries"] == 7
    assert strategy2.attrs["evidence_gate"]["minimum_entries"] == 20
