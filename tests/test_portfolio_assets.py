from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_github_social_preview_meets_github_image_contract() -> None:
    preview = PROJECT_ROOT / "report" / "figures" / "github_social_preview.png"

    assert preview.stat().st_size < 1_000_000
    with Image.open(preview) as image:
        assert image.format == "PNG"
        assert image.size == (1280, 640)
        assert image.mode == "RGBA"
