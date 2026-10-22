from PIL import Image

from scripts.mode_decision import select_actual_mode


def test_rate_distortion_selector_can_trade_bytes_for_quality():
    original = Image.new("RGB", (64, 64), (80, 120, 160))
    exact = Image.new("RGB", original.size, (80, 120, 160))
    degraded = Image.new("RGB", original.size, (0, 0, 0))

    selected, costs = select_actual_mode(
        {"small": (1_000, degraded), "better": (1_001, exact)},
        original,
        lambda_value=0.013,
    )

    assert selected == "better"
    assert costs["better"] < costs["small"]


def test_rate_distortion_selector_uses_actual_size_when_quality_matches():
    original = Image.new("RGB", (64, 64), (80, 120, 160))
    same = Image.new("RGB", original.size, (80, 120, 160))

    selected, _ = select_actual_mode(
        {"small": (900, same), "large": (1_000, same)}, original, 0.013,
    )

    assert selected == "small"
