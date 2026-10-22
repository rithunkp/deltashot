from __future__ import annotations

from pathlib import Path
from random import Random

from PIL import Image

from data.bucketing import bucket_from_ratio, compute_changed_pixel_ratio
from data.mutations import MUTATION_OPERATORS, HtmlPage, mutate_replace


def sample_page() -> HtmlPage:
    return HtmlPage(
        page_id="sample",
        html="<html><body><main><h1>Hello</h1><p style='color:#111827'>Body text</p><button>Go</button></main></body></html>",
    )


def test_mutation_operators_change_html_when_possible() -> None:
    page = sample_page()
    rng = Random(7)

    for name, operator in MUTATION_OPERATORS.items():
        if name == "mutate_replace":
            continue
        result = operator(page, rng)
        assert result.mutation_type == name
        assert result.page.html != page.html
        assert result.details["changed"] is True


def test_mutate_replace_uses_different_page() -> None:
    page = sample_page()
    replacement = HtmlPage(page_id="replacement", html="<html><body><h1>Other</h1></body></html>")

    result = mutate_replace(page, Random(1), [page, replacement])

    assert result.page.page_id == "replacement"
    assert result.details["explicit_unrelated"] is True


def test_bucket_thresholds_match_project_spec() -> None:
    assert bucket_from_ratio(0.0) == "identical"
    assert bucket_from_ratio(0.01) == "minor"
    assert bucket_from_ratio(0.10) == "moderate"
    assert bucket_from_ratio(0.30) == "major"
    assert bucket_from_ratio(0.51) == "unrelated"
    assert bucket_from_ratio(0.01, explicit_unrelated=True) == "unrelated"


def test_pixel_difference_ratio(tmp_path: Path) -> None:
    ref = Image.new("RGB", (10, 10), "white")
    target = Image.new("RGB", (10, 10), "white")
    target.putpixel((0, 0), (0, 0, 0))
    ref_path = tmp_path / "ref.png"
    target_path = tmp_path / "target.png"
    ref.save(ref_path)
    target.save(target_path)

    assert compute_changed_pixel_ratio(ref_path, target_path) == 0.01
