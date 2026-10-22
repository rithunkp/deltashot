from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


BUCKETS = ("identical", "minor", "moderate", "major", "unrelated")


@dataclass(frozen=True)
class ChangeStats:
    changed_pixel_ratio: float
    bucket: str


def compute_changed_pixel_ratio(reference_path: Path, target_path: Path, tolerance: int = 3) -> float:
    """Return the fraction of pixels whose RGB values changed beyond a small tolerance."""
    with Image.open(reference_path) as ref_image, Image.open(target_path) as target_image:
        ref = ref_image.convert("RGB")
        target = target_image.convert("RGB")
        if ref.size != target.size:
            target = target.resize(ref.size, Image.Resampling.BICUBIC)

        total_pixels = ref.size[0] * ref.size[1]
        changed_pixels = 0
        for ref_pixel, target_pixel in zip(ref.get_flattened_data(), target.get_flattened_data()):
            if any(abs(ref_channel - target_channel) > tolerance for ref_channel, target_channel in zip(ref_pixel, target_pixel)):
                changed_pixels += 1
        return changed_pixels / total_pixels


def bucket_from_ratio(changed_pixel_ratio: float, explicit_unrelated: bool = False) -> str:
    if explicit_unrelated or changed_pixel_ratio > 0.50:
        return "unrelated"
    if changed_pixel_ratio <= 0.0001:
        return "identical"
    if changed_pixel_ratio < 0.02:
        return "minor"
    if changed_pixel_ratio < 0.15:
        return "moderate"
    return "major"


def compute_change_stats(reference_path: Path, target_path: Path, explicit_unrelated: bool = False) -> ChangeStats:
    ratio = compute_changed_pixel_ratio(reference_path, target_path)
    return ChangeStats(changed_pixel_ratio=ratio, bucket=bucket_from_ratio(ratio, explicit_unrelated))
