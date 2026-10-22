"""Image-quality metrics for DeltaShot evaluation."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image


def psnr(reference: Image.Image, target: Image.Image) -> float:
    """Compute PSNR in dB between two PIL images (assumed same size after resize)."""
    ref = np.asarray(reference.convert("RGB"), dtype=np.float64)
    tgt = np.asarray(target.convert("RGB"), dtype=np.float64)
    if ref.shape != tgt.shape:
        target_resized = target.convert("RGB").resize(
            (ref.shape[1], ref.shape[0]), Image.Resampling.BICUBIC
        )
        tgt = np.asarray(target_resized, dtype=np.float64)

    mse = float(np.mean((ref - tgt) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(255.0 ** 2 / mse)


def bits_per_pixel(
    compressed_size_bytes: int, image_width: int, image_height: int, num_channels: int = 3
) -> float:
    """Convert a compressed byte count into bits per pixel."""
    total_pixels = image_width * image_height
    return (compressed_size_bytes * 8) / total_pixels


def bd_rate(
    rate_psnr_a: list[tuple[float, float]],
    rate_psnr_b: list[tuple[float, float]],
) -> float | None:
    """Compute the BD-rate of curve B relative to curve A (Bjontegaard delta rate).

    Both inputs are lists of (bits_per_pixel, psnr_db) points. A positive BD-rate
    means B is *worse* than A (needs more bits to reach the same quality).
    """
    if len(rate_psnr_a) < 3 or len(rate_psnr_b) < 3:
        return None

    log_r1 = [math.log(r) for r, _ in rate_psnr_a]
    log_r2 = [math.log(r) for r, _ in rate_psnr_b]
    p1 = [p for _, p in rate_psnr_a]
    p2 = [p for _, p in rate_psnr_b]

    p_low = max(min(p1), min(p2))
    p_high = min(max(p1), max(p2))
    if p_high <= p_low:
        return None

    poly1 = np.polyfit(p1, log_r1, 3)
    poly2 = np.polyfit(p2, log_r2, 3)

    p1_int = np.polyint(poly1)
    p2_int = np.polyint(poly2)

    int1 = np.polyval(p1_int, p_high) - np.polyval(p1_int, p_low)
    int2 = np.polyval(p2_int, p_high) - np.polyval(p2_int, p_low)

    avg1 = int1 / (p_high - p_low)
    avg2 = int2 / (p_high - p_low)

    return (math.exp(avg2 - avg1) - 1.0) * 100.0
