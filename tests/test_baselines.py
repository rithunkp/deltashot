"""Tests for the classical-codec baselines (JPEG, WebP, delta+PNG)."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from scripts.baselines.delta_png import (
    compress_delta_png,
    compute_pixel_difference,
    decompress_delta_png,
)
from scripts.baselines.jpeg_webp import (
    compress_jpeg,
    compress_webp,
    decompress_jpeg,
    decompress_webp,
)
from scripts.baselines.metrics import bd_rate, bits_per_pixel, psnr


def _synth_image(seed: int, size: tuple[int, int] = (64, 64)) -> Image.Image:
    """Deterministic RGB image with simple gradient and a few blocks."""
    rng = np.random.default_rng(seed)
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for y in range(size[1]):
        for x in range(size[0]):
            arr[y, x] = (x * 4 % 256, y * 4 % 256, (x + y) * 2 % 256)
    arr += rng.integers(0, 16, arr.shape, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_psnr_identical_is_infinite() -> None:
    img = _synth_image(1)
    assert psnr(img, img) == float("inf")


def test_psnr_different_is_finite() -> None:
    a = _synth_image(1)
    b = _synth_image(2)
    p = psnr(a, b)
    assert 0.0 < p < 100.0


def test_bits_per_pixel_basic() -> None:
    bpp = bits_per_pixel(1000, 64, 64)
    assert abs(bpp - 1000 * 8 / (64 * 64)) < 1e-9


def test_bd_rate_positive_when_b_needs_more_bits() -> None:
    a = [(0.1, 30.0), (0.2, 33.0), (0.3, 35.0), (0.4, 37.0)]
    # Curve B needs 1.5x the bits at every PSNR point.
    b = [(0.15, 30.0), (0.3, 33.0), (0.45, 35.0), (0.6, 37.0)]
    rate = bd_rate(a, b)
    assert rate is not None
    assert rate > 30.0  # ~50% more bits => ~40% BD-rate


def test_jpeg_round_trip_decompresses() -> None:
    img = _synth_image(3, (96, 96))
    comp = compress_jpeg(img, 75)
    recon = decompress_jpeg(comp.buffer)
    assert recon.size == img.size
    p = psnr(img, recon.resize(img.size, Image.Resampling.BICUBIC))
    assert 20.0 < p < 60.0


def test_webp_round_trip_decompresses() -> None:
    img = _synth_image(4, (96, 96))
    comp = compress_webp(img, 75)
    recon = decompress_webp(comp.buffer)
    assert recon.size == img.size
    p = psnr(img, recon.resize(img.size, Image.Resampling.BICUBIC))
    assert 20.0 < p < 60.0


def test_delta_png_identical_pair_is_small() -> None:
    a = _synth_image(5, (96, 96))
    result = compress_delta_png(a, a)
    # Re-decode round trip — original == reconstructed by construction.
    recon = decompress_delta_png(a, result)
    assert psnr(a, recon) == float("inf")
    # Identical pair: low byte = 128 everywhere, PNG compresses that very well.
    raw_png_size = 96 * 96 * 3  # worst case: no compression.
    assert result.delta_size_bytes < raw_png_size // 4


def test_delta_png_fully_changed_pair_is_larger() -> None:
    a = _synth_image(6, (96, 96))
    b = _synth_image(7, (96, 96))
    res_a = compress_delta_png(a, a)
    res_b = compress_delta_png(a, b)
    assert res_b.delta_size_bytes > res_a.delta_size_bytes


def test_delta_png_round_trip_lossless() -> None:
    a = _synth_image(8, (96, 96))
    b = _synth_image(9, (96, 96))
    res = compress_delta_png(a, b)
    recon = decompress_delta_png(a, res)
    # The split into 8-bit low + 8-bit residual is exact — no quantization.
    assert psnr(b, recon.resize(b.size, Image.Resampling.BICUBIC)) == float("inf")


def test_compute_pixel_difference_shape() -> None:
    a = _synth_image(10, (32, 32))
    b = _synth_image(11, (32, 32))
    diff = compute_pixel_difference(a, b)
    assert diff.shape == (32, 32, 3)
    assert diff.dtype == np.int16
