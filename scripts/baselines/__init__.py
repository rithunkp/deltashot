"""Baseline codec implementations for DeltaShot evaluation."""
from scripts.baselines.jpeg_webp import (
    CompressionResult,
    compress_jpeg,
    compress_webp,
    decompress_jpeg,
    decompress_webp,
    quality_sweep_jpeg,
    quality_sweep_webp,
)
from scripts.baselines.delta_png import (
    DeltaPngResult,
    compute_pixel_difference,
    compress_delta_png,
    decompress_delta_png,
)

__all__ = [
    "CompressionResult",
    "DeltaPngResult",
    "compress_delta_png",
    "compress_jpeg",
    "compress_webp",
    "compute_pixel_difference",
    "decompress_delta_png",
    "decompress_jpeg",
    "decompress_webp",
    "quality_sweep_jpeg",
    "quality_sweep_webp",
]
