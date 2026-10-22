"""JPEG and WebP compression wrappers for DeltaShot baselines."""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class CompressionResult:
    """Result of compressing an image to a buffer."""
    buffer: bytes
    format: str  # 'JPEG' or 'WEBP'
    quality: int
    original_size: tuple[int, int]
    compressed_size_bytes: int


def compress_jpeg(image: Image.Image, quality: int, optimize: bool = True) -> CompressionResult:
    """Compress a PIL image as JPEG."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=optimize)
    buf.seek(0)
    data = buf.read()
    return CompressionResult(
        buffer=data,
        format="JPEG",
        quality=quality,
        original_size=image.size,
        compressed_size_bytes=len(data),
    )


def compress_webp(image: Image.Image, quality: int, lossless: bool = False) -> CompressionResult:
    """Compress a PIL image as WebP."""
    buf = io.BytesIO()
    image.save(buf, format="WEBP", quality=quality, lossless=lossless)
    buf.seek(0)
    data = buf.read()
    return CompressionResult(
        buffer=data,
        format="WEBP",
        quality=quality,
        original_size=image.size,
        compressed_size_bytes=len(data),
    )


def decompress_jpeg(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def decompress_webp(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def quality_sweep_jpeg(
    image: Image.Image,
    qualities: list[int] | None = None,
) -> list[CompressionResult]:
    """Run a quality sweep for JPEG. Returns results sorted by quality ascending."""
    if qualities is None:
        qualities = list(range(1, 101))
    return sorted([compress_jpeg(image, q) for q in qualities], key=lambda r: r.quality)


def quality_sweep_webp(
    image: Image.Image,
    qualities: list[int] | None = None,
) -> list[CompressionResult]:
    """Run a quality sweep for WebP. Returns results sorted by quality ascending."""
    if qualities is None:
        qualities = list(range(1, 101))
    return sorted([compress_webp(image, q) for q in qualities], key=lambda r: r.quality)
