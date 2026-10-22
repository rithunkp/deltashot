"""Naive delta+PNG baseline for DeltaShot.

Computes the per-channel unsigned difference between the reference and target,
shifts the result so 0 maps to 128 (centered), then compresses the resulting
16-bit-ish buffer with PNG. Decompression reverses the shift and adds it back
to the reference to reconstruct the target.

This is intentionally the dumbest possible reference-conditioning baseline
so we can show, in the report, that naive pixel-difference coding is not
sufficient — which is the same point DCVC makes about residual coding vs.
conditional coding.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DeltaPngResult:
    """Result of compressing a (reference, target) pair as a delta+PNG bitstream."""
    png_buffer: bytes
    reference_size: tuple[int, int]
    delta_size_bytes: int
    target_shape: tuple[int, int, int]


def compute_pixel_difference(reference: Image.Image, target: Image.Image) -> np.ndarray:
    """Compute the per-channel signed difference reference - target as int16.

    Both images are converted to RGB and resized to match the reference.
    """
    ref = np.asarray(reference.convert("RGB"), dtype=np.int16)
    tgt = np.asarray(target.convert("RGB"), dtype=np.int16)
    if ref.shape != tgt.shape:
        target_resized = target.convert("RGB").resize(
            (ref.shape[1], ref.shape[0]), Image.Resampling.BICUBIC
        )
        tgt = np.asarray(target_resized, dtype=np.int16)
    return ref - tgt


def _delta_to_uint8(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a signed delta into its low 8 bits and a residual (high bits).

    PNG is an 8-bit-per-channel format, but a full signed diff can be in
    [-255, 255]. We store the low 8 bits and a small residual image that
    represents the remaining carry. This keeps the encoding lossless.
    """
    shifted = delta + 128
    low = np.mod(shifted, 256)
    residual = np.floor_divide(shifted, 256)
    return low.astype(np.uint8), residual.astype(np.int8)


def _uint8_to_delta(low: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Inverse of _delta_to_uint8."""
    return low.astype(np.int16) + 256 * residual.astype(np.int16) - 128


def compress_delta_png(reference: Image.Image, target: Image.Image) -> DeltaPngResult:
    """Encode a target as a delta relative to a reference, using PNG for the delta."""
    delta = compute_pixel_difference(reference, target)
    low, residual = _delta_to_uint8(delta)

    low_img = Image.fromarray(low, mode="RGB")
    residual_img = Image.fromarray((residual.astype(np.int16) + 128).astype(np.uint8), mode="RGB")

    buf = io.BytesIO()
    low_img.save(buf, format="PNG", optimize=True)
    low_bytes = buf.getvalue()

    buf2 = io.BytesIO()
    residual_img.save(buf2, format="PNG", optimize=True)
    residual_bytes = buf2.getvalue()

    # Pack both PNGs into one buffer (length-prefixed).
    out = io.BytesIO()
    out.write(len(low_bytes).to_bytes(4, "little"))
    out.write(len(residual_bytes).to_bytes(4, "little"))
    out.write(low_bytes)
    out.write(residual_bytes)

    return DeltaPngResult(
        png_buffer=out.getvalue(),
        reference_size=reference.size,
        delta_size_bytes=len(out.getvalue()),
        target_shape=delta.shape,
    )


def decompress_delta_png(
    reference: Image.Image, encoded: DeltaPngResult | bytes
) -> Image.Image:
    """Decode a delta buffer and add it back to the reference to reconstruct the target."""
    if isinstance(encoded, DeltaPngResult):
        data = encoded.png_buffer
    else:
        data = encoded

    low_len = int.from_bytes(data[:4], "little")
    residual_len = int.from_bytes(data[4:8], "little")
    low_bytes = data[8:8 + low_len]
    residual_bytes = data[8 + low_len:8 + low_len + residual_len]

    low_img = Image.open(io.BytesIO(low_bytes)).convert("RGB")
    residual_img = Image.open(io.BytesIO(residual_bytes)).convert("RGB")

    low = np.asarray(low_img, dtype=np.uint8)
    residual = np.asarray(residual_img, dtype=np.uint8) - 128
    delta = _uint8_to_delta(low, residual)

    ref = np.asarray(reference.convert("RGB"), dtype=np.int16)
    target = ref - delta
    target = np.clip(target, 0, 255).astype(np.uint8)
    return Image.fromarray(target, mode="RGB")
