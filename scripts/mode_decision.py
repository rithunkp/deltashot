"""Actual-bitstream rate-distortion mode selection for the local demo."""
from __future__ import annotations

import numpy as np


def distortion_mse(original, reconstruction):
    first = np.asarray(original.convert("RGB"), dtype=np.float32) / 255.0
    second = np.asarray(reconstruction.convert("RGB"), dtype=np.float32) / 255.0
    return float(np.mean((first - second) ** 2))


def select_actual_mode(candidates, original, lambda_value, mode_overhead_bytes=5):
    """Return the minimum actual-bpp + lambda*MSE candidate and all costs.

    ``candidates`` maps codec names to ``(inner_file_bytes, reconstruction)``.
    The fixed framing overhead is included in every candidate's coded rate.
    """
    pixels = original.width * original.height
    if pixels <= 0 or not candidates:
        raise ValueError("mode selection requires candidates and a non-empty image")
    costs = {}
    for codec, (file_bytes, reconstruction) in candidates.items():
        if reconstruction.size != original.size:
            raise ValueError(f"{codec} reconstruction dimensions do not match original")
        actual_bpp = (file_bytes + mode_overhead_bytes) * 8 / pixels
        costs[codec] = actual_bpp + lambda_value * distortion_mse(original, reconstruction)
    return min(costs, key=costs.get), costs
