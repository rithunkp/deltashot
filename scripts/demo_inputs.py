"""Input validation and fixed-canvas preparation for the local demo."""
from __future__ import annotations

import base64
import binascii
import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

WORKING_SIZE = (512, 512)
MAX_UPLOAD_BYTES = 24 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_777_216


class DemoInputError(ValueError):
    """A specific, user-correctable upload or request error."""


def prepare_working_image(image: Image.Image) -> Image.Image:
    """Fit the full image into a 512-square canvas, preserving aspect ratio."""
    rgb = image.convert("RGB")
    fitted = ImageOps.contain(rgb, WORKING_SIZE, method=Image.Resampling.LANCZOS)
    array = np.asarray(fitted)
    pad_x = WORKING_SIZE[0] - fitted.width
    pad_y = WORKING_SIZE[1] - fitted.height
    left, top = pad_x // 2, pad_y // 2
    padded = np.pad(
        array,
        ((top, pad_y - top), (left, pad_x - left), (0, 0)),
        mode="edge",
    )
    return Image.fromarray(padded, "RGB")


def load_upload(item, name: str):
    if not isinstance(item, dict) or not isinstance(item.get("data"), str):
        raise DemoInputError(f"{name} image upload is missing image data")
    try:
        raw = base64.b64decode(item["data"], validate=True)
    except (binascii.Error, ValueError) as error:
        raise DemoInputError(f"{name} image data is not valid base64") from error
    if not raw:
        raise DemoInputError(f"{name} image file is empty")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise DemoInputError(f"{name} image file exceeds the {MAX_UPLOAD_BYTES:,}-byte limit")
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.format not in {"PNG", "JPEG", "WEBP"}:
                raise DemoInputError(f"{name} must be a PNG, JPEG, or WebP image")
            if opened.width <= 0 or opened.height <= 0:
                raise DemoInputError(f"{name} image has invalid dimensions")
            if opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise DemoInputError(f"{name} image exceeds the {MAX_IMAGE_PIXELS:,}-pixel limit")
            opened.load()
            image = opened.convert("RGB")
    except DemoInputError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise DemoInputError(f"{name} is not a readable PNG, JPEG, or WebP image") from error
    return image, len(raw)
