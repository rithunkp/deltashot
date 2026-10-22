import base64
import io

import pytest
from PIL import Image

from scripts.demo_inputs import DemoInputError, load_upload, prepare_working_image


def test_prepare_working_image_preserves_aspect_ratio_and_pads_to_square():
    image = Image.new("RGB", (1280, 800), (30, 80, 140))
    working = prepare_working_image(image)
    assert working.size == (512, 512)
    assert working.getpixel((256, 256)) == (30, 80, 140)


def test_1280x800_target_uses_512_square_working_canvas():
    working = prepare_working_image(Image.new("RGB", (1280, 800), (8, 24, 40)))
    assert working.size == (512, 512)
    assert working.getpixel((256, 256)) == (8, 24, 40)


def test_load_upload_accepts_large_non_square_png_and_converts_rgb():
    image = Image.new("RGBA", (1280, 800), (20, 40, 60, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    decoded, byte_count = load_upload({"data": base64.b64encode(buffer.getvalue()).decode()}, "target")
    assert decoded.mode == "RGB"
    assert decoded.size == (1280, 800)
    assert byte_count == len(buffer.getvalue())


@pytest.mark.parametrize(
    "data, message",
    [
        ("", "target image file is empty"),
        (base64.b64encode(b"not an image").decode(),
         "target is not a readable PNG, JPEG, or WebP image"),
    ],
)
def test_load_upload_returns_specific_errors(data, message):
    with pytest.raises(DemoInputError, match=message):
        load_upload({"data": data}, "target")
