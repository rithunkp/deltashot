import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("compressai")

from models.baseline_codec import get_baseline_codec
from models.conditional_codec import ReferenceConditionedCodec
from scripts.local_demo import (
    _model_decompress,
    encode_neural_file,
    image_tensor,
    padded_dimensions,
    psnr_db,
)
from scripts.bitstream import read_neural_bitstream
from PIL import Image


@pytest.mark.parametrize("conditional", [False, True], ids=["noref", "reference-conditioned"])
def test_compressai_round_trip_decodes_from_saved_bin(tmp_path, conditional):
    torch.manual_seed(17)
    image = Image.new("RGB", (64, 64), (78, 142, 203))
    target = image_tensor(image, padded_dimensions(*image.size))
    model = get_baseline_codec(quality=3)
    reference = None
    codec = "noref"
    if conditional:
        reference = image_tensor(Image.new("RGB", image.size, (180, 80, 40)), (64, 64))
        model = ReferenceConditionedCodec(model)
        codec = "reference_conditioned"
    model.eval()
    model.update(force=True)

    path = tmp_path / f"{codec}.bin"
    encoded_path, estimate = encode_neural_file(
        model, target, path, image.size, codec, reference,
    )
    assert encoded_path.is_file() and encoded_path.stat().st_size > 0
    container = read_neural_bitstream(path)
    decoded = _model_decompress(model, container, reference)

    assert decoded.size == image.size
    assert math.isfinite(psnr_db(image, decoded))
    assert math.isfinite(estimate) and estimate >= 0
    assert container["codec"] == codec
    assert container["padded_size"] == (64, 64)
