import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("compressai")
pytest.importorskip("pytorch_msssim")

from models.baseline_codec import get_baseline_codec
from models.conditional_codec import ReferenceConditionedCodec
from scripts.evaluate_codecs import restore


def _round_trip(model, inputs, checkpoint_path):
    model.eval()
    model.update(force=True)
    with torch.inference_mode():
        expected = model(*inputs)["x_hat"]
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    fresh = restore(type(model)(get_baseline_codec(quality=3))
                    if isinstance(model, ReferenceConditionedCodec)
                    else get_baseline_codec(quality=3), checkpoint_path, "cpu")
    with torch.inference_mode():
        actual = fresh(*inputs)["x_hat"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_no_reference_checkpoint_round_trip(tmp_path):
    target = torch.rand(1, 3, 64, 64)
    _round_trip(get_baseline_codec(quality=3), (target,), tmp_path / "noref.pth")


def test_reference_checkpoint_round_trip_resizes_compressai_buffers(tmp_path):
    model = ReferenceConditionedCodec(get_baseline_codec(quality=3))
    reference = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    _round_trip(model, (reference, target), tmp_path / "reference.pth")
