import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("compressai")

from data.dataset import ScreenshotDataset
from models.baseline_codec import get_baseline_codec
from models.conditional_codec import ReferenceConditionedCodec
from models.reference_encoder import ReferenceFeatureExtractor


def test_reference_features_share_analysis_and_have_three_scales():
    baseline = get_baseline_codec(quality=3)
    extractor = ReferenceFeatureExtractor(baseline.g_a)
    reference = torch.rand(2, 3, 64, 64)

    features = extractor(reference)

    assert [tuple(feature.shape[-2:]) for feature in features] == [(16, 16), (8, 8), (4, 4)]
    assert extractor._analysis_transform_ref() is baseline.g_a


def test_conditional_codec_forward_and_gradient():
    model = ReferenceConditionedCodec(get_baseline_codec(quality=3))
    reference = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)

    output = model(reference, target)

    assert output["x_hat"].shape == target.shape
    assert set(output["likelihoods"]) == {"y", "z"}
    loss = (output["x_hat"] - target).square().mean() - sum(
        likelihood.log().mean() for likelihood in output["likelihoods"].values()
    )
    loss.backward()
    assert model.analysis_fusion.weight.grad is not None
    assert model.entropy_fusion[0].weight.grad is not None


def test_reference_dataset_tiny_balanced_slice():
    dataset = ScreenshotDataset(
        "data/synthetic_pairs_full",
        split="train",
        patch_size=64,
        no_reference=False,
        max_pairs_per_bucket=5,
    )
    assert len(dataset) == 25
    for index in range(len(dataset)):
        reference, target = dataset[index]
        assert reference.shape == target.shape == (3, 64, 64)
