"""Multi-scale reference features sharing the No-Reference analysis transform."""
import weakref

import torch.nn as nn


class ReferenceFeatureExtractor(nn.Module):
    """Return reference features at 1/4, 1/8, and 1/16 input resolution.

    The first seven modules of CompressAI's ScaleHyperprior analysis transform
    are reused by reference, so the feature extractor adds no duplicate trunk.
    """

    def __init__(self, analysis_transform):
        super().__init__()
        layers = list(analysis_transform.children())
        if len(layers) < 7:
            raise ValueError("analysis_transform must expose the 7-layer ScaleHyperprior g_a")
        object.__setattr__(self, "_analysis_transform_ref", weakref.ref(analysis_transform))

    def forward(self, reference):
        analysis = self._analysis_transform_ref()
        if analysis is None:
            raise RuntimeError("The shared analysis transform no longer exists")

        x = analysis[0](reference)
        x = analysis[1](x)
        x = analysis[2](x)
        feature_4 = analysis[3](x)
        x = analysis[4](feature_4)
        feature_8 = analysis[5](x)
        feature_16 = analysis[6](feature_8)
        return feature_4, feature_8, feature_16
