"""No-Reference Codec baseline (Phase 3).

For training from scratch we use the architecture class directly from
compressai.models rather than the zoo helpers (which are for pretrained weights).
The ScaleHyperprior (Ballé et al. 2018 hyperprior) is the standard starting
point recommended in Section 4.2 step 2.
"""
from compressai.models import ScaleHyperprior


# Channel widths from the original Ballé 2018 paper.
# N=128, M=192 is the standard "medium" config used throughout CompressAI examples.
_QUALITY_PARAMS = {
    1: dict(N=128, M=192),
    2: dict(N=128, M=192),
    3: dict(N=128, M=192),
    4: dict(N=128, M=192),
    5: dict(N=128, M=192),
    6: dict(N=192, M=320),
    7: dict(N=192, M=320),
    8: dict(N=192, M=320),
}


def get_baseline_codec(quality: int = 3):
    """Return an untrained ScaleHyperprior model for training from scratch.

    Args:
        quality: Integer 1-8. Higher quality = larger model (more channels).
                 For the smoke-test run, quality=3 (N=128, M=192) is fine.
    Returns:
        A compressai.models.ScaleHyperprior instance with random weights.
    """
    params = _QUALITY_PARAMS.get(quality, dict(N=128, M=192))
    return ScaleHyperprior(**params)

