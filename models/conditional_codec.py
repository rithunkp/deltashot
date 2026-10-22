"""V1 reference-conditioned hyperprior codec using simple feature concatenation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.alignment import ReferenceConcatV1
from models.reference_encoder import ReferenceFeatureExtractor


class ReferenceConditionedCodec(nn.Module):
    """ScaleHyperprior whose analysis, entropy scales, and synthesis use reference context.

    This is the architecture-only v1: no cross-attention and no mode selection.
    Its ``forward`` follows the CompressAI training interface and returns x_hat
    plus likelihoods for use with RateDistortionLoss.
    """

    def __init__(self, base_codec):
        super().__init__()
        self.base_codec = base_codec
        self.reference_encoder = ReferenceFeatureExtractor(base_codec.g_a)
        self.reference_to_latent = ReferenceConcatV1(base_codec.M, base_codec.M)
        self.analysis_fusion = nn.Conv2d(base_codec.M * 2, base_codec.M, kernel_size=1)
        self.entropy_fusion = nn.Sequential(
            nn.Conv2d(base_codec.M * 2, base_codec.M, kernel_size=1),
            nn.Softplus(),
        )
        self.synthesis_fusion = nn.Conv2d(base_codec.M * 2, base_codec.M, kernel_size=1)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Resize CompressAI's dynamic entropy buffers before loading wrapper weights."""
        from compressai.models.utils import update_registered_buffers

        update_registered_buffers(
            self.base_codec.entropy_bottleneck,
            "base_codec.entropy_bottleneck",
            ("_quantized_cdf", "_offset", "_cdf_length"),
            state_dict,
            policy="resize",
        )
        update_registered_buffers(
            self.base_codec.gaussian_conditional,
            "base_codec.gaussian_conditional",
            ("_quantized_cdf", "_offset", "_cdf_length", "scale_table"),
            state_dict,
            policy="resize",
        )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _features(self, reference, latent_size):
        pyramid = self.reference_encoder(reference)
        return self.reference_to_latent(pyramid, latent_size)

    def _entropy_scales(self, z_hat, reference_feature):
        scales = self.base_codec.h_s(z_hat)
        return self.entropy_fusion(torch.cat((scales, reference_feature), dim=1)).clamp_min(1e-6)

    def forward(self, reference, target):
        y_base = self.base_codec.g_a(target)
        reference_feature = self._features(reference, y_base.shape[-2:])
        y = self.analysis_fusion(torch.cat((y_base, reference_feature), dim=1))

        z = self.base_codec.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.base_codec.entropy_bottleneck(z)
        scales = self._entropy_scales(z_hat, reference_feature)
        y_hat, y_likelihoods = self.base_codec.gaussian_conditional(y, scales)

        decoder_input = self.synthesis_fusion(torch.cat((y_hat, reference_feature), dim=1))
        x_hat = self.base_codec.g_s(decoder_input)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def update(self, force=False):
        return self.base_codec.update(force=force)

    def compress(self, reference, target):
        y_base = self.base_codec.g_a(target)
        reference_feature = self._features(reference, y_base.shape[-2:])
        y = self.analysis_fusion(torch.cat((y_base, reference_feature), dim=1))
        z = self.base_codec.h_a(torch.abs(y))
        z_strings = self.base_codec.entropy_bottleneck.compress(z)
        z_hat = self.base_codec.entropy_bottleneck.decompress(z_strings, z.shape[-2:])
        scales = self._entropy_scales(z_hat, reference_feature)
        indexes = self.base_codec.gaussian_conditional.build_indexes(scales)
        y_strings = self.base_codec.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.shape[-2:]}

    def decompress(self, strings, shape, reference):
        return self.decompress_with_reference(strings, shape, reference)

    def decompress_with_reference(self, strings, shape, reference):
        z_hat = self.base_codec.entropy_bottleneck.decompress(strings[1], shape)
        reference_feature = self._features(reference, (z_hat.shape[-2] * 4, z_hat.shape[-1] * 4))
        scales = self._entropy_scales(z_hat, reference_feature)
        indexes = self.base_codec.gaussian_conditional.build_indexes(scales)
        y_hat = self.base_codec.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        decoder_input = self.synthesis_fusion(torch.cat((y_hat, reference_feature), dim=1))
        return {"x_hat": self.base_codec.g_s(decoder_input).clamp_(0, 1)}
