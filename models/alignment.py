"""Reference alignment interfaces. V1 intentionally performs no spatial alignment."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReferenceConcatV1(nn.Module):
    """Resize the deepest reference feature to latent resolution and concatenate it."""

    def __init__(self, reference_channels, latent_channels):
        super().__init__()
        self.project = nn.Conv2d(reference_channels, latent_channels, kernel_size=1)

    def forward(self, features, spatial_size):
        reference = features[-1]
        reference = F.interpolate(reference, size=spatial_size, mode="bilinear", align_corners=False)
        return self.project(reference)
