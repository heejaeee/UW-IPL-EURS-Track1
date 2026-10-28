"""Restormer conditioned on the degradation labels (plan 01_restoration/05; user decision 2026-10-16).

Track 1 predictors may read the water type and depth from the filename. This wrapper feeds them to the vendored
Restormer (src/models/restormer.py, MIT, not edited) as constant maps beside the RGB image at the first
convolution; the network has to learn the per-pixel distance itself. The global residual stays on the RGB image.
The label weights of the widened first convolution start at zero (`widen_first_convolution`), so the wrapper with
the released weights computes exactly what Restormer computes on the image alone.
"""
from __future__ import annotations

import torch
from torch import nn

from .e034_arch import Restormer

RGB_CHANNELS = 3
FIRST_CONVOLUTION = "patch_embed.proj.weight"


def widen_first_convolution(key: str, value: torch.Tensor, label_channels: int) -> torch.Tensor:
    """A released Restormer weight for the wrapper: the first convolution gets zero weights for the label maps."""
    if key == FIRST_CONVOLUTION:
        zeros = value.new_zeros(value.shape[0], label_channels, *value.shape[2:])
        return torch.cat([value, zeros], dim=1)
    return value


class LabelConditionedRestormer(nn.Module):
    def __init__(self, label_channels: int, **restormer_options):
        super().__init__()
        self.label_channels = label_channels
        self.restormer = Restormer(**restormer_options)
        projection = self.restormer.patch_embed.proj
        self.restormer.patch_embed.proj = nn.Conv2d(RGB_CHANNELS + label_channels, projection.out_channels,
                                                    kernel_size=3, stride=1, padding=1,
                                                    bias=projection.bias is not None)

    def load_released(self, state: dict) -> None:
        """Load a released (RGB-only) Restormer state dict, label weights at zero."""
        self.restormer.load_state_dict({key: widen_first_convolution(key, value, self.label_channels)
                                        for key, value in state.items()})

    def forward(self, image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """image (B, 3, H, W), labels (B, label_channels): Restormer.forward (single-image branch) with the labels
        as constant maps beside the image at the first convolution."""
        net = self.restormer
        maps = labels.to(image.dtype)[:, :, None, None].expand(-1, -1, *image.shape[-2:])
        embedded = net.patch_embed(torch.cat([image, maps], dim=1))
        encoded1 = net.encoder_level1(embedded)
        encoded2 = net.encoder_level2(net.down1_2(encoded1))
        encoded3 = net.encoder_level3(net.down2_3(encoded2))
        latent = net.latent(net.down3_4(encoded3))
        decoded3 = net.decoder_level3(net.reduce_chan_level3(torch.cat([net.up4_3(latent), encoded3], 1)))
        decoded2 = net.decoder_level2(net.reduce_chan_level2(torch.cat([net.up3_2(decoded3), encoded2], 1)))
        decoded1 = net.decoder_level1(torch.cat([net.up2_1(decoded2), encoded1], 1))
        return net.output(net.refinement(decoded1)) + image
