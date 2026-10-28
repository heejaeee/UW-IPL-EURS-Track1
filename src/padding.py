"""Pad images to a size a network accepts, and crop its output back.

Some networks downsample by fixed factors (WWE-UIE: PixelUnshuffle(2) twice, so H and W must be
multiples of 4). Evaluation sizes such as 1024x683 or 1024x518 are not, so the input is reflect-padded
at the bottom and right, and the output is cropped back to the original size.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def padding_needed(length: int, multiple: int) -> int:
    """How many pixels make `length` a multiple of `multiple`."""
    return (-length) % multiple


def pad_to_multiple(images: torch.Tensor, multiple: int) -> torch.Tensor:
    """Reflect-pad the bottom and right of N x C x H x W `images` to multiples of `multiple`."""
    height, width = images.shape[-2:]
    return F.pad(images, (0, padding_needed(width, multiple), 0, padding_needed(height, multiple)), mode="reflect")


def crop_to(images: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """The top-left height x width region (undoes pad_to_multiple)."""
    return images[..., :height, :width]
