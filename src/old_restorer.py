"""Small EURS Restormer variant using the pinned MIT-licensed official layers."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .old_arch import Restormer


class RestormerRestorer(Restormer):
    # Deliberately smaller than the published dim48/[4,6,6,8] configuration.
    channels = (24,48,96,192)

    def __init__(self):
        super().__init__(dim=24, num_blocks=[1,2,2,4], num_refinement_blocks=2,
                         heads=[1,2,4,8], ffn_expansion_factor=2.66)
        nn.init.zeros_(self.output.weight)

    def forward(self, x):
        def run(block, value):
            # Recompute activations to retain FP32 and batch4 on a 16GB GPU.
            if self.training and torch.is_grad_enabled():
                return checkpoint(block, value, use_reentrant=False, preserve_rng_state=False)
            return block(value)
        f1 = run(self.encoder_level1, self.patch_embed(x))
        f2 = run(self.encoder_level2, self.down1_2(f1))
        f3 = run(self.encoder_level3, self.down2_3(f2))
        f4 = run(self.latent, self.down3_4(f3))
        z = run(self.decoder_level3, self.reduce_chan_level3(torch.cat([self.up4_3(f4),f3],1)))
        z = run(self.decoder_level2, self.reduce_chan_level2(torch.cat([self.up3_2(z),f2],1)))
        z = run(self.refinement, run(self.decoder_level1, torch.cat([self.up2_1(z),f1],1)))
        return (self.output(z)+x).clamp(0,1), (f1,f2,f3,f4)
