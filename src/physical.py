from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, io, ast, types
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
@dataclass
class FormationParameters:
    kd: torch.Tensor          # (10, 3)
    attenuation: torch.Tensor  # (10, 3), c
    binf: torch.Tensor        # (10, 20, 3), B∞ per water type and depth

    def to(self, device) -> FormationParameters:
        return FormationParameters(self.kd.to(device), self.attenuation.to(device), self.binf.to(device))

def load_parameters(path: Path) -> FormationParameters:
    stored = json.loads(Path(path).read_text())
    return FormationParameters(torch.tensor(stored["kd"], dtype=torch.float32),
                               torch.tensor(stored["attenuation"], dtype=torch.float32),
                               torch.tensor(stored["binf"], dtype=torch.float32))
@torch.inference_mode()
def refine(observed, restored, water, depth, parameters):
    device=parameters.kd.device
    x=torch.from_numpy(observed).permute(2,0,1).to(device)
    clean=torch.from_numpy(restored).permute(2,0,1).to(device)
    attenuation=parameters.attenuation[water-1,:,None,None]
    direct=torch.exp(-parameters.kd[water-1,:,None,None]*depth/10)
    background=parameters.binf[water-1,depth//5-1,:,None,None]
    z_values=torch.linspace(.1,1.,32,device=device)
    best_error=torch.full_like(x[0],float('inf'));distance=torch.full_like(x[0],.55)
    for z in z_values:
        transmission=torch.exp(-attenuation*z)
        rendered=background+(clean*direct-background)*transmission
        error=((rendered-x)**2).mean(0)
        better=error<best_error
        best_error=torch.where(better,error,best_error)
        distance=torch.where(better,z,distance)
    # Remove per-pixel noise in the estimated distance; maintain full-size RGB.
    distance=F.avg_pool2d(F.pad(distance[None,None],(2,2,2,2),mode='replicate'),5,stride=1)[0,0]
    transmission=torch.exp(-attenuation*distance)
    jacobian=direct*transmission
    rendered=background+(clean*direct-background)*transmission
    gain=(jacobian/(jacobian.square()+.05**2)).clamp(max=4.)
    correction=(gain*(x-rendered)).clamp(-.1,.1)
    # Low-information channels cannot support reliable inversion.
    reliability=((jacobian-.05)/.15).clamp(0,1)
    result=(clean+.5*reliability*correction).clamp(0,1)
    assert torch.isfinite(result).all()
    return result.permute(1,2,0).cpu().numpy()
