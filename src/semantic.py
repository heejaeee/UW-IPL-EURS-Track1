PROMPTS={1: ['coral'], 2: ['fish'], 3: ['shark', 'stingray', 'manta ray'], 4: ['sea turtle', 'starfish', 'seahorse', 'jellyfish', 'octopus', 'crab']}
from pathlib import Path
import json, io, ast, types
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
def letterbox(im, size, mask=False):
    w, h = im.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w*scale)), max(1, round(h*scale))
    resample = Image.Resampling.NEAREST if mask else Image.Resampling.BICUBIC
    im = im.resize((nw, nh), resample)
    canvas = Image.new('L' if mask else 'RGB', (size, size), 255 if mask else (0,0,0))
    left, top = (size-nw)//2, (size-nh)//2
    canvas.paste(im, (left, top))
    return canvas, (left, top, nw, nh)

def rgb_tensor(im):
    return torch.from_numpy(np.array(im, dtype=np.float32).transpose(2,0,1).copy()) / 255
def project_scores(scores, box, target):
    left, top, width, height = box
    return F.interpolate(scores[:, top:top+height, left:left+width][None],
                         size=(target[1], target[0]), mode='bilinear', align_corners=False)[0]

class PromptScores:
    def __init__(self, checkpoint, device):
        self.device = device
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self.model = build_sam3_image_model(checkpoint_path=checkpoint, load_from_HF=False, device=self.device)
        self.model.eval().requires_grad_(False)
        self.processor = Sam3Processor(self.model, confidence_threshold=.3, device=self.device)
        self.text = {}
        with torch.inference_mode(), torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16, enabled=str(self.device).startswith('cuda')):
            for phrases in PROMPTS.values():
                for phrase in phrases:
                    self.text[phrase] = self.model.backbone.forward_text([phrase], device=self.device)

    @torch.inference_mode()
    def predict(self, image):
        with torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16, enabled=str(self.device).startswith('cuda')):
            state = self.processor.set_image(image)
            hard = torch.zeros(5, image.height, image.width, device=self.device)
            soft = torch.zeros_like(hard)
            hard[0].fill_(.3)
            soft[0].fill_(.3)
            for cls, phrases in PROMPTS.items():
                for phrase in phrases:
                    self.processor.reset_all_prompts(state)
                    state['backbone_out'].update(self.text[phrase])
                    state['geometric_prompt'] = self.model._get_dummy_prompt()
                    result = self.processor._forward_grounding(state)
                    if result['scores'].numel() == 0:
                        continue
                    probabilities = result['masks_logits'].reshape(-1, image.height, image.width).float()
                    confidence = result['scores'].reshape(-1, 1, 1).float()
                    hard[cls] = torch.maximum(hard[cls], ((probabilities > .5)*confidence).amax(0))
                    soft[cls] = torch.maximum(soft[cls], (probabilities*confidence).amax(0))
            assert torch.isfinite(hard).all() and torch.isfinite(soft).all()
        return hard, soft
def selective_fuse(original,restored,old,submitted,concept,confidence,margin,calibration):
    device=original.device
    weight=torch.tensor(calibration['original_weight'],device=device)[:,None,None]
    scale=torch.tensor(calibration['scale'],device=device)[:,None,None]
    bias=torch.tensor(calibration['bias'],device=device)[:,None,None]
    scores=restored.clone()
    foreground=weight*original[1:]+(1-weight)*scores[1:]
    scores[1:]=foreground*scale+(foreground>.01)*bias
    dual=scores.argmax(0).cpu().numpy().astype(np.uint8)
    candidate=old.copy()
    gate=np.isin(old,[0,2,3])&np.isin(dual,[0,2,3]);candidate[gate]=dual[gate]
    gate=np.isin(candidate,[0,1])&np.isin(concept,[0,1])&(confidence>=.5)&(margin>=.15)
    candidate[gate]=concept[gate]
    # The jointly held-out data cannot measure rare-class recall. Preserve all
    # submitted coral/elasmobranch/other pixels and do not introduce new ones.
    result=submitted.copy()
    gate=np.isin(submitted,[0,2])&np.isin(candidate,[0,2]);result[gate]=candidate[gate]
    for label in [1,3,4]:assert np.array_equal(result==label,submitted==label)
    return result
