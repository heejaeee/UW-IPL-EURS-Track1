from pathlib import Path
import json, io, ast, types
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
ARMS=('original16','restored16','consensus16','consensus64')
def fallback(base):
    legal=base<3
    if not legal.any():return np.zeros_like(base),False
    _,labels=cv2.distanceTransformWithLabels((~legal).astype(np.uint8),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
    table=np.zeros(int(labels.max())+1,np.uint8);table[labels[legal]]=base[legal]
    result=table[labels]
    assert np.array_equal(result[legal],base[legal]) and result.max()<3
    return result,True

def fuse(base,original,restored,p):
    assert base.shape==original.shape==restored.shape
    seed=(base==4).astype(np.uint8)
    if not seed.any():return {a:base.copy() for a in ARMS}
    di=cv2.distanceTransform(seed,cv2.DIST_L2,5);do=cv2.distanceTransform(1-seed,cv2.DIST_L2,5)
    repl,can_delete=fallback(base);results={}
    for arm in ARMS:
        radius=max(1,round((64 if arm=='consensus64' else 16)*max(base.shape)/1024))
        band=((seed==1)&(di<=radius))|((seed==0)&(do<=radius))
        if arm=='original16':positive=original>=p['thresholds']['add'];negative=original<=p['thresholds']['remove']
        elif arm=='restored16':positive=restored>=p['thresholds']['add'];negative=restored<=p['thresholds']['remove']
        else:positive=np.minimum(original,restored)>=p['thresholds']['add'];negative=np.maximum(original,restored)<=p['thresholds']['remove']
        pred=base.copy();pred[band&(base<3)&positive]=4
        if can_delete:pred[band&(base==4)&negative]=repl[band&(base==4)&negative]
        assert np.array_equal(pred==3,base==3)
        assert np.array_equal(pred[~band],base[~band])
        assert not np.any((pred!=base)&(base!=4)&(pred!=4))
        results[arm]=pred
    return results
def gate(base,proposal,p):
    out=base.copy()
    for cls in (3,4):
        seed=base==cls
        if not seed.any():continue
        allowed=(proposal==cls)&(base<3)
        n,labels,stats,_=cv2.connectedComponentsWithStats((allowed|seed).astype(np.uint8),connectivity=8)
        seed_counts=np.bincount(labels[seed],minlength=n)
        good=(seed_counts>0)&(stats[:,cv2.CC_STAT_AREA]<=p['max_area_ratio']*seed_counts)
        good[0]=False
        out[allowed&good[labels]]=cls
    assert np.array_equal(out[base>=3],base[base>=3])
    assert not np.any((out!=base)&(out<3))
    assert not np.any((out==1)&(base!=1))
    return out

def refine_boundary(base, original, restored, p):
    refined = fuse(base, original, restored, p)['consensus16']
    additions = (refined == 4) & (base != 4)
    output = gate(base, np.where(additions,4,0).astype(np.uint8), p)
    output[(base == 4) & (refined == 0)] = 0
    return output
