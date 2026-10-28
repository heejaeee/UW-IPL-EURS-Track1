"""Submission 19050 inference. Only trailing water/depth labels are consumed."""
from pathlib import Path
import gc, io, json, os, random, sys, tempfile, time
import numpy as np
from PIL import Image
import torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from src.semantic import PromptScores, project_scores, letterbox, rgb_tensor, selective_fuse
from src.physical import load_parameters, refine
from src.label_restormer import LabelConditionedRestormer
from src.padding import pad_to_multiple
from src.old_restorer import RestormerRestorer
from src.boundary import refine_boundary


def scoring_size(size):
    factor = min(1., 1024 / max(size))
    return tuple(max(1, round(v * factor)) for v in size)


def labels_only(stem):
    _, water, depth = stem.rsplit('_', 2)
    water, depth = int(water), int(depth)
    if not 1 <= water <= 10 or depth not in range(5, 101, 5):
        raise ValueError('Expected trailing water type 1..10 and depth 5..100 in steps of 5')
    return water, depth


def concept_runner(predictor, weights, device):
    from src.concept import SAM3_OVSS
    bank = torch.load(weights / 'concept_bank.pt', map_location='cpu', weights_only=False)
    runner = SAM3_OVSS.__new__(SAM3_OVSS)
    torch.nn.Module.__init__(runner)
    runner.model = predictor.model; runner.processor = predictor.processor
    runner.device = torch.device(device)
    runner.num_queries = 5; runner.num_cls = 5; runner.confidence_threshold = .5
    runner.topk_inst = 30; runner.use_sem_seg = True; runner.use_presence_instance = True
    runner.prompt_chunk_size = 1
    runner.text_batch_dim = bank['text_batch_dim']; runner.text_bank = bank['text_outputs']
    return runner.eval().requires_grad_(False)


@torch.inference_mode()
def semantic_mask(predictor, runner, raw_native, old, physical, cfg, device):
    target = scoring_size(raw_native.size)
    raw = raw_native.resize(target, Image.Resampling.BILINEAR)
    w,h = (max(1, round(v * 1008 / max(raw_native.size))) for v in raw_native.size)
    box = ((1008-w)//2, (1008-h)//2, w, h)
    canvas = Image.new('RGB', (1008,1008)); canvas.paste(raw.resize((w,h), Image.Resampling.BICUBIC), box[:2])
    _,soft = predictor.predict(canvas)
    original = project_scores(soft, box, target).half().float()
    scores = runner._inference_single_view(raw)
    values,labels = scores.float().topk(2,dim=0)
    concept = labels[0].cpu().numpy().astype(np.uint8)
    confidence = values[0].cpu().numpy().astype(np.float16)
    margin = (values[0]-values[1]).cpu().numpy().astype(np.float16)
    w,h = (max(1,round(v*384/max(raw_native.size))) for v in raw_native.size)
    box = ((384-w)//2,(384-h)//2,w,h); left,top = box[:2]
    hard,soft = predictor.predict(old)
    hard = hard.argmax(0).byte().cpu().numpy()[top:top+h,left:left+w]
    baseline = np.array(Image.fromarray(hard).resize(target, Image.Resampling.NEAREST))
    old_scores = project_scores(soft,box,target).half().float()
    calibration = cfg['calibration']
    weight = torch.tensor(calibration['original_weight'], device=device)[:,None,None]
    scale = torch.tensor(calibration['scale'], device=device)[:,None,None]
    bias = torch.tensor(calibration['bias'], device=device)[:,None,None]
    fg = weight*original[1:] + (1-weight)*old_scores[1:]
    old_scores[1:] = fg*scale + (fg>.01)*bias
    dual = old_scores.argmax(0).cpu().numpy().astype(np.uint8)
    submitted = baseline.copy()
    gate = np.isin(baseline,[0,2,3]) & np.isin(dual,[0,2,3]); submitted[gate] = dual[gate]
    gate = np.isin(submitted,[0,1]) & np.isin(concept,[0,1]) & (confidence>=.5) & (margin>=.15)
    submitted[gate] = concept[gate]
    canvas,physical_box = letterbox(physical,1008)
    _,soft = predictor.predict(canvas)
    corrected = project_scores(soft,physical_box,target).half().float()
    final = selective_fuse(original,corrected,baseline,submitted,concept,confidence,margin,calibration)
    refined_rgb = old.crop((left,top,left+w,top+h)).resize(target,Image.Resampling.BILINEAR)
    return final, np.array(raw), np.array(refined_rgb)


@torch.inference_mode()
def main(model_dir, input_dir, output_dir, device):
    weights = Path(model_dir); output = Path(output_dir)
    cfg = json.loads((weights/'inference.json').read_text())
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    # The historical restorers and SAM3 ran in separate processes. SAM3's
    # import enables TF32 globally; reset the original restorer precision on
    # every entry so repeated calls and input order cannot change predictions.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    paths = sorted(p for p in Path(input_dir).iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg'))
    if not paths: return
    if len({p.stem for p in paths}) != len(paths): raise ValueError('Duplicate input stems')
    for kind in ('restored','seg_masks'): (output/kind).mkdir(parents=True,exist_ok=True)
    e034 = LabelConditionedRestormer(9,LayerNorm_type='BiasFree')
    e034.load_state_dict(torch.load(weights/'e034.pt',map_location='cpu',weights_only=True))
    e034.to(device).eval().requires_grad_(False)
    old = RestormerRestorer()
    old.load_state_dict(torch.load(weights/'restormer384.pt',map_location='cpu',weights_only=True))
    old.to(device).eval().requires_grad_(False)
    params_cpu = load_parameters(weights/'rsuigm_params.json')
    params = params_cpu.to(device)
    began = time.monotonic()
    # Separate fixed-model phases keep peak memory low. No information or statistics
    # from one image is used to predict another; temporary RGBs are per-image only.
    with tempfile.TemporaryDirectory(prefix='uwipl_t1_') as temporary:
        cache = Path(temporary)
        for index,path in enumerate(paths):
            water,depth = labels_only(path.stem)
            with Image.open(path) as decoded: raw = decoded.convert('RGB')
            target = scoring_size(raw.size)
            x = np.asarray(raw.resize(target,Image.Resampling.BILINEAR),dtype=np.float32)/255
            tensor = torch.from_numpy(np.ascontiguousarray(x)).permute(2,0,1)[None].to(device)
            wi,di = water-1,depth//5-1
            labels = torch.cat((params_cpu.kd[wi:wi+1]*depth/10/16,
                                params_cpu.binf[wi:wi+1,di],params_cpu.attenuation[wi:wi+1]/5),1).to(device)
            with torch.autocast(torch.device(device).type,dtype=torch.bfloat16,
                                enabled=str(device).startswith('cuda') and torch.cuda.is_bf16_supported()):
                result = e034(pad_to_multiple(tensor,8),labels)[...,:target[1],:target[0]]
            restored = result.float()[0].clamp(0,1).permute(1,2,0).cpu().numpy()
            physical = np.round(refine(x,restored,water,depth,params).clip(0,1)*255).astype(np.uint8)
            Image.fromarray(physical).save(cache/f'{index}_physical.png')
            stream = io.BytesIO()
            Image.fromarray(physical).save(stream,format='JPEG',quality=cfg['jpeg_quality'],subsampling=0,optimize=True)
            stream.seek(0)
            with Image.open(stream) as jpeg: jpeg.convert('RGB').save(output/'restored'/f'{path.stem}.png')
            canvas,_ = letterbox(raw,384)
            # Historical Restormer384 used a batch of four. Duplicate only this
            # image to retain identical kernels/rounding, with no other image
            # influencing its result. This exactly matches the cached RGBs.
            rgb,_ = old(rgb_tensor(canvas)[None].to(device).repeat(4,1,1,1))
            pixels = (rgb[0].clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
            Image.fromarray(pixels).save(cache/f'{index}_old.png')
            print(json.dumps(dict(stage='restoration',completed=index+1,total=len(paths))),flush=True)
        del e034,old,result,rgb,tensor,params
        gc.collect()
        if str(device).startswith('cuda'): torch.cuda.empty_cache()
        torch.manual_seed(cfg['seed'])
        if str(device).startswith('cuda'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        predictor = PromptScores(str(weights/'sam3.pt'),device)
        runner = concept_runner(predictor,weights,device)
        from segmentation_refinement import Refiner
        torch.manual_seed(cfg['boundary']['seed'])
        boundary = Refiner(device=device,model_folder=str(weights/'cascadepsp'),download_and_check_model=False)
        boundary.model.eval().requires_grad_(False)
        for index,path in enumerate(paths):
            with Image.open(path) as decoded: raw = decoded.convert('RGB')
            with Image.open(cache/f'{index}_old.png') as decoded: old_rgb = decoded.convert('RGB')
            with Image.open(cache/f'{index}_physical.png') as decoded: physical = decoded.convert('RGB')
            mask,raw_rgb,refined_rgb = semantic_mask(predictor,runner,raw,old_rgb,physical,cfg,device)
            if (mask==4).any():
                coarse=(mask==4).astype(np.uint8)*255
                a=boundary.refine(raw_rgb,coarse,fast=False,L=cfg['boundary']['local_size'])
                b=boundary.refine(refined_rgb,coarse,fast=False,L=cfg['boundary']['local_size'])
                mask=refine_boundary(mask,a,b,cfg['boundary'])
            if mask.dtype!=np.uint8 or mask.max()>4: raise RuntimeError('Invalid class mask')
            Image.fromarray(mask).save(output/'seg_masks'/f'{path.stem}.png')
            print(json.dumps(dict(stage='segmentation',completed=index+1,total=len(paths))),flush=True)
    print(json.dumps(dict(stage='complete',images=len(paths),seconds=time.monotonic()-began,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30 if str(device).startswith('cuda') else 0)),flush=True)
