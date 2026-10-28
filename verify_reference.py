"""Compare fixed predictions with an organizer ZIP; never fits model settings."""
from pathlib import Path
import argparse, io, json, random, shutil, tempfile, zipfile
import numpy as np
from PIL import Image
from run import main


def verify(weights, inputs, reference_zip, report, device='cuda', count=20):
    paths = sorted(p for p in Path(inputs).rglob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg'))
    rng = random.Random(19050)
    groups = {}
    for p in paths:
        scene, _, _ = p.stem.rsplit('_', 2)
        groups.setdefault(scene, []).append(p)
    # Scene prefixes are used only for stratified verification sampling, never inference.
    chosen = []
    for group in groups.values(): rng.shuffle(group)
    while len(chosen) < min(count, len(paths)):
        for scene in sorted(groups):
            if groups[scene] and len(chosen) < count: chosen.append(groups[scene].pop())
    with tempfile.TemporaryDirectory(prefix='uwipl_verify_') as tmp:
        root=Path(tmp); original=root/'input'; renamed=root/'renamed'
        original.mkdir(); renamed.mkdir(); mapping={}
        for p in chosen:
            shutil.copy2(p,original/p.name)
            _,water,depth=p.stem.rsplit('_',2)
            new=f'anon_{rng.getrandbits(64):016x}_{water}_{depth}{p.suffix}'
            shutil.copy2(p,renamed/new); mapping[p.stem]=Path(new).stem
        main(weights,original,root/'out',device)
        main(weights,renamed,root/'renamed_out',device)
        results=[]
        with zipfile.ZipFile(reference_zip) as archive:
            names=archive.namelist()
            for p in chosen:
                row={'name':p.stem}
                for kind in ('restored','seg_masks'):
                    matches=[n for n in names if '/'+kind+'/' in '/'+n and Path(n).stem==p.stem]
                    if len(matches)!=1: raise ValueError(f'Ambiguous reference for {p.stem}/{kind}')
                    with Image.open(io.BytesIO(archive.read(matches[0]))) as im:
                        expected=np.array(im.convert('RGB') if kind=='restored' else im)
                    actual=np.array(Image.open(root/'out'/kind/(p.stem+'.png')))
                    renamed_actual=np.array(Image.open(root/'renamed_out'/kind/(mapping[p.stem]+'.png')))
                    if actual.shape!=expected.shape: raise ValueError('Shape mismatch')
                    equal=actual==expected
                    row[kind]={'pixel_agreement':float((equal.all(-1) if equal.ndim==3 else equal).mean()),
                               'renamed_identical':bool(np.array_equal(actual,renamed_actual))}
                results.append(row)
        Path(report).write_text(json.dumps({'seed':19050,'images':len(results),'results':results},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--weights',default='weights');p.add_argument('--input',required=True)
    p.add_argument('--reference-zip',required=True);p.add_argument('--report',default='reference_comparison.json')
    p.add_argument('--device',default='cuda');p.add_argument('--count',type=int,default=20)
    a=p.parse_args();verify(a.weights,a.input,a.reference_zip,a.report,a.device,a.count)
