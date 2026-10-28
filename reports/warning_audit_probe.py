import sys,json,traceback,socket,tempfile,importlib.metadata as metadata
from pathlib import Path
root=Path.cwd();sys.path[:0]=[str(root),str(root/'vendor')]
report={'package_folder':str(root),'socket_attempts':0,'checks':{},'versions':{}}
for name in ['torch','numpy','pillow','einops','opencv-python-headless','timm','scipy','pyyaml','huggingface_hub']:
 report['versions'][name]=metadata.version(name)
def deny(*args,**kwargs):
 report['socket_attempts']+=1
 raise AssertionError('Network connection attempted')
socket.create_connection=deny;socket.socket.connect=deny;socket.socket.connect_ex=deny
try:
 import run
 from segmentation_refinement.download import download_model,download_and_or_check_model_file
 from sam3.model_builder import build_sam3_image_model,download_ckpt_from_hf
 report['checks']['offline_imports']=True
 with tempfile.TemporaryDirectory() as d:
  missing=Path(d)/'missing.pt'
  for name,call in [('cascade_missing',lambda:download_and_or_check_model_file(str(missing))),('cascade_download_disabled',lambda:download_model(str(missing))),('sam3_missing',lambda:build_sam3_image_model(checkpoint_path=str(missing),device='cpu')),('sam3_default',lambda:build_sam3_image_model(device='cpu')),('sam3_download_disabled',lambda:download_ckpt_from_hf())]:
   try:call()
   except FileNotFoundError as exc:report['checks'][name]=str(exc)
   else:raise AssertionError(name+' did not reject missing weights')
  corrupt=Path(d)/'corrupt.pt';corrupt.write_bytes(b'not model data')
  try:download_and_or_check_model_file(str(corrupt))
  except ValueError as exc:report['checks']['cascade_corrupt']=str(exc)
  else:raise AssertionError('Corrupt weight accepted')
  assert corrupt.read_bytes()==b'not model data'
  assert not missing.exists()
 download_and_or_check_model_file(str(root/'weights/cascadepsp/model'))
 report['checks']['cascade_published_checksum']=True
 assert report['socket_attempts']==0
 report['passed']=True
except Exception:
 report['passed']=False;report['error']=traceback.format_exc()
print(json.dumps(report,indent=2))
sys.exit(0 if report['passed'] else 1)
