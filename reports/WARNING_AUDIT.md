# Track1 reproduction warning audit

The official checker is unchanged. All three historical static warnings are resolved; the current result is **0 errors, 0 warnings**.

## Original warnings (verbatim)

```text
warning vendor/segmentation_refinement/download.py seems to use the network; the reproduction runs with NO network (weights must come from weights/)
warning vendor/sam3/model_builder.py seems to use the network; the reproduction runs with NO network (weights must come from weights/)
warning requirements.txt lists packages the base image already has (numpy, pillow, einops, opencv-python-headless, timm, scipy, pyyaml): installing them would replace the organizers' versions (PyTorch 2.4.0 etc.) and change your results. List ONLY extra packages; leave the file empty if you need none
```

## Judgment and changes

1. **CascadePSP network code.** `run.py` already passes the package-local `weights/cascadepsp` folder and disables automatic download. With the valid archive, this warning did not affect the inference path. The unused fallback could nevertheless attempt a download or overwrite a corrupt local file. It now raises explicit local errors; the published checksum remains enforced for callers requesting verification. No numerical inference code changed.
2. **SAM3 network code.** The runtime already passes `weights/sam3.pt` with `load_from_HF=False`; the tokenizer is bundled. This path neither needed network access nor depended on the original workspace. Removed automatic online resolution and added an early local-checkpoint check. Missing assets fail before model allocation. Network-denial probes confirm zero socket attempts, including explicit calls to both old download APIs.
3. **Default-image dependency overrides.** This was a real risk when installing the old requirements into the default image: NumPy, OpenCV and other versions would be replaced. The existing root Dockerfile intentionally used those pins. Its complete original pin list is now `docker/requirements-runtime.txt`, and both custom Dockerfiles use that file. Root `requirements.txt` contains only missing extra distributions. Actual installation into the PyTorch 2.4.0 base preserved its core package versions (see JSON). The root Dockerfile still governs the submission; pixel equality across the default and custom image is not asserted.

## Actual verification

- The root PyTorch 2.5.1 image, the PyTorch 2.4.0 base plus extras, and the additional CUDA 12.8 validation image all rebuilt successfully.
- The first two images passed imports, missing-weight and corrupt-weight probes with `--network none`, a read-only root filesystem, and package location `/relocated/package`. Explicit connection denial recorded zero socket attempts. The valid downloaded CascadePSP file passed its original checksum; an invalid fixture was not overwritten.
- The CUDA 12.8 image ran `python repro_check.py . --run --input /inputs --output /results --n 5 --device cuda` from the relocated folder, with actual HF-downloaded weights, `--network none`, and a read-only root filesystem. Only `/tmp` and the result mount were writable.
- Before/after runs used identical inputs and compared all **10 PNG files**. Both pixels and complete PNG bytes were identical. Peak allocation was **5.5970 GiB**. The after-fix dynamic checker returned **0 errors, 0 warnings**.
- Missing NVIDIA Container Toolkit support on this workstation was handled for validation with explicit GPU device and driver-library mounts; the source does not rely on those host paths.
- The root 2.5.1/CUDA 12.4 image still cannot execute kernels on this workstation's RTX 5080 (`sm_120`): the actual kernel probe returned `no kernel image is available`. This hardware limitation is separate from the three static warnings. Full GPU inference in that image requires a supported GPU; the 2.4.0 image also has not had full GPU inference validated here. Root 2.5.1 was retained as previously requested.
- Python emits upstream CUDA-absent/deprecation warnings in CPU probes; those are separate from the three official static checker warnings. The checker was not changed or filtered.
- No model weights, inference settings or HF archive bytes changed. No GitHub push was performed.

See `warning_audit.json`, `warning_gpu_pixel_comparison.json`, `warning_*_build.log`, `warning_*_offline_probe.log`, and `warning_after_gpu_run.log` for machine-readable evidence. Historical HF-download logs retain their original three warnings.
