# UW-IPL EURS Track1 reproduction

Local reproduction package using the unmodified official [template](https://github.com/CeviKle/EURS-MaCVi2027-repro-template). No public push or upload has been performed.

Read METHOD.md, FACTSHEET.json, DATA_AND_LICENSES.md and STATUS_KO.md. Runtime entrypoints use package-relative paths and local weights. Network access is not needed during inference/export.

```sh
python repro_check.py .
```

The weight archive and a local extracted copy are kept outside this link-based Git repository. The archive is `UW-IPL-Track1-19050-weights.tar.gz` (3,525,648,308 bytes), SHA-256:

```text
87497b7a5b854ffb79739267fe29423b06f3addc42c7f46e07970d07f473569c
```

After the owner fills the HTTPS URL, use `python repro_check.py . --download-weights`. The official downloader records the extracted files so they are not mistaken for committed weights. Do not change the archive SHA-256. The static checker cannot pass with a blank URL or contact; functional checks are reported separately.

```sh
docker build -t uwipl-track1-repro:final .
docker run --rm --gpus all --network none --read-only --tmpfs /tmp:rw,size=4g -v "$PWD:/pkg:ro" -v "$INPUT:/input:ro" uwipl-track1-repro:final python repro_check.py . --run --input /input --n 5 --device cuda
```

The requested Docker base is PyTorch 2.5.1/CUDA 12.4. Its build and offline imports passed, but it lacks GPU kernels for the local RTX 5080. GPU execution in this specific image still requires validation on compatible hardware.

For the RTX 5080 development machine, `docker/Dockerfile.cuda128-validation` defines an additional PyTorch 2.7.1/CUDA 12.8 test environment. It does not replace the requested Dockerfile. Its offline five-image run passed the output contract; restoration pixels were identical and segmentation differed by three pixels in total. Host validation of 20 images and randomized scene prefixes was pixel-identical. See reports for exact outcomes. The upstream SAM3 implementation requires CUDA; CPU-only inference is unsupported.

A portable verification tool is included:

```sh
python verify_reference.py --weights WEIGHTS_DIR --input INPUT_DIR --reference-zip RESULTS_ZIP --report comparison.json
```

It compares outputs and randomized filename prefixes without fitting parameters. After completing owner metadata, rerun the checker and refresh `SHA256SUMS.txt` before hand-in.

The three static warnings concern upstream optional download functions and pinned packages that overlap the default image. Runtime supplies local weights, disables network access and uses the explicitly requested custom Docker base. These warnings are retained, not suppressed.
