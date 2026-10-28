# UW-IPL EURS Track1 reproduction

Local final-package draft using the unmodified official [template](https://github.com/CeviKle/EURS-MaCVi2027-repro-template). No publicpush or upload has been performed.

Read METHOD.md, FACTSHEET.json and DATA_AND_LICENSES.md. Runtime entrypoints use only relative package paths and local weights; network access is unnecessary.

```sh
python repro_check.py .
```

The weight archive and a local extracted copy are kept outside this link-based Git repository. After the owner fills the HTTPS URL, use `python repro_check.py . --download-weights`; the official downloader records the extracted files so they are not mistaken for committed weights. Do not change the archive SHA256. The static checker cannot pass with a blank URL or contact; functional checks are reported separately.

```sh
docker build -t uwipl-track1-repro:final .
docker run --rm --gpus all --network none --read-only --tmpfs /tmp:rw,size=4g -v "$PWD:/pkg:ro" -v "$INPUT:/input:ro" uwipl-track1-repro:final python repro_check.py . --run --input /input --n 5 --device cuda
```

The Docker base is PyTorch2.5.1CUDA12.4 as requested; actual compatibility findings are in reports. Local CUDA12.8verification is not claimed as successful CUDA12.4verification.

For this RTX5080 development machine only, docker/Dockerfile.cuda128-validation defines an additionalTorch2.7.1/CUDA12.8testenvironment. It does not replace the requested submissionDockerfile. DockerNVIDIAContainerToolkit is absent locally; any direct-device validation is explicitly identified in reports. The upstreamSAM3imagebuilder requiresCUDA; CPU-only inference is not supported.

A portable verification tool is included: `python verify_reference.py --weights WEIGHTS_DIR --input INPUT_DIR --reference-zip RESULTS_ZIP --report comparison.json`. It compares outputs and randomized filename prefixes without fitting any parameters.
