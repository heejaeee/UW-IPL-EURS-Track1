FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /pkg
COPY . /pkg
RUN pip install --no-cache-dir -r requirements.txt
ENV HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
ENV XDG_CACHE_HOME=/tmp/cache CUDA_CACHE_PATH=/tmp/cuda-cache TRITON_CACHE_DIR=/tmp/triton
