# syntax=docker/dockerfile:1
# Deployable serving image: pinned vLLM + code + the five models baked in as layers. A target host only pulls and runs;
# it never downloads models or installs anything.
#
# Layer order is chosen so that a code change re-pushes only a few MB:
#   vLLM base (pinned by commit tag + digest) -> nginx/zstd/hf_xet -> models (one layer per weight shard, so no layer
#   exceeds the 5 GB shards of the checkpoints) -> stack code + FP8 patch -> [optional] compile-cache seed (Dockerfile.warm)
# The vLLM nightly tag moves daily and the FP8 patch anchors on this build's flashinfer.py, hence the pin.
ARG VLLM_IMAGE=vllm/vllm-openai:nightly-7c3a9302f5ac2441ad199134129e973d22a46797@sha256:9ce07a9b067f80bf1613890fbd6a16724086595215d76d6b3ca5a7a5455459f2
FROM ${VLLM_IMAGE}

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nginx zstd curl procps \
 && rm -rf /var/lib/apt/lists/* /etc/nginx/sites-enabled/default
RUN python3 -c "import hf_xet" 2>/dev/null || pip install --no-cache-dir "hf_xet>=1.1"

# --- models: revisions pinned in models.lock; changing it rebuilds only from here down ---
ENV MODEL_DIR=/opt/secai/models HF_XET_HIGH_PERFORMANCE=1
COPY stack/fetch_models.py stack/models.lock /opt/secai/stack/
RUN python3 /opt/secai/stack/fetch_models.py LLM_MODEL --include 'model-00001-of-00005.safetensors'
RUN python3 /opt/secai/stack/fetch_models.py LLM_MODEL --include 'model-00002-of-00005.safetensors'
RUN python3 /opt/secai/stack/fetch_models.py LLM_MODEL --include 'model-00003-of-00005.safetensors'
RUN python3 /opt/secai/stack/fetch_models.py LLM_MODEL --include 'model-00004-of-00005.safetensors'
RUN python3 /opt/secai/stack/fetch_models.py LLM_MODEL --include 'model-00005-of-00005.safetensors' --include 'model-extra-*.safetensors'
RUN python3 /opt/secai/stack/fetch_models.py EMBED_MODEL RERANK_MODEL OCR_MODEL MT_MODEL
# Full pass: fetches the small remaining LLM files, verifies every file size against the Hub, writes the markers.
RUN python3 /opt/secai/stack/fetch_models.py && python3 /opt/secai/stack/fetch_models.py --offline

# --- code ---
COPY stack /opt/secai/stack
COPY docker/boot.sh /opt/secai/docker/boot.sh
# Fail the build, not the first boot, if the patch no longer matches this vLLM.
RUN python3 /opt/secai/stack/patches/sm120_fp8_prefill.py apply \
 && grep -q _sm120_fp8_paged_prefill /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py \
 && python3 -m py_compile /opt/secai/stack/*.py \
 && bash -n /opt/secai/stack/up.sh /opt/secai/stack/down.sh /opt/secai/stack/kv_gc.sh /opt/secai/docker/boot.sh \
 && bash /opt/secai/stack/up.sh --dry-run >/dev/null \
 && chmod +x /opt/secai/docker/boot.sh

ARG GIT_SHA=unknown
LABEL org.opencontainers.image.source="https://github.com/elonma-china/secai-serving" \
      org.opencontainers.image.revision="${GIT_SHA}"
# /workspace (a volume) holds only mutable state: logs, compile cache, KV tier, keys. Models come from the image.
ENV SECAI_IMAGE_REVISION=${GIT_SHA} \
    CACHE_DIR=/workspace/cache \
    CACHE_SEED=/opt/secai/cache-seed \
    HF_HOME=/workspace/hf-cache \
    LOG_DIR=/workspace/logs \
    VLLM_SM120_FP8_PREFILL=1
WORKDIR /workspace
EXPOSE 8000 8002 8004 8005
ENTRYPOINT ["/opt/secai/docker/boot.sh"]
