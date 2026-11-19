# syntax=docker/dockerfile:1
# Prebuilt serving image: pinned vLLM + everything up.sh needs, with the SM120 FP8 prefill patch applied and checked at
# build time. Models and compile caches are NOT baked in (32 GB of weights would make every pull slow and every model
# update a rebuild); they live on the /workspace volume, fetched in parallel and reused across restarts.
#
# The base is pinned by commit tag + digest: `vllm/vllm-openai:nightly` moves daily and the FP8 patch anchors on the
# exact flashinfer.py of this build (the version measured in stack/benchmarks).
ARG VLLM_IMAGE=vllm/vllm-openai:nightly-7c3a9302f5ac2441ad199134129e973d22a46797@sha256:9ce07a9b067f80bf1613890fbd6a16724086595215d76d6b3ca5a7a5455459f2
FROM ${VLLM_IMAGE}

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nginx zstd curl procps \
 && rm -rf /var/lib/apt/lists/* /etc/nginx/sites-enabled/default
# Xet transfers (HF's chunked storage) are several times faster than plain LFS downloads.
RUN python3 -c "import hf_xet" 2>/dev/null || pip install --no-cache-dir "hf_xet>=1.1"

COPY stack /opt/secai/stack
COPY docker/boot.sh /opt/secai/docker/boot.sh
# Fail the build, not the first boot, if the patch no longer matches this vLLM.
RUN python3 /opt/secai/stack/patches/sm120_fp8_prefill.py apply \
 && grep -q _sm120_fp8_paged_prefill /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py \
 && python3 -m py_compile /opt/secai/stack/*.py \
 && bash -n /opt/secai/stack/up.sh /opt/secai/stack/down.sh /opt/secai/stack/kv_gc.sh /opt/secai/docker/boot.sh \
 && MODEL_DIR=/workspace/models bash /opt/secai/stack/up.sh --dry-run >/dev/null \
 && chmod +x /opt/secai/docker/boot.sh

ARG GIT_SHA=unknown
LABEL org.opencontainers.image.source="https://github.com/elonma-china/secai-serving" \
      org.opencontainers.image.revision="${GIT_SHA}"
ENV SECAI_IMAGE_REVISION=${GIT_SHA} \
    MODEL_DIR=/workspace/models \
    CACHE_DIR=/workspace/cache \
    HF_HOME=/workspace/hf-cache \
    LOG_DIR=/workspace/logs \
    HF_XET_HIGH_PERFORMANCE=1 \
    VLLM_SM120_FP8_PREFILL=1
WORKDIR /workspace
EXPOSE 8000 8002 8004 8005
ENTRYPOINT ["/opt/secai/docker/boot.sh"]
