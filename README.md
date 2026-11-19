# SecAI serving: 2x RTX PRO 6000 (Blackwell) vLLM stack

One host, two RTX PRO 6000 (96 GB), everything on vLLM. Context 131072. Tuned for many concurrent multi-turn agent sessions.

| Port | Service | Where |
|---|---|---|
| 8005 | LLM (nginx, session-affine) -> 8001 / 8003 | `secai1999/LLM-Autoscan-v1`, one TP1 replica per GPU |
| 8000 / 8002 | embedding / rerank | `secai1999/Embedding-Autoscan-v1`, `secai1999/Rerank-Autoscan-v1` (GPU1) |
| 8004 | translation adapter -> vLLM | `secai1999/Translate-Autoscan-v1` (GPU1) |
| 18083 | OCR adapter (loopback only) | `secai1999/OCR-Autoscan-v1` (GPU1) |

## Run

```bash
API_KEY=<key> HF_TOKEN=<token> bash stack/up.sh            # start/ensure everything (skips healthy services)
bash stack/up.sh --dry-run                                 # print the commands only
python3 stack/tests/test_stack.py                          # offline tests (or run its test_* functions)
```

Needs 2 GPUs of >= 90 GB, >= 142 GiB free host RAM (2x55 GiB KV offload + headroom), `nginx`, and ~160 GB free disk for the NVMe KV tier.
Models are pulled from the public `secai1999/*` repos on Hugging Face; `HF_TOKEN` is optional (higher rate limits). `ocr_adapter.py` also downloads its prompt from `secai1999/OCR-Autoscan-v1` at startup.
Clients send `X-Session-Id` so a session stays on one replica and reuses its cached KV.

## Deploy (prebuilt image)

The image (`docker/Dockerfile`) contains everything: vLLM pinned to the measured build (`7c3a9302f`, by commit tag +
digest), nginx, this `stack/` with the FP8 prefill patch applied, and **all five models baked in** (`/opt/secai/models`,
revisions pinned in `stack/models.lock`, one layer per weight shard, below the code layers). A target host only needs
Docker + the NVIDIA runtime; it never installs packages or downloads models. `/workspace` holds only mutable state
(logs, compile cache, KV tier, API key).

```bash
# 1. on the target host: verify it (GPUs, CUDA >= 13.2, nvidia runtime, RAM, disk, ports, key, image pullable)
docker/preflight.sh --image ghcr.io/elonma-china/secai-serving:<tag>
# 2. deploy (key in $SECAI_WORKSPACE/api.key, chmod 600)
SECAI_SERVING_IMAGE=ghcr.io/elonma-china/secai-serving:<tag> SECAI_WORKSPACE=/srv/secai-serving docker compose -f docker/compose.yaml up -d
# 3. READY only when every model answers a real request (chat via nginx + both replicas, embedding, rerank, translate, OCR)
docker exec secai-serving python3 /opt/secai/stack/smoke.py
```
Vast.ai: `docker/vast.sh offers | create <offer> --yes | key <id> api.key | tunnel <id>` (nothing is rented without `--yes`).
Roll back by deploying the previous tag; tags are `<git sha>-<date>`, never rely on `latest` in production.

**Build and release** (needs ~100 GB free disk; the weights come from the public `secai1999/*` repos at the pinned revisions):
```bash
docker/build.sh release [--push]            # vLLM + code + models; code-only changes re-push a few MB
docker/warm.sh <image> cache-seed.tar.zst   # on a 2x PRO 6000 host: cold start, warm every kernel, pack the compile cache
docker/build.sh warm cache-seed.tar.zst [--push]   # release + cache seed: boot skips recompilation
```
What a boot still does (cannot be baked, needs the GPUs): load weights onto the GPUs, capture CUDA graphs, pin 2x55 GiB of
RAM for KV offload. Boot sequence: seed/restore compile cache -> wait for the key -> `up.sh` (replica A starts at once,
the GPU1 services follow one by one so replica B sizes itself on what is left) -> watchdog (unhealthy 60 s -> `down.sh`
+ `up.sh`, 3 times, then exit so `restart: unless-stopped` takes over). Service logs go to stdout (`docker logs`, prefixed
`[llm-a]`...) and to `/workspace/logs`; per-service ready times are in `startup-timeline.log`; vLLM serves Prometheus
`/metrics` on 8000/8001/8002/8003. `BIND_HOST=127.0.0.1` limits the public ports to loopback (tunnel / reverse proxy).

Without Docker: `MODEL_DIR=/workspace/models python3 stack/fetch_models.py && MODEL_DIR=/workspace/models bash stack/up.sh`.
Compile caches can also be shared without the image through `stack/cache_bundle.py` (private HF repo). Not covered: one
host is a single point of failure, and the API key is the only access control (no TLS) - keep the ports off the internet.

## What is optimized

- **KV offload**: GPU -> pinned RAM (55 GiB/replica) -> NVMe (`TieringOffloadingSpec`, trimmed to 80 GiB/replica by `kv_gc.sh`).
- **MTP speculative decoding**, 2 draft tokens (+86% decode at c=1, +64% at c=16).
- **FP8 prefill attention** for SM120 (`stack/patches/sm120_fp8_prefill.py`, applied by `up.sh`): 120K-token TTFT -37%.
- 16K prefill batches with CUDA graphs, vision tower dropped (OCR is a separate model), nginx weights 5:4 by KV capacity.

Results vs the previous setup (32 concurrent agents, each running a 16-turn tool-calling session whose context grows from 2.6K to ~112K tokens; 29.4M prompt tokens, ~35 output tokens per turn, so this measures prefill and KV reuse, not decode): run time 1983 s -> 586 s, TTFT p95 254 s -> 61.6 s. Full tables and caveats: [stack/benchmarks/BENCHMARKS.md](stack/benchmarks/BENCHMARKS.md).

## Knobs (environment variables of `up.sh`)

| Variable | Default | Meaning |
|---|---|---|
| `KV_OFFLOAD` | `tier` | `tier`, `native`, `lmcache`, `none` |
| `KV_OFFLOAD_GIB` / `KV_FS_MAX_GIB` / `KV_FS_DIR` | 55 / 80 / `/workspace/kvcache` | RAM and NVMe KV tier sizes per replica |
| `KV_EVICTION` | `lru` | RAM tier policy, `lru` or `arc` (ARC untested here) |
| `LLM_MTP` | `2` | draft tokens, `0` disables |
| `LLM_KV_DTYPE` | empty (checkpoint fp8) | `nvfp4` = +57-64% KV, but disables the FP8 prefill kernel |
| `VLLM_SM120_FP8_PREFILL` | `1` | `0` skips the attention patch (`python3 stack/patches/sm120_fp8_prefill.py revert` restores the original file) |
| `LLM_UTIL_A` / `LLM_UTIL_B` | 0.90 / 0.74 | GPU memory fraction; B shares GPU1 with the four aux models |
| `LLM_BATCHED_TOKENS`, `LLM_MAX_NUM_SEQS`, `LLM_MAX_MODEL_LEN` | 16384, 64, 131072 | |
| `MODEL_DIR` | empty (image: `/opt/secai/models`) | serve local copies under the Hub names, offline |
| `BIND_HOST` | `0.0.0.0` | interface of the public ports; `127.0.0.1` = loopback only |
| `CACHE_DIR` | `/workspace/cache` | torch.compile / Triton / FlashInfer caches (persist across restarts) |

## Benchmark and test tools (`stack/`)

`agent_sim.py` (multi-agent load), `decode_test.py`, `needle.py`, `nll.py` (quality gates), `kv_check.py`, `ttft_check.py` (offload correctness/latency),
`atomic.sh`, `patch_ab.sh`, `sweep.sh`, `profile.sh` (nsys), `clocks.sh`, `glue_bench.py`. They expect the layout of the test instance (`/workspace/...`).
