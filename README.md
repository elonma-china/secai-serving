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

## Prebuilt image (fast deploy)

`docker/Dockerfile` = pinned vLLM nightly (`7c3a9302f`, the build measured in BENCHMARKS) + nginx/zstd/hf_xet + this
`stack/`, with the FP8 prefill patch applied and verified at build time. It adds ~7.5 MB on top of the vLLM image, so a
host that already has that base pulls almost nothing. Weights and compile caches stay on the `/workspace` volume:

| Boot step (`docker/boot.sh`) | What it removes from the critical path |
|---|---|
| `stack/fetch_models.py`: 5 models (~32 GB) in parallel over Xet, pinned in `stack/models.lock`, skipped when present | vLLM pulling each model lazily, one service after another |
| `stack/cache_bundle.py restore`: torch.compile/Triton/FlashInfer caches from a private HF repo, keyed by vLLM version + GPU + the full `up.sh --dry-run` plan | recompiling every service on a new machine |
| `up.sh` starts replica A (GPU0) immediately, alongside the GPU1 chain | A waiting for embed/rerank/OCR/MT to finish |
| watchdog: an unhealthy service for 60 s triggers `down.sh` + `up.sh` (max 3, then exit for the restart policy) | no auto-restart before |

```bash
docker/build.sh [--push]                                   # tag ghcr.io/elonma-china/secai-serving:<sha>-<date> + latest
# Local/on-prem 2x RTX PRO 6000 (key in $SECAI_WORKSPACE/api.key, chmod 600):
SECAI_WORKSPACE=/srv/secai-serving docker compose -f docker/compose.yaml up -d
# Vast.ai (nothing is rented without --yes):
docker/vast.sh offers; docker/vast.sh create <offer_id> --yes
docker/vast.sh key <instance_id> ./api.key                 # boot waits for it; models download meanwhile
docker/vast.sh tunnel <instance_id>                        # 38005/38000/38002/38004/38083 on localhost
# First boot of a new image/config: add CACHE_SAVE=1 and an HF_TOKEN with write access to publish the cache bundle
# (it is saved from boot.sh so the key matches the env the stack ran with).
```

Progress is in `/workspace/logs/boot.log`; per-service ready times in `/workspace/logs/startup-timeline.log`.
Without the image, `MODEL_DIR=/workspace/models python3 stack/fetch_models.py && MODEL_DIR=/workspace/models bash stack/up.sh`
gives the same parallel fetch on a plain host. `stack/down.sh` stops everything `up.sh` started.

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
| `MODEL_DIR` | empty (image: `/workspace/models`) | serve local copies from `fetch_models.py` under the Hub names, offline |
| `CACHE_DIR` | `/workspace/cache` | torch.compile / Triton / FlashInfer caches (persist across restarts) |

## Benchmark and test tools (`stack/`)

`agent_sim.py` (multi-agent load), `decode_test.py`, `needle.py`, `nll.py` (quality gates), `kv_check.py`, `ttft_check.py` (offload correctness/latency),
`atomic.sh`, `patch_ab.sh`, `sweep.sh`, `profile.sh` (nsys), `clocks.sh`, `glue_bench.py`. They expect the layout of the test instance (`/workspace/...`).
