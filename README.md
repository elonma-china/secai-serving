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

## Benchmark and test tools (`stack/`)

`agent_sim.py` (multi-agent load), `decode_test.py`, `needle.py`, `nll.py` (quality gates), `kv_check.py`, `ttft_check.py` (offload correctness/latency),
`atomic.sh`, `patch_ab.sh`, `sweep.sh`, `profile.sh` (nsys), `clocks.sh`, `glue_bench.py`. They expect the layout of the test instance (`/workspace/...`).
