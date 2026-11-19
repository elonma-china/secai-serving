# Benchmarks

Hardware: 2x RTX PRO 6000 Blackwell (96 GB, **400 W cap**), 249 GB host RAM, NVMe. vLLM 0.34.0rc1.dev776, FlashInfer 0.7.4.
Model: `orcarouter/Qwen3.8-27B-Uncensored-NVFP4`, context 131072. Raw outputs: [`raw/`](raw/).

## 1. Agentic end-to-end (32 concurrent agents, same seed/workload in every row)

**What one "agent" does (measured, not nominal).** `agent_sim.py --agents 32 --hops 14 --file-words 1500 --tool-min 1 --tool-max 4 --max-turns 40`.
This is a long-context multi-turn tool-calling workload, **not** short prompts:

- The user question is short (a few dozen tokens, not measured separately: "what is the final value of `timeout_ms` for service X, follow the `ref:` chain"), but the first request already carries a **2.6K-token** system prompt shared by all agents.
- Each agent has a private synthetic repo with a 14-hop reference chain and tools `read_file`, `search`, `submit_answer`. Every turn the model calls one tool, the harness waits 1-4 s (simulated tool latency), appends the tool result (one **~7.3K-token file**), and **re-sends the whole conversation**.
- Every agent finished in exactly **16 turns**, so the prompt grows from **2.6K to ~112K tokens** (median 112.3K, max 112.6K) per agent. 32 agents x 16 turns = 512 requests, **29.4M prompt tokens in total** (~57K per request on average).
- **Output is tiny**: 34.5 tokens per turn on average (max 66), 17.7K total, because each reply is just a tool call (thinking disabled, temperature 0.2). So this benchmark stresses **prefill and KV-cache reuse**, not decode or long generations. Decode is covered separately in section 2.
- 32 sessions run concurrently with a 0.2 s stagger, sessions pinned per replica by `X-Session-Id` (sha1-hashed IDs so they spread across both replicas).
- Same seed and workload in every row below. All rows: 32/32 agents correct, 0 errors. TTFT is measured per request (time to first streamed token), over all 512 requests.

| Config | Run time | TTFT p50 | TTFT p95 | TTFT max | Tokens served from cache |
|---|---|---|---|---|---|
| Baseline: 2 replicas, GPU prefix cache only (no offload), default batch size, no MTP, no FP8 patch | 1983 s | 28.0 s | 254 s | 300 s | 36.5% |
| + FP8 prefill attention, 16K batches, MTP 2, RAM+NVMe KV tier | 652 s (3.0x) | 13.1 s | 70.9 s | 127 s | ~81% |
| + replica B util 0.68 -> 0.74, nginx weights 5:4, fixed `kv_gc.sh` (**final**) | **586 s (3.4x)** | **13.1 s** | **61.6 s** | 142 s | ~82% |

16 agents, baseline only: 239 s, TTFT p50 9.0 s / p95 21.8 s, 88.7% hit. Cache share for the last two rows = GPU prefix hits + external (RAM/NVMe) hits over prompt tokens.
The two optimized rows were not ablated individually on the agent workload; per-optimization numbers come from the atomic tests below.
The last row changed three things at once (B capacity, nginx weights, gc), so its -10% run time is not attributable to any one of them.

## 2. Atomic single-GPU tests (GPU0, 131K context, one change at a time)

| Profile | GPU KV tokens | Decode c=1 | Decode c=16 | NLL (19.3K tok) | Needle 3/3 |
|---|---|---|---|---|---|
| baseline | 1.70M | 65 tok/s | 837 tok/s | 0.5728 | yes |
| FP8 prefill attention | 1.70M | 65 | 838 | 0.5684 | yes |
| MTP 1 | 1.46M | 99 | 1243 | 0.5684 | yes |
| **MTP 2 (shipped)** | 1.40M | 121 | 1372 | 0.5684 | yes |
| MTP 3 | 1.36M | 130 | 1376 | 0.5684 | yes |
| NVFP4 KV cache | 2.79M | 65 | 820 | 0.5724 | yes |
| MTP 2 + NVFP4 KV | 2.20M | 120 | 1371 | 0.5709 | yes |

Notes: MTP 3 gives +7% at c=1 and nothing at c=16 versus MTP 2, and costs KV; MTP 2 is shipped. NLL differences are within noise of each other
(NLL is lower with the patch, not higher). NVFP4 KV adds 57-64% KV capacity but disables the FP8 prefill kernel (118K-token prefill 25.6 s vs 16.1 s), so it is off by default (`LLM_KV_DTYPE=nvfp4` to enable).

## 3. FP8 prefill attention (patch `patches/sm120_fp8_prefill.py`), same server, patch off vs on

| Prompt | Off | On | Change |
|---|---|---|---|
| 32K TTFT | 4.04 s | 3.42 s | -15% |
| 120K TTFT | 21.1 s | 13.3 s | -37% |
| 118K needle (3 depths) | 24.2 s, 3/3 | 16.1 s, 3/3 | -33% |

## 4. KV offload telemetry (final agent run, both replicas summed)

| Metric | Value |
|---|---|
| Loaded RAM -> GPU | 435 GB, 7.6 s total transfer time |
| Read NVMe -> RAM | 40 GB, 6.4 s total |
| Stored | 237 GB to RAM tier, 221 GB to NVMe |
| Prompt tokens: GPU prefix hits / external hits / computed | 12.2M / 11.8M / 5.4M of 29.4M |

About 4M of the computed tokens are the new file each turn and cannot be cached, so at most ~1.5M tokens of the recomputation are avoidable by cache policy.

## 5. Other measurements

- Profiling: nsys works; **ncu does not** (`ERR_NVGPUCTRPERM`, host driver restricts counters), so the roofline was derived analytically, not measured.
- Per-layer K/V gather (`index_select`) in the FP8 path costs 20.8 ms per 120K prefill (0.16%); the glue-free patch from `sm120-vllm-kernels` was therefore not ported.
- Prefill is power-capped (median SM clock ~1774 MHz); decode runs at ~2685 MHz.

## Caveats

- The agent workload is synthetic; its tool calls are repetitive, so the MTP draft acceptance (99.9%) is far above what real traffic will see.
- Single run per row (no repeats), so differences of a few percent are not significant. p95/max are sensitive to scheduling noise.
- Not measured: LMCache, ARC eviction (`KV_EVICTION=arc`), long-duration stability, OCR under load.
- The NVMe tier is trimmed by `kv_gc.sh` by file mtime (FIFO by write time), not LRU.
- The FP8 patch rewrites the installed `vllm/v1/attention/backends/flashinfer.py` and is tested only against vLLM 0.34.0rc1.dev776 / FlashInfer 0.7.4.
