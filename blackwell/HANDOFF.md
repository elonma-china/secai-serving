# Blackwell serving handoff — 2026-11-09

## User decisions and scope

- Serve `orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4` on four RTX PRO 6000 Blackwell Server Edition GPUs. Use the current local Hugging Face token; do not request a new one.
- RTX 5090 is **connectivity only for now**. Do not start embedding, rerank or OCR models until the user asks. Translation goes through the main LLM.
- Tune serving flags/config and request/expert routing. Do not port or tune custom kernels in this pass.
- The user asked to preserve this handoff because quota may run out.
- **Post-screen plan (user, 2026-11-09):** keep **CPU PLE + 2×TP2+EP2 + MTP2**. Do not optimize PLE further. Park topology-aware expert routing / DeepEP. Optimize inside vLLM on SM120 before custom kernels or TensorRT-LLM. Sweep MoE NVFP4 backends and linear/BF16 backends **independently**. Stretch goals (~20–25%, not promised): prefill 8K TTFT <400 ms, 32K <1100 ms, decode C1 MTP2 >220 tok/s, mixed C8 >650 tok/s. Protocol: Prefill 32K/1/C1, Decode 2K/1024/C1, Mixed 8K/512/C8; ≥3 seeds after warmup; keep a change only if gains repeat without material mixed/KV/correctness loss.

## Status snapshot (2026-11-09 ~17:00 UTC / 00:00 +07)

- **Live serving:** A+B matched MTP2 baseline — TP2+EP2, CPU PLE, batch 16K, MTP2, custom AR on, **no** async / kernel / compilation overrides. Health 8001/8003/8005 = 200.
- **Locked topology:** CPU PLE + 2×TP2+EP2 + MTP2. Tuning screens closed for this pass: reject async, keep Marlin MoE, keep linear `auto`, **reject** `--disable-custom-all-reduce` (decode ~165 vs ~199), **reject** lean CUDA graph sizes (decode ~175 vs ~199).
- **SSH access for client hosts:** `gx10-training-space` and `aiqk3-esc8000-e11 intramind-migrate` ed25519 keys added to `/root/.ssh/authorized_keys` on the instance (2026-11-09).
- **Bench note:** `BENCH_TOKENIZER=/workspace/qwen38-orcarouter`.

## Client connection (saved)

Vast maps **SSH only** — API is loopback on the instance. Tunnel then call OpenAI-compatible `/v1`.

```sh
# From a host that has one of the authorized keys (or secai2026):
ssh -N -L 18005:127.0.0.1:8005 -p 40353 root@16.59.164.128

# Interactive shell (same host/port):
ssh -p 40353 root@16.59.164.128
```

| Field | Value |
|---|---|
| SSH | `root@16.59.164.128` port **40353** |
| Instance | Vast **59887402** (`secai-qwen38-bw4-bench2`) |
| Base URL (via tunnel) | `http://127.0.0.1:18005/v1` |
| Base URL (on instance) | `http://127.0.0.1:8005/v1` |
| Model | `orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4` |
| Auth | Bearer key in `/tmp/secai-serving-api-key-59887402` on the laptop that started serving (do not commit) |

```sh
curl -sS http://127.0.0.1:18005/v1/models -H "Authorization: Bearer $API_KEY"
curl -sS http://127.0.0.1:18005/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4","messages":[{"role":"user","content":"ping"}],"max_tokens":32}'
```

## Current live infrastructure

- Vast 4-GPU instance **59887402**, label `secai-qwen38-bw4-bench2`, **running**, about **$7.2667/hour**. Four 96-GB Server Edition GPUs, all SM120. Topology: GPUs 0–1 are `PIX`, 2–3 are `PIX`, cross-pair `NODE`; no NVLink. Host has about 1 TB RAM.
- **Use direct SSH**: `ssh -i ~/.ssh/secai2026 -o IdentitiesOnly=yes -p 40353 root@16.59.164.128` (or the two client keys above). Local tunnel: `127.0.0.1:18005` → remote `127.0.0.1:8005`.
- Checkpoint at `/workspace/qwen38-orcarouter`, HF revision `9c6f1345b523a9d9d70b41542a63b39e3c9b7ca6`. Runtime `0.33.0rc1.dev1098+g85b52c9f5`.
- Two vLLM replicas: A GPUs 0–1 `:8001`, B GPUs 2–3 `:8003`; nginx `least_conn` on `:8005`. Helpers: `/workspace/llm.sh`, `/workspace/bench_tuning.sh`, `/workspace/restart_a_flags.sh`, `/workspace/restart_a_baseline.sh`.
- Vast 5090 instance **59893527** is **stopped**. No aux models.

## Secret locations (never print their contents)

- Vast API key: `~/.config/vastai/vast_api_key`.
- Existing Hugging Face token: `~/.cache/huggingface/token`; used for the checkpoint download, not written to the instance filesystem.
- Current LLM serving Bearer key: `/tmp/secai-serving-api-key-59887402` (mode 0600). It was passed to remote startup and benchmarks over SSH stdin. vLLM receives it as `VLLM_API_KEY` environment variable; avoid putting it in argv or logs.

## Measured config and routing results

All numbers below are single screening runs with random prompts unless otherwise noted. JSON results are copied locally under ignored `results/instance-59887402/`; remote JSONs are under `/workspace/results/`.

- PLE TP4+EP4 on GPU: KV 2.86M logical tokens; 8K input/512 output/C16 **568 output tok/s**; 32K/512/C8 **231 tok/s**.
- PLE TP4+EP4 on CPU: KV 4.66M; corresponding **575** and **233 tok/s**. The speed difference is within the short-run noise; CPU PLE offers much more cache.
- TP4 without EP, CPU PLE: **570** and **228 tok/s** for those cases.
- Two TP2+EP2 replicas on PCIe pairs, CPU PLE: each KV 3.14M. Combined under simultaneous load: 8K/512/total C16 **~758 tok/s**; 32K/512/total C8 **~386 tok/s**. These combined totals approximate sum of both replicas' output tokens divided by the longer measured duration; benchmark starts are not perfectly synchronized.
- Isolated same-pair EP toggle: EP2 **647 tok/s** versus EP off **416 tok/s** for 8K/512/C8. At 32K/512/C4: EP2 **368** versus EP off **193 tok/s**. Hence keep EP2 on this checkpoint.
- Live vLLM logs show linear expert placement, 256/512 routed experts per EP rank, MARLIN NVFP4 MoE. DeepGEMM `nv_dev` supports SM120 grouped GEMM but explicitly excludes Mega MoE on SM120; checkpoint NVFP4 K-group 16 does not directly match the SM120 UE8M0 scale contract. See `blackwell/EXPERT_ROUTING.md`.

### Fixed-seed tuning screen on replica A (B loaded but idle)

`/workspace/bench_tuning.sh` came from `blackwell/bench_tuning.sh`; random dataset seed 42. Cases: prefill 8K/1/C1; prefill 32K/1/C1; decode 2K/1024/C1; mixed 8K/512/C8. Metrics: TTFT for prefill, output tok/s and TPOT for decode/mixed.

| Profile | Prefill 8K TTFT | Prefill 32K TTFT | Decode C1 tok/s / TPOT | Mixed C8 tok/s | KV tokens |
|---|---:|---:|---:|---:|---:|
| Baseline TP2+EP2, CPU PLE, batch 16K, no MTP | 511 ms | 1341 ms | 130 / 7.44 ms | 499 | 3.14M |
| Batch tokens 32K, otherwise baseline | 510 ms | 1319 ms | 130 / 7.45 ms | 502 | 2.85M |
| MTP depth 1, batch 16K | 501 ms | 1382 ms | 170 / 5.67 ms | 528 | 2.40M |
| MTP depth 2, batch 16K | 500 ms | 1381 ms | 182 / 5.27 ms | 549 | 2.29M |
| MTP depth 3, batch 16K | 499 ms | 1387 ms | 203 / 4.79 ms | 516 | 2.20M |
| Dynamic MTP (depth 3 at batch 1; depth 2 at batch 2–64) | 500 ms | 1380 ms | 188 / 5.16 ms | 536 | 2.19M |

The MTP method for this installed vLLM is `{"method":"mtp","num_speculative_tokens":N}`, not the method string from earlier notes. The model has one MTP layer; depths above 1 repeat that layer and vLLM warns acceptance may fall. MTP2 improves single decode 40% over baseline but reduces KV about 27%; MTP3 is faster C1 yet slower in the mixed C8 screen. Dynamic MTP was slower than MTP3 for C1 and MTP2 for C8 despite equal or lower KV, so reject it. Batch 32K has negligible prefill gain and loses KV capacity, so reject it. Prefill has not improved meaningfully with the tested flags.

### Dual-replica and CCU measurements (both MTP2)

- A direct side-by-side screen with A=MTP2 and B=baseline, both under load, gave 8K/512/C8 per replica **444 vs 413 output tok/s**; 32K/512/C4 **203 vs 193**. JSON: `results/instance-59887402/dual/mtp2_vs_baseline/`. These are single random-prompt runs and the earlier seed was reused, so prefer the fresh-seed sweep below for absolute capacity.
- A first CCU sweep reused seed 42 and returned a spurious high result on A due prefix cache warmed by prior tests. Its output is **invalid**. `blackwell/bench_ccu.sh` now uses a new seed for each concurrency level; results below are in `results/instance-59887402/ccu/mtp2_fresh_seeds/`.

| Total client CCU | Aggregate output tok/s, approximate | Worst replica TTFT p99 | Worst replica TPOT p50 | Completed |
|---:|---:|---:|---:|---:|
| 16 | 891 | 3.9 s | 14.7 ms | 32/32 |
| 32 | 1088 | 8.3 s | 22.5 ms | 64/64 |
| 64 | 1274 | 17.0 s | 39.3 ms | 128/128 |
| 128 | 1399 | 34.3 s | 70.8 ms | 256/256 |
| 256 (128 per replica, queued above `max-num-seqs`) | 1398 | 80.9 s | 84.3 ms | 512/512 |

Workload per request: random 8K input, 512 output, ignore EOS, seed `42000 + per-replica CCU`, 2×CCU prompts per replica, 4 warmups. Both pairs measured simultaneously. Aggregate rate is total output tokens divided by the longer of the two client benchmark durations, so treat it as approximate. The 64-per-replica `max-num-seqs` permits at most 128 *running* requests across both replicas; it is not a proven latency-safe CCU. The C256 queued screen (`results/instance-59887402/ccu/mtp2_queue256/`) reached the same ~1.4K output tok/s while TTFT p50 rose to ~47 s and p99 to ~81 s; throughput saturated near C128 for this workload. For interactive use the tested CCU16 has the lowest tail latency; CCU32 offers more output throughput if ~8 s TTFT p99 is tolerable. A true maximum or production SLA still depends on prompt lengths, output lengths, cache reuse, tool calls, and acceptable latency.

### Observed load and access, after C256 screen (2026-11-09 15:00:52 UTC)

- **During C256:** one point sample showed all four GPUs at 100% utilization; GPU memory used 91,879 / 91,879 / 95,795 / 95,795 MiB, power 354 / 359 / 370 / 370 W. Host RAM `free -h`: 312/999 GiB used, 687 GiB available. This is a point sample under benchmark load, not a time average.
- **Idle after C256:** GPUs at 0% utilization; same GPU memory allocations 91,879 / 91,879 / 95,795 / 95,795 MiB, power ~84–85 W. Host RAM 310/999 GiB used, 689 GiB available. `nvidia-smi --query-compute-apps` attributed those allocations to the four vLLM workers (91,742 MiB each on pair A, 95,658 MiB each on pair B); no benchmark GPU process remained. Both replicas reported 2,358,023 logical KV cache tokens each, despite the different resident allocation. Investigate the ~3.9 GiB/pair difference before raising GPU memory utilization.
- Ports 8001, 8003 and nginx 8005 all returned `/health` HTTP 200; authenticated `/v1/models` over local SSH tunnel `127.0.0.1:18005` returned HTTP 200 with one model.
- Nsight Systems profiling is complete for the first 32K prefill and C1 decode screens. The traces and current routing state are detailed below.

## Nsight Systems bottleneck investigation

- NVIDIA `nsight-systems-2026.5.2` was installed on the four-GPU instance from the CUDA APT repository. A tiny CUDA smoke trace succeeded; CPU perf sampling is blocked by the container's paranoid level, so traces use `--sample=none --cpuctxsw=none`.
- **Live routing restored:** nginx config `/etc/nginx/conf.d/blackwell-llm.conf` again includes A (`8001`) and B (`8003`); the pre-profile copy remains at `/etc/nginx/conf.d/blackwell-llm.conf.pre-nsys`. On 2026-11-09 15:27 UTC, both replicas and frontend `/health` returned 200, `nginx -t` passed, and authenticated `/v1/models` through local tunnel returned 200 with one model.
- Original A was safely stopped. First profiled A was launched under `nsys launch --session-new=qwen_a --trace=cuda,nvtx,nccl,cublas` (default CUDA Graph granularity), then profiled with two `nsys start/stop` windows. Trace files: `/workspace/results/nsys/vllm-a/prefill32k.nsys-rep` and `decode2k.nsys-rep`, also copied locally to `results/instance-59887402/nsys/`. Bench JSONs are adjacent remotely. First profiled A was stopped safely.
- CUDA Graph node-level decode was captured under `qwen_a_node` and saved as `/workspace/results/nsys/vllm-a/decode2k-nodes.nsys-rep`, also copied locally to `results/instance-59887402/nsys/`. The profiled process group was stopped safely, normal MTP2 A was relaunched through `/workspace/llm.sh`, and its authenticated inference smoke passed. The serving key remains at `/tmp/secai-serving-api-key-59887402`; pass by SSH stdin, never print it.
- Why node trace is necessary: first prefill trace has 17.55 s summed kernel time across both GPUs and only 0.031 s in whole CUDA Graph records; kernel summary is useful. First decode trace has **18.206 s of whole CUDA Graph records** but only 2.071 s in individually visible kernels, so its kernel summary misses the dominant decode work. Do not cite its naive 38.6% CUTLASS/14.4% sampling percentages as decode bottleneck.
- Prefill 32K/1/C1 trace: each GPU ~8.7–8.9 s summed kernel time over 4 prompts; corrected per-GPU category shares (broad kernel-name heuristics): NCCL collectives **27.6–28.0%**, other GEMM **20.3–20.8%**, Marlin/fused MoE expert compute **15.4–16.7%**, QSA including indexer **12.0–12.3%**, Hyperconnection **11.4–11.7%**, GDN **3.6–3.7%**, GPU PLE lookup **0.4%**, MoE router+dispatch **0.6%**. `blackwell/analyze_nsys.py` generates this from exported SQLite. These are summed GPU kernel durations, not causal wall-time shares; streams may overlap. NCCL time likely includes model-wide TP communication, not just expert dispatch.
- Decode node trace: one fresh-seed 2K-input/256-output/C1 request, no warmups, ~1.78 s client duration. Per-GPU summed kernel shares: generic GEMM **44.0–44.8%** (not assigned to model component; may include shared expert/draft/dense projections), MoE expert compute **17.9–18.7%**, Hyperconnection **9.8–10.0%**, cross-device reduce plus NCCL **8.8–10.9%**, QSA **2.9–3.0%**, GDN **2.9–3.0%**, MoE router+dispatch **4.6–4.7%**, PLE GPU lookup **0.1%**. This trace supports that sparse attention itself is small in decode, but it does **not** prove generic GEMMs are unrelated to MoE. Neither expert routing nor CPU PLE is established as the dominant decode bottleneck.
- CPU-wait check on the node-level decode trace: union of GPU kernel intervals per device occupied **1.641/1.781 s (92.1%)** on GPU0 and **1.674/1.781 s (94.0%)** on GPU1; idle gaps were ~0.140/0.107 s, maximum single gap ~12 ms. This strongly argues against a large CPU scheduling stall in this C1 request. The gaps are an upper bound for CPU waiting because they may include inter-kernel sync or memory operations. CPU perf sampling itself was unavailable, so the exact CPU cause of any gap remains unassigned.
- **Current bottleneck assessment:** prefill 32K has a large NCCL share (~28% of summed GPU kernel time) plus GEMM/MoE/QSA; decode C1 is primarily generic GEMM (~44–45%) and MoE expert compute (~18–19%). Router+dispatch is ~4–5% of decode kernel time; QSA and GDN are each ~3%; PLE GPU lookup is ~0.1%. QSA is not the main decode limiter on this workload. Under C256, four GPUs hit 100% util and throughput plateaued near 1.4K output tok/s. These are single synthetic traces, not general production shares.

### GEMM / NCCL attribution from existing nsys (2026-11-09)

Sources: `/workspace/results/nsys/vllm-a/decode2k-nodes.sqlite` and `prefill32k.sqlite` (local copies under `results/instance-59887402/nsys/`). Summed GPU kernel time; streams may overlap. NVTX here is mostly NCCL/cublas — **not** per-layer vLLM ranges — so GEMM roles use kernel families + checkpoint shapes (`hidden_size=2560`, MoE/shared intermediate 640, GDN `in_proj_qkv` 10240×2560).

**Decode generic GEMM (~44–45%, ~1.62 s both GPUs)** by family:

| Family | Share of generic GEMM | Notes |
|---|---:|---|
| CUTLASS WMMA BF16 | ~40% | Small-tile BF16 decode GEMM |
| Humming BF16 | ~35% | BF16 path; not Marlin |
| cuBLAS GEMV | ~20% | Classic M≈1 decode |
| Other CUTLASS/cublas | ~5% | Includes some relu tensorop |

Routed experts and shared-expert `gate/up/down` are NVFP4-scaled and already sit mostly in the separate **Marlin MoE** bucket (~18–19%). Kernel names do **not** split shared-expert vs MTP draft vs dense proj inside generic GEMM; that residual is dominated by **BF16/linear-style** GEMM/GEMV. Changing only the MoE backend cannot remove most of the 45%. MTP appears explicitly in a few percent of kernels (`gdn_decode_post_conv_mtp_kernel`, sampling); draft linears may still hide inside the BF16 GEMM families.

**Prefill 32K NCCL (~28%):** `AllReduce_bf16_RING` **~97.4%** of NCCL time; `AllGather_RING` ~2.6%. That is **TP all-reduce**, not EP dispatch. Decode NCCL is ~4% of kernels (AllReduce ~74%, AllGather ~26%) plus custom `cross_device_reduce_1stage` ~6%.

### MoE NVFP4 backend sweep (2026-11-09 ~16:20–16:31 UTC)

Forced `LLM_KERNEL_CONFIG={"moe_backend":...}` on replica A (TP2+EP2+MTP2). Compatibility with this host/checkpoint:

| `moe_backend` | Result | Reason |
|---|---|---|
| `flashinfer_b12x` / `b12x` | **Fail load** | Kernel does not support `ep_size=2` |
| `flashinfer_cutlass` / `cutlass` | **Fail load** | Quant scheme unsupported: `GroupShape(1,16)` + f8e4m3 scales × act `None` |
| `flashinfer_cutedsl` / `flashinfer_trtllm` | **Fail load** | Kernel does not support current device (SM120) |
| `humming` | **Loads** — `Using HUMMING NvFp4 MoE` | No Marlin weight-only warning |
| `auto` → Marlin | Baseline | Weight-only FP4 Marlin path |

Fixed-seed screen (`results/instance-59887402/tuning/moe_humming/`):

| Profile | Prefill 8K | Prefill 32K | Decode C1 | Mixed C8 |
|---|---:|---:|---:|---:|
| Marlin (`auto`, mtp2_recheck) | 501 ms | 1385 ms | **199** tok/s | 525 |
| Humming MoE | 494 ms | 1362 ms | **176** tok/s | **567** |

**Keep Marlin as default** for this EP2 serving stack: humming is ~11% slower on C1 decode; mixed C8 +8% on one seed is not enough to switch. FlashInfer “native” MoE paths are not usable here with EP2 and/or SM120 and/or K-group-16 layout. Optional later: re-test humming under high-CCU only, or try non-EP (would fight the measured EP2 win).

### Linear backend sweep (2026-11-09 ~15:55–16:10 UTC)

`llm.sh` now accepts `LLM_KERNEL_CONFIG` → `--kernel-config`. Fixed-seed screens on replica A (B idle), MTP2 unchanged. JSON: `results/instance-59887402/tuning/{linear_b12x,linear_humming}/`.

| Profile | Prefill 8K | Prefill 32K | Decode C1 tok/s | Mixed C8 | Notes |
|---|---:|---:|---:|---:|---|
| MTP2 recheck (`linear_backend=auto`) | 501 ms | 1385 ms | 199 | 525 | Baseline |
| `flashinfer_b12x` | 500 ms | 1380 ms | 204 | 547 | Log: *no kernel for this linear layer type; using automatic selection* — largely invalid as a forced-b12x trial |
| `humming` | 499 ms | 1378 ms | 196 | 540 | Loaded; no clear win vs auto on one seed |

Do **not** lock either override: gains are within short-run noise / MTP accept variance, and b12x did not stick on the critical linear path. MoE stayed Marlin NVFP4 in both runs. Replica A restored to matched MTP2 without `LLM_KERNEL_CONFIG`. Next linear candidates if retrying: `cutlass`, `flashinfer_cutlass`, or `linear_backend_per_quant` once quant-specific mapping is clearer. Prefer starting the **MoE NVFP4** sweep next (`moe_backend`), which targets a different bucket.

### Async-scheduling trial (2026-11-09 ~15:30–15:40 UTC)

Fixed-seed `bench_tuning.sh` (seed 42, `BENCH_TOKENIZER=/workspace/qwen38-orcarouter`) on A; B idle. JSON: `results/instance-59887402/tuning/{mtp2_recheck,mtp2_async}/`.

| Profile | Prefill 8K | Prefill 32K | Decode C1 tok/s / TPOT | Mixed C8 | MTP accept (decode) |
|---|---:|---:|---:|---:|---:|
| MTP2 recheck | 501 ms | 1385 ms | 199 / 4.81 ms | 525 | 58.8% |
| MTP2 + async-scheduling | 500 ms | 1381 ms | 181 / 5.31 ms | 549 | 48.8% |

**Reject** `--async-scheduling` here: prefill flat, C1 decode ~9% slower with lower MTP acceptance, mixed +4.5% on one screen. A was restored without async so both replicas match again.

## Safe next actions

1. **Closed this pass:** NCCL custom-AR keep (nocustom decode regress); lean CUDA graph sizes reject. MoE Marlin; linear `auto`; no async.
2. Optional later only if revisiting speed: `NCCL_ALGO=Tree`, MTP3-on-A / MTP2-on-B, FP8 KV, richer NVTX GEMM split, humming under high-CCU only.
3. Park PLE / DeepEP / expert-routing / custom SM120 / TRT-LLM / GDN-conv until new evidence.
4. Keep 5090 stopped. Client keys already on the instance — see **Client connection** above.

**Auto-destroy scheduled:** `2026-11-11 07:00` Asia/Ho_Chi_Minh via local systemd user timer `vast-destroy-59887402.timer` → `vastai destroy instance 59887402 -y`. Cancel: `systemctl --user disable --now vast-destroy-59887402.timer` (notes in `~/.local/share/vast-destroy/CANCEL.md`). Machine must stay on / linger enabled for the timer to fire.
