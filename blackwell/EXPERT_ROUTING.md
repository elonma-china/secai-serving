# Expert routing on the measured 4× RTX PRO 6000 Server Edition host

The host has two `PIX` PCIe pairs, GPUs 0–1 and 2–3. Traffic between pairs is
`NODE`; there is no NVLink. Keep each TP2+EP2 replica within one pair and
balance independent requests across the two replicas. The tested checkpoint
is `orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4` with PLE on host RAM,
131072 maximum model length, 16384 batched tokens, 64 sequences, and vLLM
nightly commit `85b52c9f5309c2f179b40cafa563de0c2f714922`.

| Profile | 8K in / 512 out, total C16 | 32K in / 512 out, total C8 |
|---|---:|---:|
| TP4+EP4, PLE CPU | 575 output tok/s; TTFT 4.03 s | 233 tok/s; TTFT 6.86 s |
| TP4 without EP, PLE CPU | 570 tok/s; TTFT 4.14 s | 228 tok/s; TTFT 7.18 s |
| 2×TP2+EP2, PLE CPU | ~758 tok/s; TTFT 3.41 s | ~386 tok/s; TTFT 3.55 s |

The two-replica totals are the sum of both replicas' output tokens divided by
the longer benchmark duration. The two benchmark clients started together,
but the durations exclude their warmups and their exact start times may
differ. Treat these as screening results, not production SLO measurements.
Each TP2 engine has 3.14M tokens of KV capacity; TP4+EP4 with CPU PLE has
4.66M. At single concurrency, one TP2 engine produced 119 tok/s (2K input)
and 46.6 tok/s (127K input); both requests completed.

With the other replica loaded but idle, the same 0–1 GPU pair gave the
following isolated EP toggle results. This confirms EP2 is beneficial for
this checkpoint even though EP4 had little effect in the TP4 screen:

| One TP2 engine | 8K input, C8 | 32K input, C4 |
|---|---:|---:|
| EP2 on | 647 output tok/s; TTFT 0.32 s | 368 tok/s; TTFT 0.45 s |
| EP off | 416 tok/s; TTFT 2.72 s | 193 tok/s; TTFT 3.62 s |

The live vLLM logs show **linear placement**, 256 of 512 routed experts per
EP rank, and **MARLIN NVFP4 MoE**. We did not change the model's top-k router,
and this benchmark did not enable EPLB or a DeepEP communication backend.

## What the DeepSeek repos imply

- [DeepInfer](https://github.com/deepseek-ai/DeepInfer) currently lists
  DeepSeek-V4.1-Flash and NVIDIA SM100 support. Its production serving load
  balancers and KV cache system are not in the public release. It is a design
  reference, not a drop-in Qwen3.8/SM120 engine.
- [DeepInfer's deployment notes](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md)
  describe expert load balancing with redundant hot experts and periodic
  re-placement over NVLink. That mechanism needs measured expert load and
  spare weight memory. It should be evaluated only if vLLM reports meaningful
  expert imbalance on this workload.
- [DeepGEMM `nv_dev`](https://github.com/deepseek-ai/DeepGEMM/tree/nv_dev)
  **does support SM120**: BF16/FP8/FP4 dense and grouped GEMM, among other
  paths. The branch README explicitly excludes Mega MoE on SM120. Its SM120
  FP4 GEMM uses UE8M0 scales with K granularity 32 or 128, while this
  checkpoint declares NVFP4 tensor groups of 16; using that GEMM would need
  a verified format conversion and integration. The SM100 NVFP4 Mega MoE
  path does not make SM120 Mega MoE available. DeepGEMM's GEMM library alone
  does not choose expert placement.
- [DeepEP](https://github.com/deepseek-ai/DeepEP) supplies dispatch/combine.
  Its [experimental `hybrid-ep` PCIe mode](https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep)
  routes through an RDMA NIC even for intra-node traffic; the current Vast
  container exposes no InfiniBand device. It is not a validated dispatch
  replacement for this host. vLLM's
  [EP deployment guide](https://github.com/vllm-project/vllm/blob/main/docs/serving/expert_parallel_deployment.md)
  warns that DeepEP's high-throughput and low-latency modes target
  disaggregated serving and can perform poorly on mixed workloads.

For this config/routing pass, keep the measured 2×TP2+EP2 placement. A later
expert-optimization pass would first record per-expert token counts, rank
imbalance and communication time under real agent/RAG traffic. It would then
compare vLLM's EPLB and all-to-all choices on this PCIe topology, with model
output checks. The random-token benchmark cannot establish hot-expert skew.
