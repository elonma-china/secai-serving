#!/usr/bin/env python3
"""Summarize Nsight Systems CUDA kernel time by broad serving component.

Run `nsys export --type sqlite --output trace.sqlite trace.nsys-rep` first.
Kernel durations are summed GPU time, not elapsed wall time; streams may overlap.
"""

import argparse
import sqlite3
from collections import defaultdict


def category(name: str) -> str:
    name = name.lower()
    if "nccl" in name:
        if "allreduce" in name or "all_reduce" in name:
            return "NCCL AllReduce"
        if "allgather" in name or "all_gather" in name:
            return "NCCL AllGather"
        if "reducescatter" in name or "reduce_scatter" in name:
            return "NCCL ReduceScatter"
        return "NCCL other"
    if "cross_device_reduce" in name:
        return "Cross-device reduce"
    if "topkgating" in name:
        return "MoE router"
    if any(x in name for x in ("moe_align", "count_and_sort_expert", "finalizemoerouting")):
        return "MoE dispatch"
    if any(x in name for x in ("marlin_moe", "moe_sum", "fused_moe", "moefcgemm")):
        return "MoE expert compute"
    if "qsa" in name or "persistent_topk" in name:
        return "QSA"
    if any(x in name for x in ("flashinfergdn", "causal_conv", "fused_post_conv", "delta_rule", "gdn_decode")):
        return "GDN"
    if "ple_embedding" in name or "engram" in name:
        return "PLE GPU lookup"
    if any(x in name for x in ("topk_topp", "_rejection", "_resample", "draft_tokens")):
        return "Sampling/speculation"
    if "_hc_" in name or "hc_down_silu" in name:
        return "Hyperconnection"
    # Prefer fine GEMM families before the residual bucket.
    if "humming<" in name or name.startswith("void humming"):
        return "GEMM humming BF16"
    if "gemvx" in name:
        return "GEMM cuBLAS GEMV"
    if "wmma_tensorop" in name:
        return "GEMM CUTLASS WMMA BF16"
    if "cutlass" in name and "tensorop" in name:
        return "GEMM CUTLASS tensorop"
    if "cublas" in name:
        return "GEMM cuBLAS other"
    if any(x in name for x in ("cutlass", "humming", "gemm", "cublas")):
        return "Other GEMM"
    return "Other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    db = sqlite3.connect(args.sqlite)
    grouped = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    names = defaultdict(lambda: [0, 0])
    rows = db.execute(
        "SELECT k.deviceId, s.value, k.end-k.start "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id=k.demangledName"
    )
    for device, name, duration in rows:
        value = grouped[device][category(name)]
        value[0] += duration
        value[1] += 1
        names[name][0] += duration
        names[name][1] += 1
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for device, categories in sorted(grouped.items()):
        total = sum(value[0] for value in categories.values())
        print(f"GPU {device}: {total / 1e9:.3f}s summed kernel time")
        for name, (duration, count) in sorted(categories.items(), key=lambda x: -x[1][0]):
            print(f"  {name:23} {duration / total * 100:5.1f}% {duration / 1e9:8.3f}s {count:7d} calls")
        if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables:
            graph_count, graph_time = db.execute(
                "SELECT count(*), coalesce(sum(end-start),0) "
                "FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE WHERE deviceId=?", (device,)
            ).fetchone()
            print(f"  CUDA graphs as whole: {graph_time / 1e9:.3f}s, {graph_count} launches")
    print("Top kernels across traced GPUs:")
    for name, (duration, count) in sorted(names.items(), key=lambda x: -x[1][0])[: args.top]:
        print(f"  {duration / 1e9:8.3f}s {count:7d} {name[:130]}")


if __name__ == "__main__":
    main()
