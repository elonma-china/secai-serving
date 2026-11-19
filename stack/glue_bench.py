"""Cost of the per-layer K/V page gather (index_select) used by the FP8 prefill path, at real geometry.
A 120K-token prompt is prefilled in 16K-token chunks; each chunk gathers every page of its context for K and V, on 16 layers."""
import torch

page, hkv, d, layers, chunk, total = 64, 4, 256, 16, 16384, 120_000
pages_total = -(-total // page)
cache = torch.empty(pages_total, 2, page, hkv, d, dtype=torch.float8_e4m3fn, device="cuda")  # vLLM packs K|V per block
k, v = cache[:, 0], cache[:, 1]
def gather_ms(n_pages):
    idx = torch.arange(n_pages, device="cuda")
    for _ in range(3): k.index_select(0, idx); v.index_select(0, idx)
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(20): k.index_select(0, idx); v.index_select(0, idx)
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / 20
tot = 0.0
for c in range(-(-total // chunk)):
    ctx_pages = min(pages_total, -(-((c + 1) * chunk) // page))
    tot += gather_ms(ctx_pages) * layers
print(f"gather cost for one 120K prefill (all chunks, {layers} layers): {tot:.1f} ms  vs measured TTFT 13283 ms -> {tot / 13283 * 100:.2f}%")
