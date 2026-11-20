# Startup time: serial vs parallel GPU1 services (2026-11-20)

Same Vast.ai host as the deployment benchmark (2x RTX PRO 6000 WS, machine 15336), image `ghcr.io/elonma-china/secai-serving:cbe8629-20261120` with the
`stack/` of this change copied over it. READY = every port healthy and `stack/smoke.py` passes (one real request per model). Times are seconds from the
start command, polled every 2 s (`startup-20261120/times.txt`). "Cold" = compile caches wiped; "warm" = caches kept from the previous run of the same mode.

## Why the GPU1 services started one after another, and what changed

By default vLLM sizes each KV cache from the GPU memory it *measures* after warm-up (budget minus everything that became resident since the process
started). Services starting at the same time on one GPU count each other's allocations as their own and get a smaller, unpredictable cache, so `up.sh`
started embedding, rerank, OCR, translation and replica B in a chain (about 430 s of the 516 s cold start). `--kv-cache-memory-bytes` fixes the size and skips
the measurement, so the order stops mattering. `stack/kv_plan.py` learns the sizes from the "Extended KV cache to N blocks (X GiB)" line of a serial run:
llm-b 32.8, OCR 6.4, rerank 1.97, translation 1.69 GiB; embedding is an encoder model without a KV cache (no flag). Each size is used minus 1%.

| `STARTUP` | what starts at t=0 | cold READY | warm READY |
|---|---|---|---|
| `serial` (previous behaviour) | llm-a; the GPU1 services one by one | 516 s | 177 s |
| `parallel_aux` | llm-a + the four auxiliary services; llm-b after them | 165 s (-68%) | 108 s (-39%) |
| `parallel` | all six | **135 s (-74%)** | **74 s (-58%)** |
| `auto` (new default) = `parallel` when the plan matches this configuration | all six | 144 s | 81 s (baked seed, below) |

Repeatability (`parallel`): cold 135, 135, 135, 138 s; warm 74, 81, 81 s; the serial runs reproduced the earlier deployment benchmark (510 s / 179 s).
In `parallel` every service is up when replica A is: llm-a answers at 73-78 s warm / 130-141 s cold, which is now the floor (its own engine start, weights
load in 1-3 s). Replica B is ready at 71-78 s warm / 120-127 s cold.

## Cost of fixed KV sizes

| | serial | parallel |
|---|---|---|
| KV tokens: llm-a / llm-b | 1,251,264 / 853,994 | 1,251,264 / 844,536 (-1.1%) |
| KV tokens: OCR / translation / rerank | 481,590 / 27,648 / 18,432 | 477,121 / 27,392 / 18,240 (-1%) |
| Peak GPU memory while booting (GPU0 / GPU1) | 85.3 / 89.8 GiB | 85.3 / 89.8-93.3 GiB (varies with how the transient peaks overlap) |
| Errors / OOM in the logs | none | none (warnings identical to serial) |

Serving after a `parallel` start (32 agents x 16 turns, 29.4M prompt tokens; the same load as the deployment benchmark): 32/32 correct, 0 errors,
run time 389 s, TTFT p50 / p95 9.5 / 36.4 s, decode 124 tok/s at c=1 and 1330 tok/s at c=16, needle 5/5 at ~118.6K tokens, peak GPU memory 95.8 / 92.4 GiB
(`agent32-summary.json`). Previous runs on this host: old flow 387 s, p95 37.9 s; prebuilt image 366 s, p95 30.0 s. Single runs: differences of a few percent are noise.

## Safety: `auto`

A fixed KV size is only valid for the configuration it was learned on. `kv_plan.json` carries a fingerprint (GPU model + vLLM version + the full serial
launch plan from `STARTUP=serial up.sh --dry-run`). `STARTUP=auto` (default) starts in parallel only if it matches, otherwise it starts serially and says why on
stderr; changing e.g. `LLM_UTIL_B` was verified to fall back. Explicit `serial` / `parallel_aux` / `parallel` always win. After changing launch arguments (flags,
memory fractions, context length, models) re-learn: run `STARTUP=serial`, then `python3 stack/kv_plan.py learn --logs /workspace/logs --out stack/kv_plan.json`.

## Other results

- Watchdog with `auto` (replica B SIGKILLed): restart began after 62 s, READY after 153 s (241 s with the serial start).
- Compile-cache seed (`stack/make_seed.sh`, 12 MB zstd / 178 MB, 2,859 files) copied to `/opt/secai/cache-seed`, cache dir empty: `boot.sh` logged
  "seeding compile cache from the image", `auto -> parallel`, READY after 81 s, i.e. a cold machine starts like a warm one. The seed matches only this
  image's configuration and GPU model.
- **The released image, end to end** (`ghcr.io/elonma-china/secai-serving:2fe2825-20261120-warm`, digest `sha256:00b9b4842a4d...`, the previous image plus one 26.5 MB
  layer with this code and the seed; its `up.sh`, `kv_plan.py`, `kv_plan.json` and `smoke.py` were checked identical to the repo by sha256): a fresh Vast
  instance, empty `/workspace`, only an API key, no environment variables, `boot.sh` as shipped: "seeding compile cache from the image", `auto -> parallel`,
  READY after **118 s** (embedding 45, rerank 78, OCR 102, translation/llm-a/llm-b 117 s), smoke 7/7. It is slower than the 81 s above because it is the first
  read of the 32 GB of weights from disk (page cache cold). Against the old flow's 553 s on the same machine that is -79%.
- Image load on a host that already cached the layers: 46-49 s (655 s on a host that had none). The release is a registry-side extra layer
  (`docker/append_layer.py`), so only 26.5 MB were pushed and nothing was pulled locally.

Limits: one host class; the plan and seed are specific to RTX PRO 6000 WS + vLLM `7c3a9302f` (the image pins both); `kv_check.py` still INCONCLUSIVE (not re-run).
