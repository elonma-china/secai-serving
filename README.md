# SecAI serving: 2x RTX PRO 6000 (Blackwell) vLLM stack

One host, two RTX PRO 6000 (96 GB), everything on vLLM. Context 131072. Tuned for many concurrent multi-turn agent sessions.

| Port | Service | Where |
|---|---|---|
| 8005 | LLM (nginx, session-affine) -> 8001 / 8003 | `secai1999/LLM-Autoscan-v1`, one TP1 replica per GPU |
| 8000 / 8002 | embedding / rerank | `secai1999/Embedding-Autoscan-v1`, `secai1999/Rerank-Autoscan-v1` (GPU1) |
| 8004 | translation adapter -> vLLM | `secai1999/Translate-Autoscan-v1` (GPU1) |
| 18083 | OCR adapter (`POST /v1/ocr`) | `secai1999/OCR-Autoscan-v1` (GPU1) |

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

Run this **on the 2x RTX PRO 6000 host** (the check refuses any other machine before pulling anything):
```bash
git clone https://github.com/elonma-china/secai-serving && cd secai-serving
docker/deploy.sh --generate-key          # creates /srv/secai-serving/api.key (mode 600) if missing; never printed
```
`docker/deploy.sh` does, in order: (1) `docker/preflight.sh` - exactly 2x RTX PRO 6000 >= 90 GB, both idle, driver CUDA >= 13.2,
nvidia container runtime, >= 142 GiB free RAM, disk (110 GB image + 250 GB workspace; summed if they share a filesystem), free
ports, key mode, image pullable. Any FAIL prints a fix and the script exits **without pulling**; (2) `docker pull` of the image
pinned by digest in `docker/release.env` (public package, no login); (3) `docker compose up -d` and wait for health;
(4) `stack/smoke.py` inside the container: one real request per model (chat via nginx + both replicas, embedding, rerank,
translate, OCR). It prints `DEPLOYED` only if every model answered. `--check-only` runs step 1 alone; `--down` stops the stack;
`--workspace DIR`, `--image REF`, `--timeout SEC`, `--no-smoke` are available.

Vast.ai (the instance runs the image itself, so there is no Docker host to check): the machine is checked twice.
```bash
docker/vast.sh offers                       # offers that meet every criterion, cheapest first
docker/vast.sh check-offer <offer>          # before renting (free): 2x RTX PRO 6000 >= 90 GB, CUDA >= 13.2, RAM, disk, bandwidth, reliability
docker/vast.sh create <offer> --yes         # re-checks the offer, then rents; boot waits for the API key, no model is loaded yet
docker/vast.sh key <id> api.key             # first `check <id>` = preflight.sh --in-container over ssh (GPUs idle, RAM, /dev/shm,
                                            # workspace, models in the image); the key is installed (and the stack starts) only if it passes
docker/vast.sh ports <id>                   # public URL of each service, incl. OCR (all need the API key, none has TLS)
docker/vast.sh tunnel <id>                  # optional private access via SSH; `destroy <id>` stops the billing
```
The compile-cache seed is GPU code plus a little CPU code (FlashInfer CuTe DSL kernels). On a CPU without the seed CPU's
instructions (e.g. no AVX-512) boot drops those objects and rebuilds them (`docker/host_code.sh`, ~+30 s); without that the
LLM replicas die with "Illegal instruction".
Released image: `ghcr.io/elonma-china/secai-serving:2fe2825-20261120-warm` (code of this repo + a compile-cache seed; starts in 118 s on a fresh 2x RTX PRO 6000 host) (digest in `docker/release.env`, which is what gets deployed; `latest` is never used). Roll back by putting the previous tag/digest in `release.env` or passing `--image`.

**Build and release** (needs ~100 GB free disk; the weights come from the public `secai1999/*` repos at the pinned revisions):
```bash
docker/build.sh release [--push]            # vLLM + code + models; code-only changes re-push a few MB
docker/warm.sh <image> cache-seed.tar.zst   # on a 2x PRO 6000 host: cold start, warm every kernel, pack the compile cache
docker/build.sh warm cache-seed.tar.zst [--push]   # release + cache seed: boot skips recompilation
# Code / seed change only (no local 36 GB pull): one extra layer straight on the registry image, tag + digest printed, then update docker/release.env
python3 docker/append_layer.py --base <repo@digest> --tag <sha>-<date>-warm --revision <sha> --add <dir>/stack=/opt/secai/stack --add <seed dir>=/opt/secai/cache-seed --keep-mtime
```
What a boot still does (cannot be baked, needs the GPUs): load weights onto the GPUs, capture CUDA graphs, pin 2x55 GiB of
RAM for KV offload. Boot sequence: seed/restore compile cache -> wait for the key -> `up.sh` (all six vLLM services start at once with a fixed
KV size per service from `stack/kv_plan.json`; `STARTUP=serial` restores the one-by-one start) -> watchdog (unhealthy 60 s -> `down.sh`
+ `up.sh`, 3 times, then exit so `restart: unless-stopped` takes over). Service logs go to stdout (`docker logs`, prefixed
`[llm-a]`...) and to `/workspace/logs`; per-service ready times are in `startup-timeline.log`; vLLM serves Prometheus
`/metrics` on 8000/8001/8002/8003. Every service port (8005, 8000, 8002, 8004, 18083) listens on `BIND_HOST` and needs the same API key (`Authorization: Bearer <key>`), so a client on any machine only needs the URL and the key; `BIND_HOST=127.0.0.1` limits them to loopback (tunnel / reverse proxy). The vLLM behind OCR (18303) and Hy-MT (18304) stay on loopback; only the adapters are public.

Without Docker: `MODEL_DIR=/workspace/models python3 stack/fetch_models.py && MODEL_DIR=/workspace/models bash stack/up.sh`.
Compile caches can also be shared without the image through `stack/cache_bundle.py` (private HF repo). Not covered: one
host is a single point of failure, and the API key is the only access control (no TLS) - keep the ports off the internet.

## Hướng dẫn chạy trên Vast.ai (tiếng Việt, đã chạy thật 2026-11-23)

Mọi lệnh dưới đây chạy từ thư mục `docker/` và đã kiểm trên máy thuê thật. Cần `vastai` đã đăng nhập và đủ credit
(`vastai show user`). Máy 2x RTX PRO 6000 giá khoảng 3,1 USD/giờ: **xong việc phải `destroy`**.

**1. Tìm và kiểm offer (miễn phí)**
```bash
./vast.sh offers                    # cột đầu là ID offer, rẻ nhất lên trên
./vast.sh check-offer 61282662      # phải in "OFFER OK" (GPU, CUDA, RAM, đĩa, mạng, độ tin cậy)
```
Nên chọn offer có CPU AVX-512 (AMD EPYC 9xxx); CPU khác vẫn chạy nhưng boot chậm thêm ~30 giây.

**2. Thuê (mất tiền) rồi chờ `running`**
```bash
./vast.sh create 61282662 --yes     # in "Started. {... 'new_contract': 62088958}" -> 62088958 là ID instance
vastai show instance 62088958 --raw | python3 -c 'import json,sys; print(json.load(sys.stdin)["actual_status"])'
```
Chạy `create` **một lần**; nếu lỗi, kiểm `vastai show instances` trước khi thử lại. Trạng thái `loading` (kéo image 36 GB) mất vài phút.

**3. Kiểm máy, cài key, chờ boot**
```bash
./vast.sh check 62088958            # phải in "PREFLIGHT OK"
./vast.sh key 62088958 api.key      # api.key = file chứa đúng key (mode 600); tự chạy lại check rồi mới cài key
ssh -p <ssh_port> root@<ssh_host> tail -f /workspace/logs/boot.log   # lệnh in ra ở bước trên; xong khi thấy "stack up in ~150s"
```

**4. Kiểm 5 model bằng smoke chính thức** (một request thật cho mỗi model)
```bash
ssh -p <ssh_port> root@<ssh_host> 'cd /opt/secai/stack && python3 smoke.py'   # kết quả cuối: READY
```

**5. Lấy URL công khai và thử từng dịch vụ**
```bash
./vast.sh ports 62088958
# LLM (nginx)  http://<ip>:45419   embedding :44782   rerank :44044   translate :44073   OCR :44409
curl -s -H "Authorization: Bearer $(cat api.key)" http://<ip>:45419/v1/models      # -> secai1999/LLM-Autoscan-v1
curl -s -H "Authorization: Bearer $(cat api.key)" -H 'Content-Type: application/json' \
  http://<ip>:45419/v1/chat/completions \
  -d '{"model":"secai1999/LLM-Autoscan-v1","messages":[{"role":"user","content":"Say OK"}],"max_tokens":32}'
```
Tên model phải đúng: `secai1999/LLM-Autoscan-v1`, `secai1999/Embedding-Autoscan-v1`, `secai1999/Rerank-Autoscan-v1`
(sai tên -> 404). Cổng OCR công khai chỉ có trong image build **sau** PR #9; image `2fe2825-20261120-warm` giữ OCR ở loopback,
nên `ports` vẫn liệt kê nhưng không kết nối được. Dùng tunnel:
```bash
./vast.sh tunnel 62088958 &         # 38005 LLM, 38000 embed, 38002 rerank, 38004 translate, 38083 OCR (chỉ localhost)
curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $(cat api.key)" http://127.0.0.1:38083/health   # 200
```

**6. Trỏ gateway (`secai-docker/.env`) sang máy mới**
```bash
cp -p .env .env.bak-$(date +%Y%m%d)                       # luôn sao lưu trước
sed -i -E 's#^QWEN_BASE_URL=.*#QWEN_BASE_URL=http://<ip>:45419/v1#; s#^EMBEDDING_API_URL=.*#EMBEDDING_API_URL=http://<ip>:44782#; s#^RERANKER_API_URL=.*#RERANKER_API_URL=http://<ip>:44044#' .env
docker compose up -d --no-deps --no-build gateway
docker compose config --services | grep -E '^pentestcode' | xargs docker compose up -d --no-deps --no-build   # pentestcode đọc QWEN_BASE_URL
curl -s http://127.0.0.1:8131/health   # llm/qwen/embedding/reranker = true
```
Embedding/rerank **không** có `/v1` ở cuối URL; LLM thì có. Dùng `xargs` (không dùng `$(...)` trần) vì zsh không tách từ.
`/health` có thể vẫn `degraded` vì lý do khác (ví dụ `kb_sync`); xem trường `problems`. Dịch/OCR của gateway dùng cấu hình
riêng (loopback 18868/18083), không đổi bằng ba biến trên.

**7. Dừng tính tiền / quay lại**
```bash
./vast.sh destroy 62088958 --yes    # mất toàn bộ dữ liệu trên máy
cp -p .env.bak-<ngày> .env && docker compose up -d --no-deps --no-build gateway   # quay lại endpoint cũ (rồi pentestcode như bước 6)
```

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
| `BIND_HOST` | `0.0.0.0` | interface of the public ports (incl. OCR); `127.0.0.1` = loopback only |
| `STARTUP` | `auto` | `auto` = all services start at once when `stack/kv_plan.json` matches this configuration (else serial); `serial`, `parallel_aux`, `parallel` force a mode. Cold start 516 s -> 135-144 s, warm 177 s -> 74-81 s: [stack/benchmarks/STARTUP.md](stack/benchmarks/STARTUP.md) |
| `CACHE_DIR` | `/workspace/cache` | torch.compile / Triton / FlashInfer caches (persist across restarts) |

## Benchmark and test tools (`stack/`)

`agent_sim.py` (multi-agent load), `decode_test.py`, `needle.py`, `nll.py` (quality gates), `kv_check.py`, `ttft_check.py` (offload correctness/latency),
`atomic.sh`, `patch_ab.sh`, `sweep.sh`, `profile.sh` (nsys), `clocks.sh`, `glue_bench.py`. They expect the layout of the test instance (`/workspace/...`).
