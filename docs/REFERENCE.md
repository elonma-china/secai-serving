# Tài liệu tham khảo: triển khai Docker, build, biến môi trường

Phần chi tiết tách khỏi [README](../README.md). Các đường dẫn `docker/...` và `stack/...` tính từ gốc repo.

## Triển khai trên máy 2x RTX PRO 6000 có Docker

Image `docker/Dockerfile` chứa mọi thứ: vLLM ghim đúng bản đã đo (`7c3a9302f`, theo tag và digest), nginx, thư mục `stack/` đã áp patch FP8 prefill, và **cả năm model nướng sẵn** (`/opt/secai/models`, revision ghim trong `stack/models.lock`, mỗi shard một layer). Máy đích chỉ cần Docker và NVIDIA runtime, không cài gói và không tải model. `/workspace` chỉ giữ trạng thái thay đổi (log, compile cache, KV tier, API key).

Chạy **trên máy 2x RTX PRO 6000** (kiểm tra từ chối mọi máy khác trước khi kéo gì về):

```bash
git clone https://github.com/elonma-china/secai-serving && cd secai-serving
docker/deploy.sh --generate-key          # tạo /srv/secai-serving/api.key (mode 600) nếu chưa có, không in ra màn hình
```

`docker/deploy.sh` chạy theo thứ tự:
1. `docker/preflight.sh`: đúng 2x RTX PRO 6000 >= 90 GB và đang rảnh, driver CUDA >= 13.2, nvidia container runtime, RAM trống >= 142 GiB, đĩa (image 110 GB + workspace 250 GB, cộng dồn nếu cùng filesystem), cổng trống, key mode 600, image kéo được. Có mục FAIL thì in cách sửa và thoát **trước khi kéo**.
2. `docker pull` image ghim theo digest trong `docker/release.env` (package công khai, không cần login).
3. `docker compose up -d` và chờ healthy.
4. `stack/smoke.py` trong container: một request thật cho mỗi model. Chỉ in `DEPLOYED` khi cả 5 model trả lời.

Tùy chọn: `--check-only` (chỉ bước 1), `--down` (dừng stack), `--workspace DIR`, `--image REF`, `--timeout SEC`, `--no-smoke`.

### Các lệnh khác của `docker/vast.sh`

```bash
./vast.sh offers                 # offer đạt mọi tiêu chí, rẻ nhất lên trên
./vast.sh check-offer <offer>    # kiểm offer trước khi thuê (miễn phí)
./vast.sh create <offer> --yes   # kiểm lại rồi thuê; thiếu --yes chỉ in lệnh (dry run)
./vast.sh check <id>             # preflight trong máy đã thuê (qua ssh)
./vast.sh key <id> api.key       # check, đạt thì cài key và stack mới boot
./vast.sh ports <id>             # URL công khai từng dịch vụ
./vast.sh tunnel <id>            # truy cập riêng qua SSH (localhost 38xxx)
./vast.sh destroy <id> --yes     # dừng tính tiền
```

### Image phát hành

`ghcr.io/elonma-china/secai-serving:2fe2825-20261120-warm` (code repo này + compile-cache seed, khởi động 118 giây trên máy 2x RTX PRO 6000 mới). Digest nằm trong `docker/release.env`, đó là thứ được triển khai (không dùng `latest`). Rollback: đặt tag/digest cũ vào `release.env` hoặc truyền `--image`.
Compile-cache seed gồm code GPU và một ít code CPU (kernel FlashInfer CuTe DSL). Trên CPU thiếu tập lệnh của CPU dựng seed (ví dụ không có AVX-512), boot bỏ các object đó và dựng lại (`docker/host_code.sh`, khoảng +30 giây); thiếu bước này replica LLM chết với "Illegal instruction".

### Build và phát hành

Cần khoảng 100 GB đĩa trống; trọng số lấy từ repo công khai `secai1999/*` đúng revision đã ghim.

```bash
docker/build.sh release [--push]            # vLLM + code + model; đổi code chỉ đẩy lại vài MB
docker/warm.sh <image> cache-seed.tar.zst   # trên máy 2x PRO 6000: boot nguội, làm nóng mọi kernel, đóng gói compile cache
docker/build.sh warm cache-seed.tar.zst [--push]   # release + cache seed: boot không phải biên dịch lại
# Chỉ đổi code/seed (không kéo 36 GB về): thêm một layer thẳng lên image trên registry, in tag + digest, rồi cập nhật docker/release.env
python3 docker/append_layer.py --base <repo@digest> --tag <sha>-<date>-warm --revision <sha> --add <dir>/stack=/opt/secai/stack --add <seed dir>=/opt/secai/cache-seed --keep-mtime
```

### Một lần boot làm gì

Những việc không nướng sẵn được (cần GPU): nạp trọng số lên GPU, capture CUDA graph, ghim 2x55 GiB RAM cho KV offload.
Trình tự: seed/khôi phục compile cache → chờ API key → `up.sh` (6 dịch vụ vLLM khởi động cùng lúc, mỗi dịch vụ có KV cố định theo `stack/kv_plan.json`; `STARTUP=serial` quay lại kiểu từng cái một) → watchdog (không healthy 60 giây thì `down.sh` + `up.sh`, tối đa 3 lần, sau đó thoát để `restart: unless-stopped` tiếp quản).
Log dịch vụ ra stdout (`docker logs`, tiền tố `[llm-a]`...) và `/workspace/logs`; thời điểm sẵn sàng từng dịch vụ trong `startup-timeline.log`; vLLM phát Prometheus `/metrics` ở 8000/8001/8002/8003.
Mọi cổng dịch vụ (8005, 8000, 8002, 8004, 18083) nghe trên `BIND_HOST` và cùng một API key. vLLM phía sau OCR (18303) và Hy-MT (18304) chỉ ở loopback; chỉ các adapter ra công khai.

### Chạy không cần Docker

```bash
MODEL_DIR=/workspace/models python3 stack/fetch_models.py && MODEL_DIR=/workspace/models bash stack/up.sh
```

Compile cache cũng chia sẻ được không qua image bằng `stack/cache_bundle.py` (repo HF riêng tư).
Chưa giải quyết: một máy là một điểm hỏng duy nhất, và API key là cơ chế kiểm soát truy cập duy nhất (không TLS), nên giữ các cổng khỏi internet.

### Chạy trực tiếp bằng script

```bash
API_KEY=<key> HF_TOKEN=<token> bash stack/up.sh            # khởi động/đảm bảo mọi dịch vụ (bỏ qua dịch vụ đang healthy)
bash stack/up.sh --dry-run                                 # chỉ in lệnh
python3 stack/tests/test_stack.py                          # test offline (hoặc chạy từng hàm test_*)
```

Cần 2 GPU >= 90 GB, RAM trống >= 142 GiB (2x55 GiB KV offload + dự phòng), `nginx`, và khoảng 160 GB đĩa trống cho tầng KV NVMe. Model lấy từ repo công khai `secai1999/*` trên Hugging Face; `HF_TOKEN` không bắt buộc (chỉ tăng giới hạn tải). `ocr_adapter.py` cũng tải prompt từ `secai1999/OCR-Autoscan-v1` lúc khởi động.

---

## Đã tối ưu gì

- **KV offload**: GPU → RAM ghim (55 GiB/replica) → NVMe (`TieringOffloadingSpec`, `kv_gc.sh` cắt còn 80 GiB/replica).
- **MTP speculative decoding**, 2 draft token (decode nhanh hơn 86% ở c=1, 64% ở c=16).
- **FP8 prefill attention** cho SM120 (`stack/patches/sm120_fp8_prefill.py`, `up.sh` tự áp): TTFT ở 120K token giảm 37%.
- Batch prefill 16K có CUDA graph, bỏ vision tower (OCR là model riêng), nginx chia trọng số 5:4 theo dung lượng KV.

Kết quả so với cấu hình trước (32 agent đồng thời, mỗi agent một phiên 16 lượt gọi tool, context tăng từ 2,6K lên khoảng 112K token; tổng 29,4M prompt token, khoảng 35 token ra mỗi lượt, nên đo prefill và tái dùng KV chứ không đo decode): thời gian chạy 1983 s → 586 s, TTFT p95 254 s → 61,6 s. Bảng đầy đủ và các lưu ý: [stack/benchmarks/BENCHMARKS.md](../stack/benchmarks/BENCHMARKS.md).

## Biến môi trường của `up.sh`

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `KV_OFFLOAD` | `tier` | `tier`, `native`, `lmcache`, `none` |
| `KV_OFFLOAD_GIB` / `KV_FS_MAX_GIB` / `KV_FS_DIR` | 55 / 80 / `/workspace/kvcache` | dung lượng tầng RAM và NVMe mỗi replica |
| `KV_EVICTION` | `lru` | chính sách tầng RAM, `lru` hoặc `arc` (ARC chưa thử ở đây) |
| `LLM_MTP` | `2` | số draft token, `0` là tắt |
| `LLM_KV_DTYPE` | rỗng (fp8 của checkpoint) | `nvfp4` = thêm 57-64% KV nhưng tắt kernel FP8 prefill |
| `VLLM_SM120_FP8_PREFILL` | `1` | `0` bỏ qua patch attention (`python3 stack/patches/sm120_fp8_prefill.py revert` khôi phục file gốc) |
| `LLM_UTIL_A` / `LLM_UTIL_B` | 0,90 / 0,74 | tỉ lệ bộ nhớ GPU; B dùng chung GPU1 với 4 model phụ |
| `LLM_BATCHED_TOKENS`, `LLM_MAX_NUM_SEQS`, `LLM_MAX_MODEL_LEN` | 16384, 64, 131072 | |
| `MODEL_DIR` | rỗng (trong image: `/opt/secai/models`) | chạy bản model cục bộ dưới tên trên Hub, hoàn toàn offline |
| `BIND_HOST` | `0.0.0.0` | giao diện của các cổng công khai (cả OCR); `127.0.0.1` = chỉ loopback |
| `STARTUP` | `auto` | `auto` = khởi động cùng lúc khi `stack/kv_plan.json` khớp cấu hình (không thì tuần tự); `serial`, `parallel_aux`, `parallel` ép chế độ. Boot nguội 516 s → 135-144 s, boot ấm 177 s → 74-81 s: [stack/benchmarks/STARTUP.md](../stack/benchmarks/STARTUP.md) |
| `CACHE_DIR` | `/workspace/cache` | cache torch.compile / Triton / FlashInfer (giữ qua các lần restart) |

## Công cụ benchmark và test (`stack/`)

`agent_sim.py` (tải nhiều agent), `decode_test.py`, `needle.py`, `nll.py` (kiểm chất lượng), `kv_check.py`, `ttft_check.py` (độ đúng/độ trễ của offload),
`atomic.sh`, `patch_ab.sh`, `sweep.sh`, `profile.sh` (nsys), `clocks.sh`, `glue_bench.py`. Các script này giả định bố cục của máy test (`/workspace/...`).
