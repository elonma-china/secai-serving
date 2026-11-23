# SecAI serving: stack vLLM trên 2x RTX PRO 6000 (Blackwell)

Một máy có 2 GPU RTX PRO 6000 (96 GB), mọi dịch vụ chạy bằng vLLM, context 131072. Tối ưu cho nhiều phiên agent nhiều lượt chạy đồng thời.

| Cổng | Dịch vụ | Model / vị trí |
|---|---|---|
| 8005 | LLM (nginx, giữ phiên theo `X-Session-Id`) → 8001 / 8003 | `secai1999/LLM-Autoscan-v1`, mỗi GPU một replica TP1 |
| 8000 / 8002 | embedding / rerank | `secai1999/Embedding-Autoscan-v1`, `secai1999/Rerank-Autoscan-v1` (GPU1) |
| 8004 | adapter dịch → vLLM | `secai1999/Translate-Autoscan-v1` (GPU1) |
| 18083 | adapter OCR (`POST /v1/ocr`) | `secai1999/OCR-Autoscan-v1` (GPU1) |

Mọi cổng công khai dùng chung **một API key** (`Authorization: Bearer <key>`), không có TLS: đừng gửi dữ liệu thật qua các cổng này và đừng mở ra internet.
`BIND_HOST=127.0.0.1` giới hạn các cổng về loopback (dùng tunnel hoặc reverse proxy). Client gửi `X-Session-Id` để một phiên luôn rơi vào một replica và tái dùng KV cache.

---

## Chạy trên Vast.ai: từng bước (đã chạy thật ngày 2026-11-23)

Máy 2x RTX PRO 6000 giá khoảng **3,1 USD/giờ**, tính tiền từ lúc bấm thuê đến lúc `destroy`. Làm một mạch từ bước 1 đến bước 7 trong **cùng một cửa sổ terminal** (các biến `ID`, `OFFER`... chỉ sống trong cửa sổ đó). Toàn bộ lệnh chạy từ thư mục `docker/` của repo này, trên zsh hoặc bash.
Tổng thời gian khoảng 30 phút, trong đó khoảng 15 phút là Vast kéo image 36 GB.

Yêu cầu: `vastai` đã đăng nhập, `python3`, `ssh`, `curl`, và (cho bước 6) `docker compose` của gateway.

### Bước 1. Chuẩn bị: kiểm tài khoản và tạo API key

```bash
cd /home/ict/AIEngines/secai/secai-serving/docker
vastai show user --raw | python3 -c 'import json,sys; print("credit:", round(json.load(sys.stdin)["credit"], 2))'   # cần còn vài chục USD
vastai show instances --raw | python3 -c 'import json,sys; print("instance đang thuê:", len(json.load(sys.stdin)))'  # nên là 0, nếu >0 thì máy cũ vẫn đang tính tiền
(umask 077; python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > api.key)   # tạo key mới, mode 600
ls -l api.key                                                                         # phải thấy -rw-------
```

`api.key` đã nằm trong `.gitignore`, không bị commit. Key này là key của **máy mới**; ở bước 6 gateway sẽ được đổi sang đúng key này.

### Bước 2. Chọn offer và kiểm tra (miễn phí)

```bash
./vast.sh offers                                                                 # danh sách offer, rẻ nhất lên trên
OFFER=$(./vast.sh offers | sed 's/\x1b\[[0-9;]*m//g' | awk '$1=="1"{print $2; exit}')   # lấy offer dòng 1
echo "OFFER=$OFFER"
./vast.sh check-offer "$OFFER"                                                   # phải in "OFFER OK" ở dòng cuối
```

- Nếu không in `OFFER OK` hoặc báo `not found`: offer đã bị người khác thuê, chạy lại cả khối này (danh sách đổi liên tục, có khi chỉ sau vài phút).
- Muốn chọn offer khác dòng 1: gán tay `OFFER=<ID ở cột đầu của bảng>`. Nên chọn CPU có AVX-512 (AMD EPYC 9xxx, Threadripper PRO 7xxx/9xxx); CPU khác vẫn chạy nhưng boot chậm thêm khoảng 30 giây.

### Bước 3. Thuê máy (bắt đầu tính tiền)

```bash
ID=$(./vast.sh create "$OFFER" --yes | grep -o "'new_contract': [0-9]*" | grep -o '[0-9]*$')
echo "ID=$ID"                       # phải là một dãy số, ví dụ 62141717
```

- Chạy `create` **đúng một lần**. Nếu `ID` rỗng (offer vừa biến mất hoặc lỗi), chạy `vastai show instances` kiểm tra trước khi thử lại, đừng thuê chồng.
- Chờ máy sang `running` (kéo image mất khoảng 15 phút, lúc này `loading` là bình thường):

```bash
until [ "$(vastai show instance "$ID" --raw | python3 -c 'import json,sys; print(json.load(sys.stdin)["actual_status"])')" = running ]; do sleep 15; done; echo running
```

### Bước 4. Kiểm tra máy rồi cài key

```bash
PORT=$(vastai show instance "$ID" --raw | python3 -c 'import json,sys; print(json.load(sys.stdin)["ssh_port"])')
HOST=$(vastai show instance "$ID" --raw | python3 -c 'import json,sys; print(json.load(sys.stdin)["ssh_host"])')
until ./vast.sh check "$ID" | grep -q 'PREFLIGHT OK'; do sleep 20; done; echo "PREFLIGHT OK"   # chờ ssh sẵn sàng rồi kiểm GPU, RAM, đĩa, model
./vast.sh key "$ID" api.key         # chạy lại kiểm tra, đạt thì mới chép key lên máy và stack mới bắt đầu boot
```

`running` chưa có nghĩa là ssh đã mở: vài lần đầu có thể thấy `Connection refused`, vòng `until` tự thử lại. Nếu quá 5 phút vẫn chưa `PREFLIGHT OK`, bấm Ctrl-C và chạy `./vast.sh check "$ID"` để xem mục nào FAIL. Máy không đạt thì `key` không cài key và không có gì chạy: chuyển thẳng sang `./vast.sh destroy "$ID" --yes`.
Chờ stack boot xong (khoảng 2 đến 3 phút):

```bash
until ssh -p "$PORT" root@"$HOST" 'grep -q "stack up in" /workspace/logs/boot.log' 2>/dev/null; do sleep 15; done; echo "stack up"
```

### Bước 5. Smoke test: một request thật cho mỗi model

```bash
ssh -p "$PORT" root@"$HOST" 'cd /opt/secai/stack && python3 smoke.py'
```

Kết quả phải có 7 dòng `PASS` (LLM qua nginx, replica A, replica B, embedding, rerank, translate, ocr) và dòng cuối `READY`.

### Bước 6. Thử từng dịch vụ từ máy của bạn qua URL công khai

```bash
./vast.sh ports "$ID"               # in URL của từng dịch vụ
eval "$(./vast.sh ports "$ID" | awk '/^embedding/{print "EMB_URL="$2} /^rerank/{print "RERANK_URL="$2} /^translate/{print "MT_URL="$2} /^LLM/{print "LLM_URL="$3} /^OCR/{print "OCR_URL="$2}')"
KEY=$(cat api.key)

curl -s -H "Authorization: Bearer $KEY" "$LLM_URL/v1/models"                                  # có "secai1999/LLM-Autoscan-v1"
curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' "$LLM_URL/v1/chat/completions" \
  -d '{"model":"secai1999/LLM-Autoscan-v1","messages":[{"role":"user","content":"Say OK"}],"max_tokens":32}'   # có "OK" trong content
curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' "$EMB_URL/v1/embeddings" \
  -d '{"model":"secai1999/Embedding-Autoscan-v1","input":["xin chao"]}' | python3 -c 'import json,sys; print("dim", len(json.load(sys.stdin)["data"][0]["embedding"]))'   # dim 1024
curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' "$RERANK_URL/v1/rerank" \
  -d '{"model":"secai1999/Rerank-Autoscan-v1","query":"capital of France","documents":["Paris is the capital of France.","Bananas are yellow."]}'   # index 0 điểm cao nhất
curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' "$MT_URL/v1/translate" \
  -d '{"text":"Good morning","source_lang":"en","target_lang":"vi"}'                       # "Chúc buổi sáng tốt lành"
```

Tên model phải đúng như trên (sai tên sẽ nhận 404). Embedding và rerank **không** có `/v1` ở cuối URL gốc; LLM thì có (`$LLM_URL/v1`).

**OCR**: image hiện tại (`2fe2825-20261120-warm`) được build trước PR #9, nên cổng OCR công khai (`$OCR_URL`) chưa kết nối được và OCR không nhận API key. Kiểm bằng tunnel (smoke ở bước 5 đã gọi OCR thật và đạt):

```bash
./vast.sh tunnel "$ID" & TUN_PID=$!; sleep 6        # 38005 LLM, 38000 embed, 38002 rerank, 38004 translate, 38083 OCR (chỉ localhost)
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:38083/health      # 200
kill $TUN_PID                                       # đóng tunnel
```

Khi dùng image build sau PR #9, `$OCR_URL` kết nối được và dùng chung API key như các cổng khác.

### Bước 7. Trỏ gateway (`secai-docker/.env`) sang máy mới

Gateway và toàn bộ container `pentestcode` bị khởi động lại, nên các phiên đang chạy sẽ đứt. Ba biến key của gateway (`QWEN_API_KEY`, `EMBEDDING_API_KEY`, `RERANKER_API_KEY`) phải đổi sang key của máy mới, nếu không gateway nhận 401.

```bash
KEY=$(cat api.key)
cd /home/ict/AIEngines/secai/secai-docker
BAK=.env.bak-$(date +%Y%m%d-%H%M%S); cp -p .env "$BAK"; ls -l "$BAK"        # sao lưu trước, nhớ tên file này cho bước 8
sed -i -E "s#^QWEN_BASE_URL=.*#QWEN_BASE_URL=$LLM_URL/v1#; s#^EMBEDDING_API_URL=.*#EMBEDDING_API_URL=$EMB_URL#; s#^RERANKER_API_URL=.*#RERANKER_API_URL=$RERANK_URL#; s#^(QWEN_API_KEY|EMBEDDING_API_KEY|RERANKER_API_KEY)=.*#\1=$KEY#" .env
docker compose up -d --no-deps --no-build gateway
docker compose config --services | grep -E '^pentestcode' | xargs docker compose up -d --no-deps --no-build   # pentestcode đọc QWEN_BASE_URL (qua ORCA_LOCAL_BASE_URL)
sleep 20; curl -s http://127.0.0.1:8131/health
```

Trong `/health`: `llm` và `qwen` phải là `true`; mục `serving` có `embedding` và `reranker` với `ok: true`, `endpoint` là IP máy mới.
`status` vẫn có thể là `degraded` vì lý do khác (ví dụ `kb_sync` quá 36 giờ): xem trường `problems`. Dịch và OCR của gateway dùng cấu hình riêng (loopback 18868 / 18083), không đổi bằng các biến trên.
Dùng `xargs` (không dùng `$(...)` trần) vì zsh không tách từ.

### Bước 8. Hoàn lại gateway và dừng tính tiền

```bash
cd /home/ict/AIEngines/secai/secai-docker
cp -p "$BAK" .env && docker compose up -d --no-deps --no-build gateway      # quay lại endpoint cũ
docker compose config --services | grep -E '^pentestcode' | xargs docker compose up -d --no-deps --no-build
sleep 20; curl -s http://127.0.0.1:8131/health                              # endpoint là máy cũ
cd /home/ict/AIEngines/secai/secai-serving/docker
./vast.sh destroy "$ID" --yes       # mất toàn bộ dữ liệu trên máy; cuối cùng in "instances left on the account: N"
```

Nếu muốn **giữ** máy mới làm endpoint chính: bỏ hai lệnh khôi phục ở trên, chỉ giữ lại file `$BAK` làm bản sao lưu.

Mở terminal mới giữa chừng: gán lại `ID=<id>` (xem bằng `vastai show instances`) rồi chạy lại hai lệnh `PORT=` / `HOST=` ở bước 4 và lệnh `eval` ở bước 6.

---

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

Kết quả so với cấu hình trước (32 agent đồng thời, mỗi agent một phiên 16 lượt gọi tool, context tăng từ 2,6K lên khoảng 112K token; tổng 29,4M prompt token, khoảng 35 token ra mỗi lượt, nên đo prefill và tái dùng KV chứ không đo decode): thời gian chạy 1983 s → 586 s, TTFT p95 254 s → 61,6 s. Bảng đầy đủ và các lưu ý: [stack/benchmarks/BENCHMARKS.md](stack/benchmarks/BENCHMARKS.md).

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
| `STARTUP` | `auto` | `auto` = khởi động cùng lúc khi `stack/kv_plan.json` khớp cấu hình (không thì tuần tự); `serial`, `parallel_aux`, `parallel` ép chế độ. Boot nguội 516 s → 135-144 s, boot ấm 177 s → 74-81 s: [stack/benchmarks/STARTUP.md](stack/benchmarks/STARTUP.md) |
| `CACHE_DIR` | `/workspace/cache` | cache torch.compile / Triton / FlashInfer (giữ qua các lần restart) |

## Công cụ benchmark và test (`stack/`)

`agent_sim.py` (tải nhiều agent), `decode_test.py`, `needle.py`, `nll.py` (kiểm chất lượng), `kv_check.py`, `ttft_check.py` (độ đúng/độ trễ của offload),
`atomic.sh`, `patch_ab.sh`, `sweep.sh`, `profile.sh` (nsys), `clocks.sh`, `glue_bench.py`. Các script này giả định bố cục của máy test (`/workspace/...`).
