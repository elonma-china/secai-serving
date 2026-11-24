# SecAI serving: 2x RTX PRO 6000 (vLLM)

Một máy 2 GPU RTX PRO 6000 (96 GB) chạy toàn bộ model bằng vLLM, context 131072.

| Cổng | Dịch vụ |
|---|---|
| 8005 | LLM `secai1999/LLM-Autoscan-v1` (nginx chia 2 replica, giữ phiên theo `X-Session-Id`) |
| 8000 | embedding `secai1999/Embedding-Autoscan-v1` |
| 8002 | rerank `secai1999/Rerank-Autoscan-v1` |
| 8004 | dịch `secai1999/Translate-Autoscan-v1` |
| 18083 | OCR `secai1999/OCR-Autoscan-v1` |

Mọi cổng dùng chung một API key (`Authorization: Bearer <key>`), không có TLS: đừng mở ra internet.

## Thuê máy trên Vast.ai và chạy thử

Giá khoảng 3,1 USD/giờ, tính tiền đến khi `destroy` (bước 7). Tổng khoảng 30 phút.
Làm một mạch trong **cùng một cửa sổ terminal** (các biến chỉ sống trong cửa sổ đó). Cần `vastai` đã đăng nhập và còn credit.

### Bước 1. Tạo key và chọn offer

```bash
cd /home/ict/AIEngines/secai/secai-serving/docker
(umask 077; python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > api.key)
OFFER=$(./vast.sh offers | sed 's/\x1b\[[0-9;]*m//g' | awk '$1=="1"{print $2; exit}')
./vast.sh check-offer "$OFFER"
```

Dòng cuối phải là `OFFER OK`. Nếu không (offer vừa bị người khác thuê), chạy lại bước này.

### Bước 2. Thuê máy (bắt đầu tính tiền)

```bash
ID=$(./vast.sh create "$OFFER" --yes | grep -o "'new_contract': [0-9]*" | grep -o '[0-9]*$'); echo "ID=$ID"
until [ "$(vastai show instance "$ID" --raw | python3 -c 'import json,sys; print(json.load(sys.stdin)["actual_status"])')" = running ]; do sleep 15; done; echo running
```

Chạy `create` một lần. `ID` rỗng thì chạy `vastai show instances` kiểm tra trước khi thử lại. Chờ `running` mất khoảng 15 phút (đang kéo image 36 GB).

### Bước 3. Kiểm máy, cài key, chờ boot

```bash
PORT=$(vastai show instance "$ID" --raw | python3 -c 'import json,sys; print(json.load(sys.stdin)["ssh_port"])')
HOST=$(vastai show instance "$ID" --raw | python3 -c 'import json,sys; print(json.load(sys.stdin)["ssh_host"])')
until ./vast.sh check "$ID" | grep -q 'PREFLIGHT OK'; do sleep 20; done; echo "PREFLIGHT OK"
./vast.sh key "$ID" api.key
until ssh -p "$PORT" root@"$HOST" 'grep -q "stack up in" /workspace/logs/boot.log' 2>/dev/null; do sleep 15; done; echo "stack up"
```

Vài lần đầu `Connection refused` là bình thường (ssh chưa mở). Quá 5 phút chưa `PREFLIGHT OK` thì Ctrl-C, chạy `./vast.sh check "$ID"` xem mục FAIL; máy không đạt thì `./vast.sh destroy "$ID" --yes`.

### Bước 4. Smoke test (một request thật cho mỗi model)

```bash
ssh -p "$PORT" root@"$HOST" 'cd /opt/secai/stack && python3 smoke.py'
```

Phải có 7 dòng `PASS` và dòng cuối `READY`.

### Bước 5. Thử từ máy của bạn

```bash
eval "$(./vast.sh ports "$ID" | awk '/^embedding/{print "EMB_URL="$2} /^rerank/{print "RERANK_URL="$2} /^translate/{print "MT_URL="$2} /^LLM/{print "LLM_URL="$3}')"
KEY=$(cat api.key)
curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' "$LLM_URL/v1/chat/completions" -d '{"model":"secai1999/LLM-Autoscan-v1","messages":[{"role":"user","content":"Say OK"}],"max_tokens":32}'
curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' "$EMB_URL/v1/embeddings" -d '{"model":"secai1999/Embedding-Autoscan-v1","input":["xin chao"]}' | head -c 150
curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' "$RERANK_URL/v1/rerank" -d '{"model":"secai1999/Rerank-Autoscan-v1","query":"capital of France","documents":["Paris is the capital of France.","Bananas are yellow."]}' | head -c 300
curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' "$MT_URL/v1/translate" -d '{"text":"Good morning","source_lang":"en","target_lang":"vi"}'
```

Mong đợi: câu trả lời chứa `OK`; vector embedding; rerank cho tài liệu số 0 điểm cao nhất; bản dịch `Chúc buổi sáng tốt lành`. Sai tên model sẽ nhận 404.
OCR đã được bước 4 kiểm. Cổng OCR công khai chưa dùng được với image hiện tại (build trước PR #9); nếu cần thì dùng `./vast.sh tunnel "$ID"`.

### Bước 6. Trỏ gateway sang máy mới (tùy chọn)

Gateway và toàn bộ container `pentestcode` bị khởi động lại, các phiên đang chạy sẽ đứt. Bỏ qua bước này nếu chỉ muốn thử máy.

```bash
cd /home/ict/AIEngines/secai/secai-docker
KEY=$(cat ../secai-serving/docker/api.key)
BAK=.env.bak-$(date +%Y%m%d-%H%M%S); cp -p .env "$BAK"
sed -i -E "s#^QWEN_BASE_URL=.*#QWEN_BASE_URL=$LLM_URL/v1#; s#^EMBEDDING_API_URL=.*#EMBEDDING_API_URL=$EMB_URL#; s#^RERANKER_API_URL=.*#RERANKER_API_URL=$RERANK_URL#; s#^(QWEN_API_KEY|EMBEDDING_API_KEY|RERANKER_API_KEY)=.*#\1=$KEY#" .env
docker compose up -d --no-deps --no-build gateway
docker compose config --services | grep -E '^pentestcode' | xargs docker compose up -d --no-deps --no-build
sleep 20; curl -s http://127.0.0.1:8131/health
```

Trong `/health`: `llm`, `qwen` là `true`, mục `serving` trỏ tới IP máy mới. `status` vẫn có thể `degraded` vì lý do khác (xem `problems`).

### Bước 7. Hoàn lại gateway và dừng tính tiền

```bash
cd /home/ict/AIEngines/secai/secai-docker
cp -p "$BAK" .env && docker compose up -d --no-deps --no-build gateway
docker compose config --services | grep -E '^pentestcode' | xargs docker compose up -d --no-deps --no-build
cd ../secai-serving/docker
./vast.sh destroy "$ID" --yes
```

Cuối cùng in `instances left on the account: N`. Bỏ qua hai dòng đầu nếu không làm bước 6.

## Tài liệu khác

[docs/REFERENCE.md](docs/REFERENCE.md): triển khai Docker, build image, biến môi trường, kết quả tối ưu. Benchmark: [stack/benchmarks/](stack/benchmarks/).
