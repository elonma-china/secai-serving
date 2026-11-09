# DEPLOY.md — legacy A100 serving trên vast.ai (version `v1.0`)

> Mục tiêu mới 4× RTX PRO 6000 Blackwell + RTX 5090 nằm ở
> [blackwell/README.md](blackwell/README.md). Tài liệu bên dưới chỉ dành cho
> profile A100 cũ.

> Dự án này CHỈ lo hạ tầng model: **LLM + embedding + reranker** (6 replica vLLM + nginx LB)
> trên máy 2×A100 thuê ở vast.ai. Kết quả là **3 URL** — dán vào script `deploy.sh` của
> bundle app (`/opt/nghiem_thu/secai-customer-2026-06-23-A100/`). App KHÔNG nằm ở đây.
>
> Code instance boot = tag **`v1.0`** (commit `ccb6ca9`) của fork
> `justtuananh/infra_serving_quick_start` — bất biến; đổi code → tạo tag mới + sửa `TAG=`
> trong `serving-up.sh`.

## Kiến trúc
```
Máy chạy app (VPS thường — bundle secai-customer)     vast.ai 2×A100 (dự án này)
┌───────────────────────────────┐                     ┌────────────────────────────┐
│ backend :4096 · web :3000     │──── Bearer API_KEY ──▶ 8001 LLM   (Qwen3-30B AWQ) │
│ qdrant 654k · neo4j · swagger │                     │ 8000 embed (bge-m3)        │
│  ./deploy.sh --llm ... ◀──────┼──── 3 URL ──────────│ 8002 rerank(Qwen3-0.6B)    │
└───────────────────────────────┘                     └────────────────────────────┘
```
Vì sao tách: vast không có A100 hỗ trợ VM-mode (khảo sát 2026-07-02) nên app compose không
chạy được trên máy vast; DB cần máy bền; app không cần GPU.

## Chuẩn bị 1 lần
```bash
~/.local/bin/vastai set api-key <KEY>    # key: https://cloud.vast.ai/account/ (nạp credit ở Billing)
```
(script tự cài `vastai` CLI nếu thiếu)

## Dựng serving — MỘT LỆNH
```bash
./serving-up.sh                 # search → chọn máy → thuê → chờ ~10-15' → IN 3 URL
./serving-up.sh --yes           # không hỏi, lấy offer rẻ nhất
./serving-up.sh --offer <ID>    # thuê đúng máy đã nhắm
./serving-up.sh --replace       # zero-downtime: cái mới healthy xong mới destroy cái cũ
./serving-up.sh --dry-run       # xem lệnh, không tốn tiền
```
Cuối script in sẵn lệnh `deploy.sh` hoàn chỉnh để copy-paste sang máy app.

Cờ khác: `--max-price` (mặc định 2.5) · `--disk` (100) · `--api-key` (sk-secai2026) ·
`--hf-token` · `--timeout` (1800) · `--thin` (dùng image bake sẵn `justtuananh/secai-serving:vast`
— build 1 lần bằng `vast/build-push.sh` trên máy có docker, boot nhanh hơn ~1-2').

## Xem / dừng
```bash
./serving-status.sh             # instance đang thuê + probe 3 endpoint
./serving-down.sh <ID>          # destroy — NGỪNG tính tiền (hỏi xác nhận)
./serving-down.sh <ID> --stop   # tạm dừng, giữ weights (phí lưu trữ nhỏ, start lại nhanh)
./serving-down.sh --all         # destroy mọi instance secai-*
```

## Model (khớp với app — ĐỪNG đổi embedding)
| Service | Model | Ghi chú |
|---|---|---|
| LLM :8001 | `cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit` | quantization ĐỂ TRỐNG (auto-detect compressed-tensors; ép `awq_marlin` là lỗi) |
| Embed :8000 | `BAAI/bge-m3` | **BẮT BUỘC** — index KB dựng bằng nó, đổi là recall ≈ 0 |
| Rerank :8002 | `Qwen/Qwen3-Reranker-0.6B` | serve-args đặc thù đã trong `vast/on_start.sh` |

## Troubleshooting
| Triệu chứng | Xử lý |
|---|---|
| Chờ port mãi | `vastai logs <ID>` — nhìn marker `═══ on_start done` / lỗi đỏ |
| Endpoint 000 lâu dù on_start done | IP/port vast có thể đổi lúc provisioning — script tự re-read và log "endpoint đổi"; probe tay: `./serving-status.sh` |
| Endpoint 401 | API key lệch — script die sớm; so `--api-key` với env `API_KEY` của instance |
| Thuê fail "offer taken" | Máy vừa bị thuê mất — chạy lại chọn offer khác |
| create lỗi nhưng nghi đã tạo | `vastai show instances` — nếu có instance mới thì nó ĐANG tính tiền |

## Chi phí (tham khảo 2026-07-02)
2×A100 80GB: $1.34–2.27/h (instance production hiện tại: `43496792`, Korea, $1.79/h).
Không dùng → `./serving-down.sh <ID>` (dựng lại chỉ 10-15').

## Bảo mật
Port vast nằm trên IP công cộng dùng chung, KHÔNG TLS — chỉ Bearer `API_KEY` bảo vệ
(vLLM validate; nginx không inject). Đặt key mạnh qua `--api-key`. Cần TLS → Cloudflare tunnel.

## File
```
serving-up.sh       # ⭐ thuê + dựng + in 3 URL
serving-status.sh   # xem + probe
serving-down.sh     # stop/destroy
lib.sh              # hàm chung
vast/               # code chạy TRÊN instance (on_start.sh, Dockerfile thin image, ...)
templates/ nginx/   # phụ thuộc của on_start (reranker jinja, LB reference)
```
