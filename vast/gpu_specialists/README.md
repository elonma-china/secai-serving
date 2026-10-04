# Năm model trên Vast

`serving-up-cyber-f1.sh` dựng Cyber-F1, **DuyTa/sec-embedding**, **DuyTa/sec-rerank** resident và hai API riêng trên loopback: MT `127.0.0.1:18868`, OCR `127.0.0.1:18083`. API phụ chỉ tải model trong child của request; hai API dùng cùng `/workspace/secai-gpu-specialists/admission.lock`. Không có hai model phụ resident đồng thời. Gateway kết nối qua SSH tunnel với hai token riêng; không publish hai cổng này bằng Vast.

```sh
# Offline: không gọi API/Vast CLI, SSH, thuê hay destroy.
./serving-up-cyber-f1.sh --dry-run

# Chỉ chạy khi thực sự muốn thuê. SSH/SCP chuyển bundle sau khi 3 API nền sẵn sàng.
./serving-up-cyber-f1.sh --yes --gpus 1 --disk 150 --ssh-identity ~/.ssh/id_ed25519

# Giữ flow 2GPU. Auxiliaries chọn GPU1; phép đo nvidia-smi chỉ quan sát GPU này.
./serving-up-cyber-f1.sh --yes --gpus 2 --specialist-gpu 1 --disk 150

# Chỉ 3 dịch vụ / tắt riêng một auxiliary.
./serving-up-cyber-f1.sh --dry-run --disable-specialists
./serving-up-cyber-f1.sh --dry-run --disable-ocr
```

Mặc định LLM dùng `--gpu-util 0.78` khi bật specialists, `0.80` khi tắt; embed/rerank giữ `0.05/0.06`. Khóa admission yêu cầu ít nhất **8192 MiB free** (6144 MiB model +2048 MiB reserve). Nếu thực tế không đủ, bootstrap dừng và báo lỗi; không chỉnh hay dừng resident. Có thể giảm `--gpu-util` và dựng lại instance được phép thuê. `--max-model-len`, `--llm-model`, `--embed-model`, `--rerank-model` hiện có vẫn được hỗ trợ. Việc override embed/rerank phải phù hợp model fine-tuned của gateway.

Repo này chạy độc lập: bundle `runtime.tar.gz` chứa nguyên các source cần thiết từ gateway commit `0650f9d7d0b40f9eefb94f3059e1516d842ac112`, gồm cả MT service/child supervisor và OCR GPU accounting helper. Kiểm SHA256 bundle và từng file theo `runtime-manifest.json`; không clone gateway trên instance, không cần parent checkout hay quyền đọc GitHub repo từ instance. Rebuild từ commit đã review:

```sh
python3 vast/gpu_specialists/make_bundle.py \
  --gateway-repo /path/to/secai-agent-gateway \
  --revision 0650f9d7d0b40f9eefb94f3059e1516d842ac112
```

## Instance hiện có

Installer chỉ chạm supervisor mới thuộc `/workspace/secai-specialist-services`. Operator phải drain và dừng **đúng hai process ASGI cũ** trước khi thay thế, kiểm GPU child đã chết và admission lock được thả; giữ ba resident chạy. Installer không kill process cũ không thuộc nó, không chạy Docker và không thuê/destroy instance. Token hiện có được giữ nguyên khi chỉ định file:

```sh
./serving-specialists-install.sh --host root@192.165.134.28 --port 10963 \
  --reuse-artifacts \
  --python /workspace/secai-translation-ocr/runtime/venv/bin/python \
  --mt-token-file /workspace/secai-gpu-specialists/runtime/mt-on-demand-58b2f62/private-token \
  --ocr-token-file /workspace/secai-translation-ocr/runtime/service.token \
  --smoke

./serving-specialists-install.sh --host root@192.165.134.28 --port 10963 --action restart --smoke
./serving-specialists-install.sh --host root@192.165.134.28 --port 10963 --action ready --smoke
./serving-specialists-install.sh --host root@192.165.134.28 --port 10963 --action stop
```

Kiểm đường dẫn token OCR trước khi dùng. `--reuse-artifacts` không tải model, cài pip/apt hay biên dịch. Nó kiểm dependencies của Python đã chỉ định, commit llama.cpp và SHA256 của binary/libraries được chấp nhận. Có thể chỉ định `--identity`, `--root`, `--gpu`, `--expected-incumbents` (mặc định3), `--shared-lock`, `--mt-port`, `--ocr-port`, đường dẫn model/binary/compat hoặc `--disable-mt`/`--disable-ocr`. Các token phải private `0600`, dài32–256 ký tự ASCII chữ/số/underscore/hyphen. Token mới sinh ngẫu nhiên nếu không có file; giá trị không xuất ra log.

Supervisor dùng stdlib, giữ lock riêng chống chạy trùng và tự restart worker crash sau5 giây. Dừng/restart đợi worker tự thu hồi child; không kill nhóm process hay GPU host PID. Nếu cleanup chưa chứng minh được, lock và evidence fail closed còn nguyên. Log ở `<root>/logs/{supervisor,mt,ocr}.log`. `config.json` private chứa GPU UUID, incumbent PIDs, các checksum và token **paths**. Incumbent restart làm runtime fail closed; chỉ recapture baseline bằng `boot` sau khi chứng minh ba resident sẵn sàng, không có child phụ. Onstart của rental mới gọi `boot` sau khi resident healthy để khôi phục sau container restart. Instance cũ cần operator thêm cùng lệnh vào startup đang dùng:

```sh
python3 /workspace/secai-specialist-services/bootstrap.py boot --root /workspace/secai-specialist-services
```

## Artifacts và readiness

- OCR: `Qwen/Qwen3-VL-2B-Instruct` revision `89644892e4d85e24eaac8bacfd4f463576704203`; weights SHA256 `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`. Config/tokenizer file digests lưu tại install, kiểm lại khi start/readiness; request chạy offline với `trust_remote_code=False`.
- MT: `tencent/Hy-MT2-7B-GGUF` revision `ab8472660ac61fac25f1af43fac2599d52a8a775`, `Hy-MT2-7B-Q4_K_M.gguf` SHA256 `9f96256500f3fc1ab4d64336b58f52a949a95ad7516b0c229476eef782f9f77b`. Backend trả model name `tencent/Hy-MT2-7B` theo contract gateway.
- llama.cpp: commit `11fe02151f79c41d0d4af7da708755d73b9c0da6`, full CUDA offload, context4096, EOS overrides được giữ nguyên. Máy mới build source pin với CUDA13/SM80, lưu SHA256 binary/libraries và compiler trong `build-cuda/secai-runtime-manifest.json` private; máy đã có build chỉ được reuse nếu trùng manifest hoặc bộ checksum đã chấp nhận trong bundle. Không giả định binary compile trên máy khác phải trùng byte.
- Python venv riêng: torch `2.11.0+cu130`, transformers `5.13.1`, FastAPI `0.136.3`, Pydantic `2.13.4`, Pillow `12.3.0`, uvicorn `0.51.0`, huggingface-hub `1.23.0`, safetensors `0.8.0`, accelerate `1.14.0`, httpx `0.28.1`. Không sửa pip env resident. MT-only bỏ dependency OCR.
- Bootstrap mới cần30 GiB disk trống. Cài build tools/CUDA toolkit13.0.2 khi thiếu; CUDA forward compat580.95.05 được lấy từ [NVIDIA package index](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/) và kiểm exact library SHA256 `f27223c58d4c0d2ead3c2d747eb30a530c449f89c42e16b23e9bef04f3e6dc2e`.

Readiness kiểm checksum, Bearer auth401 và private routes/model validation; không dùng `/health` để khẳng định model chạy. `--smoke` chạy lần lượt dịch thật và OCR ảnh chữ nhỏ, yêu cầu response200 có nội dung; rental mới luôn chạy bước này trước khi báo xong hoặc `--replace-old`. Đây là smoke về khả năng thực thi, không phải đánh giá chất lượng/dataset nghiệm thu.

```sh
python3 -m unittest discover -s tests -v
bash -n serving-up-cyber-f1.sh serving-specialists-install.sh vast/on_start_cyber_f1.sh
git diff --check
```
