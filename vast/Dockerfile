# ─────────────────────────────────────────────────────────────────────────────
#  Thin vast.ai image — secAI serving (LLM + embedding + reranker, 2× A100)
# ─────────────────────────────────────────────────────────────────────────────
#  Mục tiêu: deploy vast.ai KHÔNG git-clone, KHÔNG apt lúc boot. Chỉ thêm 1 layer
#  mỏng lên trên image vLLM chính thức (base layer đã được host vast cache sẵn →
#  chỉ pull phần delta). Weights (~21GB) KHÔNG bake vào đây — tải từ HuggingFace
#  lúc chạy bằng hf_transfer (nhanh hơn & free hơn mọi registry).
#
#  Build (context = GỐC repo, cần vast/ + templates/ + nginx/):
#      docker build -f vast/Dockerfile -t justtuananh/secai-serving:vast .
#  hoặc dùng vast/build-push.sh.
#
#  Dùng trên vast.ai: Launch mode = SSH · On-start Script = bash /opt/serving/vast/on_start.sh
# ─────────────────────────────────────────────────────────────────────────────
FROM vllm/vllm-openai:latest

# Bật tải nhanh HF cho toàn image + cache HF ổn định trên đĩa instance.
ENV HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/workspace/hf-cache \
    DEBIAN_FRONTEND=noninteractive

# 1 layer: nginx (LB) + envsubst (gettext-base) + git/curl + hf_transfer.
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends nginx gettext-base git curl \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir hf_transfer

# Bake code đúng layout để on_start.sh tự định vị template → KHÔNG clone lúc boot.
# (on_start suy REPO_DIR = dirname(dirname(script)) = /opt/serving)
COPY vast/      /opt/serving/vast/
COPY templates/ /opt/serving/templates/
COPY nginx/     /opt/serving/nginx/
RUN chmod +x /opt/serving/vast/on_start.sh

# KHÔNG đặt ENTRYPOINT/CMD serve — vast chạy Launch mode SSH và gọi on_start.sh.
# Giữ nguyên entrypoint của base image (SSH mode sẽ override nó).
