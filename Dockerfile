FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PATH="/root/.local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# uv manages Python 3.12 + virtualenv — no system Python needed
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Clone ACE-Step v1.5 and patch pyproject.toml because the 'lightning' package was removed from PyPI
RUN git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git . \
    && sed -i 's/"lightning>=2.0.0"/"pytorch-lightning>=2.0.0"/g' pyproject.toml

# Install Python 3.11 and create venv
RUN uv python install 3.11 \
    && uv venv .venv --python 3.11
ENV PATH="/app/.venv/bin:$PATH"

# Install the project in editable mode using uv
# uv automatically handles local sources (like nano-vllm) and PyTorch indices defined in pyproject.toml
RUN uv pip install -e .

# Add RunPod + I/O extras + hf_transfer for blazing fast model downloads
RUN uv pip install runpod requests soundfile hf_transfer

# Bake DiT weights (4B XL SFT — best quality) into HF cache so initialize_service
# finds them instantly; no download on cold-start.
RUN uv run python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='ACE-Step/acestep-v15-xl-sft'); \
print('DiT weights ready')"

# Bake LM weights (0.6B) to a stable path — LLMHandler reads from checkpoint_dir/lm_model_path
RUN uv run python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='ACE-Step/acestep-5Hz-lm-0.6B', \
                  local_dir='/app/checkpoints/acestep-5Hz-lm-0.6B'); \
print('LM weights ready')"

COPY runpod_handler.py .

CMD ["uv", "run", "python", "-u", "runpod_handler.py"]
