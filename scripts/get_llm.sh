#!/usr/bin/env bash
# Download a small instruction-tuned LLM (GGUF, 4-bit) for the WaveSense 3D AI assistant.
#   ./scripts/get_llm.sh            # Qwen2.5-3B-Instruct Q4_K_M (~2.0 GB, default, best accuracy)
#   ./scripts/get_llm.sh small      # Qwen2.5-1.5B-Instruct Q4_K_M (~1.0 GB, faster)
#   ./scripts/get_llm.sh llama      # Llama-3.2-3B-Instruct Q4_K_M (~2.0 GB)
# Then: pip install -r requirements-llm.txt   (on a Jetson, see README for the CUDA build)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models/llm
case "${1:-default}" in
  small) f=qwen2.5-1.5b-instruct-q4_k_m.gguf; u=https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/$f ;;
  llama) f=Llama-3.2-3B-Instruct-Q4_K_M.gguf; u=https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/$f ;;
  *)     f=qwen2.5-3b-instruct-q4_k_m.gguf;   u=https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/$f ;;
esac
curl -L --fail -C - --retry 5 -o "models/llm/$f" "$u"
echo "saved models/llm/$f - set \"assistant\": {\"model_path\": \"models/llm/$f\"} in config/room.json if it is not the default"
