#!/usr/bin/env bash
set -euo pipefail

# vllm-mistral-rocm-bench/run_vllm_server.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source vllm_env/bin/activate

python -m vllm.entrypoints.openai.api_server \
  --model "$SCRIPT_DIR/models/Mistral-7B-Instruct-v0.2" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096
