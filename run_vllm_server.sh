#!/usr/bin/env bash
# vllm-mistral-rocm-bench/run_vllm_server.sh
# Updated: 20260423  (vLLM 0.19.1 — uses `vllm serve` CLI)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source vllm_env/bin/activate

# `vllm serve` is the current recommended entry point (replaces
# `python -m vllm.entrypoints.openai.api_server` used in older releases).
vllm serve "$SCRIPT_DIR/models/Mistral-7B-Instruct-v0.2" \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --served-model-name "mistralai/Mistral-7B-Instruct-v0.2"
