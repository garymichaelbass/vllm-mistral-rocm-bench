# vllm-mistral-rocm-bench/README_old.md

# vLLM LLaMA‑7B Benchmark (ROCm, No Docker)

This package provides an end‑to‑end benchmark pipeline:

1. Run vLLM server with LLaMA‑7B (ROCm)
2. Python client sends prompts, measures:
   - latency
   - tokens/sec
3. Logs stored in JSONL
4. Logs parsed into SQLite
5. Grafana visualizes metrics (via SQLite plugin)

## Setup

sudo apt update
sudo apt install -y python3 python3-venv python3-dev gcc g++ make git wget curl

python3 -m venv vllm_env
source vllm_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

## Download LLaMA‑7B

huggingface-cli login
mkdir -p models
cd models
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

## Run vLLM Server

chmod +x run_vllm_server.sh
./run_vllm_server.sh

## Run Benchmark Client

source vllm_env/bin/activate
python client_run.py

## Parse Logs into SQLite

python parse_and_store.py

## Grafana

Install Grafana:
  wget https://dl.grafana.com/oss/release/grafana_11.0.0_amd64.deb
  sudo dpkg -i grafana_11.0.0_amd64.deb
  sudo systemctl enable grafana-server
  sudo systemctl start grafana-server

Install SQLite plugin:
  sudo grafana-cli plugins install frser-sqlite-datasource
  sudo systemctl restart grafana-server

Open Grafana:
  http://<server-ip>:3000

Add SQLite datasource:
  Path: /path/to/vllm_bench/metrics.db

Create dashboards using queries in README.
