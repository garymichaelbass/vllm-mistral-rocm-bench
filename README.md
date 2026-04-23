# vllm-mistral-rocm-bench/README.md

# vLLM Mistral-7B-Instruct Benchmark (ROCm, No Docker)

# Package Contents

# vLLM server (ROCm native, no containers)
# Mistral-7B-Instruct-v0.2 model loading
# Python benchmarking client
# Tokens/sec + latency measurement
# Log parsing
# SQL storage (SQLite for simplicity, zero dependencies)
# Grafana visualization (Grafana OSS installed natively, no Docker)

# Execution Flow

#  1. Run vLLM server with Mistral-7B-Instruct-v0.2 (ROCm)
#  2. Python client sends prompts, measures:
#     - latency
#     - tokens/sec
#  3. Logs stored in JSONL
#  4. Logs parsed into SQLite
#  5. Grafana visualizes metrics (via SQLite plugin)

## ✅ 1. System Preparation (Ubuntu ROCm 7.2)
echo "Step 1. System Preparation (Ubuntu ROCm 7.2)"
sudo apt update
sudo apt install -y python3 python3-venv python3-dev gcc g++ make git wget curl

## ✅ 2. Install ROCm‑compatible PyTorch
echo "Step 2. Install ROCm‑compatible PyTorch"
python3 -m venv vllm_env
source vllm_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2

## ✅ 3. Install vLLM (ROCm build)
echo "Step 3. Install vLLM (ROCm build)"
pip install "vllm>=0.5.0" --extra-index-url https://download.pytorch.org/whl/rocm7.2

## ✅ 4. Download Mistral-7B-Instruct-v0.2 Weights (Hugging Face, no token required)
echo "Step 4. Download Mistral-7B-Instruct-v0.2 Weights (Hugging Face)"
pip install huggingface_hub
# No login required for Mistral-7B-Instruct-v0.2
mkdir -p models
cd models

# Use huggingface-cli download for better reliability on large files
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

## ✅ 5. Start vLLM Server (ROCm, no Docker)
echo "Step 5. Start vLLM Server (ROCm, no Docker)"

# Note: Ensure run_vllm_server.sh points to ./models/Mistral-7B-Instruct-v0.2
chmod +x run_vllm_server.sh
./run_vllm_server.sh

echo "Server now listens at :  http://127.0.0.1:8000/v1"

## ✅ 6. Python Benchmark Client (latency + tokens/sec)
echo "Step 6. Python Benchmark Client (latency + tokens/sec)"

## Run Benchmark Client
source vllm_env/bin/activate
pip install openai
python client_run.py

## ✅ 7. Store Metrics in SQLite (no server required)
echo "Step 7. Store Metrics in SQLite (no server required)"

## Parse Logs into SQLite
python parse_and_store.py

## ✅ 8. Install Grafana (native, no Docker)
echo "Step 8. Install Grafana (native, no Docker)"
# Install Grafana:
wget https://dl.grafana.com/oss/release/grafana_11.0.0_amd64.deb
sudo dpkg -i grafana_11.0.0_amd64.deb

sudo systemctl enable grafana-server
sudo systemctl start grafana-server

# Install SQLite plugin:
echo "Install SQLite plugin"
sudo grafana-cli plugins install frser-sqlite-datasource
sudo systemctl restart grafana-server

# Open Grafana:
SERVER_IP=$(curl -s ifconfig.me)
echo "Server IP is $SERVER_IP"
echo "Open Grafana at http://$SERVER_IP:3000"
echo "user: admin"
echo "pass: admin"

## ✅ 9. Add SQLite as a Grafana Data Source
echo "Step 9. Add SQLite as a Grafana Data Source"
echo "Grafana supports SQLite via plugin"

sudo grafana-cli plugins install frser-sqlite-datasource
sudo systemctl restart grafana-server

echo "In Grafana:   Data Source -> Add -> SQLite"
echo "In Grafana:   Path: /home/ubuntu/vllm-mistral-rocm-bench/metrics.db"

# Path: /home/ubuntu/vllm-mistral-rocm-bench/metrics.db
# Create dashboards using queries in README.
echo "In Grafana: Create dashboards using queries in README."

## ✅ 10. Example Grafana Queries
echo "Step 10. Example Grafana Queries"

echo "Grafana Query: Tokens/sec over time"
echo "  SELECT ts AS \"time\", tokens_per_sec FROM metrics ORDER BY ts;"
echo "Gary_Not_Necessary  FROM metrics"
echo "Gary_Not_Necessary  ORDER BY ts"

echo "Grafana Query: Latency over time"
echo "  SELECT ts AS \"time\", latency_s FROM metrics ORDER BY ts;"
echo "Gary_Not_Necessary  FROM metrics"
echo "Gary_Not_Necessary  ORDER BY ts"

echo "Grafana Query: Average performance per prompt"
echo "  SELECT prompt, avg(tokens_per_sec) AS avg_tps, avg(latency_s) AS avg_latency FROM metrics GROUP BY prompt;" 
echo "Gary_Not_Necessary FROM metrics"
echo "Gary_Not_Necessary  GROUP BY prompt"

echo "Program now completed."

