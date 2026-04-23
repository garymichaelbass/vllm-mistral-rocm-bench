#!/usr/bin/env bash
# vllm-mistral-rocm-bench/deploy_all.sh
# Execution:   bash deploy_all.sh
# Updated:     20260423  (ROCm 7.2.2, vLLM 0.19.1, Grafana 13.0.1)

set -euo pipefail

# ─────────────────────────────────────────────
# Detect Ubuntu release (24.04 = noble, 22.04 = jammy)
# ─────────────────────────────────────────────
UBUNTU_CODENAME=$(lsb_release -sc)   # e.g. "noble" or "jammy"
UBUNTU_VERSION=$(lsb_release -sr)    # e.g. "24.04" or "22.04"

echo "Detected Ubuntu ${UBUNTU_VERSION} (${UBUNTU_CODENAME})"

if [[ "$UBUNTU_CODENAME" != "noble" && "$UBUNTU_CODENAME" != "jammy" ]]; then
    echo "ERROR: Only Ubuntu 24.04 (noble) and 22.04 (jammy) are supported."
    exit 1
fi

# ─────────────────────────────────────────────
# Stop unattended-upgrades immediately — it grabs the apt lock on first boot
# and will block every apt command below for 10-20 minutes if left running.
# ─────────────────────────────────────────────
echo "Stopping unattended-upgrades to free apt lock..."
sudo systemctl stop unattended-upgrades
sudo systemctl disable unattended-upgrades
sudo killall apt apt-get unattended-upgrade 2>/dev/null || true
sleep 2

ROCM_VERSION="7.2.2"
VLLM_VERSION="0.19.1"
ROCM_WHEEL_TAG="rocm721"           # vLLM wheel tag for ROCm 7.2.x

# ─────────────────────────────────────────────
# Package Contents
# ─────────────────────────────────────────────
# vLLM server (ROCm native, no containers)
# Mistral-7B-Instruct-v0.2 model loading
# Python benchmarking client
# Tokens/sec + latency measurement
# Log parsing
# SQL storage (SQLite, zero dependencies)
# Grafana 13 visualization (Grafana OSS, no Docker)

# ─────────────────────────────────────────────
# Execution Flow
# ─────────────────────────────────────────────
# 1. Install ROCm 7.2.2 (OS-specific apt repo)
# 2. Create Python venv, install vLLM 0.19.1 ROCm wheel (bundles PyTorch)
# 3. Download Mistral-7B-Instruct-v0.2 weights
# 4. Start vLLM server
# 5. Run Python benchmark client
# 6. Store metrics in SQLite
# 7. Install Grafana 13 (apt repo, no hardcoded .deb)
# 8. Configure SQLite datasource + print dashboard queries

# ════════════════════════════════════════════════════════════
## ✅ 1. System Preparation + ROCm 7.2.2
# ════════════════════════════════════════════════════════════
echo ""
echo "Step 1. System Preparation + ROCm ${ROCM_VERSION} (Ubuntu ${UBUNTU_VERSION})"
sudo apt update
sudo apt install -y python3 python3-venv python3-dev gcc g++ make git git-lfs wget curl \
                    apt-transport-https software-properties-common lsb-release gnupg \
                    dkms linux-headers-$(uname -r) sqlite3

# ── ROCm GPG key ──────────────────────────────────────────────────────────────
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

# ── ROCm apt sources (codename-specific) ─────────────────────────────────────
sudo tee /etc/apt/sources.list.d/rocm.list << EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} ${UBUNTU_CODENAME} main
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/graphics/7.2.1/ubuntu ${UBUNTU_CODENAME} main
EOF

sudo tee /etc/apt/preferences.d/rocm-pin-600 << EOF
Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 600
EOF

sudo apt update
sudo apt install -y rocm

# Verify ROCm is visible
rocm-smi || true
echo "ROCm ${ROCM_VERSION} installed."

# ════════════════════════════════════════════════════════════
## ✅ 2. Python venv + vLLM ${VLLM_VERSION} ROCm wheel
#        NOTE: The ROCm vLLM wheel bundles PyTorch.
#              Do NOT install torch separately.
# ════════════════════════════════════════════════════════════
echo ""
echo "Step 2. Python venv + vLLM ${VLLM_VERSION} (${ROCM_WHEEL_TAG} wheel)"

python3 -m venv vllm_env
source vllm_env/bin/activate

pip install --upgrade pip

# Install vLLM ROCm pre-built wheel (includes PyTorch 2.8 + ROCm 7.2 stack)
pip install "vllm==${VLLM_VERSION}+${ROCM_WHEEL_TAG}" \
    --extra-index-url "https://wheels.vllm.ai/rocm/${VLLM_VERSION}/${ROCM_WHEEL_TAG}"

# Additional benchmark / utility deps
pip install -r requirements.txt

echo "vLLM ${VLLM_VERSION} installed."

# ════════════════════════════════════════════════════════════
## ✅ 3. Download Mistral-7B-Instruct-v0.2 (Hugging Face)
#        No HF token required for this model.
# ════════════════════════════════════════════════════════════
echo ""
echo "Step 3. Download Mistral-7B-Instruct-v0.2 weights"

pip install huggingface_hub
mkdir -p models
cd models

if [ ! -d "Mistral-7B-Instruct-v0.2" ]; then
    git lfs install
    git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
else
    echo "Model directory already exists, skipping clone."
fi

cd ..
echo "Model weights ready."

# ════════════════════════════════════════════════════════════
## ✅ 4. Start vLLM Server (ROCm, no Docker)
# ════════════════════════════════════════════════════════════
echo ""
echo "Step 4. Start vLLM Server (ROCm, background)"

chmod +x run_vllm_server.sh
./run_vllm_server.sh &
VLLM_PID=$!

echo "vLLM server PID: ${VLLM_PID}"
echo "Waiting 60 seconds for server to initialise..."
sleep 60

echo "Server listening at: http://127.0.0.1:8000/v1"

# ════════════════════════════════════════════════════════════
## ✅ 5. Python Benchmark Client (latency + tokens/sec)
# ════════════════════════════════════════════════════════════
echo ""
echo "Step 5. Python Benchmark Client"

source vllm_env/bin/activate
python client_run.py

# ════════════════════════════════════════════════════════════
## ✅ 6. Store Metrics in SQLite
# ════════════════════════════════════════════════════════════
echo ""
echo "Step 6. Store Metrics in SQLite"

python parse_and_store.py

# ════════════════════════════════════════════════════════════
## ✅ 7. Install Grafana 13 (via apt repo — no hardcoded .deb)
# ════════════════════════════════════════════════════════════
echo ""
echo "Step 7. Install Grafana OSS (apt repo)"

sudo mkdir -p /etc/apt/keyrings/
wget -q -O - https://apt.grafana.com/gpg.key \
    | gpg --dearmor \
    | sudo tee /etc/apt/keyrings/grafana.gpg > /dev/null

echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" \
    | sudo tee /etc/apt/sources.list.d/grafana.list

sudo apt update
sudo apt install -y grafana

sudo systemctl daemon-reload
sudo systemctl enable grafana-server
sudo systemctl start grafana-server

# ════════════════════════════════════════════════════════════
## ✅ 8. Install SQLite plugin for Grafana
# ════════════════════════════════════════════════════════════
echo ""
echo "Step 8. Install Grafana SQLite plugin"

# grafana-cli requires --homepath in Grafana 13+
sudo grafana-cli --homepath /usr/share/grafana plugins install frser-sqlite-datasource
sudo systemctl restart grafana-server

SERVER_IP=$(curl -s ifconfig.me)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Open Grafana at:  http://${SERVER_IP}:3000"
echo "  user: admin    pass: admin"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ════════════════════════════════════════════════════════════
## ✅ 9. Provision Grafana datasource + dashboard automatically
# ════════════════════════════════════════════════════════════
echo ""
echo "Step 9. Provisioning Grafana datasource and dashboard"

# ── Datasource ────────────────────────────────────────────
sudo mkdir -p /etc/grafana/provisioning/datasources
sudo cp grafana/provisioning/datasources/sqlite.yaml \
        /etc/grafana/provisioning/datasources/sqlite.yaml

# ── Dashboard provider ────────────────────────────────────
sudo mkdir -p /etc/grafana/provisioning/dashboards
sudo cp grafana/provisioning/dashboards/dashboards.yaml \
        /etc/grafana/provisioning/dashboards/dashboards.yaml

# ── Dashboard JSON ────────────────────────────────────────
sudo mkdir -p /var/lib/grafana/dashboards
sudo cp grafana/dashboards/mistral-bench.json \
        /var/lib/grafana/dashboards/mistral-bench.json
sudo chown -R grafana:grafana /var/lib/grafana/dashboards

# ── Copy initial metrics db so datasource test passes ─────
sudo cp /root/vllm-mistral-rocm-bench/metrics.db /var/lib/grafana/metrics.db
sudo chown grafana:grafana /var/lib/grafana/metrics.db

sudo systemctl restart grafana-server
echo "Grafana provisioned. Dashboard will appear automatically."

# ════════════════════════════════════════════════════════════
## ✅ 10. Example Grafana Queries (for reference)
# ════════════════════════════════════════════════════════════
echo ""
echo "Step 10. Grafana is pre-configured. Reference queries:"
echo ""
echo "  Tokens/sec over time:"
echo "    SELECT ts AS \"time\", tokens_per_sec FROM metrics ORDER BY ts;"
echo ""
echo "  Latency over time:"
echo "    SELECT ts AS \"time\", latency_s FROM metrics ORDER BY ts;"
echo ""
echo "  Average performance per prompt:"
echo "    SELECT prompt, avg(tokens_per_sec) AS avg_tps, avg(latency_s) AS avg_latency"
echo "    FROM metrics GROUP BY prompt;"
echo ""
echo "deploy_all.sh completed."

# ════════════════════════════════════════════════════════════
## ✅ 11. Firewall — block external access to vLLM port 8000
# ════════════════════════════════════════════════════════════
echo ""
echo "Step 11. Configuring firewall (ufw)"
sudo ufw --force enable
sudo ufw allow ssh
sudo ufw allow 3000   # Grafana
sudo ufw deny 8000    # vLLM — localhost only, no external access
echo "Firewall configured. vLLM port 8000 blocked externally."
sudo ufw status
