# vllm-mistral-rocm-bench

**vLLM · Mistral-7B-Instruct-v0.2 · ROCm benchmark — no Docker**

| Component | Version |
|---|---|
| ROCm | 7.2.2 |
| vLLM | 0.19.1 (ROCm wheel, bundles PyTorch 2.8) |
| Grafana OSS | 13.x (via apt repo) |
| Ubuntu | 24.04 LTS (noble) **or** 22.04 LTS (jammy) |

---

## Package Contents

- vLLM server (ROCm native, no containers)
- Mistral-7B-Instruct-v0.2 model loading
- Python benchmarking client (`client_run.py`)
- Tokens/sec + latency measurement
- JSONL log → SQLite storage (`parse_and_store.py`)
- Grafana 13 visualization via frser-sqlite-datasource plugin

## Execution Flow

1. Detect Ubuntu release, configure ROCm 7.2.2 apt repo
2. Create Python venv, install vLLM 0.19.1 ROCm wheel (PyTorch bundled — do **not** install torch separately)
3. Download Mistral-7B-Instruct-v0.2 from Hugging Face (no token required)
4. Start vLLM server (`vllm serve`)
5. Python client sends prompts, measures latency and tokens/sec
6. Logs stored in `bench_logs.jsonl`, parsed into `metrics.db` (SQLite)
7. Grafana installed via apt, SQLite plugin added, dashboards created

---

## Quick Start

```bash
git clone https://github.com/garymichaelbass/vllm-mistral-rocm-bench.git
cd vllm-mistral-rocm-bench
bash deploy_all.sh
```

`deploy_all.sh` auto-detects Ubuntu 24.04 (`noble`) or 22.04 (`jammy`) and uses the correct ROCm apt repo.

---

## Manual Steps

### 1. System Preparation + ROCm 7.2.2

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-dev gcc g++ make git wget curl \
                    apt-transport-https software-properties-common lsb-release gnupg

# ── ROCm GPG key
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

# ── Ubuntu 24.04 (noble)
sudo tee /etc/apt/sources.list.d/rocm.list << 'EOF'
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2.2 noble main
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/graphics/7.2.1/ubuntu noble main
EOF

# ── Ubuntu 22.04 (jammy) — substitute above with:
# deb ... https://repo.radeon.com/rocm/apt/7.2.2 jammy main
# deb ... https://repo.radeon.com/graphics/7.2.1/ubuntu jammy main

sudo tee /etc/apt/preferences.d/rocm-pin-600 << 'EOF'
Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 600
EOF

sudo apt update
sudo apt install -y rocm
```

### 2. Python venv + vLLM 0.19.1 ROCm wheel

```bash
python3 -m venv vllm_env
source vllm_env/bin/activate
pip install --upgrade pip

# ROCm wheel bundles PyTorch 2.8 — do NOT add torch separately
pip install "vllm==0.19.1+rocm721" \
    --extra-index-url https://wheels.vllm.ai/rocm/0.19.1/rocm721

pip install -r requirements.txt
```

### 3. Download Model Weights

```bash
pip install huggingface_hub
mkdir -p models && cd models
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
cd ..
```

### 4. Start vLLM Server

```bash
chmod +x run_vllm_server.sh
./run_vllm_server.sh
# Server listens at http://127.0.0.1:8000/v1
```

### 5. Run Benchmark Client

```bash
source vllm_env/bin/activate
python client_run.py
```

### 6. Parse Logs → SQLite

```bash
python parse_and_store.py
```

### 7. Install Grafana 13 (apt repo)

```bash
sudo mkdir -p /etc/apt/keyrings/
wget -q -O - https://apt.grafana.com/gpg.key \
    | gpg --dearmor \
    | sudo tee /etc/apt/keyrings/grafana.gpg > /dev/null

echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" \
    | sudo tee /etc/apt/sources.list.d/grafana.list

sudo apt update && sudo apt install -y grafana
sudo systemctl enable --now grafana-server

sudo grafana-cli --homepath /usr/share/grafana plugins install frser-sqlite-datasource
sudo systemctl restart grafana-server
```

Open Grafana at `http://<server-ip>:3000` (default: admin/admin).

### 8. Provision Grafana datasource + dashboard (automated)

`deploy_all.sh` copies the provisioning files automatically — no manual clicking required:

```bash
# Datasource
sudo cp grafana/provisioning/datasources/sqlite.yaml \
        /etc/grafana/provisioning/datasources/sqlite.yaml

# Dashboard provider
sudo cp grafana/provisioning/dashboards/dashboards.yaml \
        /etc/grafana/provisioning/dashboards/dashboards.yaml

# Dashboard JSON (3 panels: tokens/sec, latency, avg per prompt)
sudo mkdir -p /var/lib/grafana/dashboards
sudo cp grafana/dashboards/mistral-bench.json \
        /var/lib/grafana/dashboards/mistral-bench.json
sudo chown -R grafana:grafana /var/lib/grafana/dashboards

sudo systemctl restart grafana-server
```

The **Mistral-7B vLLM Benchmark** dashboard appears automatically under Dashboards — no manual panel creation needed.

---

## Example Grafana Queries

**Tokens/sec over time**
```sql
SELECT ts AS "time", tokens_per_sec FROM metrics ORDER BY ts;
```

**Latency over time**
```sql
SELECT ts AS "time", latency_s FROM metrics ORDER BY ts;
```

**Average performance per prompt**
```sql
SELECT prompt,
       avg(tokens_per_sec) AS avg_tps,
       avg(latency_s)      AS avg_latency
FROM metrics
GROUP BY prompt;
```

---

## Notes

- The vLLM ROCm wheel (`+rocm721`) targets ROCm 7.2.x and includes a compatible PyTorch build. Installing a separate `torch` package will break the environment.
- If you need a different ROCm version (e.g., ROCm 7.0), use the `+rocm700` variant: `https://wheels.vllm.ai/rocm/0.19.1/rocm700`
- `vllm serve` replaces the older `python -m vllm.entrypoints.openai.api_server` entry point.
- The `openai` Python client v1.x uses `OpenAI(base_url=..., api_key=...)` instead of module-level globals.
