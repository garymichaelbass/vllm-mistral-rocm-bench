#!/usr/bin/env python3
# vllm-mistral-rocm-bench/parse_and_store.py
# Updated: 20260424 — extended schema for bench_runner.py output

import json
import os
import shutil
import sqlite3

DB         = "metrics.db"
LOG        = "bench_logs.jsonl"
GRAFANA_DB = "/var/lib/grafana/metrics.db"

conn = sqlite3.connect(DB)
cur  = conn.cursor()

# Extended schema — ttft and gpu_model added; old rows fill them as NULL
cur.execute("""
CREATE TABLE IF NOT EXISTS metrics (
    ts                TEXT,
    run_id            TEXT UNIQUE,
    gpu_model         TEXT,
    prompt            TEXT,
    ttft_s            REAL,
    e2e_latency_s     REAL,
    completion_tokens INTEGER,
    prompt_tokens     INTEGER,
    completion_tps    REAL,
    total_tps         REAL
)
""")

# Migration: add columns to existing tables created by the old client_run.py
for col, coltype in [
    ("ttft_s",       "REAL"),
    ("gpu_model",    "TEXT"),
    ("prompt_tokens","INTEGER"),
    ("total_tps",    "REAL"),
]:
    try:
        cur.execute(f"ALTER TABLE metrics ADD COLUMN {col} {coltype}")
    except sqlite3.OperationalError:
        pass  # column already exists

imported = 0
skipped  = 0

with open(LOG) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        try:
            cur.execute("""
                INSERT INTO metrics
                    (ts, run_id, gpu_model, prompt,
                     ttft_s, e2e_latency_s,
                     completion_tokens, prompt_tokens,
                     completion_tps, total_tps)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rec.get("timestamp"),
                rec.get("run_id"),
                rec.get("gpu_model"),
                rec.get("prompt"),
                rec.get("ttft_s"),
                # bench_runner uses e2e_latency_s; client_run used latency_s
                rec.get("e2e_latency_s") or rec.get("latency_s"),
                rec.get("completion_tokens"),
                rec.get("prompt_tokens"),
                # bench_runner uses completion_tps; client_run used tokens_per_sec
                rec.get("completion_tps") or rec.get("tokens_per_sec"),
                rec.get("total_tps"),
            ))
            imported += 1
        except sqlite3.IntegrityError:
            skipped += 1  # duplicate run_id

conn.commit()
conn.close()

print(f"Imported {imported} records, skipped {skipped} duplicates.")

# ── Sync to Grafana directory ─────────────────────────────────────────────────
if os.path.isdir(os.path.dirname(GRAFANA_DB)):
    shutil.copy2(DB, GRAFANA_DB)
    os.system(f"chown grafana:grafana {GRAFANA_DB} 2>/dev/null || true")
    print(f"Synced to {GRAFANA_DB}")
else:
    print(f"Grafana directory not found — skipping sync to {GRAFANA_DB}")
