# vllm-mistral-rocm-bench/parse_and_store.py
# Updated: 20260423 — auto-syncs metrics.db to /var/lib/grafana/ after import

import json
import os
import shutil
import sqlite3

DB      = "metrics.db"
LOG     = "bench_logs.jsonl"
GRAFANA_DB = "/var/lib/grafana/metrics.db"

conn = sqlite3.connect(DB)
cur  = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS metrics (
    ts                TEXT,
    run_id            TEXT UNIQUE,
    prompt            TEXT,
    latency_s         REAL,
    completion_tokens INTEGER,
    tokens_per_sec    REAL
)
""")

imported = 0
skipped  = 0

with open(LOG) as f:
    for line in f:
        rec = json.loads(line)
        try:
            cur.execute("""
                INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?)
            """, (
                rec["timestamp"],
                rec["run_id"],
                rec["prompt"],
                rec["latency_s"],
                rec["completion_tokens"],
                rec["tokens_per_sec"]
            ))
            imported += 1
        except sqlite3.IntegrityError:
            skipped += 1   # duplicate run_id — already in db

conn.commit()
conn.close()

print(f"Imported {imported} records, skipped {skipped} duplicates.")

# ── Sync to Grafana directory ─────────────────────────────
if os.path.isdir(os.path.dirname(GRAFANA_DB)):
    shutil.copy2(DB, GRAFANA_DB)
    os.system(f"chown grafana:grafana {GRAFANA_DB} 2>/dev/null || true")
    print(f"Synced to {GRAFANA_DB}")
else:
    print(f"Grafana directory not found, skipping sync to {GRAFANA_DB}")
