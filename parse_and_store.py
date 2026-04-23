# vllm-mistral-rocm-bench/parse_and_store.py

import json, sqlite3

DB = "metrics.db"
LOG = "bench_logs.jsonl"

conn = sqlite3.connect(DB)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS metrics (
    ts TEXT,
    run_id TEXT,
    prompt TEXT,
    latency_s REAL,
    completion_tokens INTEGER,
    tokens_per_sec REAL
)
""")

with open(LOG) as f:
    for line in f:
        rec = json.loads(line)
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

conn.commit()
conn.close()

print("Imported into SQLite.")
