# vllm-mistral-rocm-bench/client_run.py

import time, json, uuid
from datetime import datetime
import openai
import os

openai.base_url = "http://127.0.0.1:8000/v1"
openai.api_key = "dummy"

PROMPTS = [
    "Explain the concept of attention in transformers.",
    "Describe the architecture of Mistral-7B-Instruct-v0.2.",
    "Summarize the benefits of vLLM."
]

LOG_FILE = "bench_logs.jsonl"

def run_once(prompt):
    run_id = str(uuid.uuid4())
    t0 = time.time()

    resp = openai.chat.completions.create(
        model="Mistral-7B-Instruct-v0.2",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.1
    )

    t1 = time.time()
    latency = t1 - t0

    usage = resp.usage
    completion_tokens = usage.completion_tokens
    tokens_per_sec = completion_tokens / latency if latency > 0 else 0

    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
        "prompt": prompt,
        "latency_s": latency,
        "completion_tokens": completion_tokens,
        "tokens_per_sec": tokens_per_sec
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"{run_id}: {latency:.3f}s, {tokens_per_sec:.2f} tok/s")
    return record

def main():
    for _ in range(5):
        for p in PROMPTS:
            run_once(p)

if __name__ == "__main__":
    main()
