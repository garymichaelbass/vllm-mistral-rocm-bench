# vllm-mistral-rocm-bench/client_run.py
# Updated: 20260423  (openai>=1.0 client API)

import time, json, uuid
from datetime import datetime, timezone
from openai import OpenAI

# openai>=1.0 uses an explicit client object instead of module-level globals.
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy",   # vLLM does not enforce API keys; value is arbitrary.
)

# Model name must match --served-model-name in run_vllm_server.sh
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

PROMPTS = [
    "Explain the concept of attention in transformers.",
    "Describe the architecture of Mistral-7B-Instruct-v0.2.",
    "Summarize the benefits of vLLM.",
]

LOG_FILE = "bench_logs.jsonl"


def run_once(prompt: str) -> dict:
    run_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.1,
    )

    latency = time.perf_counter() - t0
    completion_tokens = resp.usage.completion_tokens
    tokens_per_sec = completion_tokens / latency if latency > 0 else 0.0

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "prompt": prompt,
        "latency_s": round(latency, 4),
        "completion_tokens": completion_tokens,
        "tokens_per_sec": round(tokens_per_sec, 2),
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"{run_id[:8]}…  {latency:.3f}s  {tokens_per_sec:.2f} tok/s")
    return record


def main() -> None:
    print(f"Benchmarking model: {MODEL}")
    print(f"Log file: {LOG_FILE}")
    print("-" * 60)
    for _ in range(5):
        for prompt in PROMPTS:
            run_once(prompt)
    print("-" * 60)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
