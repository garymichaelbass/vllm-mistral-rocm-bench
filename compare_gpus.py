#!/usr/bin/env python3
# vllm-mistral-rocm-bench/compare_gpus.py
# Updated: 20260424
#
# Reads all summary_*.json files in the current directory and prints a
# side-by-side comparison table.  Useful after running bench_runner.py
# on multiple AMD ROCm GPUs and copying the JSON summaries to one machine.
#
# Usage:
#   python compare_gpus.py                       # reads ./summary_*.json
#   python compare_gpus.py path/to/summary1.json path/to/summary2.json

import glob
import json
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Load summaries
# ─────────────────────────────────────────────────────────────────────────────

paths = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob("summary_*.json"))

if not paths:
    print("No summary_*.json files found.  Run bench_runner.py first.")
    sys.exit(0)

summaries: list[dict] = []
for p in paths:
    with open(p) as f:
        summaries.append(json.load(f))

print(f"\nComparing {len(summaries)} benchmark run(s)  —  {', '.join(paths)}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _v(s: dict, *keys, default="—"):
    """Safe nested key access."""
    for k in keys:
        if not isinstance(s, dict):
            return default
        s = s.get(k, {})
    return s if s not in (None, {}, "") else default


def _fmt(val, fmt=".3f"):
    if isinstance(val, (int, float)):
        return format(val, fmt)
    return str(val)

# ─────────────────────────────────────────────────────────────────────────────
# Build rows
# ─────────────────────────────────────────────────────────────────────────────

ROWS = [
    # (display label, extraction lambda)
    ("GPU model",          lambda s: _v(s, "environment", "gpu_gpu_model")),
    ("GPU count",          lambda s: _v(s, "environment", "gpu_gpu_count")),
    ("VRAM total (MB)",    lambda s: _v(s, "environment", "gpu_vram_total_mb")),
    ("ROCm version",       lambda s: _v(s, "environment", "gpu_rocm_version")),
    ("vLLM version",       lambda s: _v(s, "environment", "vllm_version")),
    ("dtype",              lambda s: _v(s, "environment", "dtype")),
    ("max_tokens",         lambda s: _v(s, "environment", "max_tokens")),
    ("N runs",             lambda s: _v(s, "aggregate",   "n_runs")),
    ("── Latency ──",      lambda s: ""),
    ("E2E lat p50 (s)",    lambda s: _fmt(_v(s, "aggregate", "e2e_latency_s", "p50"))),
    ("E2E lat p95 (s)",    lambda s: _fmt(_v(s, "aggregate", "e2e_latency_s", "p95"))),
    ("E2E lat p99 (s)",    lambda s: _fmt(_v(s, "aggregate", "e2e_latency_s", "p99"))),
    ("TTFT p50 (s)",       lambda s: _fmt(_v(s, "aggregate", "ttft_s", "p50"))),
    ("TTFT p95 (s)",       lambda s: _fmt(_v(s, "aggregate", "ttft_s", "p95"))),
    ("── Throughput ──",   lambda s: ""),
    ("Compl tok/s mean",   lambda s: _fmt(_v(s, "aggregate", "completion_tps", "mean"), ".1f")),
    ("Compl tok/s p50",    lambda s: _fmt(_v(s, "aggregate", "completion_tps", "p50"),  ".1f")),
    ("Total tok/s mean",   lambda s: _fmt(_v(s, "aggregate", "total_tps",       "mean"), ".1f")),
    ("── Efficiency ──",   lambda s: ""),
    ("Tokens/joule",       lambda s: _fmt(_v(s, "efficiency", "tokens_per_joule"), ".3f")),
    ("Power (W)",          lambda s: _fmt(_v(s, "gpu_live_snapshot", "power_w"), ".1f")),
    ("GPU util % (snap)",  lambda s: _v(s, "efficiency", "gpu_util_pct_snapshot")),
    ("VRAM used MB (snap)",lambda s: _v(s, "efficiency", "vram_used_mb_snapshot")),
    ("── Totals ──",       lambda s: ""),
    ("Total tokens gen.",  lambda s: _v(s, "aggregate", "total_tokens_generated")),
    ("Benchmark time",     lambda s: _v(s, "benchmark_timestamp")),
]

# ─────────────────────────────────────────────────────────────────────────────
# Print table
# ─────────────────────────────────────────────────────────────────────────────

COL_W     = 22
LABEL_W   = 24
separator = "-" * (LABEL_W + (COL_W + 3) * len(summaries))

# Header row: GPU model truncated to column width
print(separator)
header = f"{'Metric':<{LABEL_W}}"
for s in summaries:
    gpu = str(_v(s, "environment", "gpu_gpu_model"))[:COL_W]
    header += f"  {gpu:>{COL_W}}"
print(header)
print(separator)

for label, fn in ROWS:
    if label.startswith("──"):
        print()
        print(f"{label}")
        continue
    row = f"{label:<{LABEL_W}}"
    for s in summaries:
        val = str(fn(s))[:COL_W]
        row += f"  {val:>{COL_W}}"
    print(row)

print(separator)

# ─────────────────────────────────────────────────────────────────────────────
# Best-in-class callouts  (skip if only 1 GPU)
# ─────────────────────────────────────────────────────────────────────────────

if len(summaries) > 1:
    print("\nBest-in-class:")

    def _best_low(metric_keys, label):
        vals = []
        for i, s in enumerate(summaries):
            v = _v(s, *metric_keys)
            if isinstance(v, (int, float)):
                vals.append((v, i))
        if vals:
            best_val, best_i = min(vals)
            gpu = str(_v(summaries[best_i], "environment", "gpu_gpu_model"))[:40]
            print(f"  Lowest {label:<22}  {best_val:.3f}  →  {gpu}")

    def _best_high(metric_keys, label, fmt=".1f"):
        vals = []
        for i, s in enumerate(summaries):
            v = _v(s, *metric_keys)
            if isinstance(v, (int, float)):
                vals.append((v, i))
        if vals:
            best_val, best_i = max(vals)
            gpu = str(_v(summaries[best_i], "environment", "gpu_gpu_model"))[:40]
            print(f"  Highest {label:<21}  {format(best_val, fmt)}  →  {gpu}")

    _best_low( ["aggregate", "e2e_latency_s", "p50"],   "E2E latency p50")
    _best_low( ["aggregate", "ttft_s",        "p50"],   "TTFT p50")
    _best_high(["aggregate", "completion_tps", "mean"], "throughput (tok/s)")
    _best_high(["efficiency", "tokens_per_joule"],      "efficiency (tok/J)", ".3f")
    print()
