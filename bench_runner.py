#!/usr/bin/env python3
# vllm-mistral-rocm-bench/bench_runner.py
# Updated: 20260424
#
# Replaces client_run.py.
# Measures: TTFT, E2E latency (P50/P95/P99), tokens/sec, GPU memory,
#            GPU utilisation %, power draw (W), tokens/joule.
# Outputs:  bench_logs.jsonl  (raw per-run records, append-safe)
#           summary_<gpu>_<ts>.json  (full self-describing report)
#           gpu_comparison.csv       (one row per run, append-safe)

import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from statistics import median, stdev

from openai import OpenAI

DB = "metrics.db"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit these or override via environment variables
# ─────────────────────────────────────────────────────────────────────────────
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL         = os.environ.get("VLLM_MODEL",    "mistralai/Mistral-7B-Instruct-v0.2")
MAX_TOKENS    = int(os.environ.get("BENCH_MAX_TOKENS", "256"))
TEMPERATURE   = float(os.environ.get("BENCH_TEMPERATURE", "0.1"))

NUM_WARMUP    = int(os.environ.get("BENCH_WARMUP", "2"))   # runs excluded from stats
NUM_RUNS      = int(os.environ.get("BENCH_RUNS",   "5"))   # measured runs per prompt

LOG_FILE      = "bench_logs.jsonl"
COMPARE_CSV   = "gpu_comparison.csv"
REPORT_CACHE  = ".bench_report_cache.json"   # temp file for --defer-report

PROMPTS = [
    "Explain the concept of attention in transformers.",
    "Describe the architecture of Mistral-7B-Instruct-v0.2.",
    "Summarize the benefits of vLLM for inference serving.",
    "What is the difference between prefill and decode in LLM inference?",
    "Explain KV-cache and why it matters for GPU memory.",
]

# ─────────────────────────────────────────────────────────────────────────────
# GPU discovery via rocm-smi
# ─────────────────────────────────────────────────────────────────────────────

def _rocm_smi(*flags: str) -> str:
    """Run rocm-smi with flags, return stdout. Returns '' on failure."""
    try:
        result = subprocess.run(
            ["rocm-smi", *flags],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except Exception:
        return ""


def get_gpu_info() -> dict:
    """Return static GPU metadata from rocm-smi."""
    info: dict = {
        "gpu_model":    "unknown",
        "gpu_count":    0,
        "vram_total_mb": 0,
        "rocm_version": "unknown",
    }

    # GPU model name
    raw = _rocm_smi("--showproductname")
    names = re.findall(r"GPU\[.*?\].*?:\s*(.+)", raw)
    if names:
        info["gpu_model"] = names[0].strip()
        info["gpu_count"] = len(names)

    # Total VRAM
    raw = _rocm_smi("--showmeminfo", "vram")
    totals = re.findall(r"Total\s+Memory.*?:\s*(\d+)\s+(?:kB|Bytes)", raw, re.IGNORECASE)
    if totals:
        # rocm-smi returns bytes or kB depending on version
        val = int(totals[0])
        # If clearly in bytes (>1 GB as bytes), convert to MB
        info["vram_total_mb"] = val // (1024 * 1024) if val > 1_073_741_824 else val // 1024

    # ROCm version
    raw = _rocm_smi("--version")
    m = re.search(r"ROCm.*?(\d+\.\d+[\.\d]*)", raw, re.IGNORECASE)
    if m:
        info["rocm_version"] = m.group(1)

    return info


def get_gpu_live() -> dict:
    """Return live GPU metrics (util %, VRAM used MB, power W)."""
    live: dict = {"gpu_util_pct": None, "vram_used_mb": None, "power_w": None}

    # Utilisation
    raw = _rocm_smi("--showuse")
    m = re.search(r"GPU use \(%\)\s*:\s*(\d+)", raw)
    if m:
        live["gpu_util_pct"] = int(m.group(1))

    # VRAM used
    raw = _rocm_smi("--showmeminfo", "vram")
    used = re.findall(r"Used\s+Memory.*?:\s*(\d+)\s+(?:kB|Bytes)", raw, re.IGNORECASE)
    if used:
        val = int(used[0])
        live["vram_used_mb"] = val // (1024 * 1024) if val > 1_073_741_824 else val // 1024

    # Power draw
    raw = _rocm_smi("--showpower")
    m = re.search(r"Average\s+Graphics\s+Package\s+Power\s*\(W\)\s*:\s*([\d.]+)", raw, re.IGNORECASE)
    if not m:
        m = re.search(r"Socket Power.*?:\s*([\d.]+)\s*W", raw, re.IGNORECASE)
    if m:
        live["power_w"] = float(m.group(1))

    # GPU temperature
    raw = _rocm_smi("--showtemp")
    m = re.search(r"Temperature\s*\(Sensor\s*edge\).*?:\s*([\d.]+)", raw, re.IGNORECASE)
    if not m:
        m = re.search(r"GPU Temperature.*?:\s*([\d.]+)", raw, re.IGNORECASE)
    if m:
        live["gpu_temp_c"] = float(m.group(1))
    else:
        live["gpu_temp_c"] = None

    return live


# ─────────────────────────────────────────────────────────────────────────────
# Per-request measurement (streaming for TTFT)
# ─────────────────────────────────────────────────────────────────────────────

def run_once(client: OpenAI, prompt: str) -> dict:
    """
    Send one request in streaming mode and return timing metrics.

    Measured:
      ttft_s          Time to first token (server latency + network)
      e2e_latency_s   Total wall-clock time from send to last token
      completion_tokens
      prompt_tokens
      completion_tps  Completion tokens / e2e_latency
      total_tps       (prompt + completion) tokens / e2e_latency
    """
    run_id = str(uuid.uuid4())
    first_token_time = None
    completion_tokens = 0
    prompt_tokens = 0
    text_chunks: list[str] = []

    t_start = time.perf_counter()

    with client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=True,
    ) as stream:
        for chunk in stream:
            now = time.perf_counter()
            delta = chunk.choices[0].delta if chunk.choices else None

            if delta and delta.content:
                if first_token_time is None:
                    first_token_time = now
                text_chunks.append(delta.content)
                completion_tokens += 1   # streaming: count chunks as tokens

            # Final chunk carries usage on some vLLM versions
            if hasattr(chunk, "usage") and chunk.usage:
                completion_tokens = chunk.usage.completion_tokens or completion_tokens
                prompt_tokens     = chunk.usage.prompt_tokens or 0

    t_end = time.perf_counter()

    e2e = t_end - t_start
    ttft = (first_token_time - t_start) if first_token_time else e2e
    completion_tps = completion_tokens / e2e if e2e > 0 else 0.0
    total_tps      = (prompt_tokens + completion_tokens) / e2e if e2e > 0 else 0.0

    return {
        "run_id":            run_id,
        "prompt":            prompt,
        "ttft_s":            round(ttft, 4),
        "e2e_latency_s":     round(e2e, 4),
        "completion_tokens": completion_tokens,
        "prompt_tokens":     prompt_tokens,
        "completion_tps":    round(completion_tps, 2),
        "total_tps":         round(total_tps, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pct(values: list[float], p: float) -> float:
    """Return the p-th percentile (0–100) of values."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (p / 100) * (len(sorted_v) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_v) - 1)
    return round(sorted_v[lo] + (idx - lo) * (sorted_v[hi] - sorted_v[lo]), 4)


def aggregate(records: list[dict]) -> dict:
    """Compute aggregate stats from a list of run records."""
    lats   = [r["e2e_latency_s"]  for r in records]
    ttfts  = [r["ttft_s"]         for r in records]
    ctps   = [r["completion_tps"] for r in records]
    ttps   = [r["total_tps"]      for r in records]

    def _stats(vals: list[float]) -> dict:
        return {
            "mean":  round(sum(vals) / len(vals), 4) if vals else 0,
            "p50":   _pct(vals, 50),
            "p95":   _pct(vals, 95),
            "p99":   _pct(vals, 99),
            "min":   round(min(vals), 4) if vals else 0,
            "max":   round(max(vals), 4) if vals else 0,
        }

    return {
        "n_runs":             len(records),
        "e2e_latency_s":      _stats(lats),
        "ttft_s":             _stats(ttfts),
        "completion_tps":     _stats(ctps),
        "total_tps":          _stats(ttps),
        "total_tokens_generated": sum(r["completion_tokens"] for r in records),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────────────────────

CSV_HEADER = (
    "timestamp,gpu_model,gpu_count,vram_total_mb,rocm_version,"
    "vllm_version,model,dtype,max_tokens,n_runs,"
    "e2e_lat_p50_s,e2e_lat_p95_s,e2e_lat_p99_s,"
    "ttft_p50_s,ttft_p95_s,"
    "completion_tps_mean,completion_tps_p50,"
    "total_tps_mean,"
    "vram_used_mb,gpu_util_pct,power_w,tokens_per_joule"
)


def write_jsonl(records: list[dict], gpu_info: dict, ts: str) -> None:
    with open(LOG_FILE, "a") as f:
        for rec in records:
            row = {
                "timestamp": ts,
                "gpu_model":  gpu_info["gpu_model"],
                **rec,
            }
            f.write(json.dumps(row) + "\n")


def write_json_summary(summary: dict, gpu_info: dict) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", gpu_info["gpu_model"].lower())[:40]
    ts   = summary["benchmark_timestamp"].replace(":", "").replace("-", "")[:15]
    fname = f"summary_{slug}_{ts}.json"
    with open(fname, "w") as f:
        json.dump(summary, f, indent=2)
    return fname


def append_csv_row(summary: dict, gpu_info: dict, live: dict) -> None:
    agg   = summary["aggregate"]
    lat   = agg["e2e_latency_s"]
    ttft  = agg["ttft_s"]
    ctps  = agg["completion_tps"]
    ttps  = agg["total_tps"]
    power = live.get("power_w") or 0
    tpj   = round(ctps["mean"] / power, 4) if power > 0 else ""

    row = ",".join(str(v) for v in [
        summary["benchmark_timestamp"],
        f'"{gpu_info["gpu_model"]}"',
        gpu_info["gpu_count"],
        gpu_info["vram_total_mb"],
        gpu_info["rocm_version"],
        summary["environment"]["vllm_version"],
        f'"{summary["environment"]["model"]}"',
        summary["environment"]["dtype"],
        summary["environment"]["max_tokens"],
        agg["n_runs"],
        lat["p50"],  lat["p95"],  lat["p99"],
        ttft["p50"], ttft["p95"],
        ctps["mean"], ctps["p50"],
        ttps["mean"],
        live.get("vram_used_mb") or "",
        live.get("gpu_util_pct") or "",
        power or "",
        tpj,
    ])

    write_header = not os.path.exists(COMPARE_CSV) or os.path.getsize(COMPARE_CSV) == 0
    with open(COMPARE_CSV, "a") as f:
        if write_header:
            f.write(CSV_HEADER + "\n")
        f.write(row + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Terminal results display  (SQLite-backed, runs after writes complete)
# ─────────────────────────────────────────────────────────────────────────────

W = 68  # display width

def _bar(value: float, max_val: float, width: int = 30) -> str:
    """Simple ASCII progress bar."""
    filled = int(round((value / max_val) * width)) if max_val > 0 else 0
    return "█" * filled + "░" * (width - filled)


def show_db_results(all_records: list[dict], gpu_info: dict, live: dict,
                    bench_start: float, bench_end: float) -> None:
    """
    Print a formatted results report to the terminal.
    Combines in-memory records (for rich stats) with a SQLite query
    (to confirm what was actually persisted).

    Additional metrics beyond the original three queries:
      - Latency stddev and coefficient of variation (consistency)
      - Cold-start penalty (run 1 vs steady-state average)
      - Thermal throttle check (first-half vs second-half throughput trend)
      - TTFT breakdown per prompt
      - Tokens/VRAM-GB (throughput normalised to memory capacity)
      - GPU temperature
      - Total benchmark wall-clock time
      - Latency jitter per prompt (max − min)
    """
    div  = "─" * W
    div2 = "═" * W

    lats  = [r["e2e_latency_s"]  for r in all_records]
    ttfts = [r["ttft_s"]         for r in all_records]
    ctps  = [r["completion_tps"] for r in all_records]
    prompts_used = list(dict.fromkeys(r["prompt"] for r in all_records))

    wall_time = bench_end - bench_start

    print("\n" + div2)
    print("  BENCHMARK RESULTS REPORT")
    print(div2)

    # ── Section 1: Overall aggregate stats ───────────────────────────
    print(f"\n{'OVERALL STATISTICS':^{W}}")
    print(div)

    avg_lat  = sum(lats)  / len(lats)
    avg_ctps = sum(ctps)  / len(ctps)
    avg_ttft = sum(ttfts) / len(ttfts)
    std_lat  = stdev(lats)  if len(lats) > 1 else 0.0
    std_ctps = stdev(ctps)  if len(ctps) > 1 else 0.0
    cv_lat   = (std_lat  / avg_lat  * 100) if avg_lat  > 0 else 0.0
    cv_ctps  = (std_ctps / avg_ctps * 100) if avg_ctps > 0 else 0.0

    print(f"  {'Metric':<30}  {'Mean':>8}  {'Min':>8}  {'Max':>8}  {'StdDev':>8}  {'CV%':>6}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")
    print(f"  {'E2E Latency (s)':<30}  {avg_lat:>8.3f}  {min(lats):>8.3f}  {max(lats):>8.3f}  {std_lat:>8.3f}  {cv_lat:>5.1f}%")
    print(f"  {'TTFT (s)':<30}  {avg_ttft:>8.3f}  {min(ttfts):>8.3f}  {max(ttfts):>8.3f}  {stdev(ttfts) if len(ttfts)>1 else 0:>8.3f}  {'':>6}")
    print(f"  {'Throughput (tok/s)':<30}  {avg_ctps:>8.1f}  {min(ctps):>8.1f}  {max(ctps):>8.1f}  {std_ctps:>8.1f}  {cv_ctps:>5.1f}%")
    print()
    print(f"  Percentiles — E2E latency:   "
          f"p50={_pct(lats,50):.3f}s   p95={_pct(lats,95):.3f}s   p99={_pct(lats,99):.3f}s")
    print(f"  Percentiles — Throughput:    "
          f"p50={_pct(ctps,50):.1f}   p95={_pct(ctps,95):.1f}   p99={_pct(ctps,99):.1f} tok/s")
    print(f"  CV% note: <5% = very consistent   5–15% = normal   >15% = high jitter")

    # ── Section 2: Per-prompt breakdown ──────────────────────────────
    print(f"\n{'PER-PROMPT BREAKDOWN':^{W}}")
    print(div)
    print(f"  {'Prompt (truncated)':<46}  {'AvgLat':>6}  {'Jitter':>6}  {'AvgTPS':>7}  {'AvgTTFT':>7}")
    print(f"  {'-'*46}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}")

    for p in prompts_used:
        recs  = [r for r in all_records if r["prompt"] == p]
        p_lats = [r["e2e_latency_s"] for r in recs]
        p_ctps = [r["completion_tps"] for r in recs]
        p_ttft = [r["ttft_s"]         for r in recs]
        jitter = max(p_lats) - min(p_lats)
        print(f"  {p[:46]:<46}  {sum(p_lats)/len(p_lats):>6.3f}  "
              f"{jitter:>6.3f}  {sum(p_ctps)/len(p_ctps):>7.1f}  "
              f"{sum(p_ttft)/len(p_ttft):>7.3f}")

    # ── Section 3: Cold-start vs steady-state ────────────────────────
    print(f"\n{'COLD-START vs STEADY-STATE':^{W}}")
    print(div)
    if len(all_records) >= 2:
        first     = all_records[0]
        rest      = all_records[1:]
        rest_avg  = sum(r["e2e_latency_s"] for r in rest) / len(rest)
        rest_tps  = sum(r["completion_tps"] for r in rest) / len(rest)
        penalty   = first["e2e_latency_s"] - rest_avg
        pct_pen   = (penalty / rest_avg * 100) if rest_avg > 0 else 0
        print(f"  Run 1 (cold) latency  : {first['e2e_latency_s']:.3f}s   "
              f"tps={first['completion_tps']:.1f}")
        print(f"  Runs 2+ (warm) avg    : {rest_avg:.3f}s   tps={rest_tps:.1f}")
        print(f"  Cold-start penalty    : +{penalty:.3f}s  ({pct_pen:+.1f}%)")
    else:
        print("  Not enough runs to compute cold-start delta.")

    # ── Section 4: Thermal throttle check ────────────────────────────
    print(f"\n{'THERMAL TREND (first half vs second half)':^{W}}")
    print(div)
    mid = len(all_records) // 2
    if mid >= 2:
        first_half  = [r["completion_tps"] for r in all_records[:mid]]
        second_half = [r["completion_tps"] for r in all_records[mid:]]
        avg_f = sum(first_half)  / len(first_half)
        avg_s = sum(second_half) / len(second_half)
        delta = avg_s - avg_f
        trend = "▼ possible thermal throttle" if delta < -5 else \
                "▲ warming up / improving"    if delta >  5 else \
                "─ stable"
        print(f"  First half avg  : {avg_f:.1f} tok/s")
        print(f"  Second half avg : {avg_s:.1f} tok/s")
        print(f"  Trend           : {delta:+.1f} tok/s  {trend}")
    else:
        print("  Not enough runs for trend analysis.")

    # ── Section 5: GPU & efficiency metrics ──────────────────────────
    print(f"\n{'GPU & EFFICIENCY':^{W}}")
    print(div)
    print(f"  GPU model       : {gpu_info['gpu_model']}")
    print(f"  GPU count       : {gpu_info['gpu_count']}")
    print(f"  VRAM total      : {gpu_info['vram_total_mb']:,} MB  "
          f"({gpu_info['vram_total_mb']/1024:.1f} GB)")

    vram_used = live.get("vram_used_mb")
    vram_total = gpu_info["vram_total_mb"]
    if vram_used and vram_total:
        vram_pct = vram_used / vram_total * 100
        print(f"  VRAM used       : {vram_used:,} MB  ({vram_pct:.1f}%)  "
              f"[{_bar(vram_used, vram_total, 20)}]")

    if live.get("gpu_util_pct") is not None:
        u = live["gpu_util_pct"]
        print(f"  GPU utilisation : {u}%  [{_bar(u, 100, 20)}]")

    if live.get("power_w"):
        pw = live["power_w"]
        tpj = avg_ctps / pw
        vram_gb = vram_total / 1024 if vram_total else None
        tpvg = avg_ctps / vram_gb if vram_gb else None
        print(f"  Power draw      : {pw:.1f} W")
        print(f"  Efficiency      : {tpj:.3f} tok/joule")
        if tpvg:
            print(f"  Tok/s per GB    : {tpvg:.2f} tok/s/GB-VRAM")

    if live.get("gpu_temp_c") is not None:
        t = live["gpu_temp_c"]
        warn = "  ⚠ high — check cooling" if t > 85 else ""
        print(f"  GPU temperature : {t:.1f} °C{warn}")

    # ── Section 6: SQLite confirmation ───────────────────────────────
    print(f"\n{'SQLite CONFIRMATION (persisted records)':^{W}}")
    print(div)
    try:
        conn = sqlite3.connect(DB)
        cur  = conn.cursor()

        cur.execute("""
            SELECT
              round(avg(e2e_latency_s),3),
              round(min(e2e_latency_s),3),
              round(max(e2e_latency_s),3),
              round(avg(completion_tps),1),
              round(max(completion_tps),1),
              count(*)
            FROM metrics
        """)
        row = cur.fetchone()
        if row and row[5]:
            print(f"  Rows in metrics.db  : {row[5]}")
            print(f"  Avg latency (all)   : {row[0]}s")
            print(f"  Min / Max latency   : {row[1]}s / {row[2]}s")
            print(f"  Avg / Peak tok/s    : {row[3]} / {row[4]}")
        else:
            print("  metrics.db appears empty — check parse_and_store.py")

        print()
        cur.execute("""
            SELECT prompt,
                   round(avg(e2e_latency_s),3) AS avg_lat,
                   round(max(e2e_latency_s) - min(e2e_latency_s),3) AS jitter,
                   round(avg(completion_tps),1) AS avg_tps,
                   round(avg(ttft_s),3) AS avg_ttft
            FROM metrics
            GROUP BY prompt
            ORDER BY avg_lat DESC
        """)
        rows = cur.fetchall()
        if rows:
            print(f"  {'Prompt':<46}  {'AvgLat':>6}  {'Jitter':>6}  {'AvgTPS':>7}  {'AvgTTFT':>7}")
            print(f"  {'-'*46}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}")
            for r in rows:
                print(f"  {str(r[0])[:46]:<46}  {str(r[1]):>6}  "
                      f"{str(r[2]):>6}  {str(r[3]):>7}  {str(r[4] or '—'):>7}")

        conn.close()
    except Exception as e:
        print(f"  SQLite query failed: {e}")

    # ── Section 7: Timing & file summary ─────────────────────────────
    print(f"\n{'TIMING & FILES':^{W}}")
    print(div)
    print(f"  Total benchmark wall time  : {wall_time:.1f}s  ({wall_time/60:.1f} min)")
    print(f"  Runs completed             : {len(all_records)}")
    print(f"  Total tokens generated     : {sum(r['completion_tokens'] for r in all_records):,}")
    print(f"  Raw log  → {LOG_FILE}")
    print(f"  CSV row  → {COMPARE_CSV}")
    print(div2)


# ─────────────────────────────────────────────────────────────────────────────
# vLLM version probe
# ─────────────────────────────────────────────────────────────────────────────

def get_vllm_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "unknown")
    except ImportError:
        return "unknown"


def get_vllm_model_info(client: OpenAI) -> dict:
    """Fetch served model metadata from /v1/models."""
    try:
        models = client.models.list()
        for m in models.data:
            if MODEL in m.id:
                return {
                    "dtype":       getattr(m, "dtype", "bfloat16"),
                    "max_model_len": getattr(m, "max_model_len", 4096),
                }
    except Exception:
        pass
    return {"dtype": "bfloat16", "max_model_len": 4096}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(defer_report: bool = False) -> None:
    client = OpenAI(base_url=VLLM_BASE_URL, api_key="dummy")

    ts_iso = datetime.now(timezone.utc).isoformat()
    print(f"\nvllm-mistral-rocm-bench  |  {ts_iso}")
    print("=" * 64)

    # ── Environment metadata ────────────────────────────────────────
    gpu_info   = get_gpu_info()
    model_info = get_vllm_model_info(client)

    env = {
        "model":        MODEL,
        "vllm_version": get_vllm_version(),
        "dtype":        model_info["dtype"],
        "max_tokens":   MAX_TOKENS,
        "temperature":  TEMPERATURE,
        "max_model_len": model_info["max_model_len"],
        **{f"gpu_{k}": v for k, v in gpu_info.items()},
    }

    print(f"GPU   : {gpu_info['gpu_model']}  (×{gpu_info['gpu_count']})")
    print(f"VRAM  : {gpu_info['vram_total_mb']} MB total")
    print(f"ROCm  : {gpu_info['rocm_version']}")
    print(f"vLLM  : {env['vllm_version']}")
    print(f"Model : {MODEL}")
    print(f"Warmup: {NUM_WARMUP} runs   Measured: {NUM_RUNS} runs × {len(PROMPTS)} prompts")
    print("=" * 64)

    # ── Warmup ──────────────────────────────────────────────────────
    if NUM_WARMUP > 0:
        print(f"\nWarmup ({NUM_WARMUP} runs, not recorded)…")
        for i in range(NUM_WARMUP):
            rec = run_once(client, PROMPTS[0])
            print(f"  warmup {i+1}/{NUM_WARMUP}  "
                  f"e2e={rec['e2e_latency_s']:.3f}s  "
                  f"ttft={rec['ttft_s']:.3f}s  "
                  f"{rec['completion_tps']:.1f} tok/s")

    # ── Benchmark runs ──────────────────────────────────────────────
    print(f"\nBenchmark ({NUM_RUNS} runs × {len(PROMPTS)} prompts)…")
    all_records: list[dict] = []
    per_prompt: dict[str, list[dict]] = {p: [] for p in PROMPTS}

    bench_start = time.perf_counter()
    for run_idx in range(NUM_RUNS):
        for p_idx, prompt in enumerate(PROMPTS):
            rec = run_once(client, prompt)
            all_records.append(rec)
            per_prompt[prompt].append(rec)
            print(
                f"  run {run_idx+1}/{NUM_RUNS}  p{p_idx+1}  "
                f"e2e={rec['e2e_latency_s']:.3f}s  "
                f"ttft={rec['ttft_s']:.3f}s  "
                f"{rec['completion_tps']:.1f} tok/s  "
                f"({rec['completion_tokens']} tokens)"
            )
    bench_end = time.perf_counter()

    # ── Snapshot live GPU state after the run ───────────────────────
    live = get_gpu_live()

    # ── Aggregate stats ─────────────────────────────────────────────
    agg = aggregate(all_records)
    per_prompt_agg = {p: aggregate(recs) for p, recs in per_prompt.items()}

    # tokens/joule (efficiency) — requires power reading
    power = live.get("power_w")
    tokens_per_joule = None
    if power and power > 0:
        tokens_per_joule = round(agg["completion_tps"]["mean"] / power, 4)

    # ── Build summary dict ──────────────────────────────────────────
    summary = {
        "benchmark_timestamp": ts_iso,
        "environment":         env,
        "gpu_live_snapshot":   live,
        "aggregate":           agg,
        "per_prompt":          per_prompt_agg,
        "efficiency": {
            "tokens_per_joule":     tokens_per_joule,
            "gpu_util_pct_snapshot": live.get("gpu_util_pct"),
            "vram_used_mb_snapshot": live.get("vram_used_mb"),
        },
    }

    # ── Write outputs ───────────────────────────────────────────────
    write_jsonl(all_records, gpu_info, ts_iso)
    json_file = write_json_summary(summary, gpu_info)
    append_csv_row(summary, gpu_info, live)

    # Run parse_and_store to sync into SQLite before the display query
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("pas", "parse_and_store.py")
        pas  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pas)
    except Exception as e:
        print(f"  (parse_and_store.py auto-run failed: {e} — run it manually)")

    # ── Save report cache (used by --report-only / --defer-report) ───
    cache = {
        "all_records": all_records,
        "gpu_info":    gpu_info,
        "live":        live,
        "bench_start": bench_start,
        "bench_end":   bench_end,
        "json_file":   json_file,
    }
    with open(REPORT_CACHE, "w") as f:
        json.dump(cache, f)

    if defer_report:
        print(f"\n  Benchmark complete. Results will be shown at end of deployment.")
        print(f"  Summary JSON → {json_file}")
        return

    # ── Full results report ──────────────────────────────────────────
    show_db_results(all_records, gpu_info, live, bench_start, bench_end)
    print(f"  Summary JSON → {json_file}")


def report_only() -> None:
    """Load cached benchmark data and print the results report."""
    if not os.path.exists(REPORT_CACHE):
        print(f"ERROR: No report cache found at '{REPORT_CACHE}'.")
        print("       Run bench_runner.py first (without --report-only).")
        sys.exit(1)

    with open(REPORT_CACHE) as f:
        cache = json.load(f)

    show_db_results(
        cache["all_records"],
        cache["gpu_info"],
        cache["live"],
        cache["bench_start"],
        cache["bench_end"],
    )
    print(f"  Summary JSON → {cache['json_file']}")

    # Clean up cache after display
    try:
        os.remove(REPORT_CACHE)
    except OSError:
        pass


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        report_only()
    else:
        defer = "--defer-report" in sys.argv
        main(defer_report=defer)
