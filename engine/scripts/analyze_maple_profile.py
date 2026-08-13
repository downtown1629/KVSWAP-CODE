#!/usr/bin/env python3
"""Summarize one Maple engine run and its KVSwap Jetson profiler samples."""

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd


PATTERNS = {
    "throughput": re.compile(
        r"Throughput Total:\s*([\d.]+)\s*Prefill:\s*([\d.]+)\s*Decode:\s*([\d.]+)"
    ),
    "latency": re.compile(
        r"Latency Total:\s*([\d.]+)\s*Prefill:\s*([\d.]+)\s*Decode:\s*([\d.]+)"
    ),
    "shape": re.compile(r"input:\s*(\d+)\s*output:\s*(\d+)\s*bsz:\s*(\d+)"),
    "peak": re.compile(
        r"Peak Memory \(GB\) RSS:\s*([\d.]+|n/a)\s*"
        r"TorchAllocated:\s*([\d.]+)\s*TorchReserved:\s*([\d.]+)"
    ),
    "expert": re.compile(
        r"Expert demand I/O: calls=(\d+), reads=(\d+), logical_bytes=(\d+), "
        r"stored_bytes=(\d+), read_ms=([\d.]+), copy_ms=([\d.]+)"
    ),
    "disk_sync": re.compile(r"Decoding disk sync time:\s*([\d.]+) ms"),
    "cache_mb": re.compile(r"Cache size:\s*([\d.]+) MB"),
    "policy": re.compile(
        r"Maple KV policy: sliding_layers=(\d+), window=(\d+), "
        r"local_capacity=(\d+), global_layers=(\d+)"
    ),
}


def _last(pattern, text, required=True):
    matches = pattern.findall(text)
    if not matches:
        if required:
            raise ValueError(f"completed metric missing: {pattern.pattern}")
        return None
    return matches[-1]


def parse_log(path):
    text = Path(path).read_text(errors="ignore")
    total_tps, prefill_tps, decode_tps = map(
        float, _last(PATTERNS["throughput"], text)
    )
    total_s, prefill_s, decode_s = map(float, _last(PATTERNS["latency"], text))
    prompt, generated, batch = map(int, _last(PATTERNS["shape"], text))
    rss, allocated, reserved = _last(PATTERNS["peak"], text)
    row = {
        "prompt_tokens": prompt,
        "generated_tokens_per_sequence": generated,
        "batch_size": batch,
        "latency_total_s": total_s,
        "latency_prefill_s": prefill_s,
        "latency_decode_s": decode_s,
        "throughput_total_tps": total_tps,
        "throughput_prefill_tps": prefill_tps,
        "throughput_decode_tps": decode_tps,
        "decode_ms_per_token": 1000 * decode_s / max(batch * (generated - 1), 1),
        "peak_rss_gb": None if rss == "n/a" else float(rss),
        "peak_torch_allocated_gb": float(allocated),
        "peak_torch_reserved_gb": float(reserved),
    }
    cache_mb = _last(PATTERNS["cache_mb"], text, required=False)
    if cache_mb is not None:
        row["planned_kv_cache_mb"] = float(cache_mb)
    policy = _last(PATTERNS["policy"], text, required=False)
    if policy is not None:
        sliding, window, capacity, global_layers = map(int, policy)
        row.update({
            "sliding_layers": sliding,
            "sliding_window": window,
            "sliding_cache_capacity": capacity,
            "global_layers": global_layers,
        })
    sync_ms = _last(PATTERNS["disk_sync"], text, required=False)
    row["decode_kv_disk_sync_ms"] = float(sync_ms) if sync_ms else 0.0
    expert = _last(PATTERNS["expert"], text, required=False)
    if expert is not None:
        calls, reads, logical, stored, read_ms, copy_ms = expert
        calls, reads, logical, stored = map(int, (calls, reads, logical, stored))
        read_ms, copy_ms = float(read_ms), float(copy_ms)
        row.update({
            "expert_materialize_calls": calls,
            "expert_extent_reads": reads,
            "expert_logical_bytes": logical,
            "expert_stored_bytes": stored,
            "expert_read_s": read_ms / 1000,
            "expert_copy_s": copy_ms / 1000,
            "expert_avg_read_kib": logical / max(reads, 1) / 1024,
            "expert_effective_read_gib_s": logical / 1024**3 / max(read_ms / 1000, 1e-9),
            "expert_reads_per_model_token": (
                reads / max(batch * (prompt + generated - 1), 1)
            ),
        })
    return row


def _numeric_mean(frame, column):
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def _numeric_max(frame, column):
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.max())


def _phase_frames(frame, prefill_s, decode_s):
    decode_n = min(len(frame), max(1, math.ceil(decode_s)))
    decode = frame.tail(decode_n)
    before_decode = frame.iloc[:-decode_n]
    prefill_n = min(len(before_decode), max(1, math.ceil(prefill_s)))
    return before_decode.tail(prefill_n), decode


def _resource_stats(frame, prefix):
    cpu_columns = [name for name in frame if re.fullmatch(r"cpu\d+", name)]
    cpu = frame[cpu_columns].apply(pd.to_numeric, errors="coerce") if cpu_columns else None
    power = pd.to_numeric(frame.get("power_tot_mw"), errors="coerce")
    result = {
        f"{prefix}_samples": len(frame),
        f"{prefix}_gpu_mean_pct": _numeric_mean(frame, "gpu_pct"),
        f"{prefix}_gpu_max_pct": _numeric_max(frame, "gpu_pct"),
        f"{prefix}_emc_mean_pct": _numeric_mean(frame, "emc_pct"),
        f"{prefix}_ram_used_peak_gb": (
            None if _numeric_max(frame, "ram_used_kb") is None
            else _numeric_max(frame, "ram_used_kb") / 1024**2
        ),
        f"{prefix}_swap_used_peak_gb": (
            None if _numeric_max(frame, "swap_used_kb") is None
            else _numeric_max(frame, "swap_used_kb") / 1024**2
        ),
        f"{prefix}_power_mean_w": None if power.dropna().empty else float(power.mean() / 1000),
        f"{prefix}_energy_j": None if power.dropna().empty else float(power.sum() / 1000),
    }
    result[f"{prefix}_cpu_mean_pct"] = (
        None if cpu is None else float(cpu.mean(axis=1).mean())
    )
    return result


def _disk_stats(frame, prefix):
    read_kbps = _numeric_mean(frame, "read_kbps")
    return {
        f"{prefix}_disk_samples": len(frame),
        f"{prefix}_disk_util_mean_pct": _numeric_mean(frame, "util_pct"),
        f"{prefix}_disk_util_max_pct": _numeric_max(frame, "util_pct"),
        f"{prefix}_disk_read_mib_s_mean": None if read_kbps is None else read_kbps / 1024,
        f"{prefix}_disk_read_mib_s_max": (
            None if _numeric_max(frame, "read_kbps") is None
            else _numeric_max(frame, "read_kbps") / 1024
        ),
        f"{prefix}_disk_read_iops_mean": _numeric_mean(frame, "read_iops"),
        f"{prefix}_disk_queue_depth_mean": _numeric_mean(frame, "queue_depth"),
        f"{prefix}_disk_avg_read_kib": _numeric_mean(
            frame[pd.to_numeric(frame.get("read_iops"), errors="coerce") > 0],
            "avg_read_kb",
        ),
    }


def add_samples(row, jtop_path=None, diskio_path=None):
    if jtop_path and Path(jtop_path).is_file():
        frame = pd.read_csv(jtop_path)
        prefill, decode = _phase_frames(
            frame, row["latency_prefill_s"], row["latency_decode_s"]
        )
        row.update(_resource_stats(frame, "run"))
        row.update(_resource_stats(prefill, "prefill"))
        row.update(_resource_stats(decode, "decode"))
    if diskio_path and Path(diskio_path).is_file():
        frame = pd.read_csv(diskio_path)
        prefill, decode = _phase_frames(
            frame, row["latency_prefill_s"], row["latency_decode_s"]
        )
        row.update(_disk_stats(frame, "run"))
        row.update(_disk_stats(prefill, "prefill"))
        row.update(_disk_stats(decode, "decode"))
    return row


def plot_timeline(jtop_path, diskio_path, row, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = []
    if jtop_path and Path(jtop_path).is_file():
        frames.append((pd.read_csv(jtop_path), ["gpu_pct", "emc_pct"], "utilization (%)"))
    if diskio_path and Path(diskio_path).is_file():
        frames.append((pd.read_csv(diskio_path), ["util_pct"], "NVMe util (%)"))
    if not frames:
        return
    fig, axes = plt.subplots(len(frames), 1, figsize=(10, 3.2 * len(frames)), squeeze=False)
    for ax, (frame, columns, ylabel) in zip(axes[:, 0], frames):
        x = pd.Series(range(len(frame))) - len(frame) + 1
        for column in columns:
            if column in frame:
                ax.plot(x, pd.to_numeric(frame[column], errors="coerce"), label=column)
        ax.axvline(-row["latency_decode_s"], color="black", ls="--", lw=1,
                   label="decode start")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[-1, 0].set_xlabel("seconds before profiler stop")
    fig.suptitle("Maple Jetson profile (1 Hz samples)")
    fig.tight_layout()
    fig.savefig(output, dpi=140)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--jtop", type=Path)
    parser.add_argument("--diskio", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    prefix = args.output_prefix or args.log.with_suffix("")
    row = add_samples(parse_log(args.log), args.jtop, args.diskio)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{prefix}.summary.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    pd.DataFrame([row]).to_csv(f"{prefix}.summary.csv", index=False)
    plot_timeline(args.jtop, args.diskio, row, f"{prefix}.timeline.png")
    print(json.dumps(row, indent=2, sort_keys=True))
    print(f"wrote {prefix}.summary.json and {prefix}.summary.csv")


if __name__ == "__main__":
    main()
