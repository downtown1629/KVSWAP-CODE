#!/usr/bin/env python3
"""Collect + plot the Orin Nano (NVMe-only, Qwen3-0.6B) Table-5-style results.

scripts/eval_nano.sh now drives every baseline (flexgen/infinigen*/kvswap via
src/main.py, plus shadowkv and vllm) from one entry point, and all of them log
under one directory: $EVAL_LOG_DIR/$EVAL_USER/logs/nano/<model>/. Parses three
differently-shaped result sources out of that one tree into one frame:
  - main.py + ShadowKV runs:  $EVAL_LOG_DIR/$EVAL_USER/logs/nano/<model>/*.log
    (same <model>_<tag>_b<batch>_cl<ctx>.log naming for both; distinguished by
    whether <tag> starts with "shadowkv")
  - vLLM:                     $EVAL_LOG_DIR/$EVAL_USER/logs/nano/<model>/<model>_results.csv
  - per-run jtop samples:     <run log path>.jtop.csv  (scripts/jtop_logger.py; not
                              written for vLLM, which sweeps every batch in one process)
  - per-run disk I/O samples: <run log path>.diskio.csv (same script, sibling file)

All four report *decode-only* tokens/sec aggregated over the batch, so the
numbers are directly comparable:
  engine "Throughput Total: T Prefill: P Decode: D"  -> D
  ShadowKV "Throughput: X tokens/s"                  -> gen_len*bsz / decode_time
  vLLM run_vllm.py                                   -> 100*batch / (total-prefill)

main.py runs (not ShadowKV/vLLM) also log one "Peak Memory (GB) RSS: ...
TorchAllocated: ... TorchReserved: ..." line (see main.py's get_peak_rss_kb());
parsed into peak_rss_gb/peak_torch_alloc_gb/peak_torch_reserved_gb columns in
summary.csv, alongside jtop_summary.csv's system-wide ram_used_peak_gb.

Usage:
  .venv/bin/python scripts/analyze_nano_results.py [--model Qwen3-0.6B] [--outdir RESULTS/nano]
Writes summary.csv, jtop_summary.csv and throughput.png / resources.png to --outdir.
"""
import argparse
import csv
import os
import re
import sys
from glob import glob

import pandas as pd

# "Throughput Total: 8.48 Prefill: 2013.34 Decode: 26.76"
RE_TPUT = re.compile(
    r"Throughput Total:\s*([\d.]+)\s*Prefill:\s*([\d.]+)\s*Decode:\s*([\d.]+)")
# "Latency Total: 20.9 Prefill: 8.40 Decode: 12.50"
RE_LAT = re.compile(
    r"Latency Total:\s*([\d.]+)\s*Prefill:\s*([\d.]+)\s*Decode:\s*([\d.]+)")
# "\tSwap: (99,), avg_num: 400.0, avg_time: 0.9 ms, avg_size: 6.2 MB, avg_bw: 6929.9 MB/s"
RE_SWAP = re.compile(
    r"Swap:\s*\(\d+,\),\s*avg_num:\s*([\d.]+),\s*avg_time:\s*([\d.]+) ms,"
    r"\s*avg_size:\s*([\d.]+) MB,\s*avg_bw:\s*([\d.]+) MB/s")
# "Peak Memory (GB) RSS: 1.842 TorchAllocated: 1.401 TorchReserved: 1.520"
# (RSS is kernel-tracked VmHWM, covers the whole process incl. pinned diskio
# staging buffers; TorchAllocated/Reserved are torch.cuda's own allocator
# stats, attributable to specific GPU tensors -- see main.py's
# get_peak_rss_kb() and KVSWAP_DISK_IO_QWEN3_0.6B.md sec.8)
RE_PEAKMEM = re.compile(
    r"Peak Memory \(GB\) RSS:\s*([\d.]+|n/a)\s*TorchAllocated:\s*([\d.]+)\s*TorchReserved:\s*([\d.]+)")
# run log filename (main.py and shadowkv both use this): <model>_<tag>_b<batch>_cl<ctx>.log,
# tag itself may contain '_'
RE_ENGINE_NAME = re.compile(r"^(?P<model>.+?)_(?P<tag>.+)_b(?P<batch>\d+)_cl(?P<ctx>\d+)$")
RE_SKV_TPUT = re.compile(r"Throughput:\s*([\d.]+) tokens/s")

# tag prefix in the run log filename -> display name. Longest prefix wins
# (checked in this order) since eval_nano.sh's ablation trio all start with
# "infinigen": infinigen_ru_gp / infinigen_ru / infinigen must not collapse
# into one bucket.
METHOD_BY_TAG = [
    ("flexgen", "FlexGen"),
    ("infinigen_ru_gp", "InfiniGen*+ru+gp"),
    ("infinigen_ru", "InfiniGen*+ru"),
    ("infinigen", "InfiniGen*"),
    ("kvswap", "KVSwap"),
    ("shadowkv", "ShadowKV"),
]
ORDER = ["FlexGen", "InfiniGen*", "InfiniGen*+ru", "InfiniGen*+ru+gp", "KVSwap",
         "ShadowKV", "vLLM"]


def method_from_tag(tag):
    for prefix, name in METHOD_BY_TAG:
        if tag == prefix or tag.startswith(prefix + "_"):
            return name
    return tag


def parse_engine_log(path):
    """One run (main.py or shadowkv) -> dict, or None if no throughput line yet."""
    stem = os.path.basename(path)[: -len(".log")]
    m = RE_ENGINE_NAME.match(stem)
    if not m:
        return None
    text = open(path, errors="ignore").read()

    if m["tag"].startswith("shadowkv"):
        tput = RE_SKV_TPUT.findall(text)
        if not tput:
            return None
        return {
            "method": "ShadowKV",
            "config": m["tag"],
            "batch": int(m["batch"]),
            "ctx": int(m["ctx"]),
            "decode_tps": float(tput[-1]),
            "log": path,
        }

    tput = RE_TPUT.findall(text)
    if not tput:
        return None
    total, prefill, decode = (float(x) for x in tput[-1])
    lat = RE_LAT.findall(text)
    # Per-layer swap stats: one line per layer per run. Averaging over layers
    # gives the per-step read cost that actually gates decode throughput.
    swaps = [tuple(float(x) for x in s) for s in RE_SWAP.findall(text)]
    row = {
        "method": method_from_tag(m["tag"]),
        "config": m["tag"],
        "batch": int(m["batch"]),
        "ctx": int(m["ctx"]),
        "decode_tps": decode,
        "prefill_tps": prefill,
        "total_tps": total,
        "log": path,
    }
    if lat:
        row["latency_total_s"], row["latency_prefill_s"], row["latency_decode_s"] = (
            float(x) for x in lat[-1])
    if swaps:
        n = len(swaps)
        row["swap_layers"] = n
        row["swap_avg_num"] = sum(s[0] for s in swaps) / n
        row["swap_avg_time_ms"] = sum(s[1] for s in swaps) / n
        row["swap_avg_size_mb"] = sum(s[2] for s in swaps) / n
        row["swap_avg_bw_mbs"] = sum(s[3] for s in swaps) / n
        # what one decode step pays in disk reads, summed over all layers
        row["swap_step_time_ms"] = sum(s[1] for s in swaps)
        row["swap_step_size_mb"] = sum(s[2] for s in swaps)
    peakmem = RE_PEAKMEM.findall(text)
    if peakmem:
        rss, torch_alloc, torch_reserved = peakmem[-1]
        row["peak_rss_gb"] = float(rss) if rss != "n/a" else None
        row["peak_torch_alloc_gb"] = float(torch_alloc)
        row["peak_torch_reserved_gb"] = float(torch_reserved)
    return row


def parse_vllm_csv(path, ctx_filter=None):
    rows = []
    if not os.path.exists(path):
        return rows
    for r in csv.DictReader(open(path)):
        ctx = int(r["seqlen"])
        if ctx_filter and ctx != ctx_filter:
            continue
        rows.append({
            "method": "vLLM",
            "config": "no-offload",
            "batch": int(r["batch"]),
            "ctx": ctx,
            "decode_tps": float(r["throughput"]),
            "log": path,
        })
    return rows


def summarize_jtop_decode(path, decode_s, gen_tokens):
    """Aggregate only the decode phase of one run.

    The whole-run aggregates are dominated by prefill (a compute-bound burst
    that looks the same for every method), which hides the part these methods
    actually differ in. The sampler runs at 1 Hz and the process exits right
    after decode, so the last ceil(decode_s) samples are the decode phase.
    """
    import math

    df = pd.read_csv(path)
    n = min(len(df), max(int(math.ceil(decode_s)), 1))
    d = df.tail(n)
    if d.empty:
        return {}
    energy_j = d["power_tot_mw"].sum() / 1000  # 1 Hz => mW-seconds -> J at /1000
    return {
        "decode_samples": n,
        "decode_gpu_mean_pct": d["gpu_pct"].mean(),
        "decode_cpu_mean_pct": d[[c for c in d.columns if c.startswith("cpu")]]
                                .mean(axis=1).mean(),
        "decode_power_mean_w": d["power_tot_mw"].mean() / 1000,
        "decode_energy_j": energy_j,
        "decode_j_per_token": energy_j / gen_tokens if gen_tokens else float("nan"),
    }


def summarize_diskio(path):
    """One run's diskio samples -> whole-run aggregates (scripts/jtop_logger.py)."""
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return {
        "disk_util_mean_pct": df["util_pct"].mean(),
        "disk_util_max_pct": df["util_pct"].max(),
        "disk_read_kbps_mean": df["read_kbps"].mean(),
        "disk_read_kbps_max": df["read_kbps"].max(),
        "disk_read_iops_mean": df["read_iops"].mean(),
        "disk_queue_depth_mean": df["queue_depth"].mean(),
        "disk_avg_read_kb_mean": df.loc[df["read_iops"] > 0, "avg_read_kb"].mean(),
    }


def summarize_diskio_decode(path, decode_s):
    """Same aggregates, restricted to the last ceil(decode_s) samples."""
    import math

    df = pd.read_csv(path)
    n = min(len(df), max(int(math.ceil(decode_s)), 1))
    d = df.tail(n)
    if d.empty:
        return {}
    return {
        "decode_disk_util_mean_pct": d["util_pct"].mean(),
        "decode_disk_util_max_pct": d["util_pct"].max(),
        "decode_disk_read_kbps_mean": d["read_kbps"].mean(),
        "decode_disk_read_iops_mean": d["read_iops"].mean(),
        "decode_disk_queue_depth_mean": d["queue_depth"].mean(),
        "decode_disk_avg_read_kb_mean": d.loc[d["read_iops"] > 0, "avg_read_kb"].mean(),
    }


def summarize_jtop(path):
    """One run's jtop samples -> one row of aggregates.

    Means are taken over the whole run (prefill included), so read them as
    'what this run did to the box', not as steady-state decode behaviour.
    """
    df = pd.read_csv(path)
    if df.empty:
        return None
    cpu = df[[c for c in df.columns if c.startswith("cpu")]]
    out = {
        "samples": len(df),
        "duration_s": len(df),  # 1 Hz sampler
        "cpu_mean_pct": cpu.mean(axis=1).mean(),
        "cpu_max_pct": cpu.mean(axis=1).max(),
        "cpu_busiest_core_pct": cpu.mean().max(),
        "gpu_mean_pct": df["gpu_pct"].mean(),
        "gpu_max_pct": df["gpu_pct"].max(),
        "gpu_freq_mean_mhz": df["gpu_freq_khz"].mean() / 1000,
        "emc_mean_pct": df["emc_pct"].mean(),
        "emc_max_pct": df["emc_pct"].max(),
        "emc_freq_mean_mhz": df["emc_freq_khz"].mean() / 1000,
        "ram_used_peak_gb": df["ram_used_kb"].max() / 1024 / 1024,
        "ram_used_mean_gb": df["ram_used_kb"].mean() / 1024 / 1024,
        "ram_free_min_gb": df["ram_free_kb"].min() / 1024 / 1024,
        "ram_cached_mean_gb": df["ram_cached_kb"].mean() / 1024 / 1024,
        "swap_used_peak_gb": df["swap_used_kb"].max() / 1024 / 1024,
        "power_mean_w": df["power_tot_mw"].mean() / 1000,
        "power_peak_w": df["power_tot_mw"].max() / 1000,
        "power_cpu_gpu_cv_mean_w": df["power_cpu_gpu_cv_mw"].mean() / 1000,
        "power_soc_mean_w": df["power_soc_mw"].mean() / 1000,
        "energy_wh": df["power_tot_mw"].sum() / 1000 / 3600,  # 1 Hz => mW-seconds
        "jetson_clocks": df["jetson_clocks"].iloc[-1],
        "nvp_model": df["nvp_model"].iloc[-1],
    }
    return out


def collect(log_root, model, ctx):
    rows = []
    for p in sorted(glob(os.path.join(log_root, "logs", "nano", model, "*.log"))):
        r = parse_engine_log(p)
        if r and r["ctx"] == ctx:
            rows.append(r)
    rows += parse_vllm_csv(
        os.path.join(log_root, "logs", "nano", model, f"{model}_results.csv"), ctx_filter=ctx)
    return rows


def collect_jtop(log_root, model, runs=None, gen_len=100):
    """runs: the engine-run frame, used to slice out each run's decode phase."""
    decode_s = {}
    if runs is not None:
        for _, r in runs.iterrows():
            if pd.notna(r.get("latency_decode_s")):
                decode_s[(str(r["method"]), int(r["batch"]))] = r["latency_decode_s"]
    rows = []
    for p in sorted(glob(os.path.join(log_root, "logs", "nano", model, "*.jtop.csv"))):
        stem = os.path.basename(p)[: -len(".jtop.csv")]
        m = RE_ENGINE_NAME.match(stem)
        if not m:
            continue
        s = summarize_jtop(p)
        if not s:
            continue
        method = method_from_tag(m["tag"])
        batch = int(m["batch"])
        s.update({"method": method, "batch": batch, "ctx": int(m["ctx"]), "jtop_csv": p})
        if (method, batch) in decode_s:
            s.update(summarize_jtop_decode(p, decode_s[(method, batch)],
                                           gen_len * batch))
        diskio_p = p[: -len(".jtop.csv")] + ".diskio.csv"
        if os.path.exists(diskio_p):
            s["diskio_csv"] = diskio_p
            s.update(summarize_diskio(diskio_p))
            if (method, batch) in decode_s:
                s.update(summarize_diskio_decode(diskio_p, decode_s[(method, batch)]))
        rows.append(s)
    return rows


# pandas' own .plot() needs pkg_resources, which isn't in engine/.venv — go
# through matplotlib directly instead of adding a dependency to the eval env.
def _grouped_bars(ax, piv, ylabel, legend=False):
    import numpy as np

    x = np.arange(len(piv.index))
    cols = [c for c in ORDER if c in piv.columns]
    w = 0.8 / max(len(cols), 1)
    for i, c in enumerate(cols):
        ax.bar(x - 0.4 + w * (i + 0.5), piv[c].values, width=w, label=c)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in piv.index])
    ax.set_xlabel("batch size")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    if legend:
        ax.legend(fontsize=8)


def plot_throughput(df, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    piv = df.pivot_table(index="batch", columns="method", values="decode_tps")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    _grouped_bars(axes[0], piv, "decode throughput (tok/s)", legend=True)
    axes[0].set_title("Decode throughput — Qwen3-0.6B, 16K ctx, NVMe (Orin Nano)")

    # Same data normalized to vLLM (the no-offload upper bound) at each batch,
    # which is how the paper frames the offloading cost.
    if "vLLM" in piv.columns:
        rel = piv.div(piv["vLLM"], axis=0) * 100
        _grouped_bars(axes[1], rel.drop(columns="vLLM"), "% of vLLM throughput")
        axes[1].axhline(100, color="k", ls="--", lw=1, label="vLLM (no offload)")
        axes[1].set_title("Relative to no-offload upper bound")
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}")


def plot_resources(jt, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("gpu_mean_pct", "GPU util (%)"),
        ("cpu_mean_pct", "CPU util, mean over cores (%)"),
        ("ram_used_peak_gb", "peak RAM used (GB)"),
        ("power_mean_w", "mean power (W)"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.6))
    for i, (ax, (col, label)) in enumerate(zip(axes, metrics)):
        piv = jt.pivot_table(index="batch", columns="method", values=col)
        _grouped_bars(ax, piv, label, legend=(i == 0))
    fig.suptitle("Per-run resource usage (jtop, 1 Hz, whole run incl. prefill)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}")


def plot_disk_util(jt, out_png):
    """Decode-phase disk-side view: is the NVMe itself the bottleneck, or idle?

    Pairs with plot_resources' GPU util — a method with low GPU% and low
    disk util% is stalled on something other than either (e.g. CPU/io_uring
    submission, see the b=4 hang investigated for infinigen_ru), while low
    GPU% + high disk util% is a straightforward "waiting on the device" read.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    io = jt[jt["decode_disk_util_mean_pct"].notna()] if \
        "decode_disk_util_mean_pct" in jt.columns else jt.iloc[0:0]
    if io.empty:
        print("[disk_util] no runs with diskio.csv + decode-phase info — skipping")
        return
    metrics = [
        ("decode_disk_util_mean_pct", "decode NVMe %util"),
        ("decode_disk_read_kbps_mean", "decode read throughput (KB/s)"),
        ("decode_disk_avg_read_kb_mean", "avg read size (KB)"),
        ("decode_disk_queue_depth_mean", "avg queue depth"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.6))
    for i, (ax, (col, label)) in enumerate(zip(axes, metrics)):
        _grouped_bars(ax, io.pivot_table(index="batch", columns="method", values=col),
                      label, legend=(i == 0))
    fig.suptitle("Decode-phase disk I/O (from /proc/diskstats, 1 Hz)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}")


def plot_io(df, out_png):
    """Why the two disk-offloading engines differ, in their own log's terms.

    Both read the same bytes per decode step (same max_num_kv, same layer
    count), so the gap is entirely in how fast those reads complete.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    io = df[df["swap_step_time_ms"].notna()]
    if io.empty:
        return
    metrics = [
        ("swap_step_size_mb", "KV read per decode step (MB)"),
        ("swap_step_time_ms", "KV read time per decode step (ms)"),
        ("swap_avg_bw_mbs", "effective read bandwidth (MB/s)"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.6))
    for i, (ax, (col, label)) in enumerate(zip(axes, metrics)):
        _grouped_bars(ax, io.pivot_table(index="batch", columns="method", values=col),
                      label, legend=(i == 0))
    fig.suptitle("Disk KV traffic per decode step (summed over 28 layers, from run logs)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}")


def plot_timeline(df, log_root, model, ctx, batch, out_png):
    """Overlay each method's jtop trace for one batch size, time-aligned.

    The aggregates in jtop_summary.csv mix prefill and decode; this is the view
    that separates them. Runs are aligned at their *end* (decode finishes when
    the process exits) so the decode phases line up, and the prefill/decode
    boundary from the run's own "Latency Total:" line is drawn in.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[(df["batch"] == batch) & df["latency_decode_s"].notna()]
    if sub.empty:
        print(f"[timeline] no engine run with latency info at batch={batch}")
        return
    metrics = [("gpu_pct", "GPU util (%)"), ("power_tot_mw", "power (mW)"),
               ("ram_used_kb", "RAM used (KB)")]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 8), sharex=True)
    for _, row in sub.iterrows():
        csv_path = os.path.join(
            log_root, "logs", "nano", model,
            f"{model}_{row['config']}_b{batch}_cl{ctx}.jtop.csv")
        if not os.path.exists(csv_path):
            continue
        d = pd.read_csv(csv_path)
        # negative t = before process exit, so t=0 is end of decode
        t = pd.Series(range(-len(d) + 1, 1), dtype=float)
        for ax, (col, label) in zip(axes, metrics):
            ax.plot(t, d[col], lw=1.2, label=f"{row['method']}")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
        axes[0].axvline(-row["latency_decode_s"], ls="--", lw=1, alpha=0.6)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("seconds before process exit "
                        "(dashed line = start of decode phase)")
    fig.suptitle(f"jtop timeline — {model}, {ctx} ctx, batch {batch}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", default=None,
                    help="default: $EVAL_LOG_DIR/$EVAL_USER")
    ap.add_argument("--model", default="Qwen3-0.6B")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--outdir", default="RESULTS/nano")
    ap.add_argument("--timeline-batch", type=int, default=4,
                    help="batch size to draw the per-run jtop timeline for")
    args = ap.parse_args()

    log_root = args.log_root
    if not log_root:
        base, user = os.environ.get("EVAL_LOG_DIR"), os.environ.get("EVAL_USER")
        if not base or not user:
            sys.exit("Set EVAL_LOG_DIR/EVAL_USER (source .env_nano) or pass --log-root")
        log_root = os.path.join(base, user)

    os.makedirs(args.outdir, exist_ok=True)

    rows = collect(log_root, args.model, args.ctx)
    if not rows:
        sys.exit(f"No completed runs found under {log_root}")
    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"], ORDER + sorted(
        set(df["method"]) - set(ORDER)), ordered=True)
    df = df.sort_values(["method", "batch"])
    summary_csv = os.path.join(args.outdir, "summary.csv")
    df.to_csv(summary_csv, index=False)
    print(f"wrote {summary_csv}\n")
    print(df.pivot_table(index="batch", columns="method", values="decode_tps")
            .round(2).to_string())

    jt_rows = collect_jtop(log_root, args.model, runs=df)
    if jt_rows:
        jt = pd.DataFrame(jt_rows)
        jt["method"] = pd.Categorical(jt["method"], ORDER + sorted(
            set(jt["method"]) - set(ORDER)), ordered=True)
        jt = jt.sort_values(["method", "batch"])
        # A jtop CSV also exists for runs that crashed (OOM etc.); those samples
        # describe a partial run, so drop them rather than charting them next to
        # completed ones.
        done = set(zip(df["method"].astype(str), df["batch"]))
        dropped = [f"{m} b{b}" for m, b in
                   zip(jt["method"].astype(str), jt["batch"]) if (m, b) not in done]
        if dropped:
            print(f"\n[jtop] dropping incomplete runs: {', '.join(dropped)}")
        jt = jt[[(m, b) in done for m, b in
                 zip(jt["method"].astype(str), jt["batch"])]]
        jtop_csv = os.path.join(args.outdir, "jtop_summary.csv")
        jt.to_csv(jtop_csv, index=False)
        print(f"\nwrote {jtop_csv}")
        cols = ["method", "batch", "duration_s", "gpu_mean_pct",
                "cpu_mean_pct", "ram_used_peak_gb", "power_mean_w", "energy_wh",
                "decode_gpu_mean_pct", "decode_power_mean_w", "decode_j_per_token",
                "decode_disk_util_mean_pct", "decode_disk_read_kbps_mean",
                "decode_disk_avg_read_kb_mean", "decode_disk_queue_depth_mean"]
        cols = [c for c in cols if c in jt.columns]
        print(jt[cols].round(2).to_string(index=False))
        plot_resources(jt, os.path.join(args.outdir, "resources.png"))
        plot_disk_util(jt, os.path.join(args.outdir, "disk_util.png"))

    plot_throughput(df, os.path.join(args.outdir, "throughput.png"))
    plot_io(df, os.path.join(args.outdir, "disk_io.png"))
    plot_timeline(df, log_root, args.model, args.ctx, args.timeline_batch,
                  os.path.join(args.outdir, f"timeline_b{args.timeline_batch}.png"))


if __name__ == "__main__":
    main()
