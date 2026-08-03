#!/usr/bin/env python3
"""CPU/GPU/EMC/RAM/SWAP/power/disk-I/O sampler for KVSwap eval runs on Jetson.

Must run under jetson-stats' own venv (it ships jtop as an importable module
there, not into the system/engine Python): /home/jetson/.local/share/jtop/bin/python

Two sampling loops share one process so eval_nano.sh only has to start/stop
one PID per run:
  - jtop loop: CPU/GPU/EMC/RAM/SWAP/power, via jetson-stats.
  - diskio loop: per-device read/write IOPS, throughput, queue depth and
    %util, read straight from /proc/diskstats (the same source `iostat -x`
    uses) since jtop's own API only exposes disk *capacity*, not I/O.

Both run at 1.0s. That's not a diskio limitation — /proc/diskstats has no
lower bound — but jtop's *client* does: intervals below 1.0s were tried and
found unreliable on this jetson-stats version (7.1.5), delivering one sample
then stalling. 1.0s is the floor that actually works, so both loops share it
for directly time-aligned rows.

Usage: jtop_logger.py <jtop_csv> <diskio_csv> <disk_device> [interval_seconds]
  disk_device: as it appears in /proc/diskstats, e.g. nvme0n1p1 (matches
               NVME_DEV_NAME/--disk_dev_name elsewhere in this repo).
Runs until killed (SIGTERM/SIGINT), flushing each row as it's written.
"""
import csv
import signal
import sys
import threading
import time

from jtop import jtop

JTOP_FIELDS = [
    "time",
    "cpu1", "cpu2", "cpu3", "cpu4", "cpu5", "cpu6",
    "gpu_pct", "gpu_freq_khz",
    "emc_pct", "emc_freq_khz",
    "ram_used_kb", "ram_free_kb", "ram_buffers_kb", "ram_cached_kb", "ram_shared_kb",
    "swap_used_kb", "swap_cached_kb",
    "power_tot_mw", "power_cpu_gpu_cv_mw", "power_soc_mw",
    "jetson_clocks", "nvp_model",
]
DISKIO_FIELDS = [
    "time", "read_iops", "write_iops", "read_kbps", "write_kbps",
    "avg_read_kb", "avg_write_kb", "queue_depth", "util_pct",
]

_stop = threading.Event()


def _handle_stop(_signum, _frame):
    _stop.set()


def _read_diskstat(device):
    """One device's line from /proc/diskstats -> raw counters, or None."""
    with open("/proc/diskstats") as f:
        for line in f:
            parts = line.split()
            if parts[2] == device:
                return {
                    "reads": int(parts[3]),
                    "read_sectors": int(parts[5]),
                    "writes": int(parts[7]),
                    "write_sectors": int(parts[9]),
                    "io_ms": int(parts[12]),  # field 13: time doing I/Os
                }
    return None


def diskio_loop(out_path, device, interval):
    prev = _read_diskstat(device)
    if prev is None:
        print(f"[diskio] device '{device}' not found in /proc/diskstats — "
              "disk I/O logging disabled for this run", file=sys.stderr)
        return
    prev_t = time.monotonic()

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DISKIO_FIELDS)
        writer.writeheader()
        while not _stop.wait(interval):
            now = _read_diskstat(device)
            now_t = time.monotonic()
            dt = now_t - prev_t
            if now is None or dt <= 0:
                prev_t = now_t
                continue
            d_reads = now["reads"] - prev["reads"]
            d_writes = now["writes"] - prev["writes"]
            d_rsec = now["read_sectors"] - prev["read_sectors"]
            d_wsec = now["write_sectors"] - prev["write_sectors"]
            d_io_ms = now["io_ms"] - prev["io_ms"]
            row = {
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "read_iops": round(d_reads / dt, 1),
                "write_iops": round(d_writes / dt, 1),
                # sectors are always 512 bytes regardless of device block size
                "read_kbps": round(d_rsec * 512 / 1024 / dt, 1),
                "write_kbps": round(d_wsec * 512 / 1024 / dt, 1),
                "avg_read_kb": round(d_rsec * 512 / 1024 / d_reads, 2) if d_reads else 0,
                "avg_write_kb": round(d_wsec * 512 / 1024 / d_writes, 2) if d_writes else 0,
                # weighted time-in-flight / wall time = average queue depth
                "queue_depth": round(d_io_ms / 1000 / dt, 3),
                # ms with >=1 I/O outstanding / wall ms = %util (iostat's definition)
                "util_pct": round(min(d_io_ms / (dt * 1000) * 100, 100.0), 1),
            }
            writer.writerow(row)
            f.flush()
            prev, prev_t = now, now_t


def jtop_loop(out_path, interval):
    with open(out_path, "w", newline="") as f, jtop(interval=interval) as jetson:
        writer = csv.DictWriter(f, fieldnames=JTOP_FIELDS)
        writer.writeheader()
        while not _stop.is_set() and jetson.ok():
            stats = jetson.stats
            mem = jetson.memory
            gpu = jetson.gpu.get("gpu", {})
            row = {
                "time": stats["time"].isoformat(),
                "cpu1": stats.get("CPU1"), "cpu2": stats.get("CPU2"),
                "cpu3": stats.get("CPU3"), "cpu4": stats.get("CPU4"),
                "cpu5": stats.get("CPU5"), "cpu6": stats.get("CPU6"),
                "gpu_pct": stats.get("GPU"),
                "gpu_freq_khz": gpu.get("freq", {}).get("cur"),
                "emc_pct": stats.get("EMC"),
                "emc_freq_khz": mem.get("EMC", {}).get("cur"),
                "ram_used_kb": mem.get("RAM", {}).get("used"),
                "ram_free_kb": mem.get("RAM", {}).get("free"),
                "ram_buffers_kb": mem.get("RAM", {}).get("buffers"),
                "ram_cached_kb": mem.get("RAM", {}).get("cached"),
                "ram_shared_kb": mem.get("RAM", {}).get("shared"),
                "swap_used_kb": mem.get("SWAP", {}).get("used"),
                "swap_cached_kb": mem.get("SWAP", {}).get("cached"),
                "power_tot_mw": stats.get("Power TOT"),
                "power_cpu_gpu_cv_mw": stats.get("Power VDD_CPU_GPU_CV"),
                "power_soc_mw": stats.get("Power VDD_SOC"),
                "jetson_clocks": stats.get("jetson_clocks"),
                "nvp_model": stats.get("nvp model"),
            }
            writer.writerow(row)
            f.flush()
    _stop.set()  # jetson.ok() went false (service died) — stop diskio too


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    jtop_csv, diskio_csv, device = sys.argv[1], sys.argv[2], sys.argv[3]
    interval = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    t = threading.Thread(target=diskio_loop, args=(diskio_csv, device, interval),
                         daemon=True)
    t.start()
    jtop_loop(jtop_csv, interval)
    t.join(timeout=interval + 2)


if __name__ == "__main__":
    main()
