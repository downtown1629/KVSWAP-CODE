#!/usr/bin/env python3
"""CPU/GPU/EMC/RAM/SWAP/power/disk-I/O sampler for KVSwap eval runs on Jetson.

Must run under jetson-stats' own venv (it ships jtop as an importable module
there, not into the system/engine Python): /home/jetson/.local/share/jtop/bin/python

Three sampling loops share one process so eval_nano.sh only has to start/stop
one PID per run:
  - jtop loop: CPU/GPU/RAM/SWAP/power, via jetson-stats. These fields were
    cross-checked line-by-line against `sudo tegrastats` output and match.
  - diskio loop: per-device read/write IOPS, throughput, queue depth and
    %util, read straight from /proc/diskstats (the same source `iostat -x`
    uses) since jtop's own API only exposes disk *capacity*, not I/O.
  - emc loop: EMC (memory controller) bandwidth %, via `sudo tegrastats`
    instead of jtop. jtop's own `EMC` stat is wrong on this board — its
    `read_emc()` (jtop/core/memory.py) does
    `utilization // emc['cur']` on the raw
    /sys/kernel/debug/bpmp/debug/actmon/mc_all_avg_activity counter, which
    floors to 0 for any realistic load (confirmed: jtop reports 0 in every
    sample while `sudo tegrastats`/Jetson Power GUI show a real, varying
    EMC_FREQ% for the same load). The correct conversion isn't a simple
    `*100` fix either: on Orin/T234 this counter is produced by BPMP
    firmware (closed-source, not in NVIDIA's public L4T kernel source —
    confirmed by checking the R36.5 kernel_src.tbz2, where
    drivers/firmware/tegra/bpmp-debugfs.c is a generic passthrough with no
    activity-counter math at all), so there's no way to reimplement the
    conversion correctly outside of BPMP. tegrastats already computes this
    part correctly, so we shell out to it (needs `sudo`; this device has
    passwordless sudo configured) rather than guess at the formula.

All three run at 1.0s. That's not a diskio/tegrastats limitation — neither
has a lower bound — but jtop's *client* does: intervals below 1.0s were
tried and found unreliable on this jetson-stats version, delivering one
sample then stalling. 1.0s is the floor that actually works, so all loops
share it for directly time-aligned rows.

Usage: jtop_logger.py <jtop_csv> <diskio_csv> <disk_device> [interval_seconds]
  disk_device: as it appears in /proc/diskstats, e.g. nvme0n1p1 (matches
               NVME_DEV_NAME/--disk_dev_name elsewhere in this repo).
Runs until killed (SIGTERM/SIGINT), flushing each row as it's written.
"""
import csv
import re
import signal
import subprocess
import sys
import threading
import time

from jtop import jtop

_EMC_RE = re.compile(r"EMC_FREQ (\d+)%@(\d+)")

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
_emc_lock = threading.Lock()
_latest_emc_pct = None


def _handle_stop(_signum, _frame):
    _stop.set()


def emc_loop(interval):
    """Sample EMC bandwidth % from `sudo tegrastats`, since jtop's own value
    is wrong on this board (see module docstring). Updates the module-level
    _latest_emc_pct; jtop_loop reads it when writing each row."""
    global _latest_emc_pct
    try:
        proc = subprocess.Popen(
            ["sudo", "tegrastats", "--interval", str(int(interval * 1000))],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
    except OSError as e:
        print(f"[emc] couldn't start 'sudo tegrastats': {e} — "
              "EMC%% logging disabled for this run", file=sys.stderr)
        return

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if _stop.is_set():
                break
            m = _EMC_RE.search(line)
            if m:
                with _emc_lock:
                    _latest_emc_pct = float(m.group(1))
    finally:
        if proc.poll() is None:
            # proc runs as root (via sudo); a plain kill from this
            # unprivileged process can't touch it.
            subprocess.run(["sudo", "kill", "-TERM", str(proc.pid)], check=False)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                subprocess.run(["sudo", "kill", "-KILL", str(proc.pid)], check=False)


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
                "emc_pct": _latest_emc_pct,  # from tegrastats, not jtop (see docstring)
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

    t_diskio = threading.Thread(target=diskio_loop, args=(diskio_csv, device, interval),
                                 daemon=True)
    t_emc = threading.Thread(target=emc_loop, args=(interval,), daemon=True)
    t_diskio.start()
    t_emc.start()
    jtop_loop(jtop_csv, interval)
    t_diskio.join(timeout=interval + 2)
    t_emc.join(timeout=interval + 2)


if __name__ == "__main__":
    main()
