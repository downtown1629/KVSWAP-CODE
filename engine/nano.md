# Running on Jetson Orin Nano (8 GB, NVMe-only)

`README.md` in this directory documents the paper's target platform: **Jetson Orin AGX**, 64 GB
unified RAM, both eMMC and NVMe. `scripts/setup.sh` enforces that hardware directly — it reads
`/proc/device-tree/model` and exits unless it contains `"AGX Orin"` — and `scripts/eval.sh` and the
`fig-*.sh`/`tab-4.sh` sweeps default to models and per-batch KV budgets sized for that machine, and
unconditionally require both `EMMC_OFFLOAD_DIR`/`EMMC_DEV_NAME` and `NVME_OFFLOAD_DIR`/`NVME_DEV_NAME`.

None of that fits a **Jetson Orin Nano with 8 GB RAM and no eMMC**. This document covers the
`scripts/*_nano.sh` variants written for that machine instead. They're new files, not edits to the
originals — `setup.sh`/`eval.sh`/etc. are untouched and still describe the AGX Orin path.

The paper's own reference point for this class of hardware is **§5.2.2 / Table 5**: Orin Nano, NVMe
only, **Qwen3-0.6B and Qwen3-1.7B**, context length 16K, batch sizes 1–8. The scripts here default to
that configuration, and have since been run on this device reproducing Table 5's flexgen/InfiniGen*/
KVSwap/ShadowKV/vLLM baselines for Qwen3-0.6B at batch 1–4 (batch 8 is tight on 8GB for the
disk-offloading modes — see "Known gaps" below).

## What's different from the AGX Orin scripts

| | AGX Orin (`scripts/*.sh`) | Orin Nano (`scripts/*_nano.sh`) |
|---|---|---|
| Hardware check | hard-exits unless device-tree model contains `"AGX Orin"` | prints the detected model and warns if unrecognized; only fatal with `STRICT_HW_CHECK=1` |
| Disk | eMMC **and** NVMe, both required | NVMe only — no `EMMC_*` vars needed |
| Default model | Llama-3.1-8B-Instruct / Qwen3-14B | Qwen3-0.6B |
| Power mode / clock check | hard-exits unless `nvpmodel` reports exactly `MAXN` and CPU/GPU devfreq are pinned at a hardcoded sysfs path | prints what it finds and warns instead of exiting; `eval_nano.sh` additionally runs `sudo jetson_clocks` itself before each batch (advisory — needs passwordless sudo, otherwise clocks are left as-is) |
| Adapters | ships with the repo under `engine/data/adapters/` | ships for Qwen3-0.6B/1.7B under `$MODEL_PATH_BASE/local_adapters` (committed to the repo, see below); `prepare_adapter_nano.sh` (re)generates more |
| vLLM baseline | `run_vllm.sh` hard-exits like `setup.sh`; `run_vllm.py` hardcodes `gpu_memory_utilization=0.85`/`max_model_len=32768` | `run_vllm_nano.sh`; both knobs are env-overridable (`VLLM_GPU_MEM_UTIL`/`VLLM_MAX_MODEL_LEN`) since the AGX defaults don't fit 8GB |
| ShadowKV baseline | `src/shadowkv/run_shadowkv.sh` hard-exits like `setup.sh` | `run_shadowkv_nano.sh`, soft-check equivalent, same underlying `test/e2e_jetson.py` driver |
| Resource logging | none | `eval_nano.sh` wires in `scripts/jtop_logger.py` automatically — one CPU/GPU/EMC/RAM/power/disk-I/O sample per second per run |

## Files

- **`scripts/nano_common.sh`** — shared, sourced by the others. Defines `check_hardware_soft`,
  `check_powermode_soft`, `check_jetson_clocks_soft` — advisory versions of the checks in
  `setup.sh`/`eval.sh`. Not meant to be run directly.
- **`scripts/download_models_nano.sh`** — fetches `Qwen/Qwen3-0.6B` into `$MODEL_PATH_BASE_HF`
  (Qwen3-1.7B is present but commented out — uncomment for the paper's second Nano data point).
- **`scripts/setup_nano.sh`** — same wheel/dependency install as `setup.sh` (the prebuilt aarch64
  wheels in `wheel_pkgs/` are JetPack/CUDA-version-specific, not AGX-specific, so nothing changes
  there), but disk setup only mounts/checks NVMe, and weight conversion
  (`scripts/make_np_weights.py`, which already always saves fp16 regardless of model) defaults to
  Qwen3-0.6B via `NANO_MODEL_LIST`.
- **`scripts/prepare_adapter_nano.sh`** — generates the KVSwap low-rank adapter (and, with
  `--with-infinigen`, the InfiniGen* skew adapter) for a model that has none checked into
  `engine/data/adapters/`. Runs `quality/src/prepare_adapter.py` — the paper's own offline
  adapter-tuning code (PAPER.pdf §3.5) — through *this* directory's `.venv`, since
  `quality/scripts/install.sh` pulls x86_64 PyPI torch wheels that won't work on Jetson at all.
  Writes to `$MODEL_PATH_BASE/local_adapters` — a separate tree from the Git-LFS-managed
  `engine/data/adapters/` that `link_adapters.sh` symlinks from, but (unlike that tree) committed
  here as plain binaries, same as `engine/data/adapters/`'s own `.pt` files. Currently populated for
  Qwen3-0.6B (KVSwap low-rank ratios 1.0/0.25 + InfiniGen* skew ratio 0.125) and Qwen3-1.7B (KVSwap
  low-rank ratios 1.0/0.25).
- **`scripts/eval_nano.sh`** — NVMe-only driver for `src/main.py`, with five modes:
  - `flexgen` — full-KV baseline, no prediction, **no adapter required**. Run this first.
  - `infinigen` — InfiniGen*-style index-selecting predictor. Needs the skew adapter
    (`prepare_adapter_nano.sh --with-infinigen`).
  - `infinigen_ru` / `infinigen_ru_gp` — the same InfiniGen* predictor with KVSwap's reuse buffer
    (`+ru`), then also its grouped I/O (`+ru+gp`), added on top — reproduces the paper's
    InfiniGen*/+ru/+ru+gp ablation chain (§4.2) using the same `--reuse_budget`/`--token_group` flags
    `kvswap` mode uses (both flags are generic to `main.py`, not KVSwap-specific). Needs the same
    skew adapter as `infinigen`.
  - `kvswap` — the actual KVSwap low-rank predictor. Needs the KVSwap adapter
    (`prepare_adapter_nano.sh`, no flag needed).

  Also applies `sudo jetson_clocks` before each batch (set `APPLY_JETSON_CLOCKS=0` to skip), drops
  the page cache between batches (advisory, needs passwordless sudo), and starts/stops
  `scripts/jtop_logger.py` around each run (see below) — override its path with `JTOP_PY` if
  jetson-stats isn't at the default `/home/jetson/.local/share/jtop/bin/python`.
- **`scripts/run_vllm_nano.sh`** — soft-check counterpart to `scripts/run_vllm.sh`. Set
  `VLLM_GPU_MEM_UTIL`/`VLLM_MAX_MODEL_LEN` to fit the vLLM "no offload" baseline into 8GB (the AGX
  defaults of 0.85/32768 fail to start here); defaults to `VLLM_MAX_MODEL_LEN` = the largest seqlen
  being tested.
- **`scripts/run_shadowkv_nano.sh`** — soft-check counterpart to `src/shadowkv/run_shadowkv.sh`.
  Requires the ShadowKV CUDA extension already built
  (`cd src/shadowkv && MAX_JOBS=1 python setup.py build_ext --inplace` — `MAX_JOBS=1` matters on 8GB
  unified memory, see "Known gaps"). `budget`/`chunk_size`/`rank` default to values copied from the
  AGX sweep scripts (`tab-4.sh`/`fig-10.sh`) as a starting point, since the paper doesn't publish an
  Orin Nano/Qwen3-0.6B-specific setting.
- **`scripts/jtop_logger.py`** — must run under jetson-stats' own venv, not `engine/.venv` (it ships
  `jtop` as an importable module only there). Three sampling loops in one process, all at 1Hz (jtop's
  *client* is unreliable below 1.0s on this jetson-stats version — one sample then stalls): CPU/GPU/
  RAM/SWAP/power via `jtop`; disk read/write IOPS/throughput/queue-depth/`%util` read straight from
  `/proc/diskstats` (jtop's own API only exposes disk *capacity*, not I/O); and EMC bandwidth % via
  `sudo tegrastats` (jtop's own `EMC` stat is wrong on this board — see Known gaps below). Writes
  `<run>.jtop.csv` + `<run>.diskio.csv`.
- **`scripts/analyze_nano_results.py`** — parses `eval_nano.sh`/ShadowKV/vLLM logs plus the
  jtop+diskio CSVs into one comparison: decode-only throughput per method/batch, per-layer disk cost
  from the engine's own `Swap:` log lines, prefill/decode-split resource usage, and disk-I/O ablation
  charts. Run via `.venv/bin/python scripts/analyze_nano_results.py` (needs `pandas`/`matplotlib`,
  already in `engine/.venv`); writes CSVs + PNGs to `RESULTS/nano/` by default.

## Usage

```bash
cd engine

# 1. Set the usual env vars, minus anything eMMC-related:
export NVME_DEV_NAME='nvme0n1p1'
export NVME_OFFLOAD_DIR='/mnt/nvme/offload'
export MODEL_PATH_BASE_HF='../../data/model_weights_hf'
export MODEL_PATH_BASE='../../data/model_weights'
export EVAL_LOG_DIR='../../data/kvswap_logs'
export EVAL_USER='test0'

# 2. Fetch weights, set up venv + NVMe mount + fp16 np-weights
bash ./scripts/download_models_nano.sh
bash ./scripts/setup_nano.sh

# 3. Sanity check: full-KV baseline, no adapter needed
bash ./scripts/eval_nano.sh flexgen

# 4. Adapters for Qwen3-0.6B/1.7B already ship under $MODEL_PATH_BASE/local_adapters — only needed
#    for a model/ratio that isn't there yet:
bash ./scripts/prepare_adapter_nano.sh          # add --with-infinigen for the InfiniGen* baseline too

# 5. Run KVSwap itself
bash ./scripts/eval_nano.sh kvswap

# 6. Baselines outside main.py
bash ./scripts/run_vllm_nano.sh                     # vLLM, no offloading
bash ./scripts/run_shadowkv_nano.sh                 # ShadowKV (build its CUDA ext first, see above)

# 7. Once a batch of runs has completed logs, compare them:
.venv/bin/python scripts/analyze_nano_results.py
```

`eval_nano.sh` takes positional args: `<mode> [total_len] ["batch_list"] [model]`, e.g.:

```bash
bash ./scripts/eval_nano.sh kvswap 16384 "1 2 4 8"
RATIO=0.25 bash ./scripts/eval_nano.sh kvswap 32768 "1 4"   # tight-budget adapter
bash ./scripts/eval_nano.sh infinigen_ru_gp 16384 "1 2"     # InfiniGen* + reuse buffer + grouped I/O
```

Logs land under `$EVAL_LOG_DIR/$EVAL_USER/logs/nano/<model>/`, one file per `(mode, batch, context)`
combination; reruns skip any log that already has a `Throughput Total:` line, same convention as
`scripts/eval.sh`. Each also gets a sibling `.jtop.csv` + `.diskio.csv` pair (see
`scripts/jtop_logger.py` above) if jetson-stats is installed.

## Known gaps / things to verify

These scripts have now been run on an actual Orin Nano, but this board's 8GB **unified** memory (CPU,
GPU, and page cache all share one pool) makes a few things worth knowing before trusting a number or
running something heavy:

- **Power mode / clocks.** `nano_common.sh`'s checks print what they find but don't block execution;
  `eval_nano.sh` separately runs `sudo jetson_clocks` itself before each batch (advisory — silently
  skipped without passwordless sudo, in which case clocks can sit at a low idle point and throughput
  numbers will be lower and noisier than representative). Check a run's `.jtop.csv` — `gpu_freq_khz`
  pinned near its max rather than drifting down confirms clocks actually held for that run.
  Confirm the correct max-performance `nvpmodel` mode id for your Nano/carrier board/JetPack version
  yourself (`sudo nvpmodel -q --verbose`, `sudo nvpmodel -m <id>`) if `jetson_clocks` alone isn't
  enough.
- **Memory headroom.** 8 GB goes quickly once a model, its KV predictor, and (for `kvswap`/
  `infinigen_ru*`) a reuse buffer are all resident — `eval_nano.sh` drops the page cache between
  batches for this reason. Batch 8 is tight for every disk-offloading mode on this board; `flexgen`
  (no offloading at all) is the first to run out of room as batch grows — if it does, that's the
  expected motivation for KVSwap, not a bug. A run's `.jtop.csv` `ram_free_kb`/`swap_used_kb` columns
  show how close a given config is running to the edge.
- **`io_uring`'s locked-memory limit.** Each `DiskIO` instance registers pinned staging buffers with
  the kernel (`ulimit -l`), and a larger `--reuse_budget` needs more of them. If that limit is
  exceeded, `io_uring_queue_init_sqpoll` fails with `BlockingIOError: [Errno 11]` inside a background
  worker thread — and because that failure currently isn't propagated back to the main process, the
  run hangs indefinitely (near-zero CPU, zero disk I/O) instead of erroring out. If a run seems stuck
  with no throughput line appearing, check for exactly this in its log before assuming it's just
  slow; raising `ulimit -l` or lowering `--reuse_budget`/batch size are the two levers.
- **`prepare_adapter.py`'s ratio-formatting quirk.** It parses `--ratios` with `float()`, so a ratio
  of `1` becomes the directory suffix `_1.0`, not `_1` like the adapters already shipped in
  `engine/data/adapters/` (`main.py` loads `--lr_proj_path` with exact string concatenation, no
  globbing, so a mismatched suffix fails silently later). `prepare_adapter_nano.sh` symlinks around
  this (`normalize_ratio_dirs`), and the same symlinks are already in place for the shipped
  `local_adapters/*_mh_1` adapters — but if you invoke `quality/src/prepare_adapter.py` directly or
  add a new ratio by hand, watch for it.
- **NVMe free-space minimum** in `setup_nano.sh` (`NVME_MIN_FREE_GB`, default 20) and
  **`MAX_ALLOC_KV_SIZE`** in `eval_nano.sh` (default 1 GiB per on-disk KV file) are conservative
  estimates for Qwen3-0.6B/1.7B at batch ≤ 8, context ≤ 32K — both are disk-space knobs (NVMe has
  headroom to spare), not RAM knobs, but raise them if you hit `create_kv_file`'s
  `total_bytes <= MAX_ALLOC_KV_SIZE` assertion at larger batches/contexts.
- **jtop's own `EMC` field is wrong on this board — `jtop_logger.py` sources EMC% from `sudo
  tegrastats` instead.** jetson-stats' `read_emc()` (`jtop/core/memory.py`, confirmed present in both
  7.1.5 and 7.2.0) computes `utilization // emc['cur']` on the raw
  `/sys/kernel/debug/bpmp/debug/actmon/mc_all_avg_activity` counter with no `*100` and no unit
  reconciliation (`utilization` is unscaled, `emc['cur']` is kHz), so it floors to `0` for any
  realistic load — confirmed both by direct measurement (`jtop`'s `EMC` stat reads `0` in every
  sample regardless of load) and by cross-checking against `sudo tegrastats`/Jetson Power GUI, which
  report a real, varying `EMC_FREQ%` for the identical load. This isn't a one-line fix, either: on
  Orin/T234 that activity counter is produced by BPMP firmware (a closed-source blob on the separate
  Cortex-R5 co-processor), not by a Linux kernel driver — checked against NVIDIA's public R36.5
  `kernel_src.tbz2`, where `drivers/firmware/tegra/bpmp-debugfs.c` is a generic debugfs-to-BPMP-MRQ
  passthrough with no activity-counter math at all, so the correct conversion isn't reconstructable
  from public source. `tegrastats` already gets this right, so `jtop_logger.py`'s `emc_loop` shells
  out to `sudo tegrastats --interval <ms>` (this device has passwordless sudo) and parses `EMC_FREQ`
  from it, overriding jtop's own (still-buggy) value in the `emc_pct` column; `emc_freq_khz` stays
  sourced from jtop since that's just a direct sysfs read and was already accurate. Every other jtop
  field used here (CPU/GPU/RAM/power) was cross-checked line-by-line against `tegrastats` and matches,
  so only `EMC` needed this workaround.
  jetson-stats' client also can't sample faster than 1.0s reliably here; a `jtop(interval=<1.0)`
  request delivers exactly one sample and then stalls, so `jtop_logger.py` is pinned at 1.0s rather
  than something finer.
