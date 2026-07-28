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
that configuration.

## What's different from the AGX Orin scripts

| | AGX Orin (`scripts/*.sh`) | Orin Nano (`scripts/*_nano.sh`) |
|---|---|---|
| Hardware check | hard-exits unless device-tree model contains `"AGX Orin"` | prints the detected model and warns if unrecognized; only fatal with `STRICT_HW_CHECK=1` |
| Disk | eMMC **and** NVMe, both required | NVMe only — no `EMMC_*` vars needed |
| Default model | Llama-3.1-8B-Instruct / Qwen3-14B | Qwen3-0.6B |
| Power mode / clock check | hard-exits unless `nvpmodel` reports exactly `MAXN` and CPU/GPU devfreq are pinned at a hardcoded sysfs path | prints what it finds and warns instead of exiting — this check was only verified on AGX Orin, not Nano |
| Adapters | ships with the repo under `engine/data/adapters/` | **does not ship** for Qwen3-0.6B — a new script, `prepare_adapter_nano.sh`, generates them |

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
  Writes to `$MODEL_PATH_BASE/local_adapters` (untracked — kept out of the Git-LFS-managed
  `engine/data/adapters/` tree that `link_adapters.sh` symlinks from).
- **`scripts/eval_nano.sh`** — NVMe-only driver for `src/main.py`, with three modes:
  - `flexgen` — full-KV baseline, no prediction, **no adapter required**. Run this first.
  - `infinigen` — InfiniGen*-style index-selecting predictor. Needs the skew adapter
    (`prepare_adapter_nano.sh --with-infinigen`).
  - `kvswap` — the actual KVSwap low-rank predictor. Needs the KVSwap adapter
    (`prepare_adapter_nano.sh`, no flag needed).

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

# 4. Generate the adapter KVSwap needs (not shipped for Qwen3-0.6B)
bash ./scripts/prepare_adapter_nano.sh          # add --with-infinigen for the InfiniGen* baseline too

# 5. Run KVSwap itself
bash ./scripts/eval_nano.sh kvswap
```

`eval_nano.sh` takes positional args: `<mode> [total_len] ["batch_list"] [model]`, e.g.:

```bash
bash ./scripts/eval_nano.sh kvswap 16384 "1 2 4 8"
RATIO=0.25 bash ./scripts/eval_nano.sh kvswap 32768 "1 4"   # tight-budget adapter
```

Logs land under `$EVAL_LOG_DIR/$EVAL_USER/logs/nano/<model>/`, one file per `(mode, batch, context)`
combination; reruns skip any log that already has a `Throughput Total:` line, same convention as
`scripts/eval.sh`.

## Known gaps / things to verify on real hardware

These scripts were written by tracing `src/main.py`, `src/cache_manager.py`, `src/diskio/`, and
`quality/src/prepare_adapter.py` against the CLI contract `scripts/eval.sh` already uses — they
have **not been run on an actual Orin Nano** (no GPU available in the environment they were authored
in). Before trusting results from them:

- **Power mode / clocks.** `nano_common.sh`'s checks print what they find but don't block execution.
  Confirm the correct max-performance `nvpmodel` mode id for your Nano/carrier board/JetPack version
  yourself (`sudo nvpmodel -q --verbose`, `sudo nvpmodel -m <id>`, `sudo jetson_clocks`) before
  treating throughput numbers as representative — an un-pinned clock will silently produce lower,
  noisier numbers.
- **Memory headroom.** 8 GB is tight. `flexgen` mode (no offloading at all) is the most likely to
  OOM at larger batch sizes — if it does, that's the expected motivation for KVSwap, not a bug; drop
  to a smaller batch for the baseline comparison or rely on `kvswap` mode instead.
- **`prepare_adapter.py`'s ratio-formatting quirk.** It parses `--ratios` with `float()`, so a ratio
  of `1` becomes the directory suffix `_1.0`, not `_1` like the adapters already shipped in
  `engine/data/adapters/` (`main.py` loads `--lr_proj_path` with exact string concatenation, no
  globbing, so a mismatched suffix fails silently later). `prepare_adapter_nano.sh` symlinks around
  this (`normalize_ratio_dirs`), but if you invoke `quality/src/prepare_adapter.py` directly, watch
  for it.
- **NVMe free-space minimum** in `setup_nano.sh` (`NVME_MIN_FREE_GB`, default 20) and
  **`MAX_ALLOC_KV_SIZE`** in `eval_nano.sh` (default 1 GiB per on-disk KV file) are conservative
  estimates for Qwen3-0.6B/1.7B at batch ≤ 8, context ≤ 32K — both are disk-space knobs (NVMe has
  headroom to spare), not RAM knobs, but raise them if you hit `create_kv_file`'s
  `total_bytes <= MAX_ALLOC_KV_SIZE` assertion at larger batches/contexts.
