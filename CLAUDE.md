# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Artifact evaluation code for the MobiSys '26 paper **"KVSwap: Disk-aware KV Cache Offloading for
Long-Context On-device Inference"** (`PAPER.pdf`). KVSwap is a research system (~5K lines of Python,
built on top of FlexGen + vLLM's PagedAttention) that lets an on-device LM keep its full KV cache on
disk (NVMe/eMMC) instead of RAM, using a compact low-rank in-memory K-cache summary to predict which
grouped KV entries to prefetch each decoding step.

This is **not** a library/product repo — there is no lint config, no test suite, no CI. It's a
paper artifact: two mostly-independent evaluation tracks that reproduce the paper's tables/figures.

- **`engine/`** — throughput evaluation. Runs the actual KVSwap engine + baselines on-device and
  measures decode tokens/sec. Target hardware: **NVIDIA Jetson Orin AGX** (64 GB unified RAM, eMMC +
  NVMe). See `engine/README.md`.
- **`quality/`** — accuracy/generation-quality evaluation. Runs the same prediction algorithm as a
  plug-in on top of HF `transformers` models on an A100-class server, no disk offloading involved.
  See `quality/README.md`.

Read `PAPER.pdf` for the design rationale (predictor math in §3.1–3.3, runtime in §3.4) before
touching `engine/src/methods.py` or `engine/src/cache_manager.py` — the code closely mirrors the
paper's notation (`lr_proj_mode`, `token_group`/group size `G`, `max_num_kv`/`M`, `reuse_budget`).

## ⚠️ This machine vs. what the repo assumes

This session runs on a **Jetson Orin Nano, 8 GB RAM, no eMMC** — not the Orin AGX (64 GB, eMMC+NVMe)
that `engine/`'s scripts are written for. Concretely:

- `engine/scripts/setup.sh` reads `/proc/device-tree/model` and **hard-exits unless it contains
  "AGX Orin"**. It will refuse to run as-is on this device.
- `eval.sh`/`fig-*.sh`/`tab-4.sh` default to models (Llama-3.1-8B, Qwen3-14B) and per-batch KV
  budgets sized for 64 GB RAM, and unconditionally require both `EMMC_OFFLOAD_DIR`/`EMMC_DEV_NAME`
  and `NVME_OFFLOAD_DIR`/`NVME_DEV_NAME` to be set (this device has no eMMC block device).
- The paper's own cross-platform result for this exact class of hardware is §5.2.2 / Table 5: Orin
  Nano, **NVMe only**, small models (**Qwen3-0.6B, Qwen3-1.7B**), context length 16K, batch 1–8 — use
  that as the realistic reference point, not the AGX Orin config in the READMEs.
- `engine/scripts/*_nano.sh` (`setup_nano.sh`, `download_models_nano.sh`, `prepare_adapter_nano.sh`,
  `eval_nano.sh`, sharing helpers in `nano_common.sh`) are NVMe-only, Qwen3-0.6B-by-default
  counterparts to `setup.sh`/`download_models.sh`/`eval.sh`, written for exactly this machine — use
  these instead of patching the AGX-only originals. **See `engine/nano.md`** for the full writeup:
  run order, what each script changes relative to its AGX Orin counterpart, and known gaps to verify
  before trusting throughput numbers (none of this has been run on real Orin Nano hardware yet).

## Setup & commands

### `engine/` (Jetson, throughput)

```bash
cd engine
bash ./scripts/setup.sh          # AGX-only hardware check; creates .venv via uv, installs wheels
bash ./scripts/download_models.sh
bash ./scripts/link_adapters.sh
```

Dependencies are installed from prebuilt aarch64 wheels in `wheel_pkgs/` (torch 2.7, flash-attn
2.7.4, triton 3.2, and a **custom vLLM 0.10.1 wheel** — see `wheel_pkgs/readme.txt` — that widens
PagedAttention block sizes in `csrc/attention/paged_attention_v{1,2}.cu`; a stock vLLM wheel will not
work with `cache_manager.py`'s block layout). `src/shadowkv` and `src/Liburing` are built in-place as
part of setup (`python setup.py build_ext --inplace`, `pip install -e .`).

Required env vars before running anything (see `engine/README.md` §2.2 "User configuration"):
`EMMC_DEV_NAME`, `EMMC_OFFLOAD_DIR`, `NVME_DEV_NAME`, `NVME_OFFLOAD_DIR`, `MODEL_PATH_BASE_HF`,
`MODEL_PATH_BASE`, `EVAL_LOG_DIR`, `EVAL_USER`, `EVAL_MODE` (`quick`|`full`).

Single run of the core engine (what `eval.sh` ultimately shells out to):

```bash
source .venv/bin/activate
export MAX_ALLOC_KV_SIZE=$((1024*1024*768))   # bytes; required by diskio/disk_interface.py
python src/main.py --model_path <path> --offload_dir <dir> --disk_dev_name nvme \
  --prompt_len 24476 --gen_len 100 --gpu_batch_size 8 --num_gpu_batches 1 \
  --percent 100 0 0 0 100 0 --lr_proj_mode lr_proj_mh --lr_proj_path <adapter dir> \
  --max_num_kv 400 --token_group 4 --reuse_budget 400 --start_layer 0-curr-emb \
  --run_args L4 --test_input_path ./data/test_inputs --run_info <log path prefix>
```

`--lr_proj_mode none` runs plain FlexGen (full KV cache, no prediction). `base` = InfiniGen-style
index-selecting predictor (needs `--skew_matrix_path`/`--skew_partial_idx_path`); `lr_proj_mh` = the
KVSwap low-rank predictor (needs `--lr_proj_path`). `--reuse_budget 0` disables the reuse buffer.
`--disk_dev_name` selects which offload dir/IO tuning profile to use (`nvme`, `emmc`, or `usb`).

Full baseline sweeps reproducing specific paper artifacts (each wraps `eval.sh` + `run_vllm.sh` +
ShadowKV's own `src/shadowkv/run_shadowkv.sh`, then renders results via `scripts/utils.py`):

```bash
bash ./scripts/tab-4.sh [quick|full]      # Table 4: throughput vs. context len / batch size
bash ./scripts/fig-10.sh [quick|full]     # Figure 10: throughput vs. model size
bash ./scripts/fig-11-tp.sh [quick|full]  # Figure 11 (throughput half only)
```

Or drive everything from `engine/run_evaluation.ipynb` (used as-is for remote/reviewer access).
Outputs land in `$EVAL_LOG_DIR/$EVAL_USER/logs/...` (raw per-run logs) and `./RESULTS/$EVAL_USER/`
(aggregated figures/tables). A completed run's log contains a `Throughput Total:` line; the sweep
scripts skip re-running any `(config)` whose log already has that line, so partial reruns resume
where they left off.

vLLM baseline directly:

```bash
bash ./scripts/run_vllm.sh <model_dir_name> <seqlen_list csv> <batch_list csv>
```

### `quality/` (A100 server, accuracy — will not run on this Jetson)

```bash
cd quality
bash scripts/install.sh
bash ./scripts/download_model.sh [full]
bash ./scripts/download_dataset.sh [full]
export DS_API_KEY=...          # DeepSeek judge LLM, for NIAH/MLVU scoring
export EVAL_USER=test0 CUDA_VISIBLE_DEVICES=0
bash ./scripts/fig-9.sh / tab-2.sh / tab-3-left.sh / tab-3-right.sh / fig-11-acc.sh
```

### Cloning

The repo carries Git LFS payloads (model adapters, RULER/MLVU data). Default clone must skip LFS
smudge, or use sparse checkout per track (see root `README.md` and `engine/remote.md` /
`quality/remote.md` for exact commands and for setting up remote Jupyter+frpc access).

## Architecture

### `engine/` — the offloading engine

The core generation loop is a **fork of FlexGen** (`main.py`: `Policy`, `InputEmbed`, `OutputEmbed`,
`SelfAttention`, `MLP`, `LM`, `run_flexgen()` are FlexGen's original class/function shapes) rewired to
back its KV cache with disk instead of a GPU/CPU/compressed 3-tier hierarchy, and to use vLLM's
`PagedAttention` CUDA kernels for the actual attention compute.

Data flow per decode step, spread across `main.py` (`SelfAttention`), `cache_manager.py`, and
`methods.py`:

1. **Predict** (`methods.py: speculate_attention`) — using layer *i-1*'s hidden state as a stand-in
   for layer *i*'s query (justified in paper §6.1, "layer-to-layer approximation"), project it
   through a precomputed low-rank adapter and score it against the in-memory compressed K cache
   (`lr_proj_mode='lr_proj_mh'`) or an index-selected K cache (`lr_proj_mode='base'`, the InfiniGen*
   baseline) to pick the top KV *groups* (`token_group` consecutive tokens, not individual tokens/
   heads — this is KVSwap's key departure from InfiniGen/ShadowKV).
2. **Fetch** (`cache_manager.py: CacheManager.get_load_buffer`, `diskio/`) — diffs predicted group
   indices against the **reuse buffer** (FIFO cache of recently-loaded groups, tracked via
   `reuse_meta_indices`/a slot table) to skip re-reading groups already resident, then issues async
   disk reads for the misses. Two disk backends: `diskio/uring_io.py` (`io_uring`, `O_DIRECT`,
   pinned staging buffers — the real path used on Jetson) and `diskio/disk_interface.py`'s
   `use_mmap` mode (mmap + `posix_fadvise`/`madvise DONTNEED`, mainly for correctness testing).
3. **Compute** — PagedAttention runs over: hit groups already in the reuse buffer + newly-fetched
   groups + the most-recent tokens still sitting in the per-layer **rolling buffer** (uncommitted
   until a full group of size `token_group` accumulates — see `CacheManager.update_rolling_buffer`).
   I/O for layer *i*'s predictions is issued while layer *i-1*'s attention+FFN compute is still
   running (paper §3.4, "reduced I/O transfers").
4. **New KV** gets appended to the rolling buffer, compressed into the low-rank K-cache summary, and
   written to the on-disk full cache in the background.

`compression.py` is a *separate*, older code path: generic FlexGen-style 4-bit group-wise
quantization for weights/activations/cache (`CompressionConfig`, `TorchCompressedDevice`) — unrelated
to the KVSwap low-rank K-cache summary, which lives in `methods.py`/precomputed `lr_proj` adapters.

`model_config.py` derives model-specific dims (head_dim, GQA group count, rope config) per model
family (`llama3`, `qwen2`, `qwen3`) from the HF config — add new model families here first when
onboarding a new checkpoint. `pytorch_backend.py` is FlexGen's original device/tensor abstraction
layer (`TorchDevice`, `TorchDisk`, `TorchTensor`, `DeviceType`) still used for weights and non-KV
activations; the KV cache itself bypasses this in favor of `CacheManager`.

Baselines other than plain FlexGen live as parallel code paths, not separate binaries:
- **InfiniGen / InfiniGen\*** — `--lr_proj_mode base` in the same `main.py` (index-selecting
  predictor + skew matrix, vs. KVSwap's low-rank projection).
- **ShadowKV** — a vendored, separately-built subtree at `engine/src/shadowkv/` (own CUDA kernels
  under `kernels/`, own model impls under `models/`, own driver `run_shadowkv.sh` /
  `test/e2e_jetson.py`) — not integrated into `main.py`.
- **vLLM** — `src/run_vllm.py` / `scripts/run_vllm.sh`, used as the "ideal case, no disk overhead"
  upper-bound baseline (all spare device memory dedicated to KV cache).

`scripts/eval.sh` is the single entry point that all `fig-*.sh`/`tab-*.sh` sweep scripts call
per-configuration; it builds the `main.py` CLI args from shell variables, names the run/log file from
the config, and skips runs whose log already completed. `scripts/utils.py` parses those logs into the
paper's actual figures/tables.

### `quality/` — accuracy evaluation

Independent implementation: patches HF `transformers` modeling files directly
(`src/models/modeling_llama.py`, `modeling_qwen3.py`, `modeling_qwen2_5_vl.py`, InternVL3 under
`src/models/internvl3/`) with prediction/reuse logic injected via `src/models/method.py` (K-cache
masking, skew-mask construction) and `method_gen.py` (per-step generation hooks: `lr_kcache_func`,
`speculate_attention`, `sel_kv`). No disk I/O here — the point of this track is isolating the
*prediction algorithm's* effect on generation quality, decoupled from storage. Driven by
`src/eval_gen.py` over `bench/{LongBench,RULER,MLVU,Needle_test}`, scored via `src/dataset.py` and (for
NIAH/MLVU) a DeepSeek judge LLM. `adapters/` holds precomputed low-rank/skew adapters per model,
shared 1:1 with `engine/data/adapters/` (symlinked into place by `link_adapters.sh` on the engine
side).

### Shared conventions across both tracks

- **Adapters** (`*/adapters/{infinigen_skew,loki_proj_*,lowrank_proj_*}/`) are precomputed offline
  via SVD over a general-purpose corpus (C4) — see paper §3.5 "Offline Parameter Tuning" — and are
  looked up by `{model_name}_{mode}_{ratio}` naming convention in the eval scripts; regenerate via
  `quality/scripts/prepare_adapters.sh` / `quality/src/prepare_adapter.py`, not by hand.
- **`token_group`** (group size `G`) and **`max_num_kv`**/`reuse_budget` (`M`) are the two levers
  controlling the throughput/quality/memory trade-off described in paper §3.5 and §6.1 — expect to
  see them threaded through nearly every script and CLI in both tracks.
