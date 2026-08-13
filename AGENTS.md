# Repository Guidelines

## What KVSwap Does

KVSwap is a research system for long-context LLM inference on memory-constrained NVIDIA Jetson devices. Instead of retaining the complete key-value (KV) cache in unified memory, it stores full-precision KV blocks on NVMe. A compact, low-rank representation of keys remains in memory and is scored against each query to identify important token groups. KVSwap then fetches only the selected full KV blocks, overlaps asynchronous disk I/O with computation, and performs attention with a patched vLLM PagedAttention kernel. A reuse buffer preserves recently selected groups, while a rolling buffer keeps local context available. The main trade-off is lower memory use and longer supported contexts in exchange for approximation and storage traffic.

The artifact has two complementary evaluation paths:

- `engine/` measures end-to-end throughput on Jetson, including real NVMe offload and retrieval.
- `quality/` uses Hugging Face model modifications to measure approximation quality without modeling disk I/O.

Do not assume the two paths share an implementation merely because they expose similar KVSwap concepts.

## Code and Documentation Map

Runtime orchestration and policies are under `engine/src/`; `methods.py`, `cache_manager.py`, and `main.py` contain the central KVSwap flow. Asynchronous storage backends live in `engine/src/diskio/`, and experiment entry points are in `engine/scripts/`. Accuracy-side model patches are in `quality/src/models/`, with benchmarks in `quality/bench/`. Read `PAPER.pdf`, `README.md`, and the platform reproduction notes before changing algorithms or experiment defaults. `data/` contains machine-specific weights, caches, and results rather than source code.

## Running and Validating Changes

From `engine/`, use `bash scripts/setup_nano.sh` for Orin Nano and run a focused comparison with:

```bash
bash scripts/eval_nano.sh flexgen
bash scripts/eval_nano.sh kvswap 16384 "1 2"
```

For accuracy experiments, run `bash scripts/install.sh` and `bash scripts/quick_run.sh` from `quality/`. There is no unit-test suite or CI; validate the smallest relevant configuration and report hardware, model, context length, batch size, memory use, throughput, and accuracy impact. The engine requires the supplied patched vLLM wheel; stock PagedAttention uses an incompatible block layout.

## Contribution Conventions

Follow nearby Python and shell style and avoid broad formatting changes in legacy code. Use `snake_case` for functions and variables and `PascalCase` for classes. Keep commits narrowly scoped with imperative subjects. Pull requests should identify whether they affect selection quality, reuse/rolling policy, disk scheduling, kernels, or experiment tooling, and include the exact reproduction command plus representative logs. Never commit credentials, `.env_nano`, model weights, generated KV files, or incidental raw logs.
