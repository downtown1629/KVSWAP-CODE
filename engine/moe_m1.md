# Qwen3-MoE M1 Development Guide

Milestone 1 adds correctness-first, fully resident Qwen3-MoE layers while
leaving KVSwap's attention selection, cache manager, and disk I/O unchanged.
Resident execution is disabled unless an explicit weight budget is supplied.
Qwen3-30B-A3B BF16 requires 56.8705 GiB for checkpoint weights alone, so it
must not be loaded on an 8 GB Orin Nano.

The weight limit is user approval, not the OOM check. Before opening checkpoint
shards, the engine also compares weights, planned KV/activations, workspace,
the largest source tensor, and configurable system headroom against
`MemAvailable` and the CUDA allocator limit. The default headroom is 2 GiB.

## Safe Validation Order

Run metadata validation before downloading or opening model shards:

```bash
cd engine
.venv/bin/python scripts/preflight_qwen3_moe.py /path/to/model-metadata
```

The directory only needs `config.json` and
`model.safetensors.index.json`. The official Qwen3-30B-A3B metadata validates
18,867 tensor names and a 56.8705 GiB BF16 checkpoint.

Create the approximately 1.1 MiB deterministic fixture under `/tmp`:

```bash
FIXTURE_DIR=$(mktemp -d /tmp/kvswap-qwen3-moe.XXXXXX)
.venv/bin/python scripts/make_tiny_qwen3_moe.py "$FIXTURE_DIR"
.venv/bin/python scripts/preflight_qwen3_moe.py "$FIXTURE_DIR"
```

Verify streaming loads, router IDs, routing weights, and expert outputs on CPU
or CUDA. CPU is the allocation-safe default; CUDA exercises Orin device BF16
behavior. The fixture is small, but check system memory before CUDA execution.

```bash
.venv/bin/python scripts/verify_qwen3_moe_fixture.py "$FIXTURE_DIR"
free -h
.venv/bin/python scripts/verify_qwen3_moe_fixture.py "$FIXTURE_DIR" --device cuda
```

Run the static and reference suite without loading a full checkpoint:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

The wrapper exposes the same safe lanes. `fixture-fullkv` is explicitly CUDA
and runs a 64-token prefill plus one decode step using at most a 0.01 GiB
resident-weight budget:

```bash
bash scripts/eval_moe_m1.sh static
bash scripts/eval_moe_m1.sh fixture-cpu
bash scripts/eval_moe_m1.sh fixture-cuda
bash scripts/eval_moe_m1.sh fixture-fullkv
```

On Orin, the initial full-KV fixture run matched the HF greedy IDs `[99, 8]`.
It recorded 1.603 GiB peak process RSS, 15 MiB peak Torch allocation, and 21
MiB peak Torch reservation. These numbers include runtime overhead and are not
a capacity estimate for the real 30B checkpoint.

## Current Boundary

`ResidentExpertProvider` returns references to resident expert banks. The MoE
primitive owns routing and expert arithmetic but never calls `CacheManager` or
storage APIs. `MoEBlock` loads router/norm weights separately and obtains the
expert bank through a provider factory. M2 replaces that factory with demand
materialization while preserving the primitive; its cache and asynchronous
storage interfaces will be defined from measured M2 requirements. M1 does not
implement expert caching, prefetch, quantization, or fused kernels.

`--moe_token_chunk_size` bounds routed-token temporaries during prefill. The
default is 8192; lowering it trades extra launches for lower peak memory. A
layer-level NVTX range is available without emitting per-token traces.

This boundary is intentionally narrow for future models: fixed layer weights,
expert providers, config validation, and memory accounting may be shared.
Qwen3.5 hybrid recurrent state, Maple weight representation, and Gemma's
parallel dense/routed FFN remain model-specific adapter work.

Full engine generation and KVSwap coexistence remain integration gates. Do not
claim M1 complete from metadata or isolated-layer parity alone. Record device,
available memory, context length, batch size, peak memory, and the exact command
for any CUDA run.
