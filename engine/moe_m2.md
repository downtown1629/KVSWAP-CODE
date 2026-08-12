# Qwen3-MoE M2: Synchronous Expert Demand Loading

M2 adds a deliberately slow storage baseline behind M1's `ExpertProvider`
boundary. Router and norm weights remain resident; each routed layer reads only
the unique experts selected for the current token chunk, copies them into one
shared bounded scratch bank, computes, and invalidates the bank logically on the
next call. There is no expert cache, prefetch, overlap, quantization, or reuse.

## Artifact and Runtime

`scripts/pack_qwen3_moe_experts.py` streams BF16 gate/up/down tensors into one
4096-byte-aligned extent per `(layer, expert)`. `manifest.json` records a config
fingerprint, exact shapes and byte ranges, source revision, and per-expert
SHA-256. Startup validates coverage, overlap, alignment, file bounds, shapes,
representation, and an explicitly supplied immutable checkpoint revision before
CUDA allocation. Offline verification
rehashes every extent.

The Jetson backend uses `O_DIRECT`, a reusable page-aligned CUDA-registered host
buffer, and synchronous io_uring submit/wait. All layers share that staging
buffer and a single BF16 device scratch bank. `--moe_expert_scratch_slots`
bounds the bank; if an actual chunk selects more unique experts, it fails before
issuing any read. The capacity gate includes fixed weights, scratch, staging,
KV, activations, workspace, allocator limit, and explicit system headroom.

## Validation

```bash
cd engine
bash scripts/eval_moe_m2.sh static
bash scripts/eval_moe_m2.sh fixture-buffered
bash scripts/eval_moe_m2.sh fixture-direct
bash scripts/eval_moe_m2.sh fixture-kvswap
```

On Orin Nano, buffered and direct full-KV runs reproduced the M1/HF fixture
output `token_99 token_8`. With a 16-token chunk, each performed 10 materialize
calls, 20 cold expert reads, and 983,040 logical bytes; direct mode used 1.526
GiB peak RSS. The strengthened joint direct-I/O gate compares resident/demand
routing JSONL, output tokens, and both KV traces. It selected 128 KV tokens and
accounted for 983,040 expert bytes plus 32,768 KV bytes while preserving
`token_99 token_8`. These fixture figures validate correctness and lifecycle, not
performance.

For a real store, pack on a host with sufficient storage:

```bash
PYTHONPATH=src .venv/bin/python scripts/pack_qwen3_moe_experts.py \
  pack /models/Qwen3-30B-A3B /nvme/qwen3-30b-experts --source-revision REV
PYTHONPATH=src .venv/bin/python scripts/pack_qwen3_moe_experts.py \
  verify /models/Qwen3-30B-A3B /nvme/qwen3-30b-experts --source-revision REV
```

Real execution additionally requires `--expert_mode demand`, the store path,
an explicit fixed+scratch approval limit, and a conservative slot/chunk choice.
Do not bypass a failed capacity preflight on memory-constrained Jetson systems.
