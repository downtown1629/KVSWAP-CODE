# Maple Long-Context Cache Policy

## Goal and correctness boundary

Maple repeats three `sliding_attention` layers followed by one
`full_attention` layer. Long-context execution must preserve this schedule
instead of presenting the same full KV history to every layer.

- Sliding layers apply partial RoPE and attend to at most the current token plus
  511 prior tokens (`sliding_window=512`). Their persistent cache is a bounded,
  in-memory recent-KV buffer; older KV is discarded and never written to NVMe.
- Global layers use NoPE and retain the complete history. Full-KV runs use the
  configured cache placement. KVSwap runs may select/offload KV only for these
  global layers.
- Qwen and other existing models retain their current uniform cache behavior.

## Implementation seams

`model_adapters.py` owns layer classification and cache-capacity decisions.
`SelfAttention` uses those decisions when allocating, loading, and storing KV.
The attention backend receives an optional window for prefill masking. Decode
loads at most 511 previous sliding-layer entries and concatenates the current
KV, so the effective attention width never exceeds 512.

The KVSwap pipeline must skip sliding targets. Prediction and prefetch remain
valid only when the immediately following attention layer is global; the target
global layer's NoPE rule must also apply to predictor scoring.

## Validation gates

1. Static config tests confirm the 3:1 schedule and per-layer capacities.
2. Functional attention tests compare windowed prefill with a PyTorch mask.
3. Cache tests cover prompt truncation, decode append, and rollover at 512.
4. Existing Qwen/M2 tests remain unchanged.
5. Jetson validation starts just beyond the boundary (prompt 520, decode 2),
   then increases only while preflight retains explicit OS headroom.

The first implementation remains BF16 and synchronous for experts. It does not
invent a Maple KV predictor; KVSwap global-layer execution requires a separately
calibrated predictor artifact.
