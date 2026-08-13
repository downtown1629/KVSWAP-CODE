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

1. [x] Static config tests confirm the 3:1 schedule and per-layer capacities.
2. [x] Functional attention tests compare windowed prefill with a PyTorch mask
   and verify chunked FlashAttention receives the correct K/V prefix.
3. [x] Cache tests cover prompt truncation, decode append, and rollover.
4. [x] Existing MoE and expert-store tests remain unchanged.
5. [x] Jetson validation just beyond the boundary (prompt 520, decode 2)
   retains explicit OS headroom.

The first implementation remains BF16 and synchronous for experts. It does not
invent a Maple KV predictor; KVSwap global-layer execution requires a separately
calibrated predictor artifact.

## Jetson evidence

The pinned Maple checkpoint completed prompt 520/decode 2 with 18 bounded SWA
layers and six full-history global layers. The run planned 24.07 MiB of KV,
produced `assistant`, and peaked at 3.668 GiB RSS / 2.068 GiB CUDA allocated.
It reported no swap and no KV disk synchronization during decode. Expert demand
loading remained separate: 1,584 materializations and 50,316 extent reads.
