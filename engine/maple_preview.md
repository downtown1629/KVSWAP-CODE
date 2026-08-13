# Maple-Preview Adapter

This adapter targets `deepgrove/maple-preview` at immutable Hugging Face revision
`ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07`. The reviewed public configuration is
a 24-layer, 20B-A1B model with 256 BF16 experts per layer and eight active
experts. Its indexed Transformers checkpoint is 40,428,060,672 bytes; the
current repository does not contain the packed ternary runtime mentioned in the
model card.

## Supported slice

The engine maps Maple's `model.word_embeddings`, Q/K RMSNorm, normalized top-8
router, clamp-aware SwiGLU experts, partial RoPE on sliding layers, and NoPE on
global layers. Experts reuse M2's bounded synchronous NVMe provider, but receive
a Maple-specific artifact format. The resident fixed weights are approximately
1.65 GiB and one BF16 expert is 6 MiB.
The adapter parses the pinned `config.json` directly and does not execute the
repository's `trust_remote_code` modules during engine startup.

Layer-specific long-context handling follows
[MAPLE_LONG_CONTEXT_CACHE_PLAN.md](MAPLE_LONG_CONTEXT_CACHE_PLAN.md):

- Sliding layers retain only 511 prior KV entries and apply a 512-token prefill
  window; global NoPE layers retain the full history.
- KVSwap storage/selection is reserved for global layers. A calibrated Maple
  predictor is still required before using a non-`none` `lr_proj_mode`.
- Expert weights remain BF16. Ternary packing/dequantization is not inferred
  from the model card.
- The first real run is an engine smoke test, not Hugging Face parity. The public
  reference requires Triton/FlashAttention and cannot fit concurrently on an
  8 GiB Orin Nano.

This separation prevents sliding layers from reading or storing out-of-window
KV while preserving the complete history needed by global layers.

## Prepare and run

```bash
cd engine
bash scripts/eval_maple_preview.sh static
bash scripts/eval_maple_preview.sh download
bash scripts/eval_maple_preview.sh pack
bash scripts/eval_maple_preview.sh verify
bash scripts/eval_maple_preview.sh smoke
bash scripts/eval_maple_preview.sh long-smoke
```

`pack` refuses to overwrite an existing store. `verify` is the explicit offline
full-checkpoint and extent audit; normal startup only hashes resident tensors
and validates the manifest checksum root. Keep the revision pinned when copying
either checkpoint or store.

## Orin Nano evidence

The pinned BF16 checkpoint was packed into 6,144 extents totaling exactly 36
GiB. A deterministic prompt-32/decode-2 run produced `We need`, made 792
materialize calls and 6,336 expert reads (39,862,665,216 logical bytes), and
peaked at 3.296 GiB RSS. Total latency was 26.66 seconds. This proves bounded
startup and end-to-end execution; real-checkpoint logits still require an
independent large-memory reference parity run.

The layer-specific cache policy also passed prompt-520/decode-2. It activated
18 local SWA caches capped at 511 prior entries and six full-history global
caches, planned 24.07 MiB total KV, produced `assistant`, and peaked at 3.668
GiB RSS / 2.068 GiB CUDA allocated. This validates the 512-token rollover on
Jetson; it is not a Hugging Face logit-parity result. Maple KVSwap selection is
still blocked until a calibrated global-layer predictor is available.
