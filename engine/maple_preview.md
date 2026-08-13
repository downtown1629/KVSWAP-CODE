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

## Jetson profiling

Use the same `jtop_logger.py` sampler as the original KVSwap Nano reproduction:

```bash
cd engine
bash scripts/profile_maple_preview.sh 520 16 1
```

Arguments are `prompt_len`, `gen_len`, and `batch_size`. Results default to
`data/kvswap_logs/maple-preview/`; override this with `MAPLE_PROFILE_DIR`.
Each run keeps the raw engine log, 1 Hz jtop and NVMe samples, a JSON/CSV
summary, and a timeline PNG. The summary separates prefill and decode latency,
throughput, GPU/CPU/EMC usage, power/energy, RAM/swap, and NVMe utilization. It
also reports expert read/copy time, bytes, effective bandwidth, request counts,
KV disk synchronization, and peak process/CUDA memory. Phase resource values
are 1 Hz approximations: decode is the final `ceil(decode_latency)` samples and
prefill is the immediately preceding `ceil(prefill_latency)` samples.

For Nsight profiling, follow
[NSIGHT_JETSON_INCIDENT.md](NSIGHT_JETSON_INCIDENT.md). On the current Orin
Nano image, CUDA/NVTX software tracing is allowed but Nsight Compute and Nsight
Systems hardware GPU metrics are prohibited because they triggered Tegra HWPM
driver errors and loss of local console output.

For a token/layer Gantt trace, use the non-root software-only wrapper:

The trace hierarchy, token semantics, current naive expert-demand path, and
measured bottlenecks are documented in
[MAPLE_GANTT_TRACE_GUIDE.md](MAPLE_GANTT_TRACE_GUIDE.md).

```bash
cd engine
bash scripts/profile_maple_nsys.sh 32 4
```

The resulting `.nsys-rep` nests `KVSWAP_TOKEN` → `KVSWAP_LAYER` →
`KVSWAP_STAGE`, then attention/MoE sub-stages and prefill chunks. Open it with
Nsight Systems on another machine and expand the process's NVTX row. The wrapper
also exports a token overview (`.token-gantt.svg`), detailed Gantt SVGs split
into eight-model-layer pages (`.layers-00-07.gantt.svg`, etc.), raw ranges
(`.nvtx.csv`), per-token stage totals (`.stage-summary.csv`), and CUDA/NVTX
summaries (`.stats.txt`). Every SVG includes a legend for the colors present in
that page. SVG timing is CPU-side NVTX wall time; use the `.nsys-rep` to
correlate asynchronous CUDA kernels. To change page size without recollecting a
trace, rerun the exporter on the existing SQLite database:

Each detailed page crops its X-axis to the selected layer group so the bars use
the full image width. Sub-pixel events are drawn at a minimum width of 2 px for
visibility; their tooltip and CSV duration remain exact. Use the token overview
when absolute placement across all layers is important.

```bash
.venv/bin/python scripts/export_nsys_gantt.py TRACE.sqlite --layers-per-page 4
```

Prefill is one parallel prompt range, so a per-token prefill figure is the total
divided by prompt length, not an individually timed token. Each subsequent
`KVSWAP_TOKEN phase=decode` range is one measured output-token step. Stage and
sub-stage rows are nested and must not be added together.

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
