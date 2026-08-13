#!/usr/bin/env bash
set -euo pipefail

# Software trace only. Never add GPU metrics/HWPM or Nsight Compute here; see
# engine/NSIGHT_JETSON_INCIDENT.md.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ENGINE_DIR"

PROMPT_LEN=${1:-32}
GEN_LEN=${2:-4}
PYTHON=.venv/bin/python
REVISION=ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07
MODEL=${MAPLE_MODEL:-../data/model_weights_hf/maple-preview}
STORE=${MAPLE_EXPERT_STORE:-../data/nvme_offload/maple-preview-experts}
TRACE_ROOT=${MAPLE_NSYS_DIR:-../data/kvswap_logs/maple-preview/nsys}

if (( EUID == 0 )); then
    echo "Refusing root Nsight profiling; software CUDA/NVTX trace needs no root." >&2
    exit 2
fi
if [[ ! "$PROMPT_LEN" =~ ^[0-9]+$ ]] || (( PROMPT_LEN < 1 )) || \
   [[ ! "$GEN_LEN" =~ ^[0-9]+$ ]] || (( GEN_LEN < 2 )); then
    echo "usage: $0 [prompt_len>=1] [gen_len>=2]" >&2
    exit 2
fi
if ! command -v nsys >/dev/null; then
    echo "nsys is not installed" >&2
    exit 2
fi
if [[ ! -f "$MODEL/.kvswap_revision" ]] || \
   [[ "$(<"$MODEL/.kvswap_revision")" != "$REVISION" ]]; then
    echo "Maple checkpoint revision marker is missing or mismatched" >&2
    exit 2
fi
if [[ ! -f "$STORE/manifest.json" ]]; then
    echo "Maple expert store is missing: $STORE" >&2
    exit 2
fi

mkdir -p "$TRACE_ROOT"
STAMP=$(date +%Y%m%d-%H%M%S)
PREFIX="$TRACE_ROOT/maple-p${PROMPT_LEN}-g${GEN_LEN}-${STAMP}"
OFFLOAD_DIR=$(mktemp -d /tmp/kvswap-maple-nsys-offload.XXXXXX)
trap 'rmdir "$OFFLOAD_DIR" 2>/dev/null || true' EXIT INT TERM

echo "Collecting software-only CUDA/NVTX trace: $PREFIX.nsys-rep"
MAX_ALLOC_KV_SIZE=67108864 nsys profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --cudabacktrace=none \
    --force-overwrite=true \
    --output="$PREFIX" \
    "$PYTHON" src/main.py \
    --model_path "$MODEL" --offload_dir "$OFFLOAD_DIR" \
    --prompt_len "$PROMPT_LEN" --gen_len "$GEN_LEN" \
    --gpu_batch_size 1 --num_gpu_batches 1 \
    --percent 100 0 100 0 100 0 --test_input_path ./data/test_inputs \
    --run_args L0 --lr_proj_mode none --use_token_cache 0 \
    --dk_wr none --dk_rd none --token_group 1 --disk_dev_name nvme \
    --batch_split 1 --seed 1234 --flash_att 1 --paged_att 0 --nv_profile 1 \
    --expert_mode demand --moe_expert_store "$STORE" \
    --moe_checkpoint_revision "$REVISION" --moe_expert_reader direct \
    --moe_expert_scratch_slots 64 --moe_demand_weight_limit_gb 2.2 \
    --moe_system_headroom_gb 1.5 --moe_token_chunk_size 8 \
    >"$PREFIX.log" 2>&1

nsys stats \
    --report nvtx_pushpop_sum \
    --report nvtx_gpu_proj_sum \
    --report cuda_gpu_kern_sum \
    --report cuda_api_sum \
    --report cuda_kern_exec_sum \
    --format column \
    "$PREFIX.nsys-rep" >"$PREFIX.stats.txt"

"$PYTHON" scripts/export_nsys_gantt.py "$PREFIX.sqlite" \
    --output-prefix "$PREFIX"

echo "Trace: $PREFIX.nsys-rep"
echo "Text summaries: $PREFIX.stats.txt"
echo "Token overview: $PREFIX.token-gantt.svg"
echo "Per-stage timing: $PREFIX.stage-summary.csv"
echo "Open the .nsys-rep on a workstation with Nsight Systems for the Gantt view."
