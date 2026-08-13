#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ENGINE_DIR"

PROMPT_LEN=${1:-520}
GEN_LEN=${2:-16}
BATCH_SIZE=${3:-1}
PYTHON=.venv/bin/python
REVISION=ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07
MODEL=${MAPLE_MODEL:-../data/model_weights_hf/maple-preview}
STORE=${MAPLE_EXPERT_STORE:-../data/nvme_offload/maple-preview-experts}
PROFILE_ROOT=${MAPLE_PROFILE_DIR:-../data/kvswap_logs/maple-preview}
JTOP_PY=${JTOP_PY:-/home/jetson/.local/share/jtop/bin/python}

if [[ ! "$PROMPT_LEN" =~ ^[0-9]+$ ]] || (( PROMPT_LEN < 1 )) || \
   [[ ! "$GEN_LEN" =~ ^[0-9]+$ ]] || (( GEN_LEN < 2 )) || \
   [[ ! "$BATCH_SIZE" =~ ^[0-9]+$ ]] || (( BATCH_SIZE < 1 )); then
    echo "usage: $0 [prompt_len>=1] [gen_len>=2] [batch_size>=1]" >&2
    exit 2
fi
if [[ ! -f "$MODEL/.kvswap_revision" ]] || \
   [[ "$(<"$MODEL/.kvswap_revision")" != "$REVISION" ]]; then
    echo "Maple checkpoint is not marked as reviewed revision $REVISION" >&2
    exit 2
fi
if [[ ! -f "$STORE/manifest.json" ]]; then
    echo "Maple expert store is missing: $STORE" >&2
    exit 2
fi

SEQUENCE_LEN=$((PROMPT_LEN + GEN_LEN - 1))
STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p "$PROFILE_ROOT"
RUN_NAME="maple-preview_maple-fullkv_b${BATCH_SIZE}_cl${SEQUENCE_LEN}_${STAMP}"
PREFIX="$PROFILE_ROOT/$RUN_NAME"
LOG="$PREFIX.log"
JTOP_CSV="$PREFIX.jtop.csv"
DISKIO_CSV="$PREFIX.diskio.csv"
OFFLOAD_DIR=$(mktemp -d /tmp/kvswap-maple-profile-offload.XXXXXX)
JTOP_PID=""

PROFILE_DISK_DEVICE=${MAPLE_PROFILE_DISK_DEVICE:-}
if [[ -z "$PROFILE_DISK_DEVICE" ]]; then
    PROFILE_DISK_DEVICE=$(basename "$(findmnt -no SOURCE -T "$STORE")")
fi

cleanup() {
    if [[ -n "$JTOP_PID" ]]; then
        kill "$JTOP_PID" 2>/dev/null || true
        wait "$JTOP_PID" 2>/dev/null || true
    fi
    rmdir "$OFFLOAD_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sync
if sudo -n sysctl -w vm.drop_caches=1 >/dev/null 2>&1; then
    echo "[mem] dropped caches before $RUN_NAME"
else
    echo "[mem] no passwordless sudo for drop_caches; continuing"
fi

if [[ -x "$JTOP_PY" ]]; then
    "$JTOP_PY" "$SCRIPT_DIR/jtop_logger.py" "$JTOP_CSV" "$DISKIO_CSV" \
        "$PROFILE_DISK_DEVICE" 1.0 >"$PREFIX.profiler.err" 2>&1 &
    JTOP_PID=$!
else
    echo "[jtop] $JTOP_PY not found; engine-only metrics will still be analyzed"
fi

echo "Profiling Maple: prompt=$PROMPT_LEN gen=$GEN_LEN batch=$BATCH_SIZE"
echo "Raw output prefix: $PREFIX"
free -h

if ! MAX_ALLOC_KV_SIZE=67108864 "$PYTHON" src/main.py \
    --model_path "$MODEL" --offload_dir "$OFFLOAD_DIR" \
    --prompt_len "$PROMPT_LEN" --gen_len "$GEN_LEN" \
    --gpu_batch_size "$BATCH_SIZE" --num_gpu_batches 1 \
    --percent 100 0 100 0 100 0 --test_input_path ./data/test_inputs \
    --run_args L0 --lr_proj_mode none --use_token_cache 0 \
    --dk_wr none --dk_rd none --token_group 1 --disk_dev_name nvme \
    --batch_split 1 --seed 1234 --flash_att 1 --paged_att 0 --nv_profile 0 \
    --run_info "$PREFIX" \
    --expert_mode demand --moe_expert_store "$STORE" \
    --moe_checkpoint_revision "$REVISION" --moe_expert_reader direct \
    --moe_expert_scratch_slots 64 --moe_demand_weight_limit_gb 2.2 \
    --moe_system_headroom_gb 1.5 --moe_token_chunk_size 8 \
    >"$LOG" 2>&1; then
    echo "Maple profile run failed; see $LOG" >&2
    exit 1
fi

if [[ -n "$JTOP_PID" ]]; then
    kill "$JTOP_PID" 2>/dev/null || true
    wait "$JTOP_PID" 2>/dev/null || true
    JTOP_PID=""
fi
[[ -s "$PREFIX.profiler.err" ]] || rm -f "$PREFIX.profiler.err"

ANALYZE_ARGS=("$LOG" --output-prefix "$PREFIX")
[[ -s "$JTOP_CSV" ]] && ANALYZE_ARGS+=(--jtop "$JTOP_CSV")
[[ -s "$DISKIO_CSV" ]] && ANALYZE_ARGS+=(--diskio "$DISKIO_CSV")
"$PYTHON" "$SCRIPT_DIR/analyze_maple_profile.py" "${ANALYZE_ARGS[@]}" | tee "$PREFIX.summary.txt"
echo "Completed Maple profile: $PREFIX"
