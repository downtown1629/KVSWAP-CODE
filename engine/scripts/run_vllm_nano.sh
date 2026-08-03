#!/bin/bash
set -e
# NVMe/Orin-Nano-friendly variant of scripts/run_vllm.sh (the vLLM
# "no offloading" upper-bound baseline). That original hard-exits unless
# `nvpmodel -q --verbose` reports exactly "MAXN" and CPU/GPU devfreq are
# pinned at AGX-specific sysfs paths — this Orin Nano reports "MAXN_SUPER"
# and its GPU devfreq node lives elsewhere, so the original would abort
# immediately. This file is a new file, not an edit to the original — see
# engine/nano.md for the same pattern applied to setup.sh/eval.sh.
#
# Usage: bash scripts/run_vllm_nano.sh [model] [seqlen_list] [batch_list]
#   model:       dir name under MODEL_PATH_BASE_HF   (default: Qwen3-0.6B)
#   seqlen_list: comma-separated                     (default: 16384 — Table 5)
#   batch_list:  comma-separated                     (default: 1,2,4,8 — Table 5)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./nano_common.sh
source "${SCRIPT_DIR}/nano_common.sh"

if [ "$(basename "$(pwd)")" != "engine" ]; then
    echo "Error: Please run this script from the engine directory."
    exit 1
fi

check_hardware_soft
check_powermode_soft
check_jetson_clocks_soft

export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES="0"

source .venv/bin/activate

TEST_MODEL="${1:-Qwen3-0.6B}"
SEQLEN_LIST="${2:-16384}"
BATCH_LIST="${3:-1,2,4,8}"

# src/run_vllm.py hardcodes gpu_memory_utilization=0.85 and max_model_len=32768,
# both sized for AGX Orin's 64GB. On this 8GB unified-memory Nano:
#   - 0.85 fails outright at startup (observed: 4.15/7.43 GiB free, wants 6.31 GiB)
#   - even at a lower utilization, reserving KV cache for max_model_len=32768
#     needs ~3.5 GiB regardless of what seqlen we actually test, which alone
#     can exceed the whole budget
# Override both via env vars (added to run_vllm.py so the AGX defaults are
# unchanged when these aren't set). Sizing max_model_len to the largest
# seqlen actually being tested cuts the KV-cache reservation roughly in
# proportion — untested starting points, adjust if a run still fails.
MAX_SEQLEN="$(echo "$SEQLEN_LIST" | tr ',' '\n' | sort -n | tail -1)"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$MAX_SEQLEN}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.6}"

if [ -z "$MODEL_PATH_BASE_HF" ]; then
  echo "Error: MODEL_PATH_BASE_HF is not set"
  exit 1
fi
MODEL_PATH="${MODEL_PATH_BASE_HF}/${TEST_MODEL}"
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: $MODEL_PATH not found. Run scripts/download_models_nano.sh first."
    exit 1
fi

if [ -z "${EVAL_USER:-}" ] || [ "${EVAL_USER}" = '$EVAL_USER' ]; then
  echo "Error: EVAL_USER is not correctly set, EVAL_USER=$EVAL_USER"
  exit 1
fi
if [ -z "$EVAL_LOG_DIR" ]; then
  echo "Error: EVAL_LOG_DIR is not set"
  exit 1
fi

OUTPUT_PATH="$EVAL_LOG_DIR/$EVAL_USER/vllm_results"
mkdir -p "$OUTPUT_PATH"
LOG_OUT="$OUTPUT_PATH/$TEST_MODEL.log"

echo "Running vLLM with model: $TEST_MODEL, seqlen_list=$SEQLEN_LIST, batch_list=$BATCH_LIST, max_model_len=$VLLM_MAX_MODEL_LEN, gpu_mem_util=$VLLM_GPU_MEM_UTIL"

echo "Clearing system cache..."
sync
if echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1; then
  echo "System cache cleared."
else
  echo "[warn] No permission to clear system cache (sudo tee failed) — continuing without it."
  echo "  On the AGX-only original this is a hard error; here it's advisory since a"
  echo "  cold cache mainly affects run-to-run noise, not correctness."
fi
sleep 2

echo "Start running..."
{
  python src/run_vllm.py --model_path "$MODEL_PATH" --output_path "$OUTPUT_PATH" \
    --seqlen-list "$SEQLEN_LIST" --batch-list "$BATCH_LIST"
} > "${LOG_OUT}" 2>&1 || true
echo "Running finished."
sleep 3

echo "Done. Log: $LOG_OUT, CSV under $OUTPUT_PATH"
