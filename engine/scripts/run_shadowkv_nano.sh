#!/bin/bash
set -e
# NVMe/Orin-Nano-friendly variant of src/shadowkv/run_shadowkv.sh (the
# ShadowKV baseline). That original hard-exits unless `nvpmodel -q --verbose`
# reports exactly "MAXN" and CPU/GPU devfreq are pinned at AGX-specific sysfs
# paths — this Orin Nano reports "MAXN_SUPER" and its GPU devfreq node lives
# elsewhere, so the original would abort immediately. This file is a new
# file, not an edit to the original — see engine/nano.md for the same
# pattern applied to setup.sh/eval.sh, and scripts/run_vllm_nano.sh for the
# vLLM baseline's equivalent.
#
# Requires the shadowkv CUDA extension to already be built:
#   cd src/shadowkv && MAX_JOBS=1 python setup.py build_ext --inplace
# (MAX_JOBS=1 matters on 8GB unified memory — default parallel ninja jobs
# can spawn several multi-GB cicc processes at once and exhaust RAM.)
#
# Usage: bash scripts/run_shadowkv_nano.sh [model] [total_len] ["batch_list"] [budget] [chunk_size] [rank]
#   model:       dir name under MODEL_PATH_BASE_HF   (default: Qwen3-0.6B)
#   total_len:   context length incl. gen_len=100    (default: 16384 — Table 5)
#   batch_list:  quoted, space-separated              (default: "1 2 4 8" — Table 5)
#   budget:      ShadowKV's KV-selection budget       (default: 400 — matches MAX_NUM_KV used
#                elsewhere on this device in eval_nano.sh; the paper's own AGX sweep scripts
#                (tab-4.sh/fig-10.sh) don't cover Orin Nano/Qwen3-0.6B, so there's no
#                paper-verified Nano-specific value to copy — treat as a starting point)
#   chunk_size:  ShadowKV landmark chunk size          (default: 16 — matches tab-4.sh/fig-10.sh)
#   rank:        ShadowKV low-rank correction rank     (default: 40 — matches tab-4.sh/fig-10.sh)

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
# Required by src/diskio/disk_interface.py and src/shadowkv/models/disk_cache.py;
# see MAX_ALLOC_KV_SIZE default in eval_nano.sh.
export MAX_ALLOC_KV_SIZE="${MAX_ALLOC_KV_SIZE:-$((1024*1024*1024))}"

source .venv/bin/activate

if ! (cd src/shadowkv && python -c "import torch; from kernels import shadowkv") 2>/dev/null; then
  (cd src/shadowkv && python -c "import torch; from kernels import shadowkv") 2>&1 | tail -5
  echo "Error: shadowkv CUDA extension not importable. Build it first:"
  echo "  cd src/shadowkv && MAX_JOBS=1 python setup.py build_ext --inplace"
  exit 1
fi

TEST_MODEL="${1:-Qwen3-0.6B}"
TOTAL_LEN="${2:-16384}"
BATCH_LIST="${3:-1 2 4 8}"
BUDGET="${4:-400}"
CHUNK_SIZE="${5:-16}"
RANK="${6:-40}"
SEED="${SEED:-1234}"
DISK_TYPE="nvme"

for v in NVME_DEV_NAME NVME_OFFLOAD_DIR MODEL_PATH_BASE_HF EVAL_LOG_DIR EVAL_USER; do
    if [ -z "${!v}" ]; then
        echo "Error: $v is not set"
        exit 1
    fi
done

MODEL_PATH="${MODEL_PATH_BASE_HF}/${TEST_MODEL}"
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: $MODEL_PATH not found. Run scripts/download_models_nano.sh first."
    exit 1
fi

OFFLOAD_DIR_="$NVME_OFFLOAD_DIR"
mkdir -p "$OFFLOAD_DIR_"

log_dir="${EVAL_LOG_DIR}/${EVAL_USER}/logs/shadowkv"
mkdir -p "$log_dir"

genlen=100
model_name="$(basename "$MODEL_PATH")"

echo "TEST_MODEL=$TEST_MODEL TOTAL_LEN=$TOTAL_LEN BATCH_LIST=[$BATCH_LIST] BUDGET=$BUDGET CHUNK_SIZE=$CHUNK_SIZE RANK=$RANK"

for bsz in $BATCH_LIST; do
    echo "Evaluating bsz=$bsz"
    min_prompt_len=$((TOTAL_LEN - genlen))
    log_file="${log_dir}/${DISK_TYPE}/${model_name}/budget${BUDGET}/seed${SEED}/${min_prompt_len}_bsz${bsz}_gen${genlen}_chunk${CHUNK_SIZE}_r${RANK}.log"
    mkdir -p "$(dirname "$log_file")"

    if [ -f "$log_file" ] && grep -q "Throughput:" "$log_file" 2>/dev/null; then
        echo "Skipping bsz=$bsz (already completed — see $log_file)"
        continue
    fi

    run_log="${log_file}.run"
    echo "Start running..."
    {
      python src/shadowkv/test/e2e_jetson.py --model_path "$MODEL_PATH" --min_prompt_len "$min_prompt_len" \
        --bsz "$bsz" --budget "$BUDGET" --genlen "$genlen" --input_path ./data/test_inputs \
        --chunk_size "$CHUNK_SIZE" --rank "$RANK" --cache_dir "$OFFLOAD_DIR_" \
        --offload_device disk --log_file "$log_file" --seed "$SEED"
    } > "${run_log}" 2>&1 || echo "  Run failed — see $run_log"
    grep "Throughput:" "$log_file" 2>/dev/null || echo "  (no throughput line — check $run_log for errors, e.g. OOM at this batch size)"
    sleep 3
done

echo "--------------------------------"
echo "Done. Per-run logs under $log_dir"
