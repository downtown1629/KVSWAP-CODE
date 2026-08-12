#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ENGINE_DIR"

MODE=${1:-static}
PYTHON=.venv/bin/python

make_fixture() {
    FIXTURE_DIR=$(mktemp -d /tmp/kvswap-qwen3-moe.XXXXXX)
    "$PYTHON" scripts/make_tiny_qwen3_moe.py "$FIXTURE_DIR"
    "$PYTHON" scripts/preflight_qwen3_moe.py "$FIXTURE_DIR"
    echo "fixture: $FIXTURE_DIR"
}

make_kvswap_adapter() {
    ADAPTER_DIR=$(mktemp -d /tmp/kvswap-qwen3-moe-adapter.XXXXXX)
    "$PYTHON" scripts/make_tiny_qwen3_kvswap_adapter.py \
        "$FIXTURE_DIR" "$ADAPTER_DIR"
    echo "identity adapter: $ADAPTER_DIR"
}

case "$MODE" in
    static)
        PYTHONPATH=src "$PYTHON" -m unittest discover -s tests -v
        ;;
    fixture-cpu)
        make_fixture
        "$PYTHON" scripts/verify_qwen3_moe_fixture.py "$FIXTURE_DIR" --device cpu
        ;;
    fixture-cuda)
        free -h
        make_fixture
        "$PYTHON" scripts/verify_qwen3_moe_fixture.py "$FIXTURE_DIR" --device cuda
        ;;
    fixture-fullkv)
        free -h
        make_fixture
        OFFLOAD_DIR=$(mktemp -d /tmp/kvswap-qwen3-moe-offload.XXXXXX)
        run_fixture_engine() {
            local run_log=$1
            MAX_ALLOC_KV_SIZE=67108864 "$PYTHON" src/main.py \
                --model_path "$FIXTURE_DIR" --offload_dir "$OFFLOAD_DIR" \
                --prompt_len 64 --gen_len 2 --gpu_batch_size 1 \
                --num_gpu_batches 1 --percent 100 0 100 0 100 0 \
                --test_input_path ./data/test_inputs --run_args L0 \
                --lr_proj_mode none --use_token_cache 0 --dk_wr none \
                --dk_rd none --token_group 1 --disk_dev_name nvme \
                --batch_split 1 --seed 1234 --flash_att 0 --paged_att 0 \
                --moe_resident_weight_limit_gb 0.01 --nv_profile 0 \
                >"$run_log" 2>&1
            cat "$run_log"
            grep -Fqx "0: token_99 token_8" "$run_log"
        }
        RUN_LOG_1=$(mktemp /tmp/kvswap-qwen3-moe-run1.XXXXXX)
        RUN_LOG_2=$(mktemp /tmp/kvswap-qwen3-moe-run2.XXXXXX)
        run_fixture_engine "$RUN_LOG_1"
        run_fixture_engine "$RUN_LOG_2"
        OUTPUT_1=$(grep -Fx "0: token_99 token_8" "$RUN_LOG_1" | head -1)
        OUTPUT_2=$(grep -Fx "0: token_99 token_8" "$RUN_LOG_2" | head -1)
        test "$OUTPUT_1" = "$OUTPUT_2"
        echo "HF fixture parity and repeat determinism: $OUTPUT_1"
        echo "offload directory: $OFFLOAD_DIR"
        ;;
    fixture-kvswap)
        free -h
        make_fixture
        make_kvswap_adapter
        OFFLOAD_DIR=$(mktemp -d /tmp/kvswap-qwen3-moe-kv-offload.XXXXXX)
        RUN_LOG=$(mktemp /tmp/kvswap-qwen3-moe-kvswap.XXXXXX)
        TRACE_DIR=$(mktemp -d /tmp/kvswap-qwen3-moe-kvswap-trace.XXXXXX)
        RUN_INFO="$TRACE_DIR/run"
        MAX_ALLOC_KV_SIZE=67108864 "$PYTHON" src/main.py \
            --model_path "$FIXTURE_DIR" --offload_dir "$OFFLOAD_DIR" \
            --prompt_len 64 --gen_len 2 --gpu_batch_size 1 \
            --num_gpu_batches 1 --percent 100 0 0 0 100 0 \
            --test_input_path ./data/test_inputs --run_args L4 \
            --lr_proj_mode lr_proj_mh --lr_proj_path "$ADAPTER_DIR" \
            --max_num_kv 64 --start_layer 0-curr-emb --reuse_budget 0 \
            --use_token_cache 0 --dk_wr none --dk_rd clear \
            --token_group 2 --disk_dev_name nvme --batch_split 1 \
            --seed 1234 --flash_att 0 --paged_att 1 \
            --moe_resident_weight_limit_gb 0.01 --nv_profile 0 \
            --run_info "$RUN_INFO" >"$RUN_LOG" 2>&1
        cat "$RUN_LOG"
        grep -Fqx "0: token_99 token_8" "$RUN_LOG"
        "$PYTHON" - "$RUN_INFO" <<'PY'
import sys
import torch

trace = torch.load(sys.argv[1] + "_swap_info.pt", map_location="cpu", weights_only=False)
selected_tokens = sum(int(item[1].sum()) for item in trace)
expected_tokens = 2 * 64  # two attention layers, all 64 prompt tokens selected
if selected_tokens != expected_tokens:
    raise SystemExit(f"selected KV tokens: expected {expected_tokens}, got {selected_tokens}")
expected_bytes = expected_tokens * 2 * 1 * 64 * 2  # K/V * KV heads * head dim * BF16
print(f"KVSwap regression accounting: {selected_tokens} selected tokens, {expected_bytes} bytes")
PY
        echo "MoE/KVSwap coexistence: $OFFLOAD_DIR"
        ;;
    dense-kvswap)
        if [[ ! -f .env_nano ]]; then
            echo "dense-kvswap requires engine/.env_nano" >&2
            exit 2
        fi
        # shellcheck disable=SC1091
        source .env_nano
        : "${MODEL_PATH_BASE:?MODEL_PATH_BASE is required}"
        : "${NVME_OFFLOAD_DIR:?NVME_OFFLOAD_DIR is required}"
        : "${NVME_DEV_NAME:?NVME_DEV_NAME is required}"
        MODEL_DIR="$MODEL_PATH_BASE/Qwen3-0.6B"
        ADAPTER_DIR="$MODEL_PATH_BASE/local_adapters/lowrank_proj_post_rope_c4_20/Qwen3-0.6B_mh_1.0"
        [[ -d "$MODEL_DIR" && -d "$ADAPTER_DIR" ]] || {
            echo "missing Qwen3-0.6B model or KVSwap adapter" >&2
            exit 2
        }
        free -h
        RUN_LOG=$(mktemp /tmp/kvswap-qwen3-dense.XXXXXX)
        TRACE_DIR=$(mktemp -d /tmp/kvswap-qwen3-dense-trace.XXXXXX)
        RUN_INFO="$TRACE_DIR/run"
        MAX_ALLOC_KV_SIZE=67108864 "$PYTHON" src/main.py \
            --percent 100 0 0 0 100 0 --model_path "$MODEL_DIR" \
            --offload_dir "$NVME_OFFLOAD_DIR" --prompt_len 64 --gen_len 2 \
            --gpu_batch_size 1 --num_gpu_batches 1 \
            --test_input_path ./data/test_inputs --run_args L4 \
            --lr_proj_mode lr_proj_mh --lr_proj_path "$ADAPTER_DIR" \
            --max_num_kv 64 --start_layer 0-curr-emb --reuse_budget 0 \
            --use_token_cache 0 --dk_wr none --dk_rd clear --token_group 4 \
            --disk_dev_name "$NVME_DEV_NAME" --batch_split 1 --seed 1234 \
            --flash_att 1 --paged_att 1 --nv_profile 0 --run_info "$RUN_INFO" \
            >"$RUN_LOG" 2>&1
        cat "$RUN_LOG"
        grep -Fqx "0: The team" "$RUN_LOG"
        "$PYTHON" - "$RUN_INFO" <<'PY'
import sys
import torch

trace = torch.load(sys.argv[1] + "_swap_info.pt", map_location="cpu", weights_only=False)
selected_tokens = sum(int(item[1].sum()) for item in trace)
expected_tokens = 28 * 64
if selected_tokens != expected_tokens:
    raise SystemExit(f"selected KV tokens: expected {expected_tokens}, got {selected_tokens}")
print(f"Dense Qwen3 KVSwap regression accounting: {selected_tokens} selected tokens")
PY
        ;;
    *)
        echo "usage: $0 {static|fixture-cpu|fixture-cuda|fixture-fullkv|fixture-kvswap|dense-kvswap}" >&2
        exit 2
        ;;
esac
