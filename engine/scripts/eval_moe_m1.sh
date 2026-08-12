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
        MAX_ALLOC_KV_SIZE=67108864 "$PYTHON" src/main.py \
            --model_path "$FIXTURE_DIR" --offload_dir "$OFFLOAD_DIR" \
            --prompt_len 64 --gen_len 2 --gpu_batch_size 1 \
            --num_gpu_batches 1 --percent 100 0 100 0 100 0 \
            --test_input_path ./data/test_inputs --run_args L0 \
            --lr_proj_mode none --use_token_cache 0 --dk_wr none \
            --dk_rd none --token_group 1 --disk_dev_name nvme \
            --batch_split 1 --seed 1234 --flash_att 0 --paged_att 0 \
            --moe_resident_weight_limit_gb 0.01 --nv_profile 0
        echo "offload directory: $OFFLOAD_DIR"
        ;;
    *)
        echo "usage: $0 {static|fixture-cpu|fixture-cuda|fixture-fullkv}" >&2
        exit 2
        ;;
esac
