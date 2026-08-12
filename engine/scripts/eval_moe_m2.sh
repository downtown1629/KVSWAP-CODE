#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ENGINE_DIR"

MODE=${1:-static}
PYTHON=.venv/bin/python

make_artifacts() {
    FIXTURE_DIR=$(mktemp -d /tmp/kvswap-qwen3-moe-m2-fixture.XXXXXX)
    STORE_DIR=$(mktemp -d /tmp/kvswap-qwen3-moe-m2-store.XXXXXX)
    "$PYTHON" scripts/make_tiny_qwen3_moe.py "$FIXTURE_DIR"
    PYTHONPATH=src "$PYTHON" scripts/pack_qwen3_moe_experts.py \
        pack "$FIXTURE_DIR" "$STORE_DIR" --source-revision m2-fixture
    PYTHONPATH=src "$PYTHON" scripts/pack_qwen3_moe_experts.py \
        verify "$FIXTURE_DIR" "$STORE_DIR"
}

run_fixture() {
    local reader=$1
    local kvswap=$2
    local run_log=$3
    local offload_dir
    offload_dir=$(mktemp -d /tmp/kvswap-qwen3-moe-m2-offload.XXXXXX)
    local kv_args=(
        --percent 100 0 100 0 100 0
        --run_args L0 --lr_proj_mode none --dk_rd none
        --max_num_kv 0 --token_group 1 --flash_att 0 --paged_att 0
    )
    if [[ "$kvswap" == 1 ]]; then
        ADAPTER_DIR=$(mktemp -d /tmp/kvswap-qwen3-moe-m2-adapter.XXXXXX)
        "$PYTHON" scripts/make_tiny_qwen3_kvswap_adapter.py \
            "$FIXTURE_DIR" "$ADAPTER_DIR"
        kv_args=(
            --percent 100 0 0 0 100 0
            --run_args L4 --lr_proj_mode lr_proj_mh --lr_proj_path "$ADAPTER_DIR"
            --dk_rd clear --max_num_kv 64 --token_group 2
            --start_layer 0-curr-emb --reuse_budget 0 --flash_att 0 --paged_att 1
        )
    fi
    MAX_ALLOC_KV_SIZE=67108864 "$PYTHON" src/main.py \
        --model_path "$FIXTURE_DIR" --offload_dir "$offload_dir" \
        --prompt_len 64 --gen_len 2 --gpu_batch_size 1 --num_gpu_batches 1 \
        --test_input_path ./data/test_inputs --use_token_cache 0 --dk_wr none \
        --disk_dev_name nvme --batch_split 1 --seed 1234 --nv_profile 0 \
        --expert_mode demand --moe_expert_store "$STORE_DIR" \
        --moe_expert_reader "$reader" --moe_expert_scratch_slots 8 \
        --moe_expert_verify_reads 1 \
        --moe_demand_weight_limit_gb 0.01 --moe_system_headroom_gb 1 \
        --moe_token_chunk_size 16 "${kv_args[@]}" >"$run_log" 2>&1
    cat "$run_log"
    grep -Fqx "0: token_99 token_8" "$run_log"
    "$PYTHON" - "$run_log" <<'PY'
import re
import sys

text = open(sys.argv[1]).read()
match = re.search(
    r"Expert demand I/O: calls=(\d+), reads=(\d+), "
    r"logical_bytes=(\d+), stored_bytes=(\d+), "
    r"read_ms=([0-9.]+), copy_ms=([0-9.]+)", text
)
if not match:
    raise SystemExit("missing expert demand accounting")
calls, reads, logical_bytes, stored_bytes = map(int, match.groups()[:4])
read_ms, copy_ms = map(float, match.groups()[4:])
expert_bytes = 3 * 128 * 64 * 2
stored_per_expert = ((expert_bytes + 4095) // 4096) * 4096
if calls <= 0 or reads <= 0:
    raise SystemExit("expert demand path performed no reads")
if logical_bytes != reads * expert_bytes:
    raise SystemExit(f"logical expert bytes disagree: {logical_bytes}, reads={reads}")
if stored_bytes != reads * stored_per_expert:
    raise SystemExit(f"stored expert bytes disagree: {stored_bytes}, reads={reads}")
if read_ms <= 0 or copy_ms <= 0:
    raise SystemExit(f"invalid demand timings: read={read_ms}, copy={copy_ms}")
print(f"Demand accounting verified: calls={calls}, reads={reads}, bytes={logical_bytes}")
PY
}

case "$MODE" in
    static)
        python3 -m py_compile src/expert_store.py scripts/pack_qwen3_moe_experts.py
        PYTHONPATH=src "$PYTHON" -m unittest discover -s tests -p 'test_moe_*.py'
        PYTHONPATH=src "$PYTHON" -m unittest discover -s tests -p 'test_expert_store.py'
        ;;
    fixture-buffered|fixture-direct|fixture-fullkv)
        free -h
        make_artifacts
        RUN_LOG=$(mktemp /tmp/kvswap-qwen3-moe-m2-run.XXXXXX)
        reader=buffered
        [[ "$MODE" == fixture-direct ]] && reader=direct
        run_fixture "$reader" 0 "$RUN_LOG"
        ;;
    fixture-kvswap)
        free -h
        make_artifacts
        RUN_LOG=$(mktemp /tmp/kvswap-qwen3-moe-m2-kvswap.XXXXXX)
        run_fixture direct 1 "$RUN_LOG"
        ;;
    *)
        echo "usage: $0 {static|fixture-buffered|fixture-direct|fixture-fullkv|fixture-kvswap}" >&2
        exit 2
        ;;
esac
