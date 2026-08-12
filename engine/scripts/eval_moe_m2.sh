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
        verify "$FIXTURE_DIR" "$STORE_DIR" --source-revision m2-fixture
}

run_fixture() {
    local reader=$1
    local kvswap=$2
    local run_log=$3
    local expert_mode=${4:-demand}
    local run_info=${5:-none}
    local expert_trace=${6:-none}
    local offload_dir
    offload_dir=$(mktemp -d /tmp/kvswap-qwen3-moe-m2-offload.XXXXXX)
    if [[ "$(stat -c %d "$offload_dir")" != "$(stat -c %d "$STORE_DIR")" ]]; then
        echo "expert store and KV offload must be on the same filesystem device" >&2
        exit 2
    fi
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
            --run_info "$run_info"
        )
    fi
    local expert_args=(
        --expert_mode demand --moe_expert_store "$STORE_DIR"
        --moe_checkpoint_revision m2-fixture
        --moe_expert_reader "$reader" --moe_expert_scratch_slots 8
        --moe_expert_verify_reads 1 --moe_demand_weight_limit_gb 0.01
        --moe_expert_trace "$expert_trace"
    )
    if [[ "$expert_mode" == resident ]]; then
        expert_args=(
            --expert_mode resident --moe_resident_weight_limit_gb 0.01
            --moe_expert_trace "$expert_trace"
        )
    fi
    MAX_ALLOC_KV_SIZE=67108864 "$PYTHON" src/main.py \
        --model_path "$FIXTURE_DIR" --offload_dir "$offload_dir" \
        --prompt_len 64 --gen_len 2 --gpu_batch_size 1 --num_gpu_batches 1 \
        --test_input_path ./data/test_inputs --use_token_cache 0 --dk_wr none \
        --disk_dev_name nvme --batch_split 1 --seed 1234 --nv_profile 0 \
        --moe_system_headroom_gb 1 --moe_token_chunk_size 16 \
        "${expert_args[@]}" "${kv_args[@]}" >"$run_log" 2>&1
    cat "$run_log"
    grep -Fqx "0: token_99 token_8" "$run_log"
    if [[ "$expert_mode" == resident ]]; then
        return
    fi
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
if calls != 10 or reads != 20:
    raise SystemExit(f"expected 10 calls/20 reads, got {calls}/{reads}")
if logical_bytes != 983040:
    raise SystemExit(f"expected 983040 expert bytes, got {logical_bytes}")
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
        RESIDENT_LOG=$(mktemp /tmp/kvswap-qwen3-moe-m2-resident-kvswap.XXXXXX)
        DEMAND_LOG=$(mktemp /tmp/kvswap-qwen3-moe-m2-demand-kvswap.XXXXXX)
        TRACE_DIR=$(mktemp -d /tmp/kvswap-qwen3-moe-m2-trace.XXXXXX)
        run_fixture direct 1 "$RESIDENT_LOG" resident \
            "$TRACE_DIR/resident" "$TRACE_DIR/resident-experts.jsonl"
        run_fixture direct 1 "$DEMAND_LOG" demand \
            "$TRACE_DIR/demand" "$TRACE_DIR/demand-experts.jsonl"
        "$PYTHON" - "$TRACE_DIR" "$DEMAND_LOG" <<'PY'
import json
import re
import sys
import torch
from pathlib import Path

root = Path(sys.argv[1])
resident = [json.loads(line) for line in (root / "resident-experts.jsonl").open()]
demand = [json.loads(line) for line in (root / "demand-experts.jsonl").open()]
if resident != demand:
    raise SystemExit("resident/demand routing traces differ")
for name in ("resident", "demand"):
    trace = torch.load(root / f"{name}_swap_info.pt", map_location="cpu", weights_only=False)
    selected = sum(int(item[1].sum()) for item in trace)
    if selected != 128:
        raise SystemExit(f"{name} KV selected tokens: expected 128, got {selected}")
text = Path(sys.argv[2]).read_text()
match = re.search(r"logical_bytes=(\d+)", text)
expert_bytes = int(match.group(1))
kv_bytes = 128 * 2 * 1 * 64 * 2
print(f"Joint logical I/O verified: expert={expert_bytes}, KV={kv_bytes}, total={expert_bytes + kv_bytes}")
PY
        ;;
    real-qwen3-30b)
        : "${M2_MODEL_PATH:?set M2_MODEL_PATH to the Qwen3-MoE checkpoint}"
        : "${M2_EXPERT_STORE:?set M2_EXPERT_STORE to the packed expert store}"
        : "${M2_CHECKPOINT_REVISION:?set M2_CHECKPOINT_REVISION to its immutable revision}"
        free -h
        RUN_LOG=$(mktemp /tmp/kvswap-qwen3-moe-m2-real.XXXXXX)
        OFFLOAD_DIR=$(mktemp -d /tmp/kvswap-qwen3-moe-m2-real-offload.XXXXXX)
        MAX_ALLOC_KV_SIZE=67108864 "$PYTHON" src/main.py \
            --model_path "$M2_MODEL_PATH" --offload_dir "$OFFLOAD_DIR" \
            --prompt_len 64 --gen_len 2 --gpu_batch_size 1 --num_gpu_batches 1 \
            --percent 100 0 100 0 100 0 --test_input_path ./data/test_inputs \
            --run_args L0 --lr_proj_mode none --use_token_cache 0 \
            --dk_wr none --dk_rd none --token_group 1 --disk_dev_name nvme \
            --batch_split 1 --seed 1234 --flash_att 1 --paged_att 0 --nv_profile 0 \
            --expert_mode demand --moe_expert_store "$M2_EXPERT_STORE" \
            --moe_checkpoint_revision "$M2_CHECKPOINT_REVISION" \
            --moe_expert_reader direct --moe_expert_scratch_slots 8 \
        --moe_demand_weight_limit_gb "${M2_DEMAND_WEIGHT_LIMIT_GB:?set an explicit approval limit}" \
        --moe_system_headroom_gb "${M2_SYSTEM_HEADROOM_GB:-1.5}" \
            --moe_token_chunk_size 1 >"$RUN_LOG" 2>&1
        cat "$RUN_LOG"
        grep -Fqx "0: The team" "$RUN_LOG"
        grep -q "Peak Memory (GB)" "$RUN_LOG"
        "$PYTHON" - "$RUN_LOG" <<'PY'
import re
import sys

text = open(sys.argv[1]).read()
io = re.search(
    r"Expert demand I/O: calls=(\d+), reads=(\d+), "
    r"logical_bytes=(\d+), stored_bytes=(\d+)", text
)
if io is None or tuple(map(int, io.groups())) != (
    3120, 24960, 235552112640, 235552112640
):
    raise SystemExit(f"unexpected real-demand accounting: {io.groups() if io else None}")
peak = re.search(r"Peak Memory \(GB\) RSS: ([0-9.]+)", text)
if peak is None or float(peak.group(1)) >= 5.5:
    raise SystemExit(f"real-demand peak RSS is missing or unsafe: {peak.group(1) if peak else None}")
print(f"Real demand gate verified: output=The team, RSS={peak.group(1)} GiB")
PY
        ;;
    *)
        echo "usage: $0 {static|fixture-buffered|fixture-direct|fixture-fullkv|fixture-kvswap|real-qwen3-30b}" >&2
        exit 2
        ;;
esac
