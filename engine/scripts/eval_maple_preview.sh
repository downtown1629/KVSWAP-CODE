#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ENGINE_DIR"

MODE=${1:-static}
PYTHON=.venv/bin/python
REVISION=ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07
MODEL=${MAPLE_MODEL:-../data/model_weights_hf/maple-preview}
STORE=${MAPLE_EXPERT_STORE:-../data/nvme_offload/maple-preview-experts}

check_revision() {
    if [[ ! -f "$MODEL/.kvswap_revision" ]] || \
       [[ "$(<"$MODEL/.kvswap_revision")" != "$REVISION" ]]; then
        echo "Maple checkpoint is not marked as reviewed revision $REVISION" >&2
        exit 2
    fi
}

case "$MODE" in
    static)
        python3 -m py_compile \
            src/model_adapters.py src/model_config.py src/maple_cache.py \
            src/moe.py src/main.py src/pytorch_backend.py \
            scripts/pack_maple_experts.py scripts/download_maple_preview.py
        PYTHONPATH=src "$PYTHON" -m unittest discover \
            -s tests -p 'test_maple_adapter.py'
        ;;
    download)
        "$PYTHON" scripts/download_maple_preview.py "$MODEL"
        ;;
    pack)
        check_revision
        PYTHONPATH=src "$PYTHON" scripts/pack_maple_experts.py \
            pack "$MODEL" "$STORE" --source-revision "$REVISION"
        ;;
    verify)
        check_revision
        PYTHONPATH=src "$PYTHON" scripts/pack_maple_experts.py \
            verify "$MODEL" "$STORE" --source-revision "$REVISION"
        ;;
    smoke)
        check_revision
        free -h
        OFFLOAD_DIR=$(mktemp -d /tmp/kvswap-maple-offload.XXXXXX)
        RUN_LOG=$(mktemp /tmp/kvswap-maple-smoke.XXXXXX)
        MAX_ALLOC_KV_SIZE=67108864 "$PYTHON" src/main.py \
            --model_path "$MODEL" --offload_dir "$OFFLOAD_DIR" \
            --prompt_len 32 --gen_len 2 --gpu_batch_size 1 --num_gpu_batches 1 \
            --percent 100 0 100 0 100 0 --test_input_path ./data/test_inputs \
            --run_args L0 --lr_proj_mode none --use_token_cache 0 \
            --dk_wr none --dk_rd none --token_group 1 --disk_dev_name nvme \
            --batch_split 1 --seed 1234 --flash_att 1 --paged_att 0 --nv_profile 0 \
            --expert_mode demand --moe_expert_store "$STORE" \
            --moe_checkpoint_revision "$REVISION" --moe_expert_reader direct \
            --moe_expert_scratch_slots 8 --moe_demand_weight_limit_gb 1.8 \
            --moe_system_headroom_gb 1.5 --moe_token_chunk_size 1 \
            >"$RUN_LOG" 2>&1
        cat "$RUN_LOG"
        "$PYTHON" - "$RUN_LOG" <<'PY'
import re
import sys

text = open(sys.argv[1]).read()
if "0: We need" not in text:
    raise SystemExit("unexpected deterministic Maple smoke output")
io = re.search(
    r"Expert demand I/O: calls=(\d+), reads=(\d+), "
    r"logical_bytes=(\d+), stored_bytes=(\d+)", text
)
expected = (792, 6336, 39862665216, 39862665216)
if io is None or tuple(map(int, io.groups())) != expected:
    raise SystemExit(f"unexpected Maple expert accounting: {io.groups() if io else None}")
peak = re.search(r"Peak Memory \(GB\) RSS: ([0-9.]+)", text)
if peak is None or float(peak.group(1)) >= 4.5:
    raise SystemExit(f"missing or unsafe Maple peak RSS: {peak.group(1) if peak else None}")
print(f"Maple smoke verified: output=We need, RSS={peak.group(1)} GiB")
PY
        ;;
    long-smoke)
        check_revision
        free -h
        OFFLOAD_DIR=$(mktemp -d /tmp/kvswap-maple-long-offload.XXXXXX)
        RUN_LOG=$(mktemp /tmp/kvswap-maple-long-smoke.XXXXXX)
        MAX_ALLOC_KV_SIZE=67108864 "$PYTHON" src/main.py \
            --model_path "$MODEL" --offload_dir "$OFFLOAD_DIR" \
            --prompt_len 520 --gen_len 2 --gpu_batch_size 1 --num_gpu_batches 1 \
            --percent 100 0 100 0 100 0 --test_input_path ./data/test_inputs \
            --run_args L0 --lr_proj_mode none --use_token_cache 0 \
            --dk_wr none --dk_rd none --token_group 1 --disk_dev_name nvme \
            --batch_split 1 --seed 1234 --flash_att 1 --paged_att 0 --nv_profile 0 \
            --expert_mode demand --moe_expert_store "$STORE" \
            --moe_checkpoint_revision "$REVISION" --moe_expert_reader direct \
            --moe_expert_scratch_slots 64 --moe_demand_weight_limit_gb 2.2 \
            --moe_system_headroom_gb 1.5 --moe_token_chunk_size 8 \
            >"$RUN_LOG" 2>&1
        cat "$RUN_LOG"
        "$PYTHON" - "$RUN_LOG" <<'PY'
import re
import sys

text = open(sys.argv[1]).read()
policy = (
    "Maple KV policy: sliding_layers=18, window=512, "
    "local_capacity=511, global_layers=6, global_placement=configured"
)
if policy not in text:
    raise SystemExit("Maple long-context cache policy was not activated")
if re.search(r"^0: \S", text, re.MULTILINE) is None:
    raise SystemExit("Maple long-context run did not produce a token")
peak = re.search(r"Peak Memory \(GB\) RSS: ([0-9.]+)", text)
if peak is None or float(peak.group(1)) >= 5.5:
    raise SystemExit(f"missing or unsafe Maple peak RSS: {peak.group(1) if peak else None}")
print(f"Maple long-context smoke verified: prompt=520, RSS={peak.group(1)} GiB")
PY
        ;;
    *)
        echo "usage: $0 {static|download|pack|verify|smoke|long-smoke}" >&2
        exit 2
        ;;
esac
