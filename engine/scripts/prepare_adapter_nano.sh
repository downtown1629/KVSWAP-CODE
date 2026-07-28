#!/usr/bin/env bash
set -e
# Generates the KVSwap low-rank K-cache adapter (and, optionally, the
# InfiniGen* skew adapter) for a model with no precomputed adapter shipped in
# this repo — e.g. Qwen3-0.6B. engine/data/adapters/ only ships adapters for
# the models used in the paper's main AGX Orin experiments (Llama-3.1-8B,
# Llama-3.2-3B, Qwen3-14B). Without an adapter, eval_nano.sh's `flexgen` mode
# (full KV cache, no prediction) still runs fine, but `kvswap`/`infinigen` do
# not.
#
# Reuses quality/src/prepare_adapter.py — the paper's own offline
# adapter-tuning code (PAPER.pdf §3.5, "Offline Parameter Tuning") — but runs
# it through *this* venv (engine/.venv), which already has a Jetson-aarch64
# torch/flash-attn/transformers stack installed. quality/scripts/install.sh's
# venv would not work here: it pulls x86_64 PyPI torch wheels, not Jetson
# builds. prepare_adapter.py only needs plain `transformers` + `torch` for
# --mode kvswap (scikit-learn is only imported for the unused --mode loki
# path), so this works without installing anything from quality/requirements.txt.
#
# Usage: bash scripts/prepare_adapter_nano.sh [model_name] [--with-infinigen]
#   model_name       Directory name under MODEL_PATH_BASE_HF (default: Qwen3-0.6B)
#   --with-infinigen Also generate the InfiniGen* skew adapter (same as WITH_INFINIGEN=1)
#
# Must run from engine/, after scripts/setup_nano.sh has created .venv and
# scripts/download_models_nano.sh has fetched the HF checkpoint.

if [ "$(basename "$(pwd)")" != "engine" ]; then
    echo "Error: Please run this script from the engine directory."
    exit 1
fi
if [ -z "$MODEL_PATH_BASE_HF" ]; then echo "Error: MODEL_PATH_BASE_HF is not set"; exit 1; fi
if [ -z "$MODEL_PATH_BASE" ]; then echo "Error: MODEL_PATH_BASE is not set"; exit 1; fi
if [ ! -d .venv ]; then echo "Error: .venv not found. Run scripts/setup_nano.sh first."; exit 1; fi

MODEL="Qwen3-0.6B"
for arg in "$@"; do
    case "$arg" in
        --with-infinigen) WITH_INFINIGEN=1 ;;
        *) MODEL="$arg" ;;
    esac
done

source .venv/bin/activate

# ratio=1    -> "relaxed"-budget adapter (KVSWAP_RATIO in the paper's own scripts)
# ratio=0.25 -> "tight"-budget / "-t" adapter (KVSWAP_t_RATIO)
RATIOS="${RATIOS:-1,0.25}"
SEQ_LEN="${SEQ_LEN:-8192}"     # calibration chunk length; doesn't affect output naming, only runtime/memory during adapter generation
EVAL_SAMPLES="${EVAL_SAMPLES:-20}"
# Deliberately written outside engine/data/adapters/: that directory is
# Git-LFS-tracked repo content, and scripts/link_adapters.sh symlinks it
# wholesale into MODEL_PATH_BASE. Keep generated artifacts out of the
# tracked tree so `git status` stays clean.
SAVE_DIR="${SAVE_DIR:-${MODEL_PATH_BASE}/local_adapters}"

MODEL_HF_PATH="${MODEL_PATH_BASE_HF}/${MODEL}"
if [ ! -d "$MODEL_HF_PATH" ]; then
    echo "Error: $MODEL_HF_PATH not found. Run scripts/download_models_nano.sh first."
    exit 1
fi

mkdir -p "$SAVE_DIR"

# prepare_adapter.py parses --ratios with float(), so a ratio given as "1"
# becomes Python float 1.0 and str(1.0) == "1.0" — the output directory would
# be "..._mh_1.0", not "..._mh_1" like the adapters already shipped in
# engine/data/adapters/ (e.g. Llama-3.1-8B-Instruct_mh_1). eval_nano.sh builds
# an exact path (main.py loads it with plain string concatenation, no
# globbing), so a mismatched suffix here would fail silently later. Symlink
# away the ".0" so both spellings resolve.
normalize_ratio_dirs() {
    local base_dir="$1"
    [ -d "$base_dir" ] || return 0
    local d newname
    for d in "$base_dir"/*.0; do
        [ -e "$d" ] || continue
        newname="${d%.0}"
        if [ ! -e "$newname" ]; then
            ln -s "$(basename "$d")" "$newname"
            echo "  normalized: $(basename "$newname") -> $(basename "$d")"
        fi
    done
}

echo "Generating KVSwap (lr_proj_mh) adapter for ${MODEL}, ratios=${RATIOS}, task=c4, eval_samples=${EVAL_SAMPLES} ..."
python ../quality/src/prepare_adapter.py \
    --model_path "$MODEL_HF_PATH" \
    --seq_len "$SEQ_LEN" \
    --ratios "$RATIOS" \
    --mode kvswap \
    --save_dir "$SAVE_DIR" \
    --eval_samples "$EVAL_SAMPLES" \
    --task c4
normalize_ratio_dirs "${SAVE_DIR}/lowrank_proj_post_rope_c4_20"

if [ "${WITH_INFINIGEN:-0}" == "1" ]; then
    echo "Generating InfiniGen* skew adapter for ${MODEL}, ratios=${RATIOS} ..."
    python ../quality/src/prepare_adapter.py \
        --model_path "$MODEL_HF_PATH" \
        --seq_len "$SEQ_LEN" \
        --ratios "$RATIOS" \
        --mode infinigen \
        --save_dir "$SAVE_DIR" \
        --eval_samples "$EVAL_SAMPLES" \
        --task c4
    normalize_ratio_dirs "${SAVE_DIR}/infinigen_skew/skewing_idx_c4_20"
fi

echo "--------------------------------"
echo "Done. Adapters written under: $SAVE_DIR"
echo "  KVSwap:    $SAVE_DIR/lowrank_proj_post_rope_c4_20/${MODEL}_mh_<ratio>/"
if [ "${WITH_INFINIGEN:-0}" == "1" ]; then
    echo "  InfiniGen: $SAVE_DIR/infinigen_skew/skewing_matrix_c4_20/${MODEL}.pt"
    echo "             $SAVE_DIR/infinigen_skew/skewing_idx_c4_20/${MODEL}_<ratio>/"
fi
echo "eval_nano.sh reads adapters from \$SAVE_DIR (default: \${MODEL_PATH_BASE}/local_adapters) — keep it set consistently, or pass the same SAVE_DIR to both scripts."
