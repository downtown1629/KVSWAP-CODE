#!/bin/bash
set -e

echo "--------------------------------"
echo "Nano-compatible setup: Jetson Orin Nano (8 GB RAM), NVMe-only offload."
echo "Adapted from scripts/setup.sh, which requires an AGX Orin (device-tree"
echo "model must contain 'AGX Orin') plus both an eMMC and an NVMe device,"
echo "and will hard-exit on this hardware."
echo "--------------------------------"

if [ "$(basename "$(pwd)")" != "engine" ]; then
    echo "Error: Please run this script from the engine directory."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./nano_common.sh
source "${SCRIPT_DIR}/nano_common.sh"

check_hardware_soft

if ! command -v uv &> /dev/null
then
    echo "Error: uv is not installed. Please install uv before running this script."
    exit 1
fi

##########################################################
echo "Installing dependencies..."
# Same wheels as scripts/setup.sh: they target JetPack 6.2 / CUDA 12.6 /
# cp310 / linux_aarch64, which is shared across the whole Orin family (AGX
# Orin, Orin NX, Orin Nano) as long as your JetPack/L4T version matches —
# nothing about them is AGX-specific. If a wheel refuses to import, your
# JetPack version has drifted from what it was built for; get matching
# builds from https://pypi.jetson-ai-lab.io/ (and wheel_pkgs/readme.txt for
# the custom vLLM wheel).

if [ ! -d .venv ]; then
    uv venv --python 3.10
    source .venv/bin/activate
    uv pip install pip setuptools
    uv pip install -r requirements.txt

    for whl in \
        torch-2.7.0-cp310-cp310-linux_aarch64.whl \
        flash_attn-2.7.4.post1-cp310-cp310-linux_aarch64.whl \
        triton-3.2.0-cp310-cp310-linux_aarch64.whl \
        vllm-0.10.1.dev271+g60523a731.cu126-cp310-cp310-linux_aarch64.whl; do
        if [ ! -f "./wheel_pkgs/${whl}" ]; then
            echo "Error: ${whl} not found"
            echo "Please refer to wheel_pkgs/readme.txt"
            exit 1
        fi
    done

    uv pip install ./wheel_pkgs/torch-2.7.0-cp310-cp310-linux_aarch64.whl
    uv pip install ./wheel_pkgs/flash_attn-2.7.4.post1-cp310-cp310-linux_aarch64.whl --no-build-isolation
    uv pip install ./wheel_pkgs/triton-3.2.0-cp310-cp310-linux_aarch64.whl --no-deps
    UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install ./wheel_pkgs/vllm-0.10.1.dev271+g60523a731.cu126-cp310-cp310-linux_aarch64.whl --no-build-isolation
    uv pip install notebook jupyterlab

    echo "Building shadowkv..."
    pushd src/shadowkv
    python setup.py build_ext --inplace
    popd

    echo "Building Liburing..."
    pushd src/Liburing
    pip install -e . --force-reinstall
    popd
    # remove torchvision to avoid conflict
    uv pip uninstall torchvision

    echo "Done: Installing dependencies"
else
    echo "Dependencies already installed"
fi

source .venv/bin/activate
echo "--------------------------------"

#############################################################
echo "Setting up NVMe (this device has no eMMC)..."

if [ -z "$NVME_OFFLOAD_DIR" ]; then
    echo "Error: NVME_OFFLOAD_DIR is not set"
    exit 1
fi

if [ -z "$NVME_DEV_NAME" ]; then
    echo "Error: NVME_DEV_NAME is not set"
    exit 1
fi

mount_dev_if_needed() {
    local _label="$1"
    local _dev_name="$2"
    local _mnt_dir="$3"
    local _dev_path="/dev/${_dev_name}"

    if [[ ! -b "${_dev_path}" ]]; then
        echo "Error: ${_label} block device ${_dev_path} not found."
        exit 1
    fi

    local _target_real
    _target_real=$(readlink -f "${_dev_path}")

    local _root_src
    _root_real=""
    _root_src=$(findmnt -n -o SOURCE / 2>/dev/null | head -1)
    if [[ -n "${_root_src}" ]]; then
        if [[ "${_root_src}" == "/dev/root" ]]; then
            _root_real=$(readlink -f /dev/root)
        elif [[ -b "${_root_src}" ]]; then
            _root_real=$(readlink -f "${_root_src}")
        elif [[ "${_root_src}" == UUID=* ]]; then
            _root_real=$(blkid -U "${_root_src#UUID=}" 2>/dev/null || true)
            [[ -n "${_root_real}" ]] && _root_real=$(readlink -f "${_root_real}")
        fi
        if [[ -n "${_root_real}" && "${_root_real}" == "${_target_real}" ]]; then
            echo "Skip ${_label}: root filesystem is already on ${_dev_path}"
            return 0
        fi
    fi

    if findmnt -n "${_mnt_dir}" &>/dev/null; then
        echo "Skip ${_label}: ${_mnt_dir} is already a mount point"
        return 0
    fi

    if findmnt -S "${_target_real}" &>/dev/null || findmnt -S "${_dev_path}" &>/dev/null; then
        echo "Skip ${_label}: ${_dev_path} is already mounted"
        return 0
    fi

    echo "Mounting ${_label} to ${_mnt_dir}"
    sudo mount -o noatime,nodiratime,data=ordered,nodelalloc,nolazytime "${_dev_path}" "${_mnt_dir}"
}

sudo mkdir -p "${NVME_OFFLOAD_DIR}"
mount_dev_if_needed "NVMe" "${NVME_DEV_NAME}" "${NVME_OFFLOAD_DIR}"
sudo chmod -R 777 "${NVME_OFFLOAD_DIR}"

echo "Done: Setting up disk"
echo "--------------------------------"

##########################################################

echo "Checking model weights..."

if [ -z "$MODEL_PATH_BASE_HF" ]; then
    echo "Error: MODEL_PATH_BASE_HF is not set"
    exit 1
fi

if [ -z "$MODEL_PATH_BASE" ]; then
    echo "Error: MODEL_PATH_BASE is not set"
    exit 1
fi

##########################################################
echo "Making np weights (fp16, via scripts/make_np_weights.py)..."
# Default: Qwen3-0.6B only — fp16 weights are ~1.2 GiB, small enough to leave
# plenty of the 8 GB budget for KV cache. Override with e.g.
# NANO_MODEL_LIST="Qwen3-0.6B Qwen3-1.7B" to also cover the paper's second
# Orin Nano data point (Table 5, Sec 5.2.2). Run
# scripts/download_models_nano.sh first so these exist under
# MODEL_PATH_BASE_HF.
read -ra NANO_MODEL_LIST_ARR <<< "${NANO_MODEL_LIST:-Qwen3-0.6B}"

for model in "${NANO_MODEL_LIST_ARR[@]}"; do
    if [ ! -d "${MODEL_PATH_BASE_HF}/${model}" ]; then
        echo "Error: ${MODEL_PATH_BASE_HF}/${model} not found. Run scripts/download_models_nano.sh first."
        exit 1
    fi
    python scripts/make_np_weights.py --hf_model_path "$MODEL_PATH_BASE_HF/$model" --save_dir "$MODEL_PATH_BASE"
done

echo "Done: Making np weights at $MODEL_PATH_BASE"
echo "--------------------------------"

##########################################################

echo "Checking remaining NVMe disk space..."

_GB=$((1000 * 1000 * 1000))
# 20 GB is a conservative floor for Qwen3-0.6B/1.7B at batch<=8, context<=16K
# (see MAX_ALLOC_KV_SIZE in eval_nano.sh). Raise NVME_MIN_FREE_GB if you plan
# larger batches/contexts.
_NVME_MIN_FREE=$(( ${NVME_MIN_FREE_GB:-20} * _GB ))

if [[ ! -d "${NVME_OFFLOAD_DIR}" ]]; then
    echo "Error: NVMe mount directory ${NVME_OFFLOAD_DIR} does not exist."
    exit 1
fi

_avail=$(df -B1 "${NVME_OFFLOAD_DIR}" 2>/dev/null | tail -1 | awk '{print $4}')
if [[ -z "${_avail}" || ! "${_avail}" =~ ^[0-9]+$ ]]; then
    echo "Error: Could not read free space for ${NVME_OFFLOAD_DIR}."
    exit 1
fi

if [[ "${_avail}" -lt "${_NVME_MIN_FREE}" ]]; then
    _avail_gb=$(awk "BEGIN {printf \"%.2f\", ${_avail} / ${_GB}}")
    echo "Error: NVMe free space on ${NVME_OFFLOAD_DIR} is ${_avail_gb} GB, below the ${NVME_MIN_FREE_GB:-20} GB minimum."
    exit 1
fi

_avail_gb=$(awk "BEGIN {printf \"%.2f\", ${_avail} / ${_GB}}")
echo "  NVMe OK: /dev/${NVME_DEV_NAME} @ ${NVME_OFFLOAD_DIR} — ${_avail_gb} GB free"
echo "Disk check done."
echo "--------------------------------"

##########################################################

echo "Checking power mode and jetson clocks (advisory on this device — see nano_common.sh)..."
check_powermode_soft
check_jetson_clocks_soft
echo "--------------------------------"

##########################################################

echo "Done: Nano setup"
echo "--------------------------------"
echo "Next steps:"
echo "  1) bash ./scripts/prepare_adapter_nano.sh   # builds the KVSwap adapter Qwen3-0.6B needs (none ships in this repo)"
echo "  2) bash ./scripts/eval_nano.sh kvswap        # or 'flexgen' for the no-prediction baseline, no adapter required"
echo "--------------------------------"
