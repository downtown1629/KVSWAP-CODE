#!/usr/bin/env python3
"""Validate Qwen3-MoE config/index metadata without loading weight shards."""

import argparse
import sys
from pathlib import Path

import torch


ENGINE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ENGINE_SRC))

from model_config import get_model_config
from moe_weights import (
    SafetensorIndex,
    qwen3_moe_resident_bytes,
    validate_qwen3_moe_index,
)


def main():
    parser = argparse.ArgumentParser(
        description="Statically validate a Qwen3-MoE Hugging Face checkpoint index"
    )
    parser.add_argument(
        "model_path",
        help="directory containing config.json and model.safetensors.index.json",
    )
    args = parser.parse_args()

    config = get_model_config(args.model_path)
    if config.model_type != "qwen3_moe":
        raise ValueError(f"expected qwen3_moe, found {config.model_type}")
    index = SafetensorIndex(args.model_path)
    checkpoint_bytes = validate_qwen3_moe_index(index, config, torch.bfloat16)
    resident_bytes = qwen3_moe_resident_bytes(config, torch.bfloat16)
    print(f"validated {len(index.weight_map)} tensors")
    print(f"checkpoint: {checkpoint_bytes / (1024 ** 3):.4f} GiB")
    print(f"engine resident allocation: {resident_bytes / (1024 ** 3):.4f} GiB")


if __name__ == "__main__":
    main()
