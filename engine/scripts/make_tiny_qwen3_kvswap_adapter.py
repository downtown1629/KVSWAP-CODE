#!/usr/bin/env python3
"""Create a test-only identity KVSwap adapter for the tiny M1 fixture."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig


def create_adapter(fixture_dir, output_dir):
    fixture_dir = Path(fixture_dir)
    output_dir = Path(output_dir)
    if not (fixture_dir / "m1_fixture.json").is_file():
        raise ValueError(f"not an M1 fixture: {fixture_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")

    config = AutoConfig.from_pretrained(fixture_dir)
    if config.model_type != "qwen3_moe":
        raise ValueError(f"expected qwen3_moe, got {config.model_type}")
    kv_width = int(config.num_key_value_heads) * int(config.head_dim)
    output_dir.mkdir(parents=True, exist_ok=True)
    projection = torch.eye(kv_width, dtype=torch.bfloat16)
    files = []
    for layer_id in range(int(config.num_hidden_layers)):
        filename = f"lr_kproj_{layer_id}.pt"
        torch.save(projection, output_dir / filename)
        files.append(filename)

    manifest = {
        "format": "kvswap-qwen3-moe-m1-identity-adapter-v1",
        "purpose": "integration-regression-only",
        "model_type": config.model_type,
        "num_hidden_layers": int(config.num_hidden_layers),
        "projection_shape": [kv_width, kv_width],
        "dtype": "bfloat16",
        "files": files,
    }
    (output_dir / "adapter_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    print(json.dumps(create_adapter(args.fixture_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
