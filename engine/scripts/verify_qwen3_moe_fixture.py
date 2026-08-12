#!/usr/bin/env python3
"""Verify fixture weights and MoE arithmetic without CUDA or the full model."""

import argparse
import sys
from pathlib import Path

import torch
from safetensors import safe_open


ENGINE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ENGINE_SRC))

from model_config import get_model_config
from moe import ResidentExpertProvider, qwen3_moe_forward
from moe_weights import SafetensorCheckpoint, load_qwen3_moe_layer


def load_reference(path):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name) for name in handle.keys()}


def verify_fixture(model_path, device="cpu"):
    model_path = Path(model_path)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA verification requested, but CUDA is unavailable")
    config = get_model_config(str(model_path))
    checkpoint = SafetensorCheckpoint(model_path)
    reference = load_reference(model_path / "m1_reference.safetensors")
    reports = []
    for layer_id in range(config.num_hidden_layers):
        weights = load_qwen3_moe_layer(
            checkpoint, config, layer_id, device=device, dtype=torch.bfloat16
        )
        provider = ResidentExpertProvider(
            weights.gate_proj, weights.up_proj, weights.down_proj
        )
        moe_input = reference[f"moe_input.{layer_id}"].to(device)
        expected = reference[f"moe_output.{layer_id}"].to(device)
        actual, routing = qwen3_moe_forward(
            moe_input,
            weights.router,
            provider,
            top_k=config.num_experts_per_tok,
            norm_topk_prob=config.norm_topk_prob,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            routing.router_logits,
            reference[f"router_logits.{layer_id}"].to(device),
            rtol=0,
            atol=0,
        )
        if not torch.equal(
            routing.topk_ids, reference[f"topk_ids.{layer_id}"].to(device)
        ):
            raise AssertionError(f"top-k expert mismatch in layer {layer_id}")
        torch.testing.assert_close(
            routing.topk_weights,
            reference[f"topk_weights.{layer_id}"].to(device),
            rtol=0,
            atol=0,
        )
        reports.append((layer_id, float((actual.float() - expected.float()).abs().max())))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    for layer_id, max_abs_error in verify_fixture(args.model_path, args.device):
        print(f"layer {layer_id}: exact top-k, max_abs_error={max_abs_error:.1f}")


if __name__ == "__main__":
    main()
