#!/usr/bin/env python3
"""Create a deterministic, CPU-only Qwen3-MoE M1 fixture."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedTokenizerFast,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)


FIXTURE_INPUT_IDS = [1, 17, 33, 49, 65, 81, 97, 113]


def build_config():
    return Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=384,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        max_position_embeddings=32768,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
    )


def save_tokenizer(output_dir):
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    vocab.update({f"token_{token_id}": token_id for token_id in range(4, 256)})
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        chat_template=(
            "{% for message in messages %}"
            "{{ message['role'] }} {{ message['content'] }} "
            "{% endfor %}assistant "
        ),
    )
    tokenizer.save_pretrained(output_dir)


def create_fixture(output_dir, seed=1234, router_scale=8.0):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    model = Qwen3MoeForCausalLM(build_config()).to(
        device="cpu", dtype=torch.bfloat16
    )
    model.eval()
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.gate.weight.mul_(router_scale)

    input_ids = torch.tensor([FIXTURE_INPUT_IDS], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    moe_inputs = {}
    moe_outputs = {}
    hooks = []
    for layer_id, layer in enumerate(model.model.layers):
        def capture_input(_module, args, current_layer=layer_id):
            moe_inputs[current_layer] = args[0].detach().clone()

        def capture_output(_module, _args, output, current_layer=layer_id):
            moe_outputs[current_layer] = output[0].detach().clone()

        hooks.append(layer.mlp.register_forward_pre_hook(capture_input))
        hooks.append(layer.mlp.register_forward_hook(capture_output))
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            output_router_logits=True,
        )
    for hook in hooks:
        hook.remove()

    reference = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "logits": outputs.logits,
    }
    minimum_boundary_margin = float("inf")
    for layer_id, router_logits in enumerate(outputs.router_logits):
        sorted_logits = torch.topk(router_logits.float(), 3, dim=-1)
        boundary_margin = sorted_logits.values[:, 1] - sorted_logits.values[:, 2]
        minimum_boundary_margin = min(
            minimum_boundary_margin, float(boundary_margin.min())
        )
        routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(routing_weights, 2, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(moe_inputs[layer_id].dtype)
        reference[f"router_logits.{layer_id}"] = router_logits
        reference[f"topk_ids.{layer_id}"] = topk_ids
        reference[f"topk_weights.{layer_id}"] = topk_weights
        reference[f"moe_input.{layer_id}"] = moe_inputs[layer_id]
        reference[f"moe_output.{layer_id}"] = moe_outputs[layer_id]
    for hidden_id, hidden_state in enumerate(outputs.hidden_states):
        reference[f"hidden_states.{hidden_id}"] = hidden_state

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="1MB",
    )
    save_tokenizer(output_dir)
    save_file(reference, str(output_dir / "m1_reference.safetensors"))
    manifest = {
        "format": "kvswap-qwen3-moe-m1-fixture-v1",
        "seed": seed,
        "router_scale": router_scale,
        "minimum_topk_boundary_logit_margin": minimum_boundary_margin,
        "input_ids": FIXTURE_INPUT_IDS,
        "reference_file": "m1_reference.safetensors",
    }
    (output_dir / "m1_fixture.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--router-scale", type=float, default=8.0)
    args = parser.parse_args()
    manifest = create_fixture(args.output_dir, args.seed, args.router_scale)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
