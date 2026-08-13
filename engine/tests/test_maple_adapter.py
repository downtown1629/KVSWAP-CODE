import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from expert_store import ExpertStore, pack_qwen3_expert_store
from model_adapters import (
    FFNKind, get_ffn_kind, uses_rotary_position, validate_maple_config,
    validate_maple_run,
)
from moe import ResidentExpertProvider, qwen3_moe_forward
from moe_weights import (
    SafetensorCheckpoint,
    qwen3_moe_checkpoint_bytes,
    qwen3_moe_resident_expected,
)


def maple_config():
    return SimpleNamespace(
        model_type="maple",
        attention_bias=False,
        hidden_size=8,
        head_dim=4,
        num_attention_heads=2,
        num_kv_heads=1,
        num_kv_groups=2,
        num_hidden_layers=2,
        intermediate_size=16,
        moe_intermediate_size=4,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=0,
        vocab_size=32,
        max_position_embeddings=1024,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention"],
        hidden_act="silu",
        tie_word_embeddings=False,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        partial_rotary_factor=0.5,
        nope_on_global_attention=True,
        use_qk_norm=True,
        rms_norm_eps=1e-6,
        embedding_weight_name="model.word_embeddings.weight",
        expert_clamp=True,
        router_fp32=True,
        dtype=np.float16,
    )


class MapleAdapterTest(unittest.TestCase):
    def test_all_layers_are_routed_and_official_byte_count_is_exact(self):
        config = validate_maple_config(maple_config())
        self.assertTrue(all(
            get_ffn_kind(config, layer) == FFNKind.ROUTED_MOE
            for layer in range(config.num_hidden_layers)
        ))
        self.assertTrue(uses_rotary_position(config, 0))
        self.assertFalse(uses_rotary_position(config, 1))
        official = maple_config()
        official.hidden_size = 2048
        official.head_dim = 128
        official.num_attention_heads = 16
        official.num_kv_heads = 4
        official.num_kv_groups = 4
        official.num_hidden_layers = 24
        official.intermediate_size = 4096
        official.moe_intermediate_size = 512
        official.num_experts = 256
        official.num_experts_per_tok = 8
        official.vocab_size = 151936
        official.layer_types = [
            "full_attention" if (layer + 1) % 4 == 0 else "sliding_attention"
            for layer in range(24)
        ]
        self.assertEqual(qwen3_moe_checkpoint_bytes(official), 40_428_060_672)

    def test_maple_expert_clamps_match_public_reference(self):
        torch.manual_seed(7)
        hidden = torch.randn(1, 3, 8, dtype=torch.bfloat16) * 8
        router = torch.randn(4, 8, dtype=torch.bfloat16)
        gate = torch.randn(4, 4, 8, dtype=torch.bfloat16)
        up = torch.randn(4, 4, 8, dtype=torch.bfloat16)
        down = torch.randn(4, 8, 4, dtype=torch.bfloat16)
        provider = ResidentExpertProvider(gate, up, down)
        actual, routing = qwen3_moe_forward(
            hidden, router, provider, top_k=2, norm_topk_prob=True,
            expert_clamp=True, router_fp32=True,
        )
        self.assertEqual(routing.router_logits.dtype, torch.float32)
        flat = hidden.reshape(-1, 8)
        expected = torch.zeros_like(flat, dtype=torch.float32)
        for token in range(flat.shape[0]):
            for position in range(2):
                expert = int(routing.topk_ids[token, position])
                gate_value = torch.clamp(F.linear(flat[token], gate[expert]), max=7.0)
                up_value = torch.clamp(
                    F.linear(flat[token], up[expert]), min=-7.0, max=7.0
                )
                value = F.linear(F.silu(gate_value) * up_value, down[expert])
                expected[token] += value.float() * routing.topk_weights[token, position]
        torch.testing.assert_close(actual, expected.to(hidden.dtype).reshape_as(actual))

    def test_unsupported_long_context_and_kvswap_predictor_fail_closed(self):
        config = maple_config()
        validate_maple_run(config, prompt_len=15, gen_len=2, lr_proj_mode="none")
        with self.assertRaisesRegex(ValueError, "SWA cache"):
            validate_maple_run(config, prompt_len=16, gen_len=2, lr_proj_mode="none")
        with self.assertRaisesRegex(ValueError, "calibrated predictor"):
            validate_maple_run(config, prompt_len=8, gen_len=2, lr_proj_mode="lr_proj_mh")

    def test_maple_checkpoint_packs_with_family_specific_store_format(self):
        config = maple_config()
        expected = qwen3_moe_resident_expected(config)
        tensors = {
            name: torch.randn(shape, dtype=dtype)
            for name, (shape, dtype) in expected.items()
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir, tempfile.TemporaryDirectory() as root:
            save_file(tensors, str(Path(checkpoint_dir, "model.safetensors")))
            store = pack_qwen3_expert_store(
                SafetensorCheckpoint(checkpoint_dir), config, Path(root, "store"),
                source_revision="maple-fixture",
            )
            manifest = json.loads(Path(store.root, "manifest.json").read_text())
            self.assertEqual(manifest["format"], "kvswap-maple-expert-store-v1")
            self.assertEqual(len(store.extents), 8)
            ExpertStore(
                store.root, config=config,
                expected_source_revision="maple-fixture",
            )


if __name__ == "__main__":
    unittest.main()
