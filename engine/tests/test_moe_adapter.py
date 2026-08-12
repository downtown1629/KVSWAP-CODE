import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from model_adapters import (
    FFNKind,
    get_ffn_kind,
    get_qwen3_moe_layer_spec,
    is_qwen3_family,
)
from model_config import get_model_config


class Qwen3MoeAdapterTest(unittest.TestCase):
    def make_config(self):
        return SimpleNamespace(
            model_type="qwen3_moe",
            num_hidden_layers=4,
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=24,
            norm_topk_prob=True,
            decoder_sparse_step=2,
            mlp_only_layers=[3],
        )

    def test_qwen3_family(self):
        self.assertTrue(is_qwen3_family("qwen3"))
        self.assertTrue(is_qwen3_family("qwen3_moe"))
        self.assertFalse(is_qwen3_family("qwen2"))

    def test_sparse_layer_schedule_matches_transformers_rule(self):
        config = self.make_config()
        self.assertEqual(get_ffn_kind(config, 0), FFNKind.DENSE)
        self.assertEqual(get_ffn_kind(config, 1), FFNKind.ROUTED_MOE)
        self.assertEqual(get_ffn_kind(config, 2), FFNKind.DENSE)
        self.assertEqual(get_ffn_kind(config, 3), FFNKind.DENSE)

        spec = get_qwen3_moe_layer_spec(config, 1)
        self.assertEqual(spec.num_experts, 8)
        self.assertEqual(spec.top_k, 2)
        self.assertEqual(spec.intermediate_size, 24)
        self.assertIsNone(get_qwen3_moe_layer_spec(config, 0))

    def test_model_config_uses_hf_model_type_not_directory_name(self):
        raw_config = {
            "model_type": "qwen3_moe",
            "architectures": ["Qwen3MoeForCausalLM"],
            "attention_bias": False,
            "decoder_sparse_step": 2,
            "head_dim": 16,
            "hidden_act": "silu",
            "hidden_size": 64,
            "intermediate_size": 128,
            "max_position_embeddings": 32768,
            "mlp_only_layers": [3],
            "moe_intermediate_size": 32,
            "norm_topk_prob": True,
            "num_attention_heads": 4,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 4,
            "num_key_value_heads": 2,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-6,
            "rope_scaling": None,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": False,
            "vocab_size": 128,
        }
        with tempfile.TemporaryDirectory() as parent:
            tmp = Path(parent, "misleading-qwen2-model-name")
            tmp.mkdir()
            Path(tmp, "config.json").write_text(json.dumps(raw_config))
            config = get_model_config(str(tmp))

        self.assertEqual(config.model_type, "qwen3_moe")
        self.assertEqual(config.num_experts, 8)
        self.assertEqual(config.num_experts_per_tok, 2)
        self.assertEqual(config.rms_norm_eps, 1e-6)


if __name__ == "__main__":
    unittest.main()
