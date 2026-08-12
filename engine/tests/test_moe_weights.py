import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from moe_weights import (
    SafetensorCheckpoint,
    SafetensorIndex,
    load_qwen3_moe_layer,
    qwen3_moe_checkpoint_bytes,
    qwen3_moe_layer_bytes,
    qwen3_moe_layer_expected,
    qwen3_moe_resident_expected,
    qwen3_moe_resident_bytes,
    tensor_name_from_legacy_np_path,
    validate_qwen3_moe_checkpoint,
    validate_qwen3_moe_index,
    validate_resident_weight_budget,
)


class Qwen3MoeWeightsTest(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            model_type="qwen3_moe",
            num_hidden_layers=1,
            hidden_size=8,
            num_experts=3,
            num_experts_per_tok=2,
            moe_intermediate_size=6,
            norm_topk_prob=True,
            decoder_sparse_step=1,
            mlp_only_layers=[],
            head_dim=2,
            num_attention_heads=4,
            num_kv_heads=2,
            vocab_size=16,
            tie_word_embeddings=False,
            attention_bias=False,
            intermediate_size=10,
        )

    def make_checkpoint(self, directory, dtype=torch.bfloat16):
        expected = qwen3_moe_resident_expected(self.config, dtype=dtype)
        tensors = {}
        for index, (name, (shape, _)) in enumerate(sorted(expected.items())):
            values = torch.arange(
                1, 1 + torch.Size(shape).numel(), dtype=torch.float32
            ).reshape(shape)
            tensors[name] = (values + index).to(dtype)

        names = sorted(tensors)
        midpoint = len(names) // 2
        shard_names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        weight_map = {}
        for shard_name, shard_keys in zip(
            shard_names, (names[:midpoint], names[midpoint:])
        ):
            save_file(
                {name: tensors[name] for name in shard_keys},
                str(Path(directory, shard_name)),
            )
            weight_map.update({name: shard_name for name in shard_keys})
        Path(directory, "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "total_size": qwen3_moe_checkpoint_bytes(self.config)
                    },
                    "weight_map": weight_map,
                }
            )
        )
        return tensors

    def test_metadata_preflight_and_streaming_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected_tensors = self.make_checkpoint(tmp)
            checkpoint = SafetensorCheckpoint(tmp)
            expected_specs = qwen3_moe_layer_expected(self.config, 0)
            checkpoint.validate(expected_specs)
            self.assertEqual(
                validate_qwen3_moe_checkpoint(checkpoint, self.config), (0,)
            )
            self.assertEqual(
                checkpoint.total_tensor_bytes,
                sum(
                    torch.Size(shape).numel()
                    * torch.empty((), dtype=dtype).element_size()
                    for shape, dtype in qwen3_moe_resident_expected(
                        self.config
                    ).values()
                ),
            )
            weights = load_qwen3_moe_layer(checkpoint, self.config, 0)

        torch.testing.assert_close(
            weights.router, expected_tensors["model.layers.0.mlp.gate.weight"]
        )
        torch.testing.assert_close(
            weights.post_attention_norm,
            expected_tensors["model.layers.0.post_attention_layernorm.weight"],
        )
        for expert_id in range(self.config.num_experts):
            prefix = f"model.layers.0.mlp.experts.{expert_id}"
            torch.testing.assert_close(
                weights.gate_proj[expert_id],
                expected_tensors[f"{prefix}.gate_proj.weight"],
            )
            torch.testing.assert_close(
                weights.up_proj[expert_id],
                expected_tensors[f"{prefix}.up_proj.weight"],
            )
            torch.testing.assert_close(
                weights.down_proj[expert_id],
                expected_tensors[f"{prefix}.down_proj.weight"],
            )

    def test_index_only_preflight_does_not_require_weight_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_checkpoint(tmp)
            for shard in Path(tmp).glob("*.safetensors"):
                shard.unlink()
            index = SafetensorIndex(tmp)
            self.assertEqual(
                validate_qwen3_moe_index(index, self.config),
                qwen3_moe_checkpoint_bytes(self.config),
            )

    def test_index_only_preflight_rejects_name_and_size_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_checkpoint(tmp)
            index_path = Path(tmp, "model.safetensors.index.json")
            contents = json.loads(index_path.read_text())
            removed_name = next(iter(contents["weight_map"]))
            del contents["weight_map"][removed_name]
            index_path.write_text(json.dumps(contents))
            with self.assertRaisesRegex(ValueError, "tensor names disagree"):
                validate_qwen3_moe_index(SafetensorIndex(tmp), self.config)

            contents["weight_map"][removed_name] = "unused.safetensors"
            contents["metadata"]["total_size"] += 1
            index_path.write_text(json.dumps(contents))
            with self.assertRaisesRegex(ValueError, "byte count disagrees"):
                validate_qwen3_moe_index(SafetensorIndex(tmp), self.config)

    def test_preflight_reports_shape_and_dtype_before_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_checkpoint(tmp, dtype=torch.float16)
            checkpoint = SafetensorCheckpoint(tmp)
            expected = qwen3_moe_layer_expected(self.config, 0)
            name = "model.layers.0.mlp.gate.weight"
            expected[name] = ((99, 8), torch.bfloat16)
            with self.assertRaisesRegex(ValueError, "dtype mismatch") as raised:
                checkpoint.validate(expected)
        self.assertIn("shape mismatch", str(raised.exception))

    def test_index_must_match_physical_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_checkpoint(tmp)
            index_path = Path(tmp, "model.safetensors.index.json")
            index = json.loads(index_path.read_text())
            name = next(iter(index["weight_map"]))
            index["weight_map"][name] = "model-00002-of-00002.safetensors"
            index_path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "index maps"):
                SafetensorCheckpoint(tmp)

    def test_resident_budget_requires_explicit_headroom_aware_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_checkpoint(tmp)
            checkpoint = SafetensorCheckpoint(tmp)
            with self.assertRaisesRegex(ValueError, "disabled by default"):
                validate_resident_weight_budget(checkpoint.total_tensor_bytes, 0)
            with self.assertRaisesRegex(MemoryError, "exceeding"):
                validate_resident_weight_budget(
                    checkpoint.total_tensor_bytes,
                    checkpoint.total_tensor_bytes - 1,
                )
            self.assertEqual(
                validate_resident_weight_budget(
                    checkpoint.total_tensor_bytes,
                    checkpoint.total_tensor_bytes,
                ),
                checkpoint.total_tensor_bytes,
            )

    def test_tied_embeddings_still_account_for_two_engine_buffers(self):
        untied_bytes = qwen3_moe_resident_bytes(self.config)
        tied_config = SimpleNamespace(**vars(self.config))
        tied_config.tie_word_embeddings = True
        tied_bytes = qwen3_moe_resident_bytes(tied_config)
        embedding_bytes = (
            self.config.vocab_size * self.config.hidden_size * 2
        )
        unique_tied_bytes = sum(
            torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
            for shape, dtype in qwen3_moe_resident_expected(tied_config).values()
        )
        self.assertEqual(tied_bytes, unique_tied_bytes + embedding_bytes)
        self.assertEqual(tied_bytes, untied_bytes)

    def test_full_checkpoint_preflight_rejects_config_without_routed_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_checkpoint(tmp)
            checkpoint = SafetensorCheckpoint(tmp)
            dense_config = SimpleNamespace(**vars(self.config))
            dense_config.mlp_only_layers = [0]
            with self.assertRaisesRegex(ValueError, "does not contain"):
                validate_qwen3_moe_checkpoint(checkpoint, dense_config)

    def test_legacy_np_path_maps_to_hf_tensor_name(self):
        self.assertEqual(
            tensor_name_from_legacy_np_path(
                "/models/example/weights-np/model.layers.2.self_attn.q_proj.weight"
            ),
            "model.layers.2.self_attn.q_proj.weight",
        )
        with self.assertRaisesRegex(ValueError, "does not contain"):
            tensor_name_from_legacy_np_path("/models/example/model.weight")


if __name__ == "__main__":
    unittest.main()
