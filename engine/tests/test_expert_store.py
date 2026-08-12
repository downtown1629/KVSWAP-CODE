import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from expert_store import (
    DEFAULT_ALIGNMENT,
    ExpertStore,
    Qwen3DemandExpertProviderFactory,
    SynchronousExtentReader,
    estimate_qwen3_moe_demand_memory,
    pack_qwen3_expert_store,
    qwen3_expert_logical_bytes,
    qwen3_expert_store_required_bytes,
    qwen3_moe_fixed_weight_bytes,
)
from moe import ResidentExpertProvider, qwen3_moe_forward
from moe_weights import (
    SafetensorCheckpoint,
    load_qwen3_moe_expert_bank,
    qwen3_moe_resident_expected,
)


class ExpertStoreTest(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            model_type="qwen3_moe",
            num_hidden_layers=2,
            hidden_size=8,
            num_experts=4,
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

    def make_checkpoint(self, directory):
        tensors = {}
        for index, (name, (shape, _)) in enumerate(
            sorted(qwen3_moe_resident_expected(self.config).items())
        ):
            generator = torch.Generator().manual_seed(1000 + index)
            tensors[name] = torch.randn(shape, generator=generator).to(torch.bfloat16)
        save_file(tensors, str(Path(directory, "model.safetensors")))
        return tensors

    def pack(self, checkpoint_dir, store_dir):
        self.make_checkpoint(checkpoint_dir)
        checkpoint = SafetensorCheckpoint(checkpoint_dir)
        return pack_qwen3_expert_store(
            checkpoint, self.config, store_dir, source_revision="fixture-revision"
        )

    def test_pack_is_aligned_lossless_and_complete(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir, tempfile.TemporaryDirectory() as root:
            store_dir = Path(root, "store")
            store = self.pack(checkpoint_dir, store_dir)
            self.assertEqual(len(store.extents), 8)
            self.assertEqual(store.verify_checksums(), 8)
            self.assertEqual(
                Path(store_dir, "experts-000.bin").stat().st_size,
                qwen3_expert_store_required_bytes(self.config),
            )
            for extent in store.extents.values():
                self.assertEqual(extent.offset % DEFAULT_ALIGNMENT, 0)
                self.assertEqual(extent.stored_bytes % DEFAULT_ALIGNMENT, 0)
                self.assertEqual([item.name for item in extent.components], [
                    "gate_proj", "up_proj", "down_proj"
                ])

    def test_manifest_is_written_last_and_nonempty_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir, tempfile.TemporaryDirectory() as root:
            store_dir = Path(root, "store")
            self.pack(checkpoint_dir, store_dir)
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                pack_qwen3_expert_store(
                    SafetensorCheckpoint(checkpoint_dir), self.config, store_dir
                )

    def test_validation_rejects_config_corruption_overlap_and_checksum(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir, tempfile.TemporaryDirectory() as root:
            store_dir = Path(root, "store")
            self.pack(checkpoint_dir, store_dir)
            wrong = SimpleNamespace(**vars(self.config))
            wrong.num_experts_per_tok = 1
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                ExpertStore(store_dir, config=wrong)

            manifest_path = Path(store_dir, "manifest.json")
            manifest = json.loads(manifest_path.read_text())
            manifest["extents"][1]["offset"] = manifest["extents"][0]["offset"]
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "overlapping"):
                ExpertStore(store_dir, config=self.config)

    def test_reader_detects_short_read_and_closes_idempotently(self):
        with tempfile.NamedTemporaryFile() as handle:
            handle.write(b"x" * DEFAULT_ALIGNMENT)
            handle.flush()
            reader = SynchronousExtentReader(handle.name)
            view = reader.read(0, 16, DEFAULT_ALIGNMENT)
            self.assertEqual(bytes(view), b"x" * 16)
            view.release()
            with self.assertRaisesRegex(EOFError, "short expert read"):
                reader.read(DEFAULT_ALIGNMENT, 16, DEFAULT_ALIGNMENT)
            reader.close()
            reader.close()

    def test_repeated_reader_lifecycle_does_not_leak_fds(self):
        with tempfile.NamedTemporaryFile() as handle:
            handle.write(b"x" * DEFAULT_ALIGNMENT)
            handle.flush()
            before = len(os.listdir("/proc/self/fd"))
            for _ in range(20):
                with SynchronousExtentReader(handle.name) as reader:
                    view = reader.read(0, 16, DEFAULT_ALIGNMENT)
                    view.release()
            after = len(os.listdir("/proc/self/fd"))
        self.assertEqual(after, before)

    def test_demand_provider_matches_resident_and_never_caches(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir, tempfile.TemporaryDirectory() as root:
            store_dir = Path(root, "store")
            self.pack(checkpoint_dir, store_dir)
            checkpoint = SafetensorCheckpoint(checkpoint_dir)
            bank = load_qwen3_moe_expert_bank(checkpoint, self.config, 0)
            resident = ResidentExpertProvider(
                bank.gate_proj, bank.up_proj, bank.down_proj
            )
            factory = Qwen3DemandExpertProviderFactory(
                ExpertStore(store_dir, config=self.config), self.config,
                slots=self.config.num_experts, direct=False,
            )
            demand = factory.create(0, "cpu", torch.bfloat16)
            hidden = torch.randn(2, 5, self.config.hidden_size).to(torch.bfloat16)
            router = torch.randn(
                self.config.num_experts, self.config.hidden_size
            ).to(torch.bfloat16)
            expected, expected_routing = qwen3_moe_forward(
                hidden, router, resident, self.config.num_experts_per_tok, True
            )
            actual, actual_routing = qwen3_moe_forward(
                hidden, router, demand, self.config.num_experts_per_tok, True
            )
            torch.testing.assert_close(actual_routing.topk_ids, expected_routing.topk_ids)
            torch.testing.assert_close(actual, expected)
            unique_count = torch.unique(actual_routing.topk_ids).numel()
            self.assertEqual(factory.workspace.read_count, unique_count)
            second, _ = qwen3_moe_forward(
                hidden, router, demand, self.config.num_experts_per_tok, True
            )
            torch.testing.assert_close(second, expected)
            self.assertEqual(factory.workspace.read_count, 2 * unique_count)
            factory.close()

    def test_debug_read_verification_detects_data_corruption(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir, tempfile.TemporaryDirectory() as root:
            store_dir = Path(root, "store")
            self.pack(checkpoint_dir, store_dir)
            data_path = Path(store_dir, "experts-000.bin")
            with open(data_path, "r+b") as handle:
                original = handle.read(1)
                handle.seek(0)
                handle.write(bytes([original[0] ^ 0xFF]))
            factory = Qwen3DemandExpertProviderFactory(
                ExpertStore(store_dir, config=self.config), self.config,
                slots=1, direct=False, verify_reads=True,
            )
            provider = factory.create(0, "cpu", torch.bfloat16)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                provider.materialize(torch.tensor([[0]]))
            factory.close()

    def test_scratch_limit_rejects_before_any_read(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir, tempfile.TemporaryDirectory() as root:
            store_dir = Path(root, "store")
            self.pack(checkpoint_dir, store_dir)
            factory = Qwen3DemandExpertProviderFactory(
                ExpertStore(store_dir, config=self.config), self.config,
                slots=1, direct=False,
            )
            provider = factory.create(0, "cpu", torch.bfloat16)
            with self.assertRaisesRegex(MemoryError, "scratch has 1 slots"):
                provider.materialize(torch.tensor([[0, 1]]))
            self.assertEqual(factory.workspace.read_count, 0)
            factory.close()

    def test_demand_memory_plan_replaces_resident_bank_with_bounded_scratch(self):
        plan = estimate_qwen3_moe_demand_memory(
            self.config,
            scratch_slots=2,
            gpu_batch_size=1,
            num_gpu_batches=1,
            prompt_len=64,
            gen_len=2,
            cache_gpu_percent=100,
            cache_cpu_percent=0,
            activation_gpu_percent=100,
            activation_cpu_percent=0,
            flash_attention=False,
            system_headroom_bytes=1024,
        )
        self.assertEqual(
            plan.weights,
            qwen3_moe_fixed_weight_bytes(self.config)
            + 2 * qwen3_expert_logical_bytes(self.config),
        )
        self.assertEqual(plan.staging, DEFAULT_ALIGNMENT)
        self.assertGreater(plan.system_required, plan.weights)


if __name__ == "__main__":
    unittest.main()
