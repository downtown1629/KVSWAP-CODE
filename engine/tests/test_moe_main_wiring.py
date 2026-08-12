import ast
import unittest
from pathlib import Path


class MoEMainWiringStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_path = Path(__file__).parents[1] / "src" / "main.py"
        cls.tree = ast.parse(cls.source_path.read_text())
        cls.moe_tree = ast.parse(
            (Path(__file__).parents[1] / "src" / "moe.py").read_text()
        )

    def test_moe_block_implements_legacy_layer_protocol(self):
        moe_block = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MoEBlock"
        )
        methods = {
            node.name for node in moe_block.body if isinstance(node, ast.FunctionDef)
        }
        self.assertTrue(
            {
                "set_task",
                "init_weight",
                "load_weight",
                "init_cache_one_gpu_batch",
                "load_cache",
                "store_cache",
                "forward",
            }.issubset(methods)
        )

    def test_expert_provider_factory_owns_resident_expert_load(self):
        factory = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "Qwen3ResidentExpertProviderFactory"
        )
        create = next(
            node
            for node in factory.body
            if isinstance(node, ast.FunctionDef) and node.name == "create"
        )
        call_names = {
            node.func.id
            for node in ast.walk(create)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("load_qwen3_moe_expert_bank", call_names)

        moe_block = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MoEBlock"
        )
        init_weight = next(
            node
            for node in moe_block.body
            if isinstance(node, ast.FunctionDef) and node.name == "init_weight"
        )
        block_call_names = {
            node.func.id
            for node in ast.walk(init_weight)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("load_qwen3_moe_fixed_weights", block_call_names)
        self.assertNotIn("load_qwen3_moe_expert_bank", block_call_names)

    def test_resident_preflight_precedes_cuda_device_creation(self):
        run_flexgen = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_flexgen"
        )
        calls = [node for node in ast.walk(run_flexgen) if isinstance(node, ast.Call)]
        preflight_line = min(
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "validate_resident_weight_budget"
        )
        cuda_device_line = min(
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "TorchDevice"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "cuda:0"
        )
        self.assertLess(preflight_line, cuda_device_line)

    def test_budget_gate_precedes_weight_shard_open(self):
        run_flexgen = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_flexgen"
        )
        calls = [node for node in ast.walk(run_flexgen) if isinstance(node, ast.Call)]
        budget_line = min(
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "validate_resident_weight_budget"
        )
        shard_open_line = min(
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "SafetensorCheckpoint"
        )
        self.assertLess(budget_line, shard_open_line)

    def test_capacity_gate_precedes_weight_shard_open_and_cuda(self):
        run_flexgen = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_flexgen"
        )
        calls = [node for node in ast.walk(run_flexgen) if isinstance(node, ast.Call)]
        capacity_line = min(
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "validate_resident_memory_capacity"
        )
        shard_line = min(
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "SafetensorCheckpoint"
        )
        cuda_line = min(
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "TorchDevice"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "cuda:0"
        )
        self.assertLess(capacity_line, shard_line)
        self.assertLess(capacity_line, cuda_line)

    def test_checkpoint_preflight_precedes_tokenizer_loading(self):
        run_flexgen = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_flexgen"
        )
        calls = [node for node in ast.walk(run_flexgen) if isinstance(node, ast.Call)]
        checkpoint_line = min(
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "validate_qwen3_moe_checkpoint"
        )
        tokenizer_line = min(
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "AutoTokenizer"
        )
        self.assertLess(checkpoint_line, tokenizer_line)

    def test_resident_preflight_precedes_cuda_allocator_configuration(self):
        run_flexgen = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_flexgen"
        )
        calls = [node for node in ast.walk(run_flexgen) if isinstance(node, ast.Call)]
        preflight_line = min(
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "validate_resident_weight_budget"
        )
        allocator_line = min(
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Attribute)
            and node.func.attr == "set_per_process_memory_fraction"
        )
        self.assertLess(preflight_line, allocator_line)

        top_level_allocator_calls = [
            node
            for node in self.tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "set_per_process_memory_fraction"
        ]
        self.assertEqual(top_level_allocator_calls, [])

    def test_resident_limit_defaults_to_disabled(self):
        source = self.source_path.read_text()
        self.assertIn('"--moe_resident_weight_limit_gb"', source)
        self.assertIn("default=0.0", source)

    def test_qk_norm_weight_load_is_owned_by_qwen3_family_branch(self):
        attention = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SelfAttention"
        )
        init_weight = next(
            node
            for node in attention.body
            if isinstance(node, ast.FunctionDef) and node.name == "init_weight"
        )
        family_branch = next(
            node
            for node in ast.walk(init_weight)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Name)
            and node.test.func.id == "is_qwen3_family"
        )
        branch_calls = [
            node
            for statement in family_branch.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "init_weight_list"
        ]
        self.assertEqual(len(branch_calls), 1)

    def test_moe_path_does_not_own_kv_cache_or_copy_queue(self):
        protected_classes = {
            node.name: ast.unparse(node)
            for node in self.tree.body
            if isinstance(node, ast.ClassDef)
            and node.name in {"MoEBlock", "Qwen3ResidentExpertProviderFactory"}
        }
        protected_classes.update({
            node.name: ast.unparse(node)
            for node in self.moe_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name in {"ExpertProvider", "ResidentExpertProvider"}
        })
        self.assertEqual(set(protected_classes), {
            "MoEBlock", "Qwen3ResidentExpertProviderFactory",
            "ExpertProvider", "ResidentExpertProvider",
        })
        for source in protected_classes.values():
            self.assertNotIn("CacheManager", source)
            self.assertNotIn("general_copy", source)
            self.assertNotIn("prefetch_cache", source)
            self.assertNotIn("submit_copy", source)

    def test_m2_demand_mode_is_explicit_and_keeps_kv_storage_separate(self):
        source = self.source_path.read_text()
        expert_source = (
            Path(__file__).parents[1] / "src" / "expert_store.py"
        ).read_text()
        self.assertIn('"--expert_mode"', source)
        self.assertIn('choices=("resident", "demand")', source)
        self.assertIn('"--moe_expert_store"', source)
        self.assertIn("Qwen3DemandExpertProviderFactory", source)
        self.assertNotIn("CacheManager", expert_source)
        self.assertNotIn("TorchDisk", expert_source)
        self.assertNotIn("general_copy", expert_source)


if __name__ == "__main__":
    unittest.main()
