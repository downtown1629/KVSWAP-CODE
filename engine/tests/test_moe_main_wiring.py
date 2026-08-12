import ast
import unittest
from pathlib import Path


class MoEMainWiringStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_path = Path(__file__).parents[1] / "src" / "main.py"
        cls.tree = ast.parse(cls.source_path.read_text())

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


if __name__ == "__main__":
    unittest.main()
