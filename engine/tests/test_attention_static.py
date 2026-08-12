import ast
import unittest
from pathlib import Path


class AttentionStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_path = Path(__file__).parents[1] / "src" / "pytorch_backend.py"
        cls.tree = ast.parse(cls.source_path.read_text())
        cls.torch_device = next(
            node
            for node in cls.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TorchDevice"
        )

    def test_non_flash_attention_defines_repeated_key_and_value(self):
        chunk_attention = next(
            node
            for node in self.torch_device.body
            if isinstance(node, ast.FunctionDef) and node.name == "chuck_attn"
        )
        assigned_names = {
            target.id
            for node in ast.walk(chunk_attention)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertTrue({"key", "value"}.issubset(assigned_names))

    def test_gpu_kv_cache_accepts_engine_loader_arguments(self):
        method = next(
            node
            for node in self.torch_device.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "init_cache_one_gpu_batch"
        )
        argument_names = {argument.arg for argument in method.args.args}
        self.assertTrue(
            {"name", "force_bf16", "batch_split", "attr"}.issubset(argument_names)
        )

    def test_resident_kv_store_has_direct_tensor_copy_path(self):
        source = (Path(__file__).parents[1] / "src" / "main.py").read_text()
        self.assertIn("kv_home.device.device_type == DeviceType.DISK", source)
        self.assertIn("destination[..., :hidden_width].copy_(k_new)", source)
        self.assertIn("destination[..., hidden_width:].copy_(v_new)", source)


if __name__ == "__main__":
    unittest.main()
