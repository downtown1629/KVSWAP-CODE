import ast
import unittest
from pathlib import Path


class MoEFixtureStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_path = (
            Path(__file__).parents[1] / "scripts" / "make_tiny_qwen3_moe.py"
        )
        cls.source = source_path.read_text()
        cls.tree = ast.parse(cls.source)

    def test_fixture_is_small_kernel_compatible_and_cpu_only(self):
        self.assertIn("hidden_size=128", self.source)
        self.assertIn("head_dim=64", self.source)
        self.assertIn("num_experts=8", self.source)
        self.assertIn("num_experts_per_tok=2", self.source)
        self.assertIn("norm_topk_prob=True", self.source)
        self.assertIn('device="cpu"', self.source)
        self.assertNotIn(".cuda(", self.source)
        self.assertIn("PreTrainedTokenizerFast", self.source)

    def test_fixture_refuses_to_overwrite_nonempty_directory(self):
        raises = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == "FileExistsError"
        ]
        self.assertTrue(raises)

    def test_fixture_verifier_defaults_to_cpu_and_allows_cuda(self):
        verifier = (
            Path(__file__).parents[1]
            / "scripts"
            / "verify_qwen3_moe_fixture.py"
        ).read_text()
        self.assertIn('choices=("cpu", "cuda")', verifier)
        self.assertIn('default="cpu"', verifier)
        self.assertNotIn("Qwen3MoeForCausalLM", verifier)

    def test_kvswap_adapter_is_identity_and_fixture_only(self):
        adapter = (
            Path(__file__).parents[1]
            / "scripts"
            / "make_tiny_qwen3_kvswap_adapter.py"
        ).read_text()
        self.assertIn('torch.eye(kv_width, dtype=torch.bfloat16)', adapter)
        self.assertIn('"integration-regression-only"', adapter)
        self.assertIn('fixture_dir / "m1_fixture.json"', adapter)
        self.assertIn("FileExistsError", adapter)


if __name__ == "__main__":
    unittest.main()
