import ast
import unittest
from pathlib import Path


class RMSNormConfigStaticTest(unittest.TestCase):
    def test_runtime_rms_norm_calls_pass_epsilon_explicitly(self):
        source_root = Path(__file__).parents[1] / "src"
        for relative_path in ("main.py", "methods.py", "pytorch_backend.py"):
            path = source_root / relative_path
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not isinstance(node.func, ast.Name) or node.func.id != "rms_norm":
                    continue
                keywords = {keyword.arg for keyword in node.keywords}
                self.assertIn(
                    "eps",
                    keywords,
                    f"{relative_path}:{node.lineno} relies on the wrong default epsilon",
                )


if __name__ == "__main__":
    unittest.main()
