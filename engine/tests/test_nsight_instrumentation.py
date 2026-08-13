import ast
import importlib.util
import sqlite3
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).parents[1]


def load_exporter():
    path = ROOT / "scripts" / "export_nsys_gantt.py"
    spec = importlib.util.spec_from_file_location("export_nsys_gantt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NsightInstrumentationTest(unittest.TestCase):
    def test_source_exposes_gantt_hierarchy(self):
        main = (ROOT / "src" / "main.py").read_text()
        moe = (ROOT / "src" / "moe.py").read_text()
        attention = (ROOT / "src" / "pytorch_backend.py").read_text()
        for marker in ("KVSWAP_TOKEN", "KVSWAP_LAYER", "KVSWAP_STAGE"):
            self.assertIn(marker, main)
        for marker in ("KVSWAP_MOE_STAGE", "KVSWAP_PREFILL_CHUNK"):
            self.assertIn(marker, moe)
        for marker in ("KVSWAP_ATTENTION_STAGE", "KVSWAP_ATTENTION_CHUNK"):
            self.assertIn(marker, attention)

    def test_generation_loops_put_layer_ranges_inside_token_range(self):
        tree = ast.parse((ROOT / "src" / "main.py").read_text())
        lm = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "LM"
        )
        for method_name in ("generation_loop_normal", "generation_loop_normal_new"):
            method = next(
                node for node in lm.body
                if isinstance(node, ast.FunctionDef) and node.name == method_name
            )
            source = ast.unparse(method)
            self.assertLess(source.index("self.profile_token_name(i)"),
                            source.index("self.profile_layer_name(i, j)"))

    def test_nsys_wrapper_is_software_trace_only(self):
        source = (ROOT / "scripts" / "profile_maple_nsys.sh").read_text()
        self.assertIn("--trace=cuda,nvtx", source)
        self.assertIn("--capture-range=cudaProfilerApi", source)
        self.assertIn("--sample=none", source)
        self.assertNotIn("--gpu-metrics-devices", source)
        self.assertNotIn("--gpu-metrics-set", source)
        self.assertNotIn(" ncu ", source)

    def test_gantt_exporter_preserves_token_layer_stage_hierarchy(self):
        exporter = load_exporter()
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "trace.sqlite"
            connection = sqlite3.connect(database)
            connection.executescript("""
                CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE NVTX_EVENTS (
                    start INTEGER NOT NULL, end INTEGER, text TEXT,
                    textId INTEGER, globalTid INTEGER
                );
            """)
            names = (
                "KVSWAP_TOKEN phase=decode step=1 tokens=1 output_position=9",
                "KVSWAP_LAYER phase=decode step=1 engine_layer=1 model_layer=0 kind=attention_swa",
                "KVSWAP_STAGE name=compute",
                "KVSWAP_ATTENTION_STAGE name=qkv_projection",
            )
            for index, name in enumerate(names, 1):
                connection.execute("INSERT INTO StringIds VALUES (?, ?)", (index, name))
            for start, end, text_id in (
                (0, 100_000_000, 1),
                (10_000_000, 90_000_000, 2),
                (20_000_000, 80_000_000, 3),
                (30_000_000, 70_000_000, 4),
            ):
                connection.execute(
                    "INSERT INTO NVTX_EVENTS VALUES (?, ?, NULL, ?, 7)",
                    (start, end, text_id),
                )
            connection.commit()
            connection.close()

            prefix = Path(directory) / "result"
            outputs = exporter.write_outputs(database, prefix)
            self.assertTrue(all(path.is_file() for path in outputs))
            summary = prefix.with_suffix(".stage-summary.csv").read_text()
            self.assertIn("decode,1,KVSWAP_STAGE,compute,1", summary)
            self.assertIn("decode,1,KVSWAP_ATTENTION_STAGE,qkv_projection,1", summary)
            detail = (Path(directory) / "result.step-001-decode.layers-00-00.gantt.svg").read_text()
            self.assertIn("L0 attention_swa", detail)
            self.assertIn("qkv_projection", detail)
            self.assertIn(">compute</text>", detail)
            self.assertIn(">80.0 ms</text>", detail)
            overview = prefix.with_suffix(".token-gantt.svg").read_text()
            self.assertIn(">attention_swa</text>", overview)
            self.assertIn(">100.0 ms</text>", overview)
            ET.parse(Path(directory) / "result.step-001-decode.layers-00-00.gantt.svg")


if __name__ == "__main__":
    unittest.main()
