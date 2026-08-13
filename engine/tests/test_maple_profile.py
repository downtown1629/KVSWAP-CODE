import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analyze_maple_profile import add_samples, parse_log


class MapleProfileTest(unittest.TestCase):
    def test_engine_and_phase_metrics_are_summarized(self):
        log = """
Maple KV policy: sliding_layers=18, window=512, local_capacity=511, global_layers=6, global_placement=configured
Cache size: 24.07 MB
Decoding disk sync time: 3.5 ms
Expert demand I/O: calls=10, reads=20, logical_bytes=40960, stored_bytes=40960, read_ms=8.000, copy_ms=2.000
input: 520 output: 3 bsz: 1
Latency Total: 4.0 Prefill: 2.0 Decode: 2.0
Throughput Total: 0.75 Prefill: 260.0 Decode: 1.0
Peak Memory (GB) RSS: 3.5 TorchAllocated: 2.0 TorchReserved: 2.1
"""
        samples = pd.DataFrame({
            "gpu_pct": [10, 20, 30, 40],
            "cpu1": [1, 2, 3, 4],
            "emc_pct": [5, 6, 7, 8],
            "ram_used_kb": [1024, 2048, 3072, 4096],
            "swap_used_kb": [0, 0, 0, 0],
            "power_tot_mw": [1000, 2000, 3000, 4000],
        })
        disk = pd.DataFrame({
            "read_iops": [1, 2, 3, 4],
            "read_kbps": [1024, 2048, 3072, 4096],
            "avg_read_kb": [4, 4, 4, 4],
            "queue_depth": [0.1, 0.2, 0.3, 0.4],
            "util_pct": [10, 20, 30, 40],
        })
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "run.log"
            jtop_path = root / "run.jtop.csv"
            disk_path = root / "run.diskio.csv"
            log_path.write_text(log)
            samples.to_csv(jtop_path, index=False)
            disk.to_csv(disk_path, index=False)
            row = add_samples(parse_log(log_path), jtop_path, disk_path)

        self.assertEqual(row["sliding_layers"], 18)
        self.assertEqual(row["throughput_prefill_tps"], 260.0)
        self.assertEqual(row["decode_ms_per_token"], 1000.0)
        self.assertEqual(row["expert_avg_read_kib"], 2.0)
        self.assertEqual(row["prefill_gpu_mean_pct"], 15.0)
        self.assertEqual(row["decode_gpu_mean_pct"], 35.0)
        self.assertEqual(row["decode_disk_read_mib_s_mean"], 3.5)


if __name__ == "__main__":
    unittest.main()
