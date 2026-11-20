import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "cache_accounting",
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/observability/hisparse_cache_metrics.py",
)
accounting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accounting)


class TestCacheMetricsAccounting(unittest.TestCase):
    def test_layers_conserve_selected_and_logical_payload(self):
        report = accounting.summarize_cache_metrics(
            [[2, 6, 3, 2, 1], [2, 6, 1, 2, 3]], 1024
        )
        self.assertEqual(report["totals"]["resolve_calls"], 4)
        self.assertEqual(report["totals"]["miss_entries"], 4)
        self.assertEqual(report["logical_miss_kv_payload_bytes"], 8192)
        self.assertAlmostEqual(report["miss_fraction"], 1 / 3)
        self.assertFalse(report["hardware_transaction_bytes_measured"])

    def test_zero_work_has_no_inferred_hit_rate(self):
        self.assertIsNone(
            accounting.summarize_cache_metrics([[0, 0, 0, 0, 0]], 32)["miss_fraction"]
        )

    def test_malformed_counters_are_rejected(self):
        for rows in (
            [],
            [[1, 3, 1, 1, 0]],
            [[1, 3, 1, 1, -1]],
            [[1, 3, 1, 1]],
            [[True, 3, 1, 1, 1]],
            [[1, 3.0, 1, 1, 1]],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                accounting.summarize_cache_metrics(rows, 32)


if __name__ == "__main__":
    unittest.main()
