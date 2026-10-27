"""Evidence/catalog tests only; synthetic reports are never GPU evidence."""

import copy
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "benchmark/mpu_stages" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MATRIX, PROBE = load("jit_matrix"), load("hardware_probe")


def cases(family, matrix="default"):
    stage = {
        "family": family,
        "variant": "cpu",
        "variants": {
            "cpu": {"hisparse_config": {"top_k": 2048, "device_buffer_size": 4096}}
        },
    }
    return MATRIX.cases_for(stage, PROBE.FAMILIES[family], matrix)


class JitMatrixTests(unittest.TestCase):
    def test_six_stages_cover_26_default_factories_and_preserve_selector_budget(self):
        total = 0
        for family, count in (
            ("qwen", 3),
            ("deepseek-v4-flash", 5),
            ("glm-5.3-flash", 5),
        ):
            default = cases(family)
            self.assertEqual(len(default), count)
            total += 2 * len(default)
            expanded = cases(family, "boundaries")
            self.assertEqual(expanded[:count], default)
            for case in expanded[count:]:
                if case["id"].endswith("hot2x"):
                    original = next(
                        c
                        for c in default
                        if c["id"] == case["id"].removesuffix("-hot2x")
                    )
                    self.assertEqual(case["args"][:3], original["args"][:3])
                    self.assertEqual(case["args"][3], original["args"][3] * 2)
                else:
                    self.assertTrue(case["component_only"])
        self.assertEqual(total, 26)

    def test_missing_substituted_and_non_native_evidence_never_passes(self):
        planned = cases("qwen")
        report = {
            "jit_cases": planned,
            "capability_scope": "production_jit_load_no_gpu",
            "gpu_execution_requested": False,
            "checks": [],
        }
        for case in planned:
            report["checks"].append(
                {
                    "name": "production_jit_factory",
                    "status": "passed",
                    "evidence": {
                        "case": case,
                        "exports": case["exports"] or ["fixture"],
                        "native_module_loaded": True,
                        "gpu_executed": False,
                        "shared_libraries": [{"path": "synthetic-only.so"}],
                    },
                }
            )
        self.assertEqual(len(MATRIX.validate_report(report, planned)), 3)
        for change in ("missing", "substituted", "unloaded", "gpu", "exports"):
            invalid = copy.deepcopy(report)
            if change == "missing":
                invalid["checks"].pop()
            elif change == "substituted":
                invalid["checks"][1]["evidence"]["case"]["args"][2] = 1
            elif change == "unloaded":
                invalid["checks"][0]["evidence"]["native_module_loaded"] = False
            elif change == "gpu":
                invalid["gpu_execution_requested"] = True
            else:
                invalid["checks"][1]["evidence"]["exports"] = ["wrong"]
            with self.subTest(change=change), self.assertRaises(ValueError):
                MATRIX.validate_report(invalid, planned)


if __name__ == "__main__":
    unittest.main()
