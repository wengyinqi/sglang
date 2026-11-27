"""Isolated experiment: run retained GPU tests against explicit MMA workspace."""

import argparse
import unittest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", required=True)
    args = parser.parse_args()
    import sglang.jit_kernel.fused_hisparse_tma as scalar
    from sglang.jit_kernel.fused_hisparse_tma_mma import FusedHiSparseTMAMMAWorkspace

    # This process is a diagnostic harness. The production backend and its
    # configured implementation are not changed by this explicit substitution.
    scalar.FusedHiSparseTMAWorkspace = FusedHiSparseTMAMMAWorkspace
    suite = unittest.defaultTestLoader.discover(args.tests, pattern="test_fused*gpu.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
