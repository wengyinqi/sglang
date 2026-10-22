"""CPU-only checks of peer-kernel input validation and compute-side dispatch.

These tests do not compile or execute CUDA and do not establish peer correctness.
"""

import importlib.util
import math
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

REPO = Path(__file__).resolve().parents[3]


def load_source(name):
    path = REPO / "python/sglang/jit_kernel" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"mpu_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    # Avoid importing the serving runtime or JIT dependencies on a CPU host.
    with mock.patch.dict(
        sys.modules,
        {
            "sglang.jit_kernel.utils": SimpleNamespace(
                load_jit=mock.Mock(), make_cpp_args=mock.Mock()
            )
        },
    ):
        spec.loader.exec_module(module)
    return module


IO = load_source("mpu_io")
SPARSE = load_source("hisparse")


class MetadataTensor:
    """Only tensor metadata: no GPU allocation or hidden CPU payload emulation."""

    def __init__(self, shape, dtype=torch.int64, device="cuda:3", contiguous=True):
        self.shape = torch.Size(shape)
        self.dtype = dtype
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
        self.ndim = len(shape)
        self.contiguous = contiguous

    def stride(self, dim):
        return math.prod(self.shape[dim + 1 :])

    def is_contiguous(self):
        return self.contiguous

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return self.dtype.itemsize

    def __getitem__(self, index):
        if index != 0:
            raise AssertionError("Unexpected payload access in metadata validation")
        return MetadataTensor(self.shape[1:], self.dtype, str(self.device))


class TestMPUKernelContracts(unittest.TestCase):
    def test_peer_setup_rejects_topology_without_loading_jit(self):
        with mock.patch.object(IO, "_module") as loader:
            for compute, peers in (
                (0, []),
                (0, [0]),
                (0, [1, 1]),
                (0, [1, 2, 3, 4, 5]),
            ):
                with (
                    self.subTest(compute=compute, peers=peers),
                    self.assertRaises(ValueError),
                ):
                    IO.enable_peer_access(compute, peers)
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "NVIDIA CUDA"):
                    IO.enable_peer_access(0, [1])
            loader.assert_not_called()

    def test_peer_setup_is_directional_and_has_no_fallback(self):
        module = mock.Mock()
        with (
            mock.patch.object(IO, "_module", return_value=module),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=4),
            mock.patch.object(torch.version, "hip", None),
            mock.patch.object(
                torch.cuda, "device", return_value=nullcontext()
            ) as guard,
            mock.patch.object(
                torch.cuda, "can_device_access_peer", return_value=True
            ) as can_access,
        ):
            IO.enable_peer_access(3, [0, 2])
            self.assertEqual(
                can_access.call_args_list, [mock.call(3, 0), mock.call(3, 2)]
            )
            self.assertEqual(
                module.enable_peer_access.call_args_list,
                [mock.call(3, 0), mock.call(3, 2)],
            )
            guard.assert_called_once_with(3)
            can_access.return_value = False
            module.reset_mock()
            with self.assertRaisesRegex(RuntimeError, "CPU KV staging"):
                IO.enable_peer_access(3, [0])
            module.enable_peer_access.assert_not_called()

    def test_peer_table_validation_rejects_host_and_bad_layout(self):
        for table, rows, record in (
            (MetadataTensor((2, 2, 3), device="cpu"), 10, 32),
            (MetadataTensor((2, 2, 3), dtype=torch.int32), 10, 32),
            (MetadataTensor((2, 2, 3), contiguous=False), 10, 32),
            (MetadataTensor((5, 2, 3)), 10, 32),
            (MetadataTensor((2, 3, 3)), 10, 32),
            (MetadataTensor((2, 2, 0)), 10, 32),
            (MetadataTensor((2, 2, 3)), 0, 32),
            (MetadataTensor((2, 2, 3)), 10, 30),
            (MetadataTensor((2, 2, 3)), True, 32),
            (MetadataTensor((2, 2, 3)), 10, 32.0),
        ):
            with (
                self.subTest(table=table.shape, rows=rows, record=record),
                self.assertRaises(ValueError),
            ):
                IO.validate_peer_table(table, rows, record)

    def test_transfer_dispatches_metadata_on_compute_and_preserves_bounds(self):
        peers, local = MetadataTensor((2, 2, 3)), MetadataTensor((2, 3))
        indices = MetadataTensor((4,))
        module = mock.Mock()
        with (
            mock.patch.object(IO, "_module", return_value=module),
            mock.patch.object(
                torch.cuda, "device", return_value=nullcontext()
            ) as guard,
        ):
            IO.transfer_peer_kv(
                peers,
                local,
                indices,
                indices,
                10,
                32,
                True,
                layer_id=2,
                local_rows=12,
                peer_rows=19,
            )
            guard.assert_called_once_with(torch.device("cuda:3"))
            module.transfer_peer_kv.assert_called_once_with(
                peers, local, indices, indices, 10, 32, True, 2, 12, 19
            )
            module.reset_mock()
            for kwargs in (
                {"local_rows": 0, "peer_rows": 19},
                {"local_rows": 12, "peer_rows": 21},
                {"local_rows": True, "peer_rows": 19},
                {"local_rows": 12, "peer_rows": 19.0},
            ):
                with self.assertRaisesRegex(ValueError, "row bounds"):
                    IO.transfer_peer_kv(
                        peers, local, indices, indices, 10, 32, True, **kwargs
                    )
            with self.assertRaisesRegex(ValueError, "compute GPU"):
                IO.transfer_peer_kv(
                    peers,
                    MetadataTensor((2, 3), device="cuda:0"),
                    indices,
                    indices,
                    10,
                    32,
                    True,
                    local_rows=12,
                    peer_rows=19,
                )
            module.transfer_peer_kv.assert_not_called()

    def resolve_args(self):
        return dict(
            top_k_tokens=MetadataTensor((2, 32), torch.int32),
            device_buffer_tokens=MetadataTensor((3, 65), torch.int32),
            mpu_cache_locs=MetadataTensor((3, 300)),
            device_buffer_locs=MetadataTensor((3, 65), torch.int32),
            peer_ptrs=MetadataTensor((2, 2, 3)),
            device_k_buffer=MetadataTensor((256, 2, 8), torch.bfloat16),
            device_v_buffer=MetadataTensor((256, 2, 8), torch.bfloat16),
            top_k_device_locs=MetadataTensor((2, 32), torch.int32),
            req_pool_indices=MetadataTensor((2,)),
            seq_lens=MetadataTensor((2,), torch.int32),
            lru_slots=MetadataTensor((3, 64), torch.int16),
            item_size_bytes=32,
            num_top_k=32,
            hot_buffer_size=64,
            shard_rows=200,
            layer_id=2,
            peer_rows=399,
            num_real_reqs=MetadataTensor((1,), torch.int32),
        )

    def test_resolve_uses_peer_export_and_compute_device(self):
        module = mock.Mock()
        args = self.resolve_args()
        with (
            mock.patch.dict(sys.modules, {"sglang.jit_kernel.mpu_io": IO}),
            mock.patch.object(SPARSE, "_jit_sparse_mpu_module", return_value=module),
            mock.patch.object(SPARSE, "_jit_sparse_module") as host_jit,
            mock.patch.object(
                torch.cuda, "device", return_value=nullcontext()
            ) as guard,
        ):
            SPARSE.load_cache_from_mpu_to_device_buffer_mha(**args)
            host_jit.assert_not_called()
            guard.assert_called_once_with(torch.device("cuda:3"))
            dispatched = module.load_cache_from_mpu_to_device_buffer.call_args.args
            self.assertIs(dispatched[4], args["peer_ptrs"])
            self.assertEqual(dispatched[-5:], (1, 32, 200, 2, 399))

    def test_resolve_rejects_incompatible_metadata_before_jit(self):
        with (
            mock.patch.dict(sys.modules, {"sglang.jit_kernel.mpu_io": IO}),
            mock.patch.object(SPARSE, "_jit_sparse_mpu_module") as loader,
        ):
            for name, value in (
                ("peer_rows", 401),
                ("layer_id", 3),
                ("block_size", 33),
                ("page_size", 16),
                ("page_size", True),
                ("num_top_k", 32.0),
                ("layer_id", True),
                ("seq_lens", MetadataTensor((2,), torch.float32)),
                (
                    "device_v_buffer",
                    MetadataTensor((256, 2, 8), torch.bfloat16, device="cuda:0"),
                ),
                ("device_k_buffer", MetadataTensor((256, 2, 4), torch.bfloat16)),
                ("device_k_buffer", MetadataTensor((256, 2, 8), torch.float16)),
            ):
                args = self.resolve_args()
                args[name] = value
                with self.subTest(name=name), self.assertRaises(ValueError):
                    SPARSE.load_cache_from_mpu_to_device_buffer_mha(**args)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
