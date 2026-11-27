"""CPU-backed Quest-summary tier for the Qwen XM-Sparse adaptation.

The original complete BF16 summaries live in pinned CPU memory. Two fixed-address
layer buffers stage them in HBM. Memcpy graphs are GPU-initiated during model graph
replay; eager execution launches their GPU producer through a parent graph on an
independent stream. This is summary offload, not a trained K-index cache.
"""

from __future__ import annotations

import torch

from sglang.jit_kernel.xm_bounds_prefetch import (
    DeviceGraphCopy,
    checked,
    mapped_host_pointer,
    native_module,
)


class QuestBoundsTier:
    def __init__(self, shape, device, *, prefetch_mode: str, protect: bool):
        if prefetch_mode not in ("on_demand", "lookahead"):
            raise ValueError(
                "Enabled Quest summary offload needs on_demand or lookahead"
            )
        if len(shape) != 6 or shape[-1] != 2 or min(shape) <= 0:
            raise ValueError("Quest bounds shape must be positive [L,R,P,H,D,2]")
        self.device = torch.device(device)
        if (
            self.device.type != "cuda"
            or not torch.cuda.is_available()
            or torch.version.hip
        ):
            raise RuntimeError("Quest summary offload currently requires NVIDIA CUDA")
        self.device_index = (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )
        self.device = torch.device("cuda", self.device_index)
        self.layers = shape[0]
        if self.layers > 120:
            raise ValueError(
                "This device-graph adaptation supports at most 120 layer launches per model graph"
            )
        self.prefetch_mode = prefetch_mode
        self.protect = protect
        self.closed = False
        self.backing_bounds = torch.empty(
            shape, dtype=torch.bfloat16, device="cpu", pin_memory=True
        )
        self.backing_bounds[..., 0].fill_(torch.finfo(torch.bfloat16).max)
        self.backing_bounds[..., 1].fill_(torch.finfo(torch.bfloat16).min)
        with torch.cuda.device(self.device):
            self.backing_uva_ptr = torch.tensor(
                [mapped_host_pointer(self.backing_bounds, self.device_index)],
                dtype=torch.int64,
                device=self.device,
            )
            self.stage_bounds = torch.empty(
                (2, *shape[1:]), dtype=torch.bfloat16, device=self.device
            )
            self.ready_keys = torch.zeros(2, dtype=torch.int64, device=self.device)
            self.issued_keys = torch.zeros(2, dtype=torch.int64, device=self.device)
            self.step_epoch = torch.zeros(1, dtype=torch.int64, device=self.device)
            self.prefetch_gate = torch.zeros(1, dtype=torch.int32, device=self.device)
            self.counters = torch.zeros(8, dtype=torch.int64, device=self.device)
            self.prefetch_stream = torch.cuda.Stream(device=self.device)
            self.module = native_module(self.device_index)
            self.copy_graphs = [
                DeviceGraphCopy(self, layer) for layer in range(self.layers)
            ]
            # Upload and initialize all immutable resources before warmup/capture.
            torch.cuda.current_stream(self.device).synchronize()

    def begin_step(self):
        stream = torch.cuda.current_stream(self.device)
        checked(
            self.module,
            "xm_bounds_begin",
            self.step_epoch.data_ptr(),
            self.prefetch_gate.data_ptr(),
            self.device_index,
            stream.cuda_stream,
        )
        self.prefetch_layer(0)

    def prefetch_layer(self, layer: int):
        if not 0 <= layer < self.layers:
            raise ValueError("Quest prefetch layer out of bounds")
        current = torch.cuda.current_stream(self.device)
        if torch.cuda.is_current_stream_capturing():
            self.copy_graphs[layer].launch(current, captured=True)
        else:
            # Depend on summary writes and epoch publication, not CPU data staging.
            self.prefetch_stream.wait_stream(current)
            self.copy_graphs[layer].launch(self.prefetch_stream, captured=False)

    def read_layer(self, layer: int):
        self.prefetch_layer(layer)
        current = torch.cuda.current_stream(self.device)
        self.copy_graphs[layer].wait(current)
        if self.prefetch_mode == "lookahead" and layer + 1 < self.layers:
            self.prefetch_layer(layer + 1)
        return self.stage_bounds[layer % 2 : layer % 2 + 1]

    def set_protected(self, protected_operation: bool):
        if self.protect:
            stream = torch.cuda.current_stream(self.device)
            checked(
                self.module,
                "xm_bounds_protect",
                self.prefetch_gate.data_ptr(),
                int(protected_operation),
                self.device_index,
                stream.cuda_stream,
            )

    def finalize(self, req_indices, running_min, running_max, counts, pages, page_size):
        from .quest_bounds_tier_kernel import finalize_bounds

        finalize_bounds(
            self, req_indices, running_min, running_max, counts, pages, page_size
        )

    def invalidate_request(self, req_pool_idx: int):
        if not 0 <= req_pool_idx < self.backing_bounds.shape[1]:
            raise ValueError("Quest request slot out of bounds")
        from .quest_bounds_tier_kernel import invalidate_bounds

        invalidate_bounds(self, req_pool_idx)

    def statistics(self):
        """Call only at a diagnostic boundary; no fabricated transfer counters."""
        torch.cuda.synchronize(self.device)
        values = self.counters.cpu().tolist()
        names = (
            "gpu_copy_launches",
            "completed_layer_copies",
            "completed_bytes",
            "completed_segments",
            "checkpoint_wait_cycles",
            "device_launch_failures",
            "host_parent_launches",
            "wait_failures",
        )
        return {
            "mechanism": "gpu_launched_cuda_graph_memcpy",
            "priority_mechanism": "gpu_kernel_checkpoints"
            if self.protect
            else "disabled",
            "prefetch_mode": self.prefetch_mode,
            "summary_kind": "quest_bf16_min_max",
            "copies_preallocated_request_and_context_rows": True,
            "backing_bytes": self.backing_bounds.numel()
            * self.backing_bounds.element_size(),
            "staging_bytes": self.stage_bounds.numel()
            * self.stage_bounds.element_size(),
            "native_graph_count": 2 * self.layers,
            "native_source_sha256": self.module._xm_source_sha256,
            "counters": dict(zip(names, values)),
        }

    def close(self):
        if not self.closed:
            torch.cuda.synchronize(self.device)
            for graph in self.copy_graphs:
                graph.close()
            self.closed = True
