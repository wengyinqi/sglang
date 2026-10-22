"""Opt-in batch CUDA timing for single-compute-GPU HiSparse experiments.

CUDA events are recorded on the stream executing forward, including graph
replay. No device synchronization or file writes occur in the forward wrapper.
Completed records are written when the scheduler becomes fully idle. Request
IDs allow an analyzer to exclude warmup and health-check traffic.
"""

import json
import os
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path


class HiSparseForwardTimer:
    def __init__(self, directory, metadata, *, event_factory=None, stream_getter=None):
        if event_factory is None or stream_getter is None:
            import torch

            def event_factory():
                return torch.cuda.Event(enable_timing=True)

            stream_getter = torch.cuda.current_stream
        self._event_factory = event_factory
        self._stream_getter = stream_getter
        self._pending = deque()
        self._completed = []
        self._sequence = 0
        self._last_written_sequence = -1
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / (
            f"forward-tp{metadata['tp_rank']}-pp{metadata['pp_rank']}"
            f"-dp{metadata['dp_rank']}-pid{os.getpid()}.jsonl"
        )
        # A trace must belong to exactly one scheduler lifetime.
        with self.path.open("x", encoding="utf-8") as output:
            output.write(
                json.dumps({"type": "session", "schema_version": 1, **metadata}) + "\n"
            )

    @classmethod
    def create(cls, directory, server_args, tp_rank, pp_rank, dp_rank):
        if not directory:
            return None
        metadata = {
            "tp_rank": tp_rank,
            "pp_rank": pp_rank,
            "dp_rank": dp_rank,
            "tp_size": server_args.tp_size,
            "pp_size": server_args.pp_size,
            "dp_size": server_args.dp_size,
            "disaggregation_mode": server_args.disaggregation_mode,
            "enable_mixed_chunk": server_args.enable_mixed_chunk,
            "enable_two_batch_overlap": server_args.enable_two_batch_overlap,
            "speculative_algorithm": server_args.speculative_algorithm,
            "disable_overlap_schedule": server_args.disable_overlap_schedule,
            "disable_radix_cache": server_args.disable_radix_cache,
        }
        return cls(directory, metadata)

    @contextmanager
    def wrap(self, batch):
        stream = self._stream_getter()
        metadata = {
            "type": "forward",
            "sequence": self._sequence,
            "forward_mode": batch.forward_mode.name.lower(),
            "request_ids": [req.rid for req in batch.reqs],
            "extend_lens": (
                list(batch.extend_lens) if batch.forward_mode.name == "EXTEND" else None
            ),
            "stream_id": stream.cuda_stream,
            "launch_monotonic_ns": time.monotonic_ns(),
            "success": False,
        }
        self._sequence += 1
        start, end = self._event_factory(), self._event_factory()
        start.record(stream)
        try:
            yield
            metadata["success"] = True
        finally:
            end.record(stream)
            self._pending.append((start, end, metadata))
            self._collect(wait=False)

    def _collect(self, *, wait):
        if wait and self._pending:
            # Only called at full scheduler idle, never on the active path.
            for _, end, _ in self._pending:
                end.synchronize()
        while self._pending:
            start, end, metadata = self._pending[0]
            if not end.query():
                break
            self._pending.popleft()
            metadata["cuda_seconds"] = start.elapsed_time(end) / 1000.0
            self._completed.append(metadata)

    def flush_idle(self):
        """Drain the final event and publish an explicit completeness marker."""
        self._collect(wait=True)
        if not self._completed:
            return
        with self.path.open("a", encoding="utf-8") as output:
            for record in self._completed:
                output.write(json.dumps(record) + "\n")
            self._last_written_sequence = self._completed[-1]["sequence"]
            output.write(
                json.dumps(
                    {
                        "type": "idle_flush",
                        "last_sequence": self._last_written_sequence,
                    }
                )
                + "\n"
            )
        self._completed.clear()
