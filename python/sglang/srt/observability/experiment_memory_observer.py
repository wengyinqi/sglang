"""Opt-in, synchronizing memory diagnostics; never formal performance timing.

Snapshots inspect tensor metadata and the native allocator. CUDA synchronization,
allocator-history recording and NVML sampling are enabled only when an experiment
sets SGLANG_EXPERIMENT_MEMORY_DIR. No KV values or allocator addresses are saved.
"""

import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace


def runner_rank(runner):
    rank = getattr(runner, "tp_rank", None)
    if rank is None:
        from sglang.srt.runtime_context import get_parallel

        rank = get_parallel().tp_rank
    return int(rank)


def runner_layer_count(runner):
    layout = getattr(runner, "layer_info", runner)
    return int(layout.end_layer - layout.start_layer)


def combined_graph_registry(runner):
    """Inspect actual full/breakable graph objects across both phase backends.

    CUDA graphs and segments can be shared; deduplicate object identity and
    preserve unavailable wrappers instead of reporting unknown pools as zero.
    """
    graphs, seen, unknown = {}, set(), []

    def visit(value, label):
        if value is None or id(value) in seen:
            return
        seen.add(id(value))
        if callable(getattr(value, "pool", None)):
            graphs[label] = value
            return
        if getattr(value, "original_graph", None) is not None:
            visit(value.original_graph, label + ".original_graph")
            return
        segments = getattr(value, "_segments", None)
        if isinstance(segments, (tuple, list)):
            for index, segment in enumerate(segments):
                visit(segment, f"{label}.segment{index}")
            return
        if getattr(value, "graph", None) is not None:
            visit(value.graph, label + ".graph")
            return
        unknown.append(label)

    phase_counts = {}
    for name in (
        "graph_runner",
        "decode_cuda_graph_runner",
        "prefill_cuda_graph_runner",
    ):
        phase = getattr(runner, name, None)
        if phase is None:
            continue
        owners = [(name, phase)]
        backend = getattr(phase, "backend", None)
        if backend is not None:
            owners.append((name + ".backend", backend))
        for label, owner in owners:
            for attribute in ("graphs", "_graphs"):
                table = getattr(owner, attribute, None)
                if isinstance(table, dict):
                    phase_counts[label + "." + attribute] = len(table)
                    for index, value in enumerate(table.values()):
                        visit(value, f"{label}.{attribute}[{index}]")
    return SimpleNamespace(
        graphs=graphs, unknown_graph_objects=unknown, phase_registry_counts=phase_counts
    )


def configured_decode_graph(runner):
    config = getattr(runner.server_args, "cuda_graph_config", None)
    if config is not None:
        config = json.loads(config) if isinstance(config, str) else config
        decode = (
            config.get("decode")
            if isinstance(config, dict)
            else getattr(config, "decode", None)
        )
        backend = (
            decode.get("backend")
            if isinstance(decode, dict)
            else getattr(decode, "backend", None)
        )
        if backend is not None:
            return str(getattr(backend, "value", backend)).lower() != "disabled"
    return not runner.server_args.disable_cuda_graph


def graph_pool_accounting(snapshot, graph_runner, device=None):
    """Measure recognized graph pool segments; reserved and allocated overlap."""
    graphs = getattr(graph_runner, "graphs", {}) if graph_runner is not None else {}
    if not isinstance(graphs, dict):
        return {"status": "unavailable_graph_registry", "reserved_bytes": None}
    pools = {tuple(graph.pool()) for graph in graphs.values()}
    segments = snapshot.get("segments")
    if not isinstance(segments, list):
        return {"status": "unavailable_allocator_segments", "reserved_bytes": None}
    selected = [
        segment
        for segment in segments
        if tuple(segment.get("segment_pool_id", ())) in pools
        and (device is None or segment.get("device") == device)
    ]
    return {
        "status": (
            "partial_unknown_graph_objects"
            if getattr(graph_runner, "unknown_graph_objects", [])
            else "measured_native_allocator_segments"
        ),
        "pool_ids": [list(pool) for pool in sorted(pools)],
        "graph_count": len(graphs),
        "segment_count": len(selected),
        "device": device,
        "unknown_graph_objects": getattr(graph_runner, "unknown_graph_objects", []),
        "phase_registry_counts": getattr(graph_runner, "phase_registry_counts", {}),
        "reserved_bytes": (
            None
            if getattr(graph_runner, "unknown_graph_objects", [])
            else sum(segment["total_size"] for segment in selected)
        ),
        "allocated_bytes": (
            None
            if getattr(graph_runner, "unknown_graph_objects", [])
            else sum(segment["allocated_size"] for segment in selected)
        ),
        "active_bytes": (
            None
            if getattr(graph_runner, "unknown_graph_objects", [])
            else sum(segment["active_size"] for segment in selected)
        ),
        "known_pool_reserved_bytes": sum(segment["total_size"] for segment in selected),
        "scope": "Segments matching actual CUDAGraph.pool(); contained in allocator totals, not additive",
    }


def allocation_callsite(frames):
    """Python allocation attribution, not CUDA-kernel or transfer attribution."""
    for frame in frames:
        filename = str(frame.get("filename", ""))
        if any(
            name in filename
            for name in (
                "quest_algorithm.py",
                "quest_score_kernel.py",
                "flashinfer_quest_backend.py",
            )
        ):
            return "quest_score_topk_bounds_code"
    for frame in frames:
        filename = str(frame.get("filename", ""))
        if "flashinfer" in filename or "attention" in filename:
            return "attention_backend_code"
    return "other_or_unattributed"


def transient_allocation_accounting(events, start_index):
    """Peak bytes of allocations both born and logically freed in this phase.

    free_requested ends tensor allocation lifetime; free_completed ends physical
    reuse waiting. Phase-crossing allocations and graph-retained scratch are not
    included in this subset. Callsite category maxima cannot be added together.
    """
    live, lifetimes = {}, []
    for index, event in enumerate(events):
        action, address = event.get("action"), event.get("addr")
        if action == "alloc":
            item = {
                "id": len(lifetimes),
                "size": event["size"],
                "start": index,
                "end": None,
                "category": allocation_callsite(event.get("frames", [])),
            }
            live[address] = item
            lifetimes.append(item)
        elif action == "free_requested" and address in live:
            live.pop(address)["end"] = index
    selected = {
        item["id"]: item
        for item in lifetimes
        if item["start"] >= start_index and item["end"] is not None
    }
    changes = []
    for item in selected.values():
        changes.extend(
            [
                (item["start"], item["category"], item["size"]),
                (item["end"], item["category"], -item["size"]),
            ]
        )
    current, peak = defaultdict(int), defaultdict(int)
    combined, combined_peak = 0, 0
    for _, category, delta in sorted(changes):
        current[category] += delta
        combined += delta
        peak[category] = max(peak[category], current[category])
        combined_peak = max(combined_peak, combined)
    return {
        "status": "measured_phase_local_transient_subset",
        "phase_local_allocation_count": len(selected),
        "peak_live_bytes": combined_peak,
        "peak_live_bytes_by_python_callsite": dict(peak),
        "phase_crossing_or_retained_allocation_count": sum(
            item["start"] >= start_index and item["end"] is None for item in lifetimes
        ),
        "scope": "Only recorded allocations born and free_requested within this phase; graph-retained and cross-phase scratch excluded",
        "category_peaks_additive": False,
        "raw_addresses_emitted": False,
    }


class ExperimentMemoryObserver:
    """Native allocator phase windows on the model-runner process."""

    @classmethod
    def create(cls, runner):
        directory = os.environ.get("SGLANG_EXPERIMENT_MEMORY_DIR")
        if not directory:
            return None
        if runner.device != "cuda":
            raise ValueError("Experiment memory diagnostics require the CUDA allocator")
        if not runner.server_args.disable_overlap_schedule:
            raise ValueError("Memory phase diagnostics require non-overlap scheduling")
        return cls(runner, directory)

    def __init__(self, runner, directory):
        import torch

        self.runner, self.torch = runner, torch
        self.device = runner.gpu_id
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.rank = runner_rank(runner)
        self.path = directory / f"memory-rank{self.rank}.jsonl"
        self.context = json.loads(
            os.environ.get("SGLANG_EXPERIMENT_MEMORY_CONTEXT", "{}")
        )
        if self.context.get("family") not in {
            "qwen",
            "deepseek-v4-flash",
            "glm-5.3-flash",
        }:
            raise ValueError("Memory context must declare the model family")
        uuid = str(torch.cuda.get_device_properties(self.device).uuid)
        self.uuid = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
        self.context.update(
            device_uuid=self.uuid,
            tp_rank=self.rank,
            gpu_id=self.device,
            diagnostic_only=True,
            eligible_for_formal_performance=False,
        )
        self._sequence, self._phase, self._steps = 0, None, 0
        self._start_index, self._phase_started = 0, None
        self._nvml_samples, self._nvml_error = [], None
        self._sample_lock, self._stop = threading.Lock(), threading.Event()
        self._nvml = self._nvml_handle = None
        self.decode_samples = {
            int(value)
            for value in os.environ.get(
                "SGLANG_EXPERIMENT_MEMORY_DECODE_SAMPLES", "1,8,64,127"
            ).split(",")
        }
        torch.cuda.memory._record_memory_history(
            enabled="all",
            context="all",
            stacks="python",
            max_entries=500000,
            clear_history=True,
        )
        with self.path.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "type": "memory_observer",
                        "schema_version": 1,
                        "context": self.context,
                        "history_start_scope": "after CUDA distributed initialization, before model load",
                    }
                )
                + "\n"
            )
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml, self._nvml_handle = (
                pynvml,
                pynvml.nvmlDeviceGetHandleByUUID(self.uuid),
            )
        except Exception as error:
            self._nvml_error = type(error).__name__ + ": " + str(error)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        self.begin_phase("initialization")

    def _sample(self):
        if self._nvml is None:
            return
        try:
            memory = self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            sample = {
                "timestamp_seconds": time.time(),
                "used_bytes": int(memory.used),
                "device_uuid": self.uuid,
            }
            with self._sample_lock:
                self._nvml_samples.append(sample)
        except Exception as error:
            self._nvml_error = type(error).__name__ + ": " + str(error)

    def _sample_loop(self):
        while not self._stop.wait(0.025):
            self._sample()

    def _events(self, snapshot):
        traces = snapshot.get("device_traces", [])
        return traces[self.device] if len(traces) > self.device else []

    def begin_phase(self, phase, metadata=None):
        if self._phase is not None:
            self.checkpoint(final=True)
        self.torch.cuda.synchronize(self.device)
        snapshot = self.torch.cuda.memory._snapshot()
        self._start_index = len(self._events(snapshot))
        self._phase, self._steps, self._phase_started = phase, 0, time.time()
        self._metadata = metadata or {}
        self.torch.cuda.reset_peak_memory_stats(self.device)
        self._sample()

    def before_forward(self, batch):
        phase = (
            "decode"
            if batch.forward_mode.is_decode()
            else "prefill" if batch.forward_mode.is_extend() else "idle"
        )
        lengths = getattr(batch, "seq_lens_cpu", None)
        if lengths is None:
            gpu_lengths = getattr(batch, "seq_lens", None)
            lengths = gpu_lengths.detach().cpu() if gpu_lengths is not None else []
        lengths = lengths.tolist() if hasattr(lengths, "tolist") else list(lengths)
        metadata = {
            "actual_batch": int(batch.batch_size),
            "seq_lens": [int(value) for value in lengths],
            "live_history_tokens": sum(lengths) or None,
            "forward_mode": batch.forward_mode.name.lower(),
        }
        if self._phase != phase:
            self.begin_phase(phase, metadata)
        else:
            self._metadata = metadata

    def after_forward(self, output):
        self.torch.cuda.synchronize(self.device)
        self._steps += 1
        self._metadata["actual_cuda_graph"] = bool(output.can_run_graph)
        if self._phase == "decode" and self._steps in self.decode_samples:
            self.checkpoint(final=False)

    def checkpoint(self, final=False):
        from sglang.srt.observability.experiment_memory import (
            allocator_metrics_from_cuda,
            capture_memory_snapshot,
            collect_model_logical_regions,
            collect_model_tensor_trees,
        )

        self.torch.cuda.synchronize(self.device)
        self._sample()
        end = time.time()
        snapshot = self.torch.cuda.memory._snapshot()
        allocator = allocator_metrics_from_cuda(
            self.torch.cuda,
            self.device,
            peak_scope="reset_at_phase_start_through_this_snapshot",
        )
        runner = self.runner
        graphs = combined_graph_registry(runner)
        trees = collect_model_tensor_trees(
            self.context["family"],
            model_runner=runner,
            coordinator=getattr(runner, "hisparse_coordinator", None),
            kv_pool=getattr(runner, "token_to_kv_pool", None),
            allocator=getattr(runner, "token_to_kv_pool_allocator", None),
            request_pool=getattr(runner, "req_to_token_pool", None),
            backend=getattr(runner, "attn_backend", None),
            graph_runner=getattr(
                runner,
                "decode_cuda_graph_runner",
                getattr(runner, "graph_runner", None),
            ),
        )
        with self._sample_lock:
            nvml = [
                sample
                for sample in self._nvml_samples
                if self._phase_started <= sample["timestamp_seconds"] <= end
            ]
        context = {
            **self.context,
            **self._metadata,
            "rank": self.rank,
            "world_size": runner.server_args.tp_size,
            "layers": runner_layer_count(runner),
            "preallocated_request_slots": int(
                runner.req_to_token_pool.req_to_token.shape[0]
            ),
            "max_context_len": int(runner.req_to_token_pool.max_context_len),
            "phase": self._phase,
            "phase_steps": self._steps,
            "snapshot_id": f"rank{self.rank}:{self._phase}:{self._sequence}",
            "phase_start_timestamp_seconds": self._phase_started,
            "snapshot_timestamp_seconds": end,
            "final_phase_snapshot": final,
            "cuda_graph_enabled": configured_decode_graph(runner),
            "cuda_graph": configured_decode_graph(runner),
            "nvml_poll_interval_seconds": 0.025,
            "nvml_error": self._nvml_error,
        }
        regions = collect_model_logical_regions(
            self.context["family"], model_runner=runner
        )
        record = capture_memory_snapshot(
            context,
            trees,
            allocator_metrics=allocator,
            regions=regions,
            nvml_samples=nvml,
            graph_pool_summary=graph_pool_accounting(snapshot, graphs, self.device),
        )
        events = self._events(snapshot)
        record["allocator_history_transients"] = (
            {"status": "unavailable_history_truncation", "peak_live_bytes": None}
            if len(events) >= 500000
            else transient_allocation_accounting(events, self._start_index)
        )
        record["type"] = "memory_snapshot"
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        self._sequence += 1
        if final:
            self._phase = None

    def finish_idle(self):
        """Close a completed request wave only after scheduler resource release."""
        if self._phase not in {"prefill", "decode"}:
            return
        self.checkpoint(final=True)
        self.begin_phase(
            "idle", {"actual_batch": 0, "seq_lens": [], "live_history_tokens": 0}
        )
        self.checkpoint(final=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "type": "memory_wave_complete",
                        "context": self.context,
                        "snapshot_count": self._sequence,
                    }
                )
                + "\n"
            )
