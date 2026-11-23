"""Summarize captured CUDA kernel intervals without equating sums to wall time."""

import argparse
import importlib.util
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"bottleneck_{name}", Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PATTERNS = {
    "kv_resolve_and_pack": (
        "load_cache_to_device_buffer",
        "load_cache_from_mpu_to_device_buffer",
        "transfer_kv_peer",
        "mpu_transfer_kernel",
        "quest_scatter_pack",
        "quest_only_gather_scatter",
        "transfer_cache",
        "hisparse_resolve",
    ),
    "quest_selection_and_bounds": (
        "quest_score",
        "quest_decode_bounds",
        "radixselect",
        "radix_select",
        "topk",
        "top_k",
    ),
    "attention": (
        "batchdecode",
        "batch_decode",
        "flash_fwd",
        "flashinfer::decode",
        "fmha",
        "attention",
    ),
    "moe_and_matrix_compute": (
        "gemm",
        "gemv",
        "fused_moe",
        "moe_align",
        "moe_sum",
        "moe::",
        "grouped_gemm",
    ),
    "normalization_and_rope": (
        "rmsnorm",
        "rms_norm",
        "layernorm",
        "rotary",
        "rope",
        "fused_add_rms",
    ),
}


def category(name):
    lowered = name.lower()
    for key, patterns in PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            return key
    return "unclassified"


def union_ns(intervals):
    total, left, right = 0, None, None
    for start, end in sorted(intervals):
        if left is None:
            left, right = start, end
        elif start > right:
            total += right - left
            left, right = start, end
        else:
            right = max(right, end)
    return total + (right - left if left is not None else 0)


def analyze(sqlite_path, profile, mapping):
    role = load("profile_analyze").analyze(sqlite_path, mapping, profile)
    if role["status"] != "analyzed":
        raise ValueError(
            "CUDA device-role analysis must pass before bottleneck reporting"
        )
    uri = Path(sqlite_path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        names = {
            row[1]
            for row in conn.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")
        }
        if not {"demangledName", "deviceId", "start", "end"} <= names:
            raise ValueError("Nsight kernel names and timestamps are required")
        records = list(
            conn.execute(
                "SELECT k.deviceId,k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "JOIN StringIds s ON k.demangledName=s.id"
            )
        )
    if not records:
        raise ValueError("No named CUDA kernel intervals")
    grouped = defaultdict(lambda: {"calls": 0, "summed_ns": 0})
    named = defaultdict(lambda: {"calls": 0, "summed_ns": 0})
    by_device = defaultdict(list)
    for device, start, end, name in records:
        if end < start:
            raise ValueError("Negative CUDA kernel duration")
        key = category(name)
        grouped[key]["calls"] += 1
        grouped[key]["summed_ns"] += end - start
        named[name]["calls"] += 1
        named[name]["summed_ns"] += end - start
        by_device[device].append((start, end))
    summed = sum(row["summed_ns"] for row in grouped.values())
    report = {
        "schema_version": 1,
        "status": "analyzed",
        "role_analysis": role,
        "profile_variant": profile.get("profile_variant", profile["stage"]["variant"]),
        "workload": profile["workload"],
        "categories": [
            {
                "category": key,
                **value,
                "fraction_of_summed_kernel_time": value["summed_ns"] / summed,
            }
            for key, value in sorted(
                grouped.items(), key=lambda item: -item[1]["summed_ns"]
            )
        ],
        "device_intervals": [
            {
                "device_id": device,
                "kernel_union_ns": union_ns(intervals),
                "first_kernel_ns": min(s for s, _ in intervals),
                "last_kernel_ns": max(e for _, e in intervals),
            }
            for device, intervals in sorted(by_device.items())
        ],
        "top_kernels": [
            {"name": name, "category": category(name), **value}
            for name, value in sorted(
                named.items(), key=lambda item: -item[1]["summed_ns"]
            )[:30]
        ],
        "classification_patterns": PATTERNS,
        "total_kv_traffic_measured": False,
        "notes": [
            "Categories are explicit kernel-name heuristics; unclassified kernels remain visible.",
            "Category fractions use summed kernel time, not wall time or TPOT; intervals can overlap.",
            "Mapped-host/P2P loads are not represented by CUDA memcpy totals; cache-hit rates and KV transaction bytes are not inferred.",
            "The upstream decode trigger bounds capture; inspect actual NVTX/CUDA ranges for capture boundaries.",
        ],
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--device-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(
        args.sqlite,
        json.loads(args.profile.read_text()),
        json.loads(args.device_map.read_text()),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
