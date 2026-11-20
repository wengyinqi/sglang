"""Validate diagnostic resolve counters, without inferring hardware KV traffic."""

FIELDS = (
    "resolve_calls",
    "selected_entries",
    "resident_hits",
    "newest_hits",
    "miss_entries",
)


def summarize_cache_metrics(rows, item_size_bytes):
    if type(item_size_bytes) is not int or item_size_bytes <= 0 or not rows:
        raise ValueError("Positive per-side KV stride and layer rows are required")
    per_layer = []
    for index, row in enumerate(rows):
        if len(row) != len(FIELDS) or any(
            type(value) is not int or value < 0 for value in row
        ):
            raise ValueError("Invalid nonnegative cache counters")
        values = dict(zip(FIELDS, row))
        if (
            values["selected_entries"]
            != values["resident_hits"] + values["newest_hits"] + values["miss_entries"]
        ):
            raise ValueError(
                "Selected entries are not conserved across hits and misses"
            )
        per_layer.append({"layer": index, **values})
    totals = {key: sum(row[key] for row in per_layer) for key in FIELDS}
    return {
        "schema_version": 1,
        "status": "diagnostic_counters",
        "scope": "Cumulative CPU-backed Qwen MHA resolve calls on real requests; includes warmup/capture calls when they execute; excludes padded CUDA Graph blocks, prefill backup and decode writeback",
        "short_sequence_policy": "Valid selections in ordered resident fast path count as resident hits",
        "totals": totals,
        "miss_fraction": (
            totals["miss_entries"] / totals["selected_entries"]
            if totals["selected_entries"]
            else None
        ),
        "logical_miss_kv_payload_bytes": totals["miss_entries"] * item_size_bytes * 2,
        "per_side_token_stride_bytes": item_size_bytes,
        "hardware_transaction_bytes_measured": False,
        "per_layer": per_layer,
        "interpretation": "Logical K+V payload requested by miss copies, not DDR transaction bytes, DMA bandwidth, full KV traffic or MPU speedup; instrumented diagnostics are separate from formal performance",
    }
