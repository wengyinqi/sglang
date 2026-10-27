"""Summarize real Nsight SQLite CUDA activity with an explicit device UUID map.

Memcpy bytes are observed CUDA copy calls, not total KV traffic. Direct P2P or
mapped-host load/store transactions are not represented by memcpy rows.
"""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def device_map(mapping, profile):
    if mapping.get("schema_version") != 1 or not mapping.get("mapping_source"):
        raise ValueError(
            "Device mapping needs schema_version=1 and an explicit mapping_source"
        )
    entries = mapping.get("devices", [])
    expected = {(d["uuid"], d["role"]) for d in profile.get("devices", [])}
    actual = {(d["uuid"], d["role"]) for d in entries}
    ids = [d["trace_device_id"] for d in entries]
    if (
        not expected
        or expected != actual
        or len(actual) != len(entries)
        or len(set(ids)) != len(ids)
        or any(type(i) is not int or i < 0 for i in ids)
        or any(role not in ("compute", "storage") for _, role in actual)
    ):
        raise ValueError(
            "Trace device IDs must map bijectively to captured CUDA UUIDs and roles"
        )
    return {d["trace_device_id"]: d for d in entries}


def analyze(database, mapping, profile):
    if profile.get("status") != "captured_pending_role_analysis":
        raise ValueError("Require an actual completed profiler capture")
    matches = [
        a for a in profile.get("artifacts", []) if Path(a["path"]).suffix == ".sqlite"
    ]
    if len(matches) != 1 or sha256(database) != matches[0]["sha256"]:
        raise ValueError("SQLite content differs from the captured artifact")
    roles = device_map(mapping, profile)
    # URI read-only mode works on Windows too; never execute schema from a trace.
    conn = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        kernel = "CUPTI_ACTIVITY_KIND_KERNEL"
        if kernel not in tables:
            return {
                "status": "blocked",
                "reason": "No CUDA kernel table; empty traces cannot prove storage inactivity",
            }
        columns = {
            r[1]
            for r in conn.execute('PRAGMA table_info("CUPTI_ACTIVITY_KIND_KERNEL")')
        }
        if not {"deviceId", "start", "end"} <= columns:
            return {
                "status": "blocked",
                "reason": "Unsupported Nsight kernel schema",
                "columns": sorted(columns),
            }
        if conn.execute(
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end < start"
        ).fetchone()[0]:
            raise ValueError("Invalid negative kernel duration")
        activity = {
            r[0]: {"kernels": r[1], "summed_kernel_ns": r[2]}
            for r in conn.execute(
                "SELECT deviceId,COUNT(*),SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY deviceId"
            )
        }
        unmapped = sorted(set(activity) - roles.keys())
        if unmapped:
            return {
                "status": "blocked",
                "reason": "Unmapped active CUDA device IDs",
                "unmapped_devices": unmapped,
            }
        per_device = [
            {**entry, **activity.get(i, {"kernels": 0, "summed_kernel_ns": 0})}
            for i, entry in sorted(roles.items())
        ]
        compute = [d for d in per_device if d["role"] == "compute"]
        storage = [d for d in per_device if d["role"] == "storage"]
        if not compute or any(d["kernels"] == 0 for d in compute):
            status, storage_check = "blocked", "incomplete_compute_rank_coverage"
        elif any(d["kernels"] for d in storage):
            status, storage_check = "failed", "storage_kernels_observed"
        else:
            status, storage_check = "analyzed", (
                "no_storage_kernels_observed"
                if storage
                else "no_storage_device_in_this_stage"
            )
        copies, copy_status = [], "table_absent_not_zero_traffic"
        kernel_names = []
        if "demangledName" in columns and "StringIds" in tables:
            kernel_names = [
                {
                    "trace_device_id": row[0],
                    "name": row[1],
                    "calls": row[2],
                    "summed_kernel_ns": row[3],
                }
                for row in conn.execute(
                    "SELECT k.deviceId,s.value,COUNT(*),SUM(k.end-k.start) "
                    "FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON k.demangledName=s.id "
                    "GROUP BY k.deviceId,k.demangledName ORDER BY SUM(k.end-k.start) DESC LIMIT 40"
                )
            ]
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
            cols = {
                r[1]
                for r in conn.execute('PRAGMA table_info("CUPTI_ACTIVITY_KIND_MEMCPY")')
            }
            if {"copyKind", "bytes", "start", "end"} <= cols:
                # Preserve CUDA's raw copyKind enum. No host/peer direction is
                # inferred from a version-dependent integer enum value.
                copies = [
                    {
                        "copy_kind": r[0],
                        "calls": r[1],
                        "bytes": r[2],
                        "summed_copy_ns": r[3],
                    }
                    for r in conn.execute(
                        "SELECT copyKind,COUNT(*),SUM(bytes),SUM(end-start) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind"
                    )
                ]
                copy_status = "observed_cuda_copy_calls_only"
                if "ENUM_CUDA_MEMCPY_OPER" in tables:
                    enum_cols = {
                        r[1]
                        for r in conn.execute(
                            'PRAGMA table_info("ENUM_CUDA_MEMCPY_OPER")'
                        )
                    }
                    if {"id", "label"} <= enum_cols:
                        labels = dict(
                            conn.execute("SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER")
                        )
                        for item in copies:
                            item["copy_kind_label"] = labels.get(item["copy_kind"])
            else:
                copy_status = "unsupported_copy_schema"
        result = {
            "schema_version": 1,
            "status": status,
            "storage_kernel_check": storage_check,
            "devices": per_device,
            "top_kernel_names_by_summed_duration": kernel_names,
            "memcpy_status": copy_status,
            "memcpy": copies,
            "sqlite_sha256": matches[0]["sha256"],
            "device_mapping": mapping,
            "mapping_verified_automatically": False,
            "actual_decode_steps_verified": False,
            "total_kv_traffic_measured": False,
            "host_kv_payload_absence_proven": False,
            "limitations": [
                "Role conclusions are conditional on the operator-supplied trace ID to UUID mapping.",
                "Inspect raw timeline for all rank/stream coverage and decode boundaries before accepting the hardware run.",
                "Summed kernel/copy durations overlap and are not elapsed time or critical path.",
                "CUDA memcpy counts omit direct P2P and mapped-host loads/stores; zero host memcpy does not prove zero CPU KV payload traffic.",
                "Use hardware transaction counters or an address-aware trace for KV byte attribution; CPU metadata copies are not necessarily KV payload.",
            ],
        }
        return result
    finally:
        conn.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--device-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = analyze(
        args.sqlite,
        json.loads(args.device_map.read_text(encoding="utf-8")),
        json.loads(args.profile.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(args.output)
    return 0 if report["status"] == "analyzed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
