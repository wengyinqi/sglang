"""Validate and compare opt-in memory evidence without importing CUDA runtime.

Named live storage, allocator counters, graph pools and NVML have overlapping
scopes. The comparison reports each independently. Unknown owners/counters stay
null, and indexer content regions are never added to physical allocation bytes.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

CONTRACT = "experiment_memory_v1"
COMPARISON_CONTROLS = (
    "model_revision",
    "weight_format",
    "parallelism",
    "context_len",
    "request_slots",
    "actual_batch",
    "request_set_fingerprint",
    "output_len",
    "cuda_graph",
)
INDEXER_CATEGORIES = (
    "indexer_k_vectors",
    "indexer_k_scales",
    "indexer_mapping_lru",
    "indexer_workspace",
    "indexer_k_tail",
    "indexer_score_tail",
    "indexer_compression_state",
)
INDEXER_COUNTERS = (
    "lookups",
    "hits",
    "misses",
    "evictions",
    "bytes_read_from_backing",
    "bytes_written_to_backing",
)


def _integer(value, name, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(
            f"{name} must be a {'positive' if positive else 'nonnegative'} integer"
        )
    return value


def validate_snapshot(snapshot):
    """Validate structural accounting; this does not establish owner completeness."""
    if not isinstance(snapshot, dict) or snapshot.get("contract") != CONTRACT:
        raise ValueError("Unknown experiment memory contract")
    if snapshot.get("schema_version") != 1:
        raise ValueError("Unknown experiment memory schema")
    json.dumps(snapshot, allow_nan=False)
    context = snapshot.get("context", {})
    if (
        not isinstance(context.get("snapshot_id"), str)
        or not context["snapshot_id"]
        or not isinstance(context.get("phase"), str)
        or not context["phase"]
    ):
        raise ValueError("Snapshot identity and phase are required")
    tensors, storages = snapshot.get("tensors"), snapshot.get("storages")
    if not isinstance(tensors, dict) or not isinstance(storages, list):
        raise ValueError("Tensor and storage inventories are required")
    allocations, amount, by_device, expected_inclusive, expected_exclusive = (
        {},
        0,
        {},
        {},
        {},
    )
    for storage in storages:
        identity = _integer(storage.get("allocation_group"), "allocation_group")
        if identity in allocations:
            raise ValueError(
                "Storage allocation_group is unique only within one snapshot"
            )
        allocations[identity] = storage
        size = _integer(storage.get("storage_bytes"), "storage_bytes")
        amount += size
        device = storage.get("device")
        if not isinstance(device, str) or not device:
            raise ValueError("A physical storage device is required")
        by_device[device] = by_device.get(device, 0) + size
        aliases = storage.get("aliases")
        if (
            not isinstance(aliases, list)
            or not aliases
            or len(set(aliases)) != len(aliases)
        ):
            raise ValueError("Every storage must have distinct tensor aliases")
        for name in aliases:
            tensor = tensors.get(name, {})
            if (
                tensor.get("allocation_group") != identity
                or tensor.get("device") != device
                or tensor.get("storage_bytes") != size
            ):
                raise ValueError("Tensor/storage alias accounting differs")
        owners = sorted({tensors[name].get("component") for name in aliases})
        if (
            not owners
            or any(not isinstance(owner, str) or not owner for owner in owners)
            or storage.get("components") != owners
        ):
            raise ValueError("Storage owner components differ from aliases")
        category = owners[0] if len(owners) == 1 else "shared_components"
        if storage.get("exclusive_component") != category:
            raise ValueError("Storage exclusive/shared category differs")
        expected_exclusive[category] = expected_exclusive.get(category, 0) + size
        for owner in owners:
            expected_inclusive[owner] = expected_inclusive.get(owner, 0) + size
    for name, tensor in tensors.items():
        storage = allocations.get(tensor.get("allocation_group"))
        if storage is None or name not in storage["aliases"]:
            raise ValueError("Tensor must refer to its snapshot-local storage alias")
    if (
        snapshot.get("named_storage_bytes") != amount
        or snapshot.get("named_storage_bytes_by_device") != by_device
    ):
        raise ValueError("Named physical storage totals differ")
    exclusive = snapshot.get("exclusive_component_bytes", {})
    for name, value in exclusive.items():
        _integer(value, "exclusive_component_bytes." + name)
    if sum(exclusive.values()) != amount:
        raise ValueError(
            "Exclusive/shared component partition must sum to unique storage"
        )
    if (
        exclusive != expected_exclusive
        or snapshot.get("inclusive_component_bytes") != expected_inclusive
    ):
        raise ValueError("Component totals differ from actual storage owners")
    for name, value in snapshot.get("inclusive_component_bytes", {}).items():
        _integer(value, "inclusive_component_bytes." + name)
    regions = snapshot.get("logical_regions", [])
    if not isinstance(regions, list):
        raise ValueError("Logical regions must be a list")
    region_ids, geometry_intervals = set(), {}
    for region in regions:
        identity = region.get("region_id")
        if not identity or identity in region_ids:
            raise ValueError("Region IDs must be distinct")
        region_ids.add(identity)
        tensor = tensors.get(region.get("tensor_name"), {})
        if (
            region.get("allocation_group") != tensor.get("allocation_group")
            or region.get("device") != tensor.get("device")
            or region.get("physical_storage_bytes") != tensor.get("storage_bytes")
            or region.get("adds_physical_allocation") is not False
        ):
            raise ValueError("Logical content must bind to existing physical storage")
        _integer(region.get("logical_content_bytes"), "logical_content_bytes")
        if region["logical_content_bytes"] > tensor.get("logical_tensor_bytes", -1):
            raise ValueError("Logical region exceeds its tensor")
        geometry = region.get("geometry", {})
        if geometry.get("layout") == "whole_tensor":
            if region["logical_content_bytes"] != tensor["logical_tensor_bytes"]:
                raise ValueError("Whole-tensor content byte count differs")
            interval = (0, tensor["logical_tensor_bytes"])
        elif geometry.get("layout") == "page_keys_then_scales":
            rows = _integer(geometry.get("rows"), "rows")
            stride = _integer(
                geometry.get("row_stride_bytes"), "row_stride_bytes", positive=True
            )
            offset = _integer(
                geometry.get("region_offset_within_row_bytes"), "region_offset"
            )
            size = _integer(
                geometry.get("region_bytes_per_row"), "region_bytes_per_row"
            )
            shape = tensor.get("shape", [])
            if (
                len(shape) != 2
                or rows != shape[0]
                or stride != shape[1] * tensor.get("element_bytes", -1)
                or tensor.get("stride") != [shape[1], 1]
                or offset + size > stride
                or region["logical_content_bytes"] != rows * size
            ):
                raise ValueError("Logical packed content geometry differs")
            interval = (offset, offset + size)
        else:
            raise ValueError("Unknown logical content geometry")
        for previous in geometry_intervals.setdefault(region["tensor_name"], []):
            if max(interval[0], previous[0]) < min(interval[1], previous[1]):
                raise ValueError("Logical packed content regions overlap")
        geometry_intervals[region["tensor_name"]].append(interval)
    return snapshot


def _difference(left, right):
    return right - left if left is not None and right is not None else None


def compare_to_dense(dense, candidate, *, controls, allow_source_difference=False):
    """Compare matched same-model observations and retain unmeasured components.

    ``controls`` maps each required field to {"dense": value, "candidate": value}.
    Values must be known and equal. Request set/actual batch matches are required;
    subtracting prefill time does not make admission and queueing equivalent.
    """
    validate_snapshot(dense)
    validate_snapshot(candidate)
    left, right = dense["context"], candidate["context"]
    if left.get("family") is None or left.get("family") != right.get("family"):
        raise ValueError("Memory mechanism comparison requires the same model family")
    if left.get("phase") != right.get("phase"):
        raise ValueError("Memory comparison must use the same capture phase")
    if left.get("rank") != right.get("rank") or left.get("world_size") != right.get(
        "world_size"
    ):
        raise ValueError("Memory comparison must preserve rank and world size")
    if left.get("device_uuid") != right.get("device_uuid") or not left.get(
        "device_uuid"
    ):
        raise ValueError("Memory comparison requires the same physical compute device")
    if not left.get("source_revision") or not right.get("source_revision"):
        raise ValueError(
            "Source revision is required for a controlled memory comparison"
        )
    if not allow_source_difference and left.get("source_revision") != right.get(
        "source_revision"
    ):
        raise ValueError(
            "Source revisions differ; explicitly record a controlled source difference"
        )
    if not isinstance(controls, dict):
        raise ValueError("Matched experiment controls are required")
    for name in COMPARISON_CONTROLS:
        pair = controls.get(name, {})
        if (
            not isinstance(pair, dict)
            or pair.get("dense") is None
            or pair.get("dense") == ""
            or pair.get("candidate") is None
            or pair["dense"] != pair["candidate"]
        ):
            raise ValueError(f"Missing or unequal memory comparison control: {name}")
    components = set(dense["inclusive_component_bytes"]) | set(
        candidate["inclusive_component_bytes"]
    )
    component_deltas = {
        name: {
            "dense_bytes": dense["inclusive_component_bytes"].get(name),
            "candidate_bytes": candidate["inclusive_component_bytes"].get(name),
            "net_delta_bytes": _difference(
                dense["inclusive_component_bytes"].get(name),
                candidate["inclusive_component_bytes"].get(name),
            ),
        }
        for name in sorted(components)
    }
    allocator = {
        name: _difference(
            dense.get("allocator", {}).get(name),
            candidate.get("allocator", {}).get(name),
        )
        for name in (
            "allocated_bytes",
            "reserved_bytes",
            "active_bytes",
            "inactive_split_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        )
    }
    peak_scope = dense.get("allocator_peak_scope")
    peak_scopes_matched = peak_scope == candidate.get(
        "allocator_peak_scope"
    ) and peak_scope not in {None, "unmeasured", "caller_must_define_reset_interval"}
    if not peak_scopes_matched:
        for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
            allocator[name] = None
    graph = {
        name: _difference(
            dense.get("graph_pools", {}).get(name),
            candidate.get("graph_pools", {}).get(name),
        )
        for name in (
            "graph_private_pool_reserved_bytes",
            "graph_private_pool_allocated_bytes",
        )
    }
    return {
        "schema_version": 1,
        "contract": "experiment_memory_comparison_v1",
        "family": left["family"],
        "phase": left["phase"],
        "dense_snapshot_id": left["snapshot_id"],
        "candidate_snapshot_id": right["snapshot_id"],
        "controls": copy.deepcopy(controls),
        "source_revisions": {
            "dense": left.get("source_revision"),
            "candidate": right.get("source_revision"),
        },
        "named_storage_net_delta_bytes": candidate["named_storage_bytes"]
        - dense["named_storage_bytes"],
        "inclusive_component_deltas": component_deltas,
        "allocator_deltas": allocator,
        "allocator_peak_scopes_matched": peak_scopes_matched,
        "graph_pool_deltas": graph,
        "nvml_sampled_peak_delta_bytes": _difference(
            dense.get("nvml", {}).get("sampled_used_peak_bytes"),
            candidate.get("nvml", {}).get("sampled_used_peak_bytes"),
        ),
        "rules": {
            "component_deltas_are_additive": False,
            "scopes_are_additive": False,
            "absent_component": "unknown_not_zero",
            "named_scope": "explicit_named_live_owners_not_complete_device_usage",
            "prefill_subtraction": "does_not_remove_admission_or_queueing_differences",
        },
    }


def aggregate_rank_snapshots(snapshots):
    """Aggregate process-owned bytes without merging local allocation group IDs.

    NVML is left per physical device and per sampling series. Summing its peaks
    across ranks can double count one device or unrelated time windows.
    """
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("A nonempty rank snapshot list is required")
    ranks, phases, families, worlds, groups = set(), set(), set(), set(), set()
    per_rank, nvml = [], {}
    for snapshot in snapshots:
        validate_snapshot(snapshot)
        context = snapshot["context"]
        rank = _integer(context.get("rank"), "rank")
        world = _integer(context.get("world_size"), "world_size", positive=True)
        if rank >= world or rank in ranks:
            raise ValueError(
                "Every process rank must be distinct and inside world_size"
            )
        if (
            not isinstance(context.get("device_uuid"), str)
            or not context["device_uuid"]
        ):
            raise ValueError("Rank aggregation requires physical device_uuid")
        ranks.add(rank)
        worlds.add(world)
        phases.add(context.get("phase"))
        families.add(context.get("family"))
        groups.add(context.get("capture_group_id"))
        per_rank.append(
            {
                "rank": rank,
                "process_id": context.get("process_id"),
                "snapshot_id": context["snapshot_id"],
                "device_uuid": context["device_uuid"],
                "named_storage_bytes": snapshot["named_storage_bytes"],
                "named_storage_bytes_by_local_device": copy.deepcopy(
                    snapshot["named_storage_bytes_by_device"]
                ),
                "allocator": copy.deepcopy(snapshot.get("allocator")),
                "graph_pools": copy.deepcopy(snapshot.get("graph_pools")),
            }
        )
        nvml.setdefault(context["device_uuid"], []).append(
            {
                "rank": rank,
                "snapshot_id": context["snapshot_id"],
                "series": copy.deepcopy(snapshot.get("nvml")),
            }
        )
    if len(phases) != 1 or len(families) != 1 or len(worlds) != 1:
        raise ValueError("Ranks must agree on phase, family and world size")
    if len(groups) != 1 or None in groups or "" in groups:
        raise ValueError("An explicit common capture_group_id is required")
    world = next(iter(worlds))
    return {
        "contract": "experiment_memory_rank_aggregate_v1",
        "schema_version": 1,
        "capture_group_id": next(iter(groups)),
        "phase": next(iter(phases)),
        "family": next(iter(families)),
        "world_size": world,
        "captured_ranks": sorted(ranks),
        "all_ranks_captured": ranks == set(range(world)),
        "named_process_owned_storage_bytes_sum": sum(
            item["named_storage_bytes"] for item in per_rank
        ),
        "ranks": sorted(per_rank, key=lambda item: item["rank"]),
        "nvml_series_by_physical_device": nvml,
        "cross_process_shared_storage_deduplication": "unmeasured",
        "allocation_group_scope": "one_snapshot_process_device_never_merge_between_ranks",
    }


def indexer_cache_accounting(
    snapshot,
    *,
    component_coverage=None,
    counters=None,
    counter_scope=None,
    indexer_layers=None,
):
    """Content/storage ledger and future cache-counter interface, not a cache.

    Coverage for absent categories is ``unmeasured`` or ``not_applicable``.
    Existing indexer K/scale tensors and regions imply ``measured``. Mapping/LRU
    and workspace must be explicitly named by a future cache implementation.
    """
    validate_snapshot(snapshot)
    context = snapshot["context"]
    coverage = {} if component_coverage is None else dict(component_coverage)
    for name, status in coverage.items():
        if name not in INDEXER_CATEGORIES or status not in {
            "measured",
            "unmeasured",
            "not_applicable",
        }:
            raise ValueError("Unknown indexer category or coverage status")
    content, content_groups = {}, set()
    for region in snapshot["logical_regions"]:
        name = region.get("content")
        if name not in INDEXER_CATEGORIES:
            continue
        tensor = snapshot["tensors"][region["tensor_name"]]
        geometry = region.get("geometry", {})
        # Tensor aliases may expose one packed allocation through several owners.
        # Do not count duplicate content geometry under another owner name.
        key = (
            region["device"],
            region["allocation_group"],
            name,
            tensor["storage_offset"] * tensor["element_bytes"],
            json.dumps(geometry, sort_keys=True),
        )
        if key in content_groups:
            continue
        content_groups.add(key)
        content[name] = content.get(name, 0) + region["logical_content_bytes"]
    components = snapshot["inclusive_component_bytes"]
    rows = {}
    for name in INDEXER_CATEGORIES:
        amount = content.get(name)
        storage = components.get(name)
        if amount is None and storage is not None:
            amount = storage
        status = coverage.get(name, "measured" if amount is not None else "unmeasured")
        if amount is not None and status != "measured":
            raise ValueError(
                "Observed indexer memory cannot be marked unmeasured/not_applicable"
            )
        if status == "measured" and amount is None:
            raise ValueError(
                "Measured category must have named storage or logical content"
            )
        rows[name] = {
            "status": status,
            "content_bytes": 0 if status == "not_applicable" else amount,
            "inclusive_component_storage_bytes": storage,
            "shares_packed_storage": name in content,
        }
    layer_count = (
        indexer_layers if indexer_layers is not None else context.get("indexer_layers")
    )
    denominators = {
        "per_indexer_layer": layer_count,
        "per_request_slot": context.get("preallocated_request_slots"),
        "per_live_history_token": context.get("live_history_tokens"),
    }
    for name, amount in denominators.items():
        if amount is not None:
            _integer(amount, name, positive=name == "per_request_slot")
    for row in rows.values():
        row["normalized_content_bytes"] = {
            name: (
                row["content_bytes"] / amount
                if row["content_bytes"] is not None and amount
                else None
            )
            for name, amount in denominators.items()
        }
    runtime = {name: None for name in INDEXER_COUNTERS}
    if counters is not None:
        if not isinstance(counters, dict) or set(counters) - set(INDEXER_COUNTERS):
            raise ValueError("Unknown indexer cache counter")
        if (
            not isinstance(counter_scope, dict)
            or counter_scope.get("phase") != context.get("phase")
            or not counter_scope.get("reset_interval")
        ):
            raise ValueError("Runtime counters require explicit phase/reset interval")
        runtime.update(counters)
        for name, value in runtime.items():
            if value is not None:
                _integer(value, name)
        if all(runtime[name] is not None for name in ("lookups", "hits", "misses")):
            if runtime["hits"] + runtime["misses"] != runtime["lookups"]:
                raise ValueError("Cache hits plus misses must equal lookups")
    lookups, hits = runtime["lookups"], runtime["hits"]
    runtime["hit_rate"] = hits / lookups if hits is not None and lookups else None
    return {
        "schema_version": 1,
        "contract": "indexer_cache_accounting_v1",
        "snapshot_id": context["snapshot_id"],
        "family": context.get("family"),
        "categories": rows,
        "runtime_counters": runtime,
        "counter_scope": copy.deepcopy(counter_scope),
        "denominators": denominators,
        "packed_indexer_physical_component_bytes": components.get(
            "indexer_packed_keys_scales"
        ),
        "logical_regions": copy.deepcopy(snapshot["logical_regions"]),
        "rules": {
            "content_and_physical_bytes_are_additive": False,
            "category_component_storage_bytes_are_additive": False,
            "qwen_quest_is_trained_indexer_baseline": False,
            "mechanism_comparison": "same_model_after_4plus0_runtime_baseline",
            "runtime_cache_mechanism_implemented": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("snapshot", type=Path)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("snapshots", nargs="+", type=Path)
    aggregate.add_argument("--output", type=Path, required=True)
    indexer = sub.add_parser("indexer")
    indexer.add_argument("snapshot", type=Path)
    indexer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate":
        value = validate_snapshot(json.loads(args.snapshot.read_text(encoding="utf-8")))
        print(
            json.dumps(
                {"status": "valid", "snapshot_id": value["context"]["snapshot_id"]}
            )
        )
        return
    if args.command == "aggregate":
        result = aggregate_rank_snapshots(
            [json.loads(path.read_text(encoding="utf-8")) for path in args.snapshots]
        )
    else:
        result = indexer_cache_accounting(
            json.loads(args.snapshot.read_text(encoding="utf-8"))
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
