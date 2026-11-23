"""Model-native sparse quality controls for DeepSeek-V4 and GLM-5.3.

No Dense algorithm variant is fabricated for these trained indexer models.
Comparisons use their same-model native sparse HBM baseline and storage paths.
"""

import json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def policy(
    stage,
    info,
    variant,
    selection,
    model,
    *,
    capacity_ratio,
    top_k,
    mpu_device_ids,
    mpu_memory_reserve_bytes,
    compute_device_id
):
    states = info.get("internal_states", [])
    require(len(states) == 4, "Quality audit needs all four actual scheduler states")
    resolved = info.get("server_args", info)
    if "model_path" not in resolved and states:
        resolved = states[0]
    require(
        resolved.get("model_path") == model["model_path"],
        "Server model path differs from manifest",
    )
    require(
        variant in ("hbm", "cpu", "mpu"),
        "Indexer quality requires native sparse HBM/CPU/MPU",
    )
    common = {
        "tp_size": 4,
        "attn_dp_size": 4,
        "ep_size": 4,
        "dp_size": 1,
        "pp_size": 1,
        "attn_cp_size": 1,
        "dcp_size": 1,
        "moe_dp_size": 1,
        "moe_dense_tp_size": 1,
        "enable_dp_lm_head": True,
        "nnodes": 1,
        "moe_a2a_backend": "deepep",
        "disable_overlap_schedule": True,
        "enable_prefill_cp": False,
        "enable_two_batch_overlap": False,
        "enable_cp_tp_group_sharing": False,
        "disable_radix_cache": True,
        "enable_hierarchical_cache": False,
        "allow_auto_truncate": False,
        "disaggregation_mode": "null",
        "speculative_algorithm": None,
        "enable_mixed_chunk": False,
        "page_size": stage["page_size"],
        "chunked_prefill_size": stage["defaults"]["chunked_prefill_size"],
        "max_total_tokens": selection["max_total_tokens"],
    }
    ranks = set()
    selector = None
    for state in states:
        require(
            all(key in state and state[key] == value for key, value in common.items()),
            "Four-rank quality control differs",
        )
        topology = state.get("mpu_parallel_metadata", {})
        rank = topology.get("attn_dp_rank")
        require(
            type(rank) is int
            and rank in range(4)
            and rank not in ranks
            and topology.get("attn_tp_size") == 1
            and topology.get("moe_tp_size") == 1
            and topology.get("tensor_parallel_audit", {}).get("status") == "passed",
            "Missing/duplicate rank or residual model TP in quality audit",
        )
        ranks.add(rank)
        current = state.get("mpu_selector_runtime")
        require(
            isinstance(current, dict)
            and type(current.get("index_topk")) is int
            and current["index_topk"] > 0,
            "Missing model-config effective indexer top-k",
        )
        require(
            selector is None or current == selector,
            "Effective indexer selector differs across ranks",
        )
        selector = current
    require(set(ranks) == set(range(4)), "Incomplete quality scheduler topology")
    family = stage["family"]
    if family == "deepseek-v4-flash":
        require(
            all(
                state.get("attention_backend") == "dsv4"
                and state.get("dsv4_attn_backend") == "flashmla"
                for state in states
            ),
            "Require native DeepSeek-V4 flashmla sparse selector",
        )
    else:
        require(
            all(
                state.get("dsa_prefill_backend")
                == state.get("dsa_decode_backend")
                == "flashmla_sparse"
                and state.get("dtype") == state.get("kv_cache_dtype") == "bfloat16"
                for state in states
            ),
            "Require native GLM DSA precision/backend controls",
        )
    configs = []
    for state in states:
        config = state.get("hisparse_config") or {}
        config = json.loads(config) if isinstance(config, str) else config
        require(isinstance(config, dict), "Malformed resolved HiSparse config")
        configs.append(config)
        if variant == "hbm":
            require(
                state.get("enable_hisparse") is False,
                "HBM native sparse reference unexpectedly offloads",
            )
        else:
            require(
                state.get("enable_hisparse") is True
                and config.get("backing_store", "cpu") == variant
                and config.get("top_k") == top_k
                and config.get("device_buffer_size")
                == stage["variants"]["cpu"]["hisparse_config"]["device_buffer_size"],
                "Storage selector/hot controls differ",
            )
            key = "host_to_device_ratio" if variant == "cpu" else "mpu_capacity_ratio"
            require(config.get(key) == capacity_ratio, "Storage capacity ratio differs")
            if variant == "mpu":
                require(
                    config.get("host_to_device_ratio")
                    == stage["variants"]["cpu"]["hisparse_config"][
                        "host_to_device_ratio"
                    ],
                    "MPU changed frozen compute sizing ratio",
                )
                require(
                    list(mpu_device_ids) == config.get("mpu_device_ids")
                    and config.get("mpu_memory_reserve_bytes")
                    == mpu_memory_reserve_bytes
                    and compute_device_id == state.get("base_gpu_id", 0),
                    "MPU storage identity differs",
                )
                require(
                    len(mpu_device_ids) == 4
                    and len(set(mpu_device_ids)) == 4
                    and not set(mpu_device_ids).intersection(
                        range(compute_device_id, compute_device_id + 4)
                    ),
                    "Four storage devices must be disjoint from all four compute GPUs",
                )
    require(
        all(config == configs[0] for config in configs),
        "Resolved sparse configuration differs across schedulers",
    )
    return {
        **states[0],
        "mpu_quality_common": {key: states[0][key] for key in common}
        | {
            "dtype": states[0].get("dtype"),
            "kv_cache_dtype": states[0].get("kv_cache_dtype"),
            "prefill_max_requests": states[0].get("prefill_max_requests"),
            "decode_graph_policy": states[0].get("cuda_graph_config"),
        },
        "mpu_quality_selector": {
            "family": family,
            "selector": stage["selector"],
            "runtime": selector,
        },
        "mpu_quality_controls_checked": True,
    }


def controls(info, variant):
    config = info.get("hisparse_config") or {}
    config = json.loads(config) if isinstance(config, str) else config
    value = {
        "common": info["mpu_quality_common"],
        "selector": info["mpu_quality_selector"],
        "cpu_hot_buffer_size": (
            config.get("device_buffer_size") if variant == "cpu" else None
        ),
    }
    if variant == "mpu":
        value["mpu_storage"] = {
            key: config.get(key)
            for key in (
                "backing_store",
                "mpu_capacity_ratio",
                "mpu_device_ids",
                "mpu_memory_reserve_bytes",
                "device_buffer_size",
            )
        }
        value["mpu_storage"]["compute_device_id"] = info.get("base_gpu_id")
    return value
