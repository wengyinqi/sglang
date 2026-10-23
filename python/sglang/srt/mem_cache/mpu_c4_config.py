"""Pure validation for the controlled single-node TP4 C4 experiment."""

import math


def c4_backing_options(extra):
    if not isinstance(extra, dict):
        raise ValueError("HiSparse backing configuration must be a JSON object")
    unknown = {k for k in extra if k.startswith("mpu_")} - {
        "mpu_device_ids",
        "mpu_capacity_ratio",
        "mpu_memory_reserve_bytes",
    }
    if unknown:
        raise ValueError(f"Unknown MPU options: {sorted(unknown)}")
    mode = extra.get("backing_store", "cpu")
    if mode not in ("cpu", "mpu"):
        raise ValueError("HiSparse backing_store must be cpu or mpu")
    ids = extra.get("mpu_device_ids", [4, 5, 6, 7])
    ratio = extra.get("mpu_capacity_ratio", 2.0)
    reserve = extra.get("mpu_memory_reserve_bytes", 1 << 30)
    if mode == "mpu":
        if ids != [4, 5, 6, 7] or any(type(i) is not int for i in ids):
            raise ValueError(
                "C4 4+4 requires compute ordinals 0..3 and MPU ordinals 4..7"
            )
        if type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("mpu_capacity_ratio must be finite and positive")
        if type(reserve) is not int or reserve < 0:
            raise ValueError("mpu_memory_reserve_bytes must be a nonnegative integer")
    elif any(k.startswith("mpu_") for k in extra):
        raise ValueError("MPU options require backing_store=mpu")
    return mode, ids, ratio, reserve


def validate_c4_mpu_launch(cfg, extra):
    mode, _, _, _ = c4_backing_options(extra)
    if mode != "mpu":
        return
    for name, value in (
        ("tp_size", 4),
        ("dp_size", 1),
        ("attn_dp_size", 1),
        ("attn_cp_size", 1),
        ("dcp_size", 1),
        ("pp_size", 1),
        ("nnodes", 1),
        ("base_gpu_id", 0),
        ("gpu_id_step", 1),
    ):
        actual = getattr(cfg, name, value)
        if actual is not None and actual != value:
            raise ValueError(f"C4 MPU requires {name}={value}, got {actual}")
    if getattr(cfg, "disaggregation_mode", "null") not in (None, "null"):
        raise ValueError("C4 MPU experiment requires NoPD")
    if getattr(cfg, "speculative_algorithm", None) not in (None, "NONE"):
        raise ValueError("C4 MPU does not support speculative decoding")
    if not getattr(cfg, "disable_radix_cache", False):
        raise ValueError("C4 MPU requires disable_radix_cache")
    for name in (
        "enable_hierarchical_cache",
        "enable_lmcache",
        "enable_memory_saver",
        "enable_dp_attention",
        "enable_mixed_chunk",
    ):
        if getattr(cfg, name, False):
            raise ValueError(f"C4 MPU does not support {name}")
    if getattr(cfg, "cpu_offload_gb", 0):
        raise ValueError("C4 MPU forbids CPU offload")
    if getattr(cfg, "dsv4_attn_backend", "flashmla") != "flashmla":
        raise ValueError("C4 MPU requires the FlashMLA V4 layout/backend")
