"""CPU-only parallelism contract for the opt-in MPU experiment instrumentation."""


def parallel_metadata(cfg, rank):
    width = cfg.tp_size
    attention_dp = getattr(cfg, "attn_dp_size", 1)
    expert = getattr(cfg, "ep_size", 1)
    cp = getattr(cfg, "attn_cp_size", 1)
    moe_dp = getattr(cfg, "moe_dp_size", 1)
    if any(
        type(n) is not int or n < 1 for n in (width, attention_dp, expert, cp, moe_dp)
    ):
        raise ValueError("Parallel widths must be positive integers")
    if width % (attention_dp * cp) or width % (expert * moe_dp):
        raise ValueError(
            "Attention and expert partitions must divide the compute width"
        )
    result = {
        "compute_world_size": width,
        "attn_dp_size": attention_dp,
        "attn_dp_rank": rank // (width // attention_dp),
        "attn_tp_size": width // attention_dp // cp,
        "ep_size": expert,
        "moe_tp_size": width // expert // moe_dp,
    }
    if width == 4:
        required = {
            "attn_dp_size": 4,
            "ep_size": 4,
            "dp_size": 1,
            "pp_size": 1,
            "attn_cp_size": 1,
            "dcp_size": 1,
            "moe_dp_size": 1,
            "moe_dense_tp_size": 1,
            "nnodes": 1,
            "enable_dp_lm_head": True,
            "disable_overlap_schedule": True,
            "enable_prefill_cp": False,
            "enable_two_batch_overlap": False,
            "enable_cp_tp_group_sharing": False,
            "moe_a2a_backend": "deepep",
        }
        for name, expected in required.items():
            if getattr(cfg, name, None) != expected:
                raise ValueError(
                    f"MPU four-compute experiments require {name}={expected}"
                )
        result["parallel_strategy"] = "attention_dp4_ep4"
    elif width == attention_dp == expert == 1:
        result["parallel_strategy"] = "single_compute"
    else:
        raise ValueError(
            "MPU experiments support one compute GPU or Attention-DP4 + EP4"
        )
    return result


def audit_model_tensor_parallelism(model):
    """Inspect instantiated linear/vocabulary partitions without reading weights.

    The outer MoE module's tp_size is a dispatch-group width, not a GEMM shard.
    Audit the actual tensor-parallel linear/embedding modules instead.
    """
    checked = []
    families = {
        "ColumnParallelLinear",
        "RowParallelLinear",
        "VocabParallelEmbedding",
        "MergedColumnParallelRepeatedLinear",
        "ColumnParallelBatchedLinear",
    }
    for name, module in model.named_modules():
        if families.intersection(cls.__name__ for cls in type(module).__mro__):
            width = getattr(module, "tp_size", None)
            # New main uses a resolved group on linear layers; the older Quest
            # layers and vocabulary embeddings expose tp_size directly. None
            # is an explicit unsharded group, distinct from a missing field.
            if hasattr(module, "tp_group"):
                group = module.tp_group
                group_width = 1 if group is None else getattr(group, "world_size", None)
                if width is not None and width != group_width:
                    raise ValueError(f"Conflicting tensor partition metadata in {name}")
                width = group_width
            if type(width) is not int or width != 1:
                raise ValueError(
                    f"Residual tensor parallelism in {name}: tp_size={width}"
                )
            checked.append(name)
    if not checked:
        raise ValueError("No instantiated linear/embedding partitions were audited")
    return {"status": "passed", "checked_modules": checked, "effective_tp_size": 1}
