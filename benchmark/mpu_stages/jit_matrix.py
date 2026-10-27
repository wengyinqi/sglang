"""Production JIT factory catalog. Planning imports no torch or CUDA modules."""

import copy


def cases_for(stage, family, matrix="default"):
    if matrix not in ("default", "boundaries"):
        raise ValueError("Unknown JIT matrix")
    config = stage["variants"][stage["variant"]]["hisparse_config"]
    topk, hot = config["top_k"], config["device_buffer_size"]
    if any(type(v) is not int or v <= 0 for v in (topk, hot)) or hot < topk:
        raise ValueError("JIT matrix needs positive top-k and a sufficient hot buffer")
    qwen = stage["family"] == "qwen"
    c4 = stage["family"] == "deepseek-v4-flash"
    if not qwen and not c4 and stage["family"] != "glm-5.3-flash":
        raise ValueError("Unsupported JIT family")
    sparse = (
        "sglang.jit_kernel.hisparse" if qwen else "sglang.kernels.ops.kvcache.hisparse"
    )
    source = "hisparse.cuh" if qwen else "kvcacheio/hisparse.cuh"
    cases = []

    def add(name, module, factory, args, exports, headers, **kwargs):
        cases.append(
            dict(
                id=name,
                module=module,
                factory=factory,
                args=args,
                kwargs=kwargs,
                exports=exports,
                headers=headers,
            )
        )

    add("mpu-load-store", family["kernel_module"], "_module", [], [], [])
    if qwen:
        add(
            "quest-cpu-resolve",
            sparse,
            "_jit_sparse_module",
            [1024, 256, topk, hot, False],
            ["load_cache_to_device_buffer"],
            [source],
        )
        add(
            "quest-mpu-resolve",
            sparse,
            "_jit_sparse_mpu_module",
            [1024, 256, topk, hot],
            ["load_cache_from_mpu_to_device_buffer"],
            [source],
        )
    else:
        block = config.get("swap_in_block_size", 256)
        tail = 0 if c4 else 3
        extra = {} if c4 else {"num_tail_slots": tail}
        for backing in ("cpu", "mpu"):
            add(
                backing + "-resolve",
                sparse,
                "_jit_sparse_module",
                [584 if c4 else 1024, block, topk + tail, hot],
                ["load_cache_to_device_buffer"],
                [source],
                is_mla=True,
                is_dsv4_layout=c4,
                record_miss_plan=backing == "mpu",
                skip_io=backing == "mpu",
                **extra
            )
        add(
            "shared-index-copy",
            sparse,
            "_jit_copy_planned_module",
            [block, True, c4, False],
            ["copy_cache_planned"],
            [source],
        )
        if c4:
            add(
                "c4-transfer",
                sparse,
                "_jit_dsv4_transfer_module",
                [1024],
                ["transfer_cache_dsv4_mla"],
                [source],
            )
        else:
            add(
                "kpool2048-tail3",
                "sglang.kernels.ops.attention.dsa.kpool_topk_transform",
                "_jit_kpool_topk_transform_module",
                [512],
                ["kpool_topk_transform"],
                ["dsa/kpool_topk_transform.cuh"],
            )
    if matrix == "boundaries":
        # Grow only a supported cache parameter; keep the model's selector budget.
        for case in list(cases):
            if case["id"].endswith("resolve"):
                larger = copy.deepcopy(case)
                larger["id"] += "-hot2x"
                larger["args"][3] = hot * 2
                cases.append(larger)
        if not qwen and not c4:
            component = copy.deepcopy(cases[4])
            component.update(
                id="kpool-component-min128", args=[128], component_only=True
            )
            cases.append(component)
    return cases


def validate_report(report, expected_cases=None):
    """Validate completeness before a successful JIT report unlocks later gates."""
    planned = report.get("jit_cases", [])
    ids = [case.get("id") for case in planned]
    checks = report.get("checks", [])
    if (
        not ids
        or len(set(ids)) != len(ids)
        or (expected_cases is not None and planned != expected_cases)
        or report.get("capability_scope") != "production_jit_load_no_gpu"
        or report.get("gpu_execution_requested") is not False
    ):
        raise ValueError("Invalid or incomplete production JIT plan")
    actual = []
    artifacts = []
    for index, check in enumerate(checks):
        evidence = check.get("evidence", {})
        if (
            check.get("name") != "production_jit_factory"
            or check.get("status") != "passed"
            or evidence.get("native_module_loaded") is not True
            or evidence.get("gpu_executed") is not False
            or not evidence.get("exports")
            or not evidence.get("shared_libraries")
        ):
            raise ValueError(
                "JIT pass requires a real loaded module and exported callables"
            )
        if index >= len(planned) or evidence.get("case") != planned[index]:
            raise ValueError("JIT specialization differs from the planned factory")
        if (
            planned[index]["exports"]
            and evidence["exports"] != planned[index]["exports"]
        ):
            raise ValueError("JIT exports differ from the planned factory")
        actual.append(evidence.get("case", {}).get("id"))
        artifacts.extend(evidence["shared_libraries"])
    if actual != ids:
        raise ValueError("Missing, duplicated or substituted JIT cases")
    return artifacts
