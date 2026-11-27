"""Real-layer same-Q/K/V diagnostics; never a fused correctness/performance proof.

The original resolver and fused TMA/refill run once. An independent original
FlashInfer TensorCore wrapper reads the resolved destinations without another
resolver call. Returning its reference output teacher-forces subsequent layers.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Direct script execution puts this directory first, where our independent
# Nsight profile.py would shadow the stdlib profile imported by cProfile/Torch.
if __name__ == "__main__" and sys.path:
    if Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
        sys.path.pop(0)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value):
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    return repr(value)[:1000]


def validate_capture(output, config):
    rows = [
        json.loads(line)
        for path in Path(output).glob("layer-stats-pid*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = {(step, layer) for step in range(config["steps"]) for layer in range(48)}
    observed = [(row.get("step"), row.get("layer")) for row in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Incomplete or duplicate real 48-layer/step same-QKV capture")
    if any(
        row.get("tool_sha256") != config["tool_sha256"]
        or row.get("batch") != config["actual_batch"]
        or row.get("selected_payload_exact") is not True
        or row.get("reference_returned") is not config["return_reference"]
        or row.get("performance_result") is not False
        for row in rows
    ):
        raise ValueError("Same-QKV tool/batch/payload/reference scope differs")
    snapshots = [row["snapshot"] for row in rows if "snapshot" in row]
    if len(snapshots) != len(config["snapshot_layers"]):
        raise ValueError("Missing first-step selected layer snapshots")
    for item in snapshots:
        path = Path(item["path"])
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError("Same-QKV snapshot changed after capture")
    return {
        "status": "captured_same_qkv",
        "layer_step_rows": len(rows),
        "snapshots": snapshots,
        "all_host_backed_payloads_exact": True,
        "fused_model_correctness_verified": False,
        "performance_result": False,
        "scope": "Independent same-input layer diagnostic with explicit returned-output mode",
    }


def install_worker(config_path):
    """Installed through diagnostic-only sitecustomize in spawned server workers."""
    import torch
    from sglang.srt.layers.attention.flashinfer_hisparse_backend import (
        FlashInferHiSparseDecodeBackend,
        quest_scatter_pack,
    )

    cls = FlashInferHiSparseDecodeBackend
    if getattr(cls, "_xm_model_probe_installed", False):
        return
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("tool_sha256") != digest(__file__):
        raise RuntimeError("Diagnostic tool changed after config freeze")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    original = cls._run_fused_attention
    states = {}
    active = {"step": -1, "record": False}

    def reference_state(backend, bs, layer):
        key = (id(backend), bs)
        if key not in states:
            workspace = torch.empty(
                384 * 1024 * 1024, dtype=torch.uint8, device=backend.device
            )
            indptr = backend._kv_indptr_template[: bs + 1].clone()
            indices = torch.zeros(
                bs * backend.top_k + 1, dtype=torch.int32, device=backend.device
            )
            last = torch.ones(bs, dtype=torch.int32, device=backend.device)
            wrapper = backend._WrapperCls(
                workspace,
                kv_layout="NHD",
                use_cuda_graph=True,
                use_tensor_cores=True,
                paged_kv_indptr_buffer=indptr,
                paged_kv_indices_buffer=indices[: bs * backend.top_k],
                paged_kv_last_page_len_buffer=last,
            )
            wrapper.plan(
                indptr=indptr,
                indices=indices[: bs * backend.top_k],
                last_page_len=last,
                num_qo_heads=backend.num_qo_heads,
                num_kv_heads=backend.num_kv_heads,
                head_dim=backend.head_dim,
                page_size=1,
                sm_scale=layer.scaling,
                data_type=backend.kv_data_type,
                q_data_type=backend.q_data_type,
                non_blocking=True,
            )
            states[key] = {
                "workspace": workspace,
                "indptr": indptr,
                "indices": indices,
                "last": last,
                "wrapper": wrapper,
            }
        state = states[key]
        lengths = backend._actual_lens_buf[:bs].cpu()
        indptr_cpu = torch.zeros(bs + 1, dtype=torch.int32)
        indptr_cpu[1:] = torch.cumsum(lengths, dim=0)
        state["indptr"].copy_(indptr_cpu)
        backend._fast_decode_plan(
            state["wrapper"],
            indptr=state["indptr"],
            indices=state["indices"][: bs * backend.top_k],
            last_page_len=state["last"],
            num_qo_heads=backend.num_qo_heads,
            num_kv_heads=backend.num_kv_heads,
            head_dim=backend.head_dim,
            page_size=1,
            sm_scale=layer.scaling,
            logits_soft_cap=getattr(layer, "logit_cap", 0.0) or 0.0,
            data_type=backend.kv_data_type,
            q_data_type=backend.q_data_type,
            non_blocking=True,
            global_override_indptr_cpu=indptr_cpu,
        )
        return state

    def probe(backend, q, layer, forward_batch, topk_token_positions):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Same-layer diagnostic supports eager only")
        # Exactly one real resolver/TMA/refill invocation; never call separate
        # attention (which would run the resolver a second time and mutate LRU).
        fused = original(backend, q, layer, forward_batch, topk_token_positions)
        bs = backend._current_bs
        raw_k, raw_v = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        state = reference_state(backend, bs, layer)
        quest_scatter_pack(
            page_table=backend._fused_tma.device_locs[:bs],
            actual_lens=backend._actual_lens_buf[:bs],
            kv_indptr=state["indptr"][:bs],
            kv_indices_buf=state["indices"],
            scratch_idx=bs * backend.top_k,
            top_k=backend.top_k,
        )
        reference, lse = state["wrapper"].run(
            q.contiguous().view(bs, backend.num_qo_heads, backend.head_dim),
            (
                raw_k.view(-1, 1, backend.num_kv_heads, backend.head_dim),
                raw_v.view(-1, 1, backend.num_kv_heads, backend.head_dim),
            ),
            k_scale=getattr(layer, "k_scale_float", None),
            v_scale=getattr(layer, "v_scale_float", None),
            return_lse=True,
        )
        reference = reference.view_as(fused)
        lengths = getattr(forward_batch, "seq_lens_cpu", None)
        if lengths is None:
            lengths = forward_batch.seq_lens.cpu()
        eligible = (
            bs == config["actual_batch"]
            and min(lengths.tolist()) >= config["min_context"]
        )
        if layer.layer_id == 0:
            if eligible:
                active["step"] += 1
            active["record"] = eligible and active["step"] < config["steps"]
        if active["record"]:
            torch.cuda.synchronize()
            locs = backend._fused_tma.device_locs[:bs].clone()
            selected_k = raw_k[locs.clamp_min(0).long()].cpu()
            selected_v = raw_v[locs.clamp_min(0).long()].cpu()
            positions = topk_token_positions[:bs].clone()
            actual_lens = backend._actual_lens_buf[:bs].cpu()
            real = int(backend.coord.num_real_reqs.item())
            valid = (torch.arange(backend.top_k)[None, :] < actual_lens[:, None]) & (
                torch.arange(bs)[:, None] < real
            )
            host_locs = backend.coord.req_to_host_pool[
                forward_batch.req_pool_indices[:bs, None].long(),
                positions.clamp_min(0).long(),
            ].cpu()
            backed = valid & (host_locs >= 0) & (positions.cpu() >= 0)
            host_k = backend._fused_tma.host_k_layers[layer.layer_id]
            host_v = backend._fused_tma.host_v_layers[layer.layer_id]
            expected_k = host_k[host_locs[backed].long()].clone()
            expected_v = host_v[host_locs[backed].long()].clone()
            payload_exact = torch.equal(selected_k[backed], expected_k) and torch.equal(
                selected_v[backed], expected_v
            )
            left, right = fused.detach().cpu().float(), reference.detach().cpu().float()
            error = left - right
            q_cpu = q.detach().cpu()
            module = state["wrapper"]._cached_module
            module_info = (
                {key: safe(value) for key, value in vars(module).items()}
                if hasattr(module, "__dict__")
                else {"type": repr(type(module))}
            )
            for key in ("run", "plan"):
                function = getattr(module, key, None)
                code = getattr(function, "__code__", None)
                module_info[key + "_code_file"] = getattr(code, "co_filename", None)
                module_info[key + "_python_module"] = getattr(
                    function, "__module__", None
                )
            row = {
                "type": "same_layer_qkv",
                "tool_sha256": config["tool_sha256"],
                "step": active["step"],
                "layer": layer.layer_id,
                "batch": bs,
                "real_requests": real,
                "seq_lens": lengths.tolist(),
                "actual_lens": actual_lens.tolist(),
                "scaling": layer.scaling,
                "logits_soft_cap": getattr(layer, "logit_cap", 0.0) or 0.0,
                "k_scale": safe(getattr(layer, "k_scale_float", None)),
                "v_scale": safe(getattr(layer, "v_scale_float", None)),
                "max_abs_output_error": error.abs().max().item(),
                "rms_output_error": error.square().mean().sqrt().item(),
                "unequal_output_fraction": (left != right).float().mean().item(),
                "reference_output_absmax": right.abs().max().item(),
                "q_absmax": q_cpu.float().abs().max().item(),
                "q_rms": q_cpu.float().square().mean().sqrt().item(),
                "k_absmax": selected_k[valid].float().abs().max().item(),
                "v_absmax": selected_v[valid].float().abs().max().item(),
                "k_rms": selected_k[valid].float().square().mean().sqrt().item(),
                "v_rms": selected_v[valid].float().square().mean().sqrt().item(),
                "selected_backed_tokens": int(backed.sum()),
                "selected_payload_exact": payload_exact,
                "miss_host_tokens": int(
                    ((backend._fused_tma.miss_host_locs[:bs].cpu() >= 0) & valid).sum()
                ),
                "fused_kv_chunk_size": int(backend._fused_tma.kv_chunk_size.item()),
                "actual_fi_plan_info": safe(state["wrapper"]._plan_info),
                "actual_fi_module": module_info,
                "reference_returned": config["return_reference"],
                "extra_reference_workspace_bytes": state["workspace"].numel(),
                "performance_result": False,
            }
            if active["step"] == 0 and layer.layer_id in config["snapshot_layers"]:
                snapshot = output / f"step0-layer{layer.layer_id}-pid{os.getpid()}.pt"
                torch.save(
                    {
                        "q": q_cpu,
                        "selected_k": selected_k,
                        "selected_v": selected_v,
                        "valid": valid,
                        "host_backed": backed,
                        "host_k_selected": expected_k,
                        "host_v_selected": expected_v,
                        "device_locs": locs.cpu(),
                        "miss_host_locs": backend._fused_tma.miss_host_locs[:bs].cpu(),
                        "topk_positions": positions.cpu(),
                        "host_locs": host_locs,
                        "actual_lens": actual_lens,
                        "kv_indptr": state["indptr"].cpu(),
                        "kv_indices": state["indices"].cpu(),
                        "partial": backend._fused_tma.partial[:bs].cpu(),
                        "fused_output": fused.detach().cpu(),
                        "reference_output": reference.detach().cpu(),
                        "flashinfer_lse_raw": lse.detach().cpu(),
                        "stats": row,
                        "scope": "Teacher-forced same-QKV diagnostic; raw FlashInfer LSE units retained, not silently converted",
                    },
                    snapshot,
                )
                row["snapshot"] = {
                    "path": str(snapshot),
                    "bytes": snapshot.stat().st_size,
                    "sha256": digest(snapshot),
                }
            with (output / f"layer-stats-pid{os.getpid()}.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if not payload_exact:
                raise RuntimeError(
                    "Same-layer diagnostic found nonexact host-backed TMA/refill payload"
                )
        return reference if config["return_reference"] else fused

    cls._run_fused_attention = probe
    cls._xm_model_probe_installed = True
    (output / f"worker-installed-pid{os.getpid()}.json").write_text(
        json.dumps(
            {
                "tool_sha256": config["tool_sha256"],
                "core_marker": config["core_marker"],
                "diagnostic_monkeypatch": True,
                "performance_result": False,
                "fused_model_correctness_verified": False,
                "return_reference": config["return_reference"],
            },
            indent=2,
        )
        + "\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--input-lens", nargs="+", type=int, default=[16384, 32768])
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--output-len", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--return-fused", action="store_true")
    parser.add_argument("--execute", action="store_true")
    control = parser.parse_args(argv)
    if (
        not 1 <= control.steps <= 6
        or control.output_len < 80
        or min(control.input_lens) <= 4096
    ):
        parser.error("Use 1-6 diagnostic steps, >=80 outputs and long sparse contexts")
    root = control.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    campaign = load(ROOT / "benchmark/xm_sparse/campaign.py", "xm_model_probe_campaign")
    native = load(ROOT / "benchmark/mpu_stages/correctness.py", "xm_model_probe_native")
    run = native.RUN
    config = {
        "tool_sha256": digest(__file__),
        "core_marker": json.loads((ROOT / ".mpu-source.json").read_text())
        if (ROOT / ".mpu-source.json").is_file()
        else None,
        "core_fingerprint": campaign.source_fingerprint(),
        "actual_batch": len(control.input_lens),
        "min_context": min(control.input_lens),
        "steps": control.steps,
        "snapshot_layers": [0, 23, 47],
        "return_reference": not control.return_fused,
        "output_dir": str(root / "same-layer"),
        "diagnostic_monkeypatch": True,
    }
    stage_path = root / "stage.json"
    campaign.write(stage_path, campaign.recipe("fused"))
    campaign.write(root / "probe-config.json", config)
    stage, args = run.STAGE.parse_args(
        [
            "--stage",
            str(stage_path),
            "--model",
            control.model,
            "--workload",
            str(control.workload.resolve()),
            "--input-lens",
            *map(str, control.input_lens),
            "--output-len",
            str(control.output_len),
            "--num-prompts",
            "16",
            "--max-total-tokens",
            "65536",
            "--capacity-ratios",
            "8",
            "--cpu-capacity-ratio",
            "8",
            "--chunked-prefill-size",
            "4096",
            "--prefill-max-requests",
            "1",
            "--mem-fraction-static",
            "0.9",
            "--seed",
            "1729",
            "--include-hbm",
            "--disable-cuda-graph",
            "--lifecycle",
            "--output-dir",
            str(root / "native-diagnostic"),
        ],
        correctness=True,
    )
    groups = native.make_plan(stage, args)
    args.output_dir.mkdir()
    report = run.new_manifest(
        stage, args, groups, "teacher_forced_attention_diagnostic"
    )
    report.update(
        claim="Diagnostic-only original FlashInfer reference is returned after fused TMA/refill. A successful native comparison does not prove fused model output correctness.",
        probe_config=config,
        fused_model_correctness_verified=False,
        performance_result=False,
    )
    injection = root / "injection"
    injection.mkdir()
    (injection / "sitecustomize.py").write_text(
        "import importlib.util,os\np=os.environ.get('SGLANG_XM_ATTENTION_PROBE_TOOL')\n"
        "if p:\n s=importlib.util.spec_from_file_location('xm_attention_probe_worker',p)\n"
        " m=importlib.util.module_from_spec(s)\n s.loader.exec_module(m)\n"
        " m.install_worker(os.environ['SGLANG_XM_ATTENTION_PROBE_CONFIG'])\n"
    )
    original_start = run.COMMON.start_server

    def start(*positional, **keyword):
        env = dict(keyword.get("env_updates") or {})
        env.update(
            SGLANG_XM_ATTENTION_PROBE_TOOL=str(Path(__file__).resolve()),
            SGLANG_XM_ATTENTION_PROBE_CONFIG=str(root / "probe-config.json"),
            PYTHONPATH=str(injection) + os.pathsep + str(ROOT / "python"),
        )
        keyword["env_updates"] = env
        return original_start(*positional, **keyword)

    run.COMMON.start_server = start
    path = args.output_dir / "manifest.json"
    try:
        if control.execute:
            if sys.platform != "linux" or config["core_marker"] is None:
                raise ValueError("Execute only in the exact Linux archived source")
            native_ok = native.execute(stage, args, report, path)
            report["same_qkv_capture"] = validate_capture(root / "same-layer", config)
            report["reference_output_hbm_match"] = native_ok
            report["status"] = (
                "diagnostic_captured" if native_ok else "failed_diagnostic"
            )
    except (ImportError, ValueError, RuntimeError, OSError, KeyboardInterrupt) as error:
        report.update(
            status="failed_diagnostic",
            error=str(error),
            diagnostic_traceback=traceback.format_exc(),
        )
    finally:
        run.COMMON.start_server = original_start
        run.COMMON.write_manifest(path, report)
    print(path)
    return 2 if report["status"] == "failed_diagnostic" else 0


if __name__ == "__main__":
    raise SystemExit(main())
