"""Prepare explicit SLO/cost inputs and reanalyze saved measurements without GPUs."""

import argparse
import importlib.util
import json
from pathlib import Path


def load(name):
    spec = importlib.util.spec_from_file_location(
        "decision_" + name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scope(stage, storage):
    return {
        "family": stage["family"],
        "selector": stage["selector"],
        "model_revision": stage["default_revision"],
        "compute_gpus": stage["tp_size"],
        "storage_device_ids": storage,
        "storage_gpus": len(storage),
    }


def template(matrix):
    return {
        "schema_version": 1,
        "topology_scope": scope(
            matrix["stage"], matrix["topology"]["storage_device_ids"]
        ),
        "minimum_repeats": matrix["matrix"]["repeats"],
        "slo_ms": {"p99_ttft_ms": None, "p99_tpot_ms": None},
        "cost_basis": "simulator_hardware",
        "currency": None,
        "source": None,
        "cpu_hourly_total": None,
        "mpu_hourly_total": None,
    }


def compile_policy(inputs, matrix):
    expected_keys = set(template(matrix))
    if set(inputs) != expected_keys or inputs["schema_version"] != 1:
        raise ValueError("Unknown or missing decision input fields")
    expected_scope = scope(matrix["stage"], matrix["topology"]["storage_device_ids"])
    if inputs["topology_scope"] != expected_scope:
        raise ValueError(
            "Decision inputs describe a different model or physical topology"
        )
    cap = load("capacity_analysis")
    minimum = inputs["minimum_repeats"]
    if type(minimum) is not int or minimum < 5:
        raise ValueError("Formal decisions require at least five repeats")
    if set(inputs["slo_ms"]) != {"p99_ttft_ms", "p99_tpot_ms"}:
        raise ValueError("Specify both P99 latency fields in milliseconds")
    provided = [v is not None for v in inputs["slo_ms"].values()]
    if any(provided) and not all(provided):
        raise ValueError("Fill both latency SLOs or leave both pending")
    slo = inputs["slo_ms"] if all(provided) else {}
    for value in slo.values():
        cap.number(value)
    costs = []
    if inputs["cost_basis"] not in ("simulator_hardware", "hypothetical_mpu"):
        raise ValueError("Explicit simulator or hypothetical MPU cost basis required")
    points = {
        (g["variant"], g["ratio"])
        for job in matrix["jobs"]
        for g in job["performance_groups"]
    }
    for variant, ratio in sorted(points):
        total = inputs[variant + "_hourly_total"]
        if total is not None:
            cap.number(total)
            if not inputs["currency"] or not inputs["source"]:
                raise ValueError("Prices require currency and an explicit source")
            costs.append({"variant": variant, "ratio": ratio, "hourly_cost": total})
    if (
        not matrix["topology"]["storage_gpus"]
        and inputs["mpu_hourly_total"] is not None
    ):
        raise ValueError(
            "CPU-only topology cannot price an unmeasured MPU configuration"
        )
    return {
        "schema_version": 1,
        "minimum_repeats": minimum,
        "slo": slo,
        "costs": costs,
        "currency": inputs["currency"],
        "cost_basis": inputs["cost_basis"],
        "source": inputs["source"] or "pending real SLO and quoted configuration costs",
        "topology_scope": expected_scope,
    }


def verify_scope(policy, stage, storage):
    supplied = policy.get("topology_scope")
    if supplied is None and not policy["costs"]:
        return  # Pending legacy empty policies carry no economic claim.
    if supplied != scope(stage, storage):
        raise ValueError(
            "Cost policy must be bound to this model and physical topology"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("template", "compile", "review"))
    parser.add_argument("--matrix-report", required=True, type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    matrix = json.loads(args.matrix_report.read_text(encoding="utf-8"))
    if matrix.get("kind") != "experiment_matrix":
        raise ValueError("Use an experiment_matrix.py report")
    if args.mode == "template":
        result = template(matrix)
    else:
        if args.inputs is None:
            raise ValueError("Compile/review requires --inputs")
        policy = compile_policy(
            json.loads(args.inputs.read_text(encoding="utf-8")), matrix
        )
        result = policy
        if args.mode == "review":
            run, checkpoint, matrix_tools = (
                load("run"),
                load("checkpoint"),
                load("experiment_matrix"),
            )
            result = {
                "kind": "decision_review",
                "policy": policy,
                "measurements_rerun": False,
                "matrix_sha256": checkpoint.sha(args.matrix_report),
                "profiles": [],
            }
            for job in matrix["jobs"]:
                path = Path(job["performance_report"])
                if not path.is_file():
                    result["profiles"].append(
                        {"id": job["id"], "status": "missing_measurements"}
                    )
                    continue
                measured = json.loads(path.read_text(encoding="utf-8"))
                if (
                    measured.get("status") != "success"
                    or measured.get("stage") != matrix["stage"]
                ):
                    result["profiles"].append(
                        {"id": job["id"], "status": "incomplete_or_wrong_stage"}
                    )
                    continue
                try:
                    matrix_tools.verify_performance(measured, matrix["stage"], job)
                    load("capacity_analysis").verify_manifest_scope(policy, measured)
                    if job.get("gates_receipt"):
                        checkpoint.verify(
                            Path(job["gate_report"]).parent, job["gates_receipt"]
                        )
                    matrix_tools.verify_gate(
                        json.loads(
                            Path(job["gate_report"]).read_text(encoding="utf-8")
                        ),
                        matrix["stage"],
                        job,
                    )
                except (OSError, ValueError, KeyError, TypeError) as error:
                    result["profiles"].append(
                        {
                            "id": job["id"],
                            "status": "invalid_measurements",
                            "reason": str(error),
                        }
                    )
                    continue
                summary = run.summarize(
                    measured["groups"], matrix["matrix"]["plateau_threshold"], policy
                )
                result["profiles"].append(
                    {
                        "id": job["id"],
                        "status": "reanalyzed",
                        "manifest_sha256": checkpoint.sha(path),
                        "summary": summary,
                        "decisions": matrix_tools.decisions(
                            summary, matrix["matrix"]["plateau_consecutive_edges"]
                        ),
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
