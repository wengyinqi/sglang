"""Low-rate host NVML and observed scheduler progress for independent diagnostics.

This sampler launches no CUDA workload. GPU totals, thermal state and clocks are
diagnostics, not allocator accounting, CE time or formal throughput samples.
"""

import argparse
import csv
import json
import signal
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

FIELDS = (
    "index",
    "uuid",
    "memory.used",
    "utilization.gpu",
    "temperature.gpu",
    "clocks.current.sm",
    "clocks.current.memory",
    "power.draw",
)
UNITS = ("ordinal", "uuid", "MiB", "percent", "C", "MHz", "MHz", "W")


def scheduler_observation(port):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(
            f"http://127.0.0.1:{port}/server_info", timeout=0.5
        ) as response:
            info = json.load(response)
        return {
            "status": "server_info_observed",
            "states": [
                {
                    key: state[key]
                    for key in (
                        "mpu_forward_progress",
                        "memory_usage",
                        "disable_cuda_graph",
                        "cuda_graph_bs",
                        "effective_max_running_requests_per_dp",
                    )
                    if key in state
                }
                for state in info.get("internal_states", [])
            ],
            "scope": "Last completed forward progress; repeated sequence does not establish current execution phase",
        }
    except (OSError, ValueError) as error:
        return {"status": "not_observed", "error": str(error)}


def sample(port):
    started = time.monotonic_ns()
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=" + ",".join(FIELDS),
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        rows = (
            list(csv.reader(query.stdout.splitlines())) if query.returncode == 0 else []
        )
        error = (
            (query.stderr.strip() or f"nvidia-smi exited with code {query.returncode}")
            if query.returncode
            else None
        )
        if any(len(row) != len(FIELDS) for row in rows):
            rows, error = [], "Malformed nvidia-smi CSV row"
    except (OSError, subprocess.SubprocessError) as exception:
        rows, error = [], str(exception)
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "monotonic_ns": started,
        "nvml_query_end_monotonic_ns": time.monotonic_ns(),
        "query_status": "failed" if error else "success",
        "query_error": error,
        "gpus": [
            dict(zip(FIELDS, (value.strip() for value in row), strict=True))
            for row in rows
        ],
        "scheduler": scheduler_observation(port),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=14400)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--coverage-note", default="Started before campaign execution")
    args = parser.parse_args(argv)
    if args.interval < 1 or args.duration <= 0:
        parser.error("Use an interval of at least one second and positive duration")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix = str(args.output_prefix)
    metadata = {
        "schema_version": 1,
        "fields": dict(zip(FIELDS, UNITS, strict=True)),
        "interval_seconds": args.interval,
        "coverage_note": args.coverage_note,
        "claim": "NVML GPU totals and observed last-forward state only; no CE attribution or allocator ownership",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "started_monotonic_ns": time.monotonic_ns(),
        "status": "running",
    }
    Path(prefix + "-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    count = 0
    deadline = time.monotonic() + args.duration
    with (
        Path(prefix + ".jsonl").open("x", encoding="utf-8") as raw,
        Path(prefix + ".csv").open("x", encoding="utf-8", newline="") as flat,
    ):
        writer = csv.DictWriter(flat, fieldnames=["utc", "monotonic_ns", *FIELDS])
        writer.writeheader()
        next_sample = time.monotonic()
        while (
            not stopping and not args.stop_file.exists() and time.monotonic() < deadline
        ):
            item = sample(args.port)
            raw.write(json.dumps(item, ensure_ascii=False) + "\n")
            raw.flush()
            for gpu in item["gpus"]:
                writer.writerow(
                    {"utc": item["utc"], "monotonic_ns": item["monotonic_ns"], **gpu}
                )
            flat.flush()
            count += 1
            next_sample += args.interval
            time.sleep(max(0, next_sample - time.monotonic()))
    metadata.update(
        status="completed",
        samples=count,
        ended_utc=datetime.now(timezone.utc).isoformat(),
        ended_monotonic_ns=time.monotonic_ns(),
        stopped_by="signal"
        if stopping
        else "stop_file"
        if args.stop_file.exists()
        else "duration",
    )
    Path(prefix + "-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
