"""Portable regression tests for lifecycle evidence and the loopback HTTP client."""

import copy
import importlib.util
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "lifecycle_tests", ROOT / "benchmark/mpu_stages/lifecycle.py"
)
L = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(L)


def response(reason="abort", tokens=3, retracts=0):
    return {
        "meta_info": {
            "finish_reason": {"type": reason},
            "completion_tokens": tokens,
            "num_retractions": retracts,
        }
    }


def facts(prefill=80, decode=0):
    return [
        {"rank": rank, "prefill_tokens": prefill, "decode_forwards": decode}
        for rank in range(4)
    ]


def test_fault_evidence_accepts_only_observed_phases():
    L.validate_fault("abort_prefill", response(tokens=0), facts(), 100, 20)
    L.validate_fault("abort_decode", response(), facts(100, 2), 100, 20)
    L.validate_fault(
        "retract_decode", response("length", 20, 1), facts(202, 19), 100, 20
    )
    invalid = [
        ("abort_prefill", response(), []),
        ("abort_prefill", response(), facts(100, 0)),
        ("abort_prefill", response(), facts(80, 1)),
        ("abort_decode", response("length", 20), facts(100, 19)),
        ("abort_decode", response(), facts(100, 0)),
        ("retract_decode", response("length", 20, 0), facts(202, 19)),
        ("retract_decode", response("length", 20, 1), facts(100, 19)),
    ]
    for case, result, evidence in invalid:
        with pytest.raises(ValueError):
            L.validate_fault(case, result, evidence, 100, 20)


def test_rank_trace_alignment_and_request_attribution():
    traces = []
    for rank in range(4):
        traces.append(
            [
                {"type": "session", "tp_rank": rank, "tp_size": 4},
                {
                    "type": "forward",
                    "request_ids": ["other", "target"],
                    "success": True,
                    "forward_mode": "extend",
                    "extend_lens": [11, 64],
                },
                {
                    "type": "forward",
                    "request_ids": ["target"],
                    "success": True,
                    "forward_mode": "decode",
                    "extend_lens": None,
                },
            ]
        )
    result = L.forward_facts(traces, "target", 4)
    assert all(f["prefill_tokens"] == 64 and f["decode_forwards"] == 1 for f in result)
    assert L.forward_facts(traces, "missing", 4) == []
    for bad in (traces[:3], traces + traces[:1]):
        with pytest.raises(ValueError):
            L.forward_facts(bad, "target", 4)
    bad = copy.deepcopy(traces)
    bad[2][1]["extend_lens"] = [64, 11]
    with pytest.raises(ValueError):
        L.forward_facts(bad, "target", 4)


@pytest.mark.parametrize("case", ["abort_decode", "retract_decode"])
def test_loopback_stream_fault_client_and_cleanup(case):
    state = {"abort": False, "retracts": 0, "routes": []}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                state["routes"].append(self.path)
                if self.path == "/abort_request":
                    state["abort"] = True
                if self.path == "/pause_generation":
                    assert payload["mode"] == "retract"
                    state["retracts"] += 1
            self.send_response(200)
            self.end_headers()
            if self.path != "/generate":
                return  # Exercise successful control endpoints with empty bodies.
            for count in range(1, 21):
                with lock:
                    reason = (
                        "abort"
                        if state["abort"]
                        else ("length" if count == 20 else None)
                    )
                    item = {
                        "meta_info": {
                            "id": payload["rid"],
                            "completion_tokens": count,
                            "num_retractions": state["retracts"],
                            "finish_reason": {"type": reason} if reason else None,
                        }
                    }
                self.wfile.write(b"data: " + json.dumps(item).encode() + b"\n\n")
                self.wfile.flush()
                if reason:
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                time.sleep(0.01)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        args = NS(port=server.server_port, output_len=20, benchmark_timeout=5)
        rid, result = L.fault_request(args, [1, 2, 3], case)
        assert result["meta_info"]["id"] == rid
        if case == "abort_decode":
            assert result["meta_info"]["finish_reason"]["type"] == "abort"
            assert "/abort_request" in state["routes"]
        else:
            assert result["meta_info"]["num_retractions"] == 1
            assert state["routes"] == [
                "/generate",
                "/pause_generation",
                "/continue_generation",
            ]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
