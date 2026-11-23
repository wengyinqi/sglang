"""Streaming complete-idle validation for four Attention-DP/EP schedulers."""

import json
import re
from itertools import zip_longest
from pathlib import Path


def trace_identity(paths):
    ranks, pids = {}, set()
    for path in paths:
        path = Path(path)
        with path.open(encoding="utf-8") as stream:
            session = json.loads(stream.readline())
        rank = session.get("attn_dp_rank")
        if (
            session.get("type") != "session"
            or session.get("schema_version") != 1
            or type(rank) is not int
            or rank not in range(4)
            or rank in ranks
        ):
            raise ValueError("Missing or duplicate four-rank trace session")
        if not (
            session.get("tp_rank") == rank
            and session.get("tp_size") == 4
            and session.get("attn_dp_size") == 4
            and session.get("attn_tp_size") == 1
            and session.get("ep_size") == 4
            and session.get("moe_tp_size") == 1
            and session.get("dp_size") == 1
            and session.get("pp_size") == 1
            and session.get("disable_overlap_schedule") is True
            and session.get("disable_radix_cache") is True
            and session.get("parallel_strategy") == "attention_dp4_ep4"
            and session.get("tensor_parallel_audit", {}).get("status") == "passed"
        ):
            raise ValueError("Trace is not audited Attention-DP4/EP4")
        match = re.search(r"-pid(\d+)\.jsonl$", path.name)
        if not match or int(match.group(1)) in pids:
            raise ValueError("Missing or duplicate scheduler PID")
        pid = int(match.group(1))
        pids.add(pid)
        ranks[rank] = {
            "path": str(path.resolve()),
            "scheduler_pid": pid,
            "session": session,
        }
    if set(ranks) != set(range(4)):
        raise ValueError("Need four complete trace ranks")
    return {
        "trace_dir": str(Path(paths[0]).resolve().parent),
        "scheduler_pids": sorted(pids),
        "per_rank": [ranks[rank] for rank in range(4)],
    }


def forwards(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get("type") == "forward":
                    yield row


def scan_idle(identity, expected, scanner):
    rows = [
        scanner(rank["path"], expected, rank["session"])
        for rank in identity["per_rank"]
    ]
    if len({row["last_sequence"] for row in rows}) != 1:
        raise ValueError("Idle EP participants disagree on completed sequence")
    owners = {}
    facts = {
        rid: {"prefill_tokens": 0, "decode_forwards": 0, "no_forward_observed": True}
        for rid in expected
    }
    for rank, row in enumerate(rows):
        for rid, fact in row["request_facts"].items():
            if fact["prefill_tokens"] or fact["decode_forwards"]:
                if rid in owners:
                    raise ValueError(
                        "Request migrated or duplicated across idle rank traces"
                    )
                owners[rid] = rank
                facts[rid] = {**fact, "attn_dp_owner": rank}
    for sequence, group in enumerate(
        zip_longest(*(forwards(row["path"]) for row in identity["per_rank"]))
    ):
        if any(row is None for row in group):
            raise ValueError("Missing global EP participant step")
        modes, counts = group[0].get("dp_forward_modes"), group[0].get("dp_num_tokens")
        if (
            not isinstance(modes, list)
            or len(modes) != 4
            or not isinstance(counts, list)
            or len(counts) != 4
        ):
            raise ValueError("Missing idle trace DP vectors")
        for rank, row in enumerate(group):
            ids = row.get("request_ids", [])
            mode = row.get("forward_mode")
            tokens = sum(row.get("extend_lens") or []) if mode == "extend" else len(ids)
            if (
                row.get("sequence") != sequence
                or row.get("dp_forward_modes") != modes
                or row.get("dp_num_tokens") != counts
                or mode != modes[rank]
                or counts[rank] != tokens
                or (mode == "idle") != (not ids)
            ):
                raise ValueError("Idle trace EP step/vector/local token mismatch")
    return {
        "last_sequence": rows[0]["last_sequence"],
        "request_facts": facts,
        "request_owners": owners,
        "capacity": {
            "used_slots": 0,
            "reserved_slots": 0,
            "per_rank": [row["capacity"] for row in rows],
        },
        "per_rank": rows,
        "scope": "All four scheduler lifetimes and complete global idle flush; no-forward cancellations remain ownerless",
    }
