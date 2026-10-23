"""Strict output normalization and comparison; independent of serving imports."""

import math


def normalize_outputs(outputs, lengths, output_len):
    if not isinstance(outputs, list) or len(outputs) != len(lengths):
        raise ValueError("Server must return exactly every request")
    normalized = []
    for output, prompt_len in zip(outputs, lengths):
        meta = output.get("meta_info") if isinstance(output, dict) else None
        if not isinstance(meta, dict):
            raise ValueError("Server response is missing request metadata")
        records = meta.get("output_token_logprobs", [])
        if (
            meta.get("prompt_tokens") != prompt_len
            or meta.get("completion_tokens") != output_len
        ):
            raise ValueError("Actual request token counts differ from planned lengths")
        if len(records) != output_len:
            raise ValueError("Missing output token logprobs")
        tokens, logprobs = [], []
        for record in records:
            if not isinstance(record, (list, tuple)) or len(record) < 2:
                raise ValueError("Malformed output token logprob record")
            logprob, token = record[:2]
            if (
                type(token) is not int
                or not isinstance(logprob, (int, float))
                or not math.isfinite(logprob)
            ):
                raise ValueError("Non-finite logprob or malformed token ID")
            tokens.append(token)
            logprobs.append(logprob)
        normalized.append({"tokens": tokens, "logprobs": logprobs})
    return normalized


def compare(reference, actual, atol):
    if len(reference) != len(actual):
        raise ValueError("Missing correctness rounds")
    max_error = 0.0
    for round_id, (left, right) in enumerate(zip(reference, actual)):
        if len(left) != len(right):
            raise ValueError(f"Request count changed in round {round_id}")
        for request_id, (expected, observed) in enumerate(zip(left, right)):
            if expected["tokens"] != observed["tokens"]:
                divergence = next(
                    (
                        i
                        for i, (a, b) in enumerate(
                            zip(expected["tokens"], observed["tokens"])
                        )
                        if a != b
                    ),
                    min(len(expected["tokens"]), len(observed["tokens"])),
                )
                raise ValueError(
                    f"Token divergence at round={round_id} request={request_id} token={divergence}; inspect saved raw responses"
                )
            if len(expected["logprobs"]) != len(observed["logprobs"]):
                raise ValueError("Missing comparison logprobs")
            for token_id, (a, b) in enumerate(
                zip(expected["logprobs"], observed["logprobs"])
            ):
                delta = abs(a - b)
                max_error = max(max_error, delta)
                if not math.isfinite(delta) or delta > atol:
                    raise ValueError(
                        f"Logprob mismatch at round={round_id} request={request_id} token={token_id}: abs_error={delta} > {atol}"
                    )
    return {
        "exact_greedy_token_match": True,
        "max_abs_logprob_error": max_error,
        "logprob_atol": atol,
    }
