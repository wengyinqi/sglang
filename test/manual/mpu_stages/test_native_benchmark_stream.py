"""CPU native-/generate protocol tests using the production benchmark client.

The HTTP stream is simulated; the actual request coroutine and output dataclass
are compiled without importing unrelated serving/CUDA dependencies.
"""

import ast
import asyncio
import contextlib
import io
import json
import sys
import time
import traceback
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[3] / "python/sglang/benchmark/serving.py"


class _Content:
    def __init__(self, frames):
        self.frames = frames

    async def __aiter__(self):
        for frame in self.frames:
            yield frame


class _Response:
    def __init__(self, frames):
        self.status = 200
        self.content = _Content(frames)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _Session:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def post(self, **kwargs):
        return self.response


def _requester(frames):
    source = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {
        "RequestFuncOutput",
        "async_request_sglang_generate",
        "_record_server_prompt_len",
    }
    nodes = [n for n in source.body if getattr(n, "name", None) in names]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + nodes,
        type_ignores=[],
    )
    namespace = {
        "__name__": __name__,
        "dataclass": dataclass,
        "field": field,
        "json": json,
        "orjson": SimpleNamespace(loads=json.loads),
        "time": time,
        "traceback": traceback,
        "sys": sys,
        "_ROUTING_KEY_HEADER": "X-SMG-Routing-Key",
        "get_request_headers": lambda: {},
        "remove_prefix": lambda text, prefix: text.removeprefix(prefix),
        "_create_bench_client_session": lambda: _Session(_Response(frames)),
        "args": SimpleNamespace(
            temperature=0.0,
            top_p=1.0,
            disable_ignore_eos=False,
            disable_stream=False,
            return_logprob=False,
            return_routed_experts=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
        ),
    }
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["async_request_sglang_generate"]


def _frame(tokens, *, text="", finished=False, prompt_tokens=3):
    return {
        "text": text,
        "meta_info": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": tokens,
            "finish_reason": {"type": "length", "length": tokens} if finished else None,
        },
    }


class TestNativeBenchmarkAccounting(unittest.TestCase):
    def _run(self, frames, *, streaming=True):
        encoded = [
            (
                (("data: " if streaming else "") + json.dumps(frame) + "\n\n").encode()
                if isinstance(frame, dict)
                else frame
            )
            for frame in frames
        ]
        request = SimpleNamespace(
            prompt=[10, 11, 12],
            api_url="http://localhost/generate",
            prompt_len=4,
            output_len=8,
            lora_name=None,
            image_data=None,
            extra_request_body={},
            routing_key=None,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            return asyncio.run(_requester(encoded)(request))

    def test_empty_text_uses_observed_counts_instead_of_requested_length(self):
        result = self._run([_frame(3, finished=True), b"data: [DONE]\n\n"])
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output_len, 3)
        self.assertEqual(result.prompt_len, 3)
        self.assertEqual(result.generated_text, "")
        self.assertGreater(result.ttft, 0)

    def test_metadata_only_final_frame_updates_count(self):
        final = _frame(8, finished=True)
        del final["text"]
        result = self._run([_frame(1, text="a"), final, b"data: [DONE]\n\n"])
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output_len, 8)
        self.assertEqual(result.generated_text, "a")
        self.assertEqual(len(result.itl), 7)

    def test_request_id_links_stream_to_scheduler_timing(self):
        first, final = _frame(1), _frame(8, finished=True)
        first["meta_info"]["id"] = final["meta_info"]["id"] = "request-a"
        result = self._run([first, final])
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.request_id, "request-a")
        final["meta_info"]["id"] = "request-b"
        result = self._run([first, final])
        self.assertFalse(result.success)
        self.assertIn("request ID changed", result.error)

    def test_truncated_stream_is_not_a_completed_request(self):
        for frames in (
            [],
            [_frame(2, text="ab")],
            [_frame(2, text="ab"), b"data: [DONE]\n\n"],
        ):
            with self.subTest(frames=frames):
                result = self._run(frames)
                self.assertFalse(result.success)
                self.assertIn("without final token accounting", result.error)

    def test_error_frame_and_abort_reason_are_not_successes(self):
        aborted = _frame(2, text="ab", finished=True)
        aborted["meta_info"]["finish_reason"] = {"type": "abort", "message": "OOM"}
        for frame in ({"error": {"message": "OOM"}}, aborted):
            with self.subTest(frame=frame):
                result = self._run([frame, b"data: [DONE]\n\n"])
                self.assertFalse(result.success)
                self.assertIn("OOM", result.error)

    def test_invalid_or_missing_counts_are_rejected(self):
        for key, value in (
            ("completion_tokens", -1),
            ("completion_tokens", True),
            ("prompt_tokens", -1),
            ("prompt_tokens", "3"),
        ):
            frame = _frame(8, finished=True)
            frame["meta_info"][key] = value
            with self.subTest(key=key, value=value):
                self.assertFalse(self._run([frame]).success)
        for key in ("completion_tokens", "prompt_tokens"):
            frame = _frame(8, finished=True)
            del frame["meta_info"][key]
            with self.subTest(missing=key):
                self.assertFalse(self._run([frame]).success)

    def test_regressing_token_count_is_rejected(self):
        result = self._run([_frame(5, text="hello"), _frame(3, finished=True)])
        self.assertFalse(result.success)

    def test_nonstreaming_json_can_finish_without_done_marker(self):
        result = self._run([_frame(8, text="answer", finished=True)], streaming=False)
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output_len, 8)

    def test_prefill_only_response_does_not_invent_output_tokens(self):
        result = self._run([_frame(0, finished=True), b"data: [DONE]\n\n"])
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output_len, 0)


if __name__ == "__main__":
    unittest.main()
