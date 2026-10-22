"""Opt-in Linux/CUDA acceptance test for the shared Quest selection rule.

SGLANG_QUEST_E2E_MODEL=/local/Qwen-model python -m unittest \
    test.manual.hisparse.test_quest_offload_equivalence -v

This runs a locally available model in four fresh processes: Quest full-HBM
and Quest + HiSparse, each in eager and CUDA-graph mode.  Prompts exceed both
top-k and the hot cache; mixed lengths, page-boundary crossings, graph batch
padding and repeated request-slot reuse are exercised.  It compares generated
token IDs and log probabilities, not only successful text generation.

Default top-k=256 is a small-model regression, not a paper reproduction.
Set SGLANG_QUEST_E2E_TOP_K=2048 with the paper checkpoint to validate its
selection budget.  This still uses short sparse prompts, not 32K/200K.
Throughput/quality reproduction remains a separate job.

Hardware and execution settings:
  SGLANG_QUEST_E2E_TP_SIZE       tensor parallel size (default 1)
  SGLANG_QUEST_E2E_MEM_FRACTION  static GPU memory fraction (default 0.75)
  SGLANG_QUEST_E2E_TIMEOUT      seconds per worker (default 1800)
  SGLANG_QUEST_E2E_OVERLAP      1 enables overlap scheduling (default 0)

For example, use TP_SIZE=2 and MEM_FRACTION=0.85 for a two-GPU validation
of the paper checkpoint after checking available memory.  Default results
cover eager/graph execution with overlap disabled; rerun with OVERLAP=1
to validate overlap separately.  Each worker prints its effective settings.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def environment_settings():
    settings = {
        "top_k": int(os.environ.get("SGLANG_QUEST_E2E_TOP_K", "256")),
        "tp_size": int(os.environ.get("SGLANG_QUEST_E2E_TP_SIZE", "1")),
        "mem_fraction": float(os.environ.get("SGLANG_QUEST_E2E_MEM_FRACTION", "0.75")),
        "timeout": int(os.environ.get("SGLANG_QUEST_E2E_TIMEOUT", "1800")),
    }
    if settings["top_k"] <= 0 or settings["top_k"] % 64:
        raise ValueError("SGLANG_QUEST_E2E_TOP_K must be a positive multiple of 64.")
    if settings["tp_size"] <= 0 or settings["timeout"] <= 0:
        raise ValueError("SGLANG_QUEST_E2E_TP_SIZE and TIMEOUT must be positive.")
    if not 0 < settings["mem_fraction"] <= 1:
        raise ValueError("SGLANG_QUEST_E2E_MEM_FRACTION must be in (0, 1].")
    overlap = os.environ.get("SGLANG_QUEST_E2E_OVERLAP", "0")
    if overlap not in ("0", "1"):
        raise ValueError("SGLANG_QUEST_E2E_OVERLAP must be 0 or 1.")
    settings["overlap"] = overlap == "1"
    return settings


def run_worker_process(command, timeout):
    """Reap the worker and its own Linux process group, including on timeout.

    Killing only the Python worker can leave its Engine tensor-parallel
    processes resident on the GPUs.  A fresh session makes the group belong
    to this invocation, so cleanup cannot signal the caller's process group.
    """
    worker_env = os.environ.copy()
    python_path = str(REPO_ROOT / "python")
    worker_env["PYTHONPATH"] = os.pathsep.join(
        [python_path, worker_env["PYTHONPATH"]]
        if worker_env.get("PYTHONPATH")
        else [python_path]
    )
    process = subprocess.Popen(
        command, start_new_session=True, cwd=REPO_ROOT, env=worker_env
    )
    try:
        return_code = process.wait(timeout=timeout)
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        # The worker may have exited before its children.  Signal the group
        # even when process.poll() would already report the worker's exit.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def run_worker(mode, result_path):
    import torch
    from transformers import AutoTokenizer

    import sglang as sgl

    sglang_source = Path(sgl.__file__).resolve()
    if not sglang_source.is_relative_to((REPO_ROOT / "python" / "sglang").resolve()):
        raise RuntimeError(
            f"Worker imported SGLang outside this checkout: {sglang_source}"
        )
    if sys.platform != "linux" or not torch.cuda.is_available():
        raise RuntimeError(
            "This acceptance test requires Linux and an NVIDIA CUDA GPU."
        )
    model_path = os.environ["SGLANG_QUEST_E2E_MODEL"]
    if not Path(model_path).is_dir():
        raise ValueError(
            "SGLANG_QUEST_E2E_MODEL must name an existing local model directory."
        )
    settings = environment_settings()
    if settings["tp_size"] > torch.cuda.device_count():
        raise ValueError("SGLANG_QUEST_E2E_TP_SIZE exceeds the visible CUDA GPU count.")
    top_k = settings["top_k"]
    page_size = 64
    output_len = 80
    context_len = 6 * top_k + 2 * page_size + output_len
    cfg = {
        "algorithm": "quest",
        "top_k": top_k,
        "quest_page_size": page_size,
        "device_buffer_size": 2 * top_k,
        "host_to_device_ratio": 2,
    }
    offload = mode.startswith("hisparse")
    kwargs = dict(
        model_path=model_path,
        tp_size=settings["tp_size"],
        hisparse_config=json.dumps(cfg),
        dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        disable_radix_cache=True,
        disable_cuda_graph=mode.endswith("eager"),
        cuda_graph_bs=[2, 4],
        max_running_requests=4,
        max_total_tokens=32 * top_k,
        context_length=context_len,
        chunked_prefill_size=2 * top_k,
        page_size=page_size,
        random_seed=42,
        mem_fraction_static=settings["mem_fraction"],
        disable_overlap_schedule=not settings["overlap"],
        log_level="warning",
    )
    if offload:
        kwargs.update(
            enable_hisparse=True,
            prefill_attention_backend="flashinfer",
            decode_attention_backend="flashinfer_hisparse",
        )
    else:
        kwargs.update(attention_backend="flashinfer_quest")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    base_ids = tokenizer.encode(
        "Consider the following numbered facts. "
        + " ".join(f"Record {i} has value {i * 13 % 97}." for i in range(256)),
        add_special_tokens=False,
    )
    lengths = [4 * top_k + 1, 4 * top_k + page_size, 6 * top_k + 1]
    rounds = [lengths, [65, 2 * top_k + 1], lengths]
    records = []
    print(
        "Quest acceptance requested configuration: "
        + json.dumps(
            {
                "mode": mode,
                "sglang_module": str(sglang_source),
                "engine": kwargs,
                "prompt_lengths": rounds,
                "output_len": output_len,
                "timeout": settings["timeout"],
            }
        ),
        flush=True,
    )
    engine = sgl.Engine(**kwargs)
    effective_config = {key: getattr(engine.server_args, key) for key in kwargs}
    print(
        "Quest acceptance effective configuration: "
        + json.dumps(
            {
                "mode": mode,
                "sglang_module": str(sglang_source),
                "engine": effective_config,
            }
        ),
        flush=True,
    )
    try:
        for lens in rounds:
            inputs = [
                (base_ids * ((n + len(base_ids) - 1) // len(base_ids)))[:n]
                for n in lens
            ]
            outputs = engine.generate(
                input_ids=inputs,
                sampling_params={
                    "temperature": 0.0,
                    "max_new_tokens": output_len,
                    "ignore_eos": True,
                },
                return_logprob=True,
            )
            if not isinstance(outputs, list) or len(outputs) != len(lens):
                raise AssertionError("The engine did not return exactly every request.")
            batch = []
            for n, output in zip(lens, outputs):
                meta = output["meta_info"]
                token_logprobs = meta["output_token_logprobs"]
                if meta["prompt_tokens"] != n or len(token_logprobs) != output_len:
                    raise AssertionError(f"Unexpected request length/count: {meta}")
                batch.append(
                    {
                        "tokens": [x[1] for x in token_logprobs],
                        "logprobs": [x[0] for x in token_logprobs],
                    }
                )
            if len(batch) != len(lens):
                raise AssertionError("The engine did not return every request.")
            records.append(batch)
    finally:
        engine.shutdown()
    Path(result_path).write_text(
        json.dumps(
            {
                "mode": mode,
                "sglang_module": str(sglang_source),
                "engine": effective_config,
                "records": records,
            }
        ),
        encoding="utf-8",
    )


@unittest.skipUnless(
    os.environ.get("SGLANG_QUEST_E2E_MODEL"), "Set SGLANG_QUEST_E2E_MODEL to opt in."
)
class TestQuestOffloadEquivalence(unittest.TestCase):
    def test_sparse_decode_and_slot_reuse(self):
        import torch

        self.assertEqual(sys.platform, "linux", "This acceptance test requires Linux.")
        settings = environment_settings()
        results = {}
        with tempfile.TemporaryDirectory(prefix="quest-equivalence-") as directory:
            for mode in (
                "quest-eager",
                "hisparse-eager",
                "quest-graph",
                "hisparse-graph",
            ):
                path = Path(directory) / f"{mode}.json"
                run_worker_process(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        mode,
                        str(path),
                    ],
                    timeout=settings["timeout"],
                )
                results[mode] = json.loads(path.read_text(encoding="utf-8"))["records"]
        reference = results["quest-eager"]
        self.assertEqual(len(reference), 3)
        for mode, batches in results.items():
            self.assertEqual(len(batches), len(reference), mode)
            self.assertEqual([len(batch) for batch in batches], [3, 2, 3], mode)
            for batch_idx, (expected, actual) in enumerate(zip(reference, batches)):
                self.assertEqual(len(actual), len(expected), (mode, batch_idx))
                for req_idx, (left, right) in enumerate(zip(expected, actual)):
                    with self.subTest(mode=mode, batch=batch_idx, request=req_idx):
                        self.assertEqual(left["tokens"], right["tokens"])
                        torch.testing.assert_close(
                            torch.tensor(left["logprobs"]),
                            torch.tensor(right["logprobs"]),
                            rtol=0,
                            atol=0.02,
                        )
            # The first and last batches use identical inputs after slot reuse.
            for before, after in zip(batches[0], batches[2]):
                self.assertEqual(before["tokens"], after["tokens"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        run_worker(sys.argv[2], sys.argv[3])
    else:
        unittest.main()
