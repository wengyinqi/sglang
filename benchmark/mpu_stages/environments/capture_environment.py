"""Run inside the finished Linux image; no model, GPU, or network access."""

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path


def capture(expected, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    (output_dir / "pip-freeze.txt").write_text(freeze, encoding="utf-8")
    packages = {}
    for name, wanted in expected["expected_packages"].items():
        actual = importlib.metadata.version(name)
        packages[name] = actual
        if actual.split("+", 1)[0] != wanted:
            raise RuntimeError(f"{name}: expected {wanted}, installed {actual}")
    origin = importlib.util.find_spec("sglang").origin
    source = Path("/sgl-workspace/sglang/python").resolve()
    if not origin or source not in Path(origin).resolve().parents:
        raise RuntimeError(f"SGLang is not loaded from archived source: {origin}")
    result = {
        "schema_version": 1,
        "platform": platform.platform(),
        "python": sys.version,
        "packages": packages,
        "sglang_source": origin,
        "source_commit": expected["source_commit"],
        "source_archive_sha256": expected["source_archive_sha256"],
        "pip_freeze": freeze,
        "validation_scope": "package metadata and source location only; no GPU/model execution",
    }
    (output_dir / "environment-runtime.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    capture(json.loads(args.expected.read_text(encoding="utf-8")), args.output_dir)


if __name__ == "__main__":
    main()
