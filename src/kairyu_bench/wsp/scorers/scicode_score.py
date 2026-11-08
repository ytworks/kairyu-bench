"""SciCode step tests: the official harness lines appended to each step's
code, run with the SciCode environment's interpreter (exit 0 = pass)."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Official limit is 1,800 s per step; a step that runs longer than this is a
# failure here so one runaway script cannot hold a timed tier.
STEP_TIMEOUT_S = 300


# Only class names from this fixed allowlist may leave the verifier. Generated
# code can print arbitrary text, so this is a stderr-summary hint, not a proven
# Python exception object or a causal diagnosis.
_EXCEPTION_CLASSES = frozenset({
    "AssertionError", "AttributeError", "AxisError", "EOFError", "FloatingPointError",
    "ImportError", "IndentationError", "IndexError", "KeyError", "LinAlgError",
    "MemoryError", "ModuleNotFoundError", "NameError", "NotImplementedError",
    "OverflowError", "RecursionError", "RuntimeError", "SyntaxError", "TabError",
    "TypeError", "UnboundLocalError", "ValueError", "ZeroDivisionError",
})


def _stream_metadata(value: str | bytes | None) -> dict:
    if value is None:
        return {"bytes": None, "sha256": None, "representation": None}
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return {
        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        "representation": "utf8_of_captured_text" if isinstance(value, str) else "raw_timeout_bytes",
    }


def _exception_class(stderr: str | bytes | None) -> str | None:
    text = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else stderr or ""
    # Exception messages/paths/source lines are deliberately never returned.
    for line in reversed(text.splitlines()):
        match = re.fullmatch(r"(?:[A-Za-z_]\w*\.)*([A-Za-z_]\w*)(?::(?: .*)?)?", line)
        if match:
            name = match.group(1)
            return name if name in _EXCEPTION_CLASSES else None
    return None


def _run(python: str, h5: str, job: dict, directory: Path) -> tuple[bool, dict]:
    script = directory / f"{job['key'].replace(':', '_')}.py"
    lines = [job["code"], "", "from scicode.parse.parse import process_hdf5_to_tuple", ""]
    lines.append(f"targets = process_hdf5_to_tuple('{job['step']}', {len(job['tests'])}, '{h5}')")
    for index, test in enumerate(job["tests"]):
        lines.append(f"target = targets[{index}]\n")
        lines.extend(test.split("\n"))
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    started = time.perf_counter()
    timed_out = False
    returncode: int | None = None
    stdout: str | bytes | None
    stderr: str | bytes | None
    try:
        completed = subprocess.run(
            [python, str(script)], check=True, capture_output=True, text=True, timeout=STEP_TIMEOUT_S, cwd=directory,
        )
        passed, returncode, stdout, stderr = True, completed.returncode, completed.stdout, completed.stderr
    except subprocess.CalledProcessError as error:
        passed, returncode, stdout, stderr = False, error.returncode, error.stdout, error.stderr
    except subprocess.TimeoutExpired as error:
        passed, timed_out, stdout, stderr = False, True, error.stdout, error.stderr
    elapsed = time.perf_counter() - started
    return passed, {
        "returncode": returncode, "timed_out": timed_out, "elapsed_seconds": elapsed,
        "exception_class": _exception_class(stderr) if not passed and not timed_out else None,
        "exception_class_origin": "stderr_allowlisted_hint",
        "stdout": _stream_metadata(stdout), "stderr": _stream_metadata(stderr),
    }


def _check_environment(python: str, h5: str, job: dict, directory: Path) -> None:
    """A failing test must mean the model's code failed: the interpreter has to
    import SciCode's harness and read this step's targets first."""

    script = directory / "environment-check.py"
    script.write_text(
        "from scicode.parse.parse import process_hdf5_to_tuple\n"
        f"process_hdf5_to_tuple('{job['step']}', {len(job['tests'])}, '{h5}')\n",
        encoding="utf-8",
    )
    completed = subprocess.run([python, str(script)], capture_output=True, text=True, timeout=STEP_TIMEOUT_S)
    if completed.returncode != 0:
        raise RuntimeError(f"SciCode test environment is broken: {completed.stderr[-1500:]}")


def main() -> int:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="wsp-scicode-") as directory, ThreadPoolExecutor(8) as pool:
        if payload["jobs"]:
            _check_environment(payload["python"], payload["h5"], payload["jobs"][0], Path(directory))
        results = pool.map(lambda job: _run(payload["python"], payload["h5"], job, Path(directory)), payload["jobs"])
        records = list(zip(payload["jobs"], results, strict=True))
        verdicts = {job["key"]: result[0] for job, result in records}
        diagnostics = {job["key"]: result[1] for job, result in records}
    Path(sys.argv[2]).write_text(
        json.dumps({"schema_version": 1, "verdicts": verdicts, "diagnostics": diagnostics}), encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
