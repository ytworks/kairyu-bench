"""SciCode step tests: the official harness lines appended to each step's
code, run with the SciCode environment's interpreter (exit 0 = pass)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Official limit is 1,800 s per step; a step that runs longer than this is a
# failure here so one runaway script cannot hold a timed tier.
STEP_TIMEOUT_S = 300


def _run(python: str, h5: str, job: dict, directory: Path) -> bool:
    script = directory / f"{job['key'].replace(':', '_')}.py"
    lines = [job["code"], "", "from scicode.parse.parse import process_hdf5_to_tuple", ""]
    lines.append(f"targets = process_hdf5_to_tuple('{job['step']}', {len(job['tests'])}, '{h5}')")
    for index, test in enumerate(job["tests"]):
        lines.append(f"target = targets[{index}]\n")
        lines.extend(test.split("\n"))
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        subprocess.run([python, str(script)], check=True, capture_output=True, text=True, timeout=STEP_TIMEOUT_S, cwd=directory)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


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
        verdicts = {job["key"]: passed for job, passed in zip(payload["jobs"], results, strict=True)}
    Path(sys.argv[2]).write_text(json.dumps(verdicts), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
