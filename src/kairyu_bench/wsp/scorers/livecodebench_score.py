"""LiveCodeBench codegen_metrics on ``[{key, sample, code}]`` (official tests)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    from lcb_runner.evaluation import codegen_metrics, extract_instance_results

    jobs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    # A failed test must mean the model's code failed: a trivial stdin
    # program has to pass in this environment first.
    canary = {"input_output": json.dumps({"inputs": ["1 2\n"], "outputs": ["3\n"], "fn_name": None})}
    check = codegen_metrics([canary], [["a, b = map(int, input().split())\nprint(a + b)"]], num_process_evaluate=1, timeout=6)
    if not extract_instance_results(check[1])[0][0]:
        raise RuntimeError(f"LiveCodeBench test environment is broken: {check[2]}")
    metrics = codegen_metrics(
        [job["sample"] for job in jobs], [[job["code"]] for job in jobs], num_process_evaluate=8, timeout=6
    )
    graded = extract_instance_results(metrics[1])
    verdicts = {job["key"]: {"passed": bool(grades[0])} for job, grades in zip(jobs, graded, strict=True)}
    Path(sys.argv[2]).write_text(json.dumps(verdicts), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
