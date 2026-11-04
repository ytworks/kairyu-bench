"""EvalPlus 0.3.1 in its own environment.

``--dump OUT``: the pinned HumanEval+ v0.1.10 / MBPP+ v0.2.0 prompts and entry
points. ``IN OUT``: sanitize each reply and run the official base and plus
tests (an item passes when both pass), as ``evalplus.evaluate`` does.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

# evalplus 0.3.1 defaults to 1 s; upstream later raised it to 4 s to avoid
# timing false negatives under load, which also keeps verdicts steadier here.
MIN_TIME_LIMIT_S = 4.0


def _problems(dataset: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash, get_mbpp_plus, get_mbpp_plus_hash
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.evaluate import get_groundtruth

    if dataset == "humaneval":
        problems = get_human_eval_plus()
        return problems, get_groundtruth(problems, get_human_eval_plus_hash(), [])
    problems = get_mbpp_plus()
    return problems, get_groundtruth(problems, get_mbpp_plus_hash(), MBPP_OUTPUT_NOT_NONE_TASKS)


def _check(dataset: str, problem: dict[str, Any], solution: str, expected: dict[str, Any]) -> dict[str, Any]:
    from evalplus.eval import PASS
    from evalplus.evaluate import check_correctness

    result = check_correctness(
        dataset, 0, problem, solution, expected, False, True, None, MIN_TIME_LIMIT_S
    )
    base, plus = result["base"][0], result["plus"][0]
    return {"base": base, "plus": plus, "passed": base == PASS and plus == PASS}


def main() -> int:
    if sys.argv[1] == "--dump":
        tasks = {}
        for dataset in ("humaneval", "mbpp"):
            problems, _ = _problems(dataset)
            tasks[dataset] = {
                task_id: {"prompt": problem["prompt"], "entry_point": problem["entry_point"]}
                for task_id, problem in problems.items()
            }
        Path(sys.argv[2]).write_text(json.dumps(tasks), encoding="utf-8")
        return 0

    from evalplus.sanitize import sanitize

    jobs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    loaded = {dataset: _problems(dataset) for dataset in sorted({job["dataset"] for job in jobs})}
    # A failed test must mean the model's code failed: each dataset's first
    # canonical solution has to pass in this environment first.
    for dataset, (problems, expected) in loaded.items():
        task_id = next(job["task_id"] for job in jobs if job["dataset"] == dataset)
        problem = problems[task_id]
        if not _check(dataset, problem, problem["prompt"] + problem["canonical_solution"], expected[task_id])["passed"]:
            raise RuntimeError(f"EvalPlus environment cannot pass the canonical solution of {task_id}")
    verdicts: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {}
        for job in jobs:
            problems, expected = loaded[job["dataset"]]
            solution = sanitize(job["content"], job["entry_point"])
            futures[job["key"]] = pool.submit(
                _check, job["dataset"], problems[job["task_id"]], solution, expected[job["task_id"]]
            )
        for key, future in futures.items():
            verdicts[key] = future.result()
    Path(sys.argv[2]).write_text(json.dumps(verdicts), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
