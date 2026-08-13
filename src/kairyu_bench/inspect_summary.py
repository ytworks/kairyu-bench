from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _metric_value(value: object, name: str) -> float:
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Inspect metric {name} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Inspect metric {name} is not finite")
    return result


def summarize_inspect_log(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Inspect log: {error}") from error
    if not isinstance(payload, dict) or payload.get("status") != "success":
        raise ValueError("Inspect evaluation did not finish successfully")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Inspect log has no samples")
    problem_ids: list[str] = []
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("id") is None:
            raise ValueError("Inspect sample has no ID")
        problem_id = str(sample["id"])
        if problem_id in problem_ids:
            raise ValueError("Inspect log contains a duplicate sample ID")
        problem_ids.append(problem_id)

    results = payload.get("results")
    scores = results.get("scores") if isinstance(results, dict) else None
    if not isinstance(scores, list):
        raise ValueError("Inspect log has no aggregate scores")
    scorer = next(
        (
            score
            for score in scores
            if isinstance(score, dict) and score.get("name") == "scicode_scorer"
        ),
        None,
    )
    if not isinstance(scorer, dict) or not isinstance(scorer.get("metrics"), dict):
        raise ValueError("Inspect log has no SciCode scorer metrics")
    raw_metrics = scorer["metrics"]
    main = _metric_value(raw_metrics.get("Problem Correctness"), "Problem Correctness")
    sub = _metric_value(
        raw_metrics.get("sub_problem_correctness"), "sub_problem_correctness"
    )
    return {
        "problem_ids": problem_ids,
        "metrics": {
            "Problem Correctness": main,
            "sub_problem_correctness": sub,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m kairyu_bench.inspect_summary")
    parser.add_argument("inspect_log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = summarize_inspect_log(args.inspect_log)
        args.output.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as error:
        print(f"kairyu-bench Inspect summary: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
