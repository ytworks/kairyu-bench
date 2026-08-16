from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kairyu_bench.benchmarks import BENCHMARK_NAMES
from kairyu_bench.results import BenchmarkResult


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_run_results(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, BenchmarkResult]]:
    metadata = _read_object(run_dir / "run.json")
    selected = metadata.get("benchmarks")
    if not isinstance(selected, list) or not all(
        isinstance(name, str) for name in selected
    ):
        raise ValueError(f"{run_dir}/run.json has no benchmark list")
    results: dict[str, BenchmarkResult] = {}
    for name in selected:
        path = run_dir / "normalized" / f"{name}.json"
        payload = _read_object(path)
        result = BenchmarkResult.from_dict(payload)
        if result.benchmark != name:
            raise ValueError(f"{path} contains result for {result.benchmark}")
        if result.data["run_id"] != metadata.get("run_id"):
            raise ValueError(f"{path} run_id differs from run.json")
        results[name] = result
    return metadata, results


def _score_text(score: float | int | None) -> str:
    return "—" if score is None else f"{float(score):.2f}"


def score_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# kairyu-bench score report: {report['run_id']}",
        "",
        f"Model: `{report['model_id']}`  ",
        f"Endpoint: `{report['endpoint']['fingerprint']}`  ",
    ]
    average = report.get("macro_average_percent")
    if average is None:
        lines.append("Macro average: unavailable")
    else:
        lines.append(f"Macro average: {average:.2f}%")
    lines.extend(
        [
            "",
            "| Benchmark | Status | Score (%) | Evaluated | Scoring | Labels | Agent |",
            "| --- | --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in report["benchmarks"]:
        labels: list[str] = []
        if row["self_judged"]:
            labels.append("self-judged")
        if row["self_simulated"]:
            labels.append("self-simulated")
        label = ", ".join(labels) or "—"
        agent = row["agent"] or "—"
        lines.append(
            f"| {row['benchmark']} | {row['status']} | {_score_text(row['score'])} | "
            f"{row['evaluated']}/{row['requested']} | {row['method']} | {label} | "
            f"{agent} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_score_report(run_dir: Path) -> dict[str, Any]:
    metadata, results = load_run_results(run_dir)
    rows: list[dict[str, Any]] = []
    completed_scores: list[float] = []
    for name in metadata["benchmarks"]:
        result = results[name].data
        primary = result["score"]["primary"]
        if result["status"] == "completed" and primary is not None:
            completed_scores.append(float(primary))
        rows.append(
            {
                "benchmark": name,
                "status": result["status"],
                "score": primary,
                "unit": result["score"]["unit"],
                "requested": result["counts"]["requested"],
                "evaluated": result["counts"]["evaluated"],
                "method": result["scoring"]["method"],
                "self_judged": result["scoring"]["self_judged"],
                "self_simulated": result["scoring"]["self_simulated"],
                "agent": result.get("agent"),
                "error": result["error"],
            }
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": metadata["run_id"],
        "status": metadata["status"],
        "endpoint": metadata["endpoint"],
        "model_id": metadata["model_id"],
        "limit": metadata.get("limit"),
        "counts": {
            "selected": len(rows),
            "completed": sum(row["status"] == "completed" for row in rows),
        },
        "macro_average_percent": (
            sum(completed_scores) / len(completed_scores) if completed_scores else None
        ),
        "benchmarks": rows,
    }
    _write_json(run_dir / "report.json", report)
    (run_dir / "report.md").write_text(score_report_markdown(report), encoding="utf-8")
    return report


def _compatibility(
    baseline: BenchmarkResult, candidate: BenchmarkResult
) -> tuple[bool, str | None]:
    left = baseline.data
    right = candidate.data
    if left["status"] != "completed" or right["status"] != "completed":
        return False, "one or both results are not completed"
    left_agent = left.get("agent")
    right_agent = right.get("agent")
    if baseline.benchmark == "terminal-bench":
        left_agent = left_agent or "terminus-2"
        right_agent = right_agent or "terminus-2"
    if left_agent != right_agent:
        return False, "agent differs"
    if left["source"] != right["source"]:
        return False, "source lock differs"
    if left["selection"]["problem_ids"] != right["selection"]["problem_ids"]:
        return False, "selected problem IDs differ"
    if left["scoring"] != right["scoring"]:
        return False, "scoring method or self-scoring policy differs"
    if left.get("conditions", {}) != right.get("conditions", {}):
        return False, "benchmark conditions differ"
    if left["score"]["unit"] != right["score"]["unit"]:
        return False, "score unit differs"
    if left["score"]["primary"] is None or right["score"]["primary"] is None:
        return False, "one or both primary scores are unavailable"
    return True, None


def comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        f"# kairyu-bench comparison: {comparison['baseline']['run_id']} → {comparison['candidate']['run_id']}",
        "",
        "| Benchmark | Baseline | Candidate | Delta | Comparable |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in comparison["rows"]:
        delta = "—" if row["delta"] is None else f"{row['delta']:+.2f}"
        comparable = "yes" if row["compatible"] else row["reason"]
        lines.append(
            f"| {row['benchmark']} | {_score_text(row['baseline_score'])} | "
            f"{_score_text(row['candidate_score'])} | {delta} | {comparable} |"
        )
    lines.append("")
    return "\n".join(lines)


def compare_runs(baseline_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    baseline_meta, baseline_results = load_run_results(baseline_dir)
    candidate_meta, candidate_results = load_run_results(candidate_dir)
    selected_names = [
        name
        for name in BENCHMARK_NAMES
        if name in baseline_results or name in candidate_results
    ]
    rows: list[dict[str, Any]] = []
    for name in selected_names:
        baseline = baseline_results.get(name)
        candidate = candidate_results.get(name)
        compatible: bool
        reason: str | None
        if baseline is None or candidate is None:
            compatible, reason = False, "benchmark is missing from one run"
        else:
            compatible, reason = _compatibility(baseline, candidate)
        left_score = baseline.data["score"]["primary"] if baseline else None
        right_score = candidate.data["score"]["primary"] if candidate else None
        delta: float | None = None
        if compatible:
            if not isinstance(left_score, (int, float)) or not isinstance(
                right_score, (int, float)
            ):
                raise ValueError("compatible results must contain numeric scores")
            delta = float(right_score) - float(left_score)
        rows.append(
            {
                "benchmark": name,
                "baseline_score": left_score,
                "candidate_score": right_score,
                "delta": delta,
                "compatible": compatible,
                "reason": reason,
            }
        )
    return {
        "schema_version": 1,
        "baseline": {
            "run_id": baseline_meta["run_id"],
            "model_id": baseline_meta["model_id"],
        },
        "candidate": {
            "run_id": candidate_meta["run_id"],
            "model_id": candidate_meta["model_id"],
        },
        "rows": rows,
    }
