"""Versioned published scores. These are references, never local run results."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from kairyu_bench.benchmarks import BENCHMARK_NAMES


def load_public_results() -> dict[str, Any]:
    path = Path(__file__).parent / "data/public_results.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    validate_catalog(catalog)
    return catalog


def validate_catalog(catalog: dict[str, Any]) -> None:
    if catalog.get("schema_version") != 1:
        raise ValueError("unsupported public result catalog schema")
    sources = catalog["sources"]
    keys = set()
    for row in catalog["results"]:
        key = (row["benchmark"], row["model"], row.get("config"))
        if key in keys or row["benchmark"] not in BENCHMARK_NAMES:
            raise ValueError("duplicate or unknown public result")
        keys.add(key)
        score = row["score_percent"]
        if score is None:
            if (
                row["status"] != "not-reported-in-reviewed-sources"
                or row["source_id"] is not None
            ):
                raise ValueError(
                    "missing public scores must remain explicitly unavailable"
                )
            continue
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 100
        ):
            raise ValueError("public score must be numeric percent")
        source = sources.get(row["source_id"])
        if (
            not source
            or not source.get("url", "").startswith("https://")
            or not source.get("retrieved_at")
        ):
            raise ValueError("published score requires a dated source")
        if row["comparable"] is not False or not row["conditions"]:
            raise ValueError(
                "published references require explicit comparison limitations"
            )
        if row["benchmark"] == "deepswe":
            if not math.isclose(
                score,
                100 * row["passed_attempts"] / row["scored_attempts"],
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "DeepSWE score disagrees with numerator and denominator"
                )
            if (
                not 0 < row["task_count"] <= 113
                or row["attempts_per_task"] != 4
                or row["scored_attempts"] > 452
            ):
                raise ValueError("DeepSWE public task or trial count differs from v1.1")


def references_for(benchmarks: list[str]) -> list[dict[str, Any]]:
    catalog = load_public_results()
    return [
        dict(row, source=catalog["sources"][row["source_id"]])
        for row in catalog["results"]
        if row["benchmark"] in benchmarks and row["score_percent"] is not None
    ]


def best_deepswe(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in catalog["results"]:
        if row["benchmark"] != "deepswe":
            continue
        previous = selected.get(row["model"])
        # Selection policy is explicit; the detail table retains all configurations.
        if previous is None or row["score_percent"] > previous["score_percent"]:
            selected[row["model"]] = row
    return sorted(selected.values(), key=lambda r: (-r["score_percent"], r["model"]))


def comparison_table(catalog: dict[str, Any]) -> str:
    models = catalog["table_models"]
    lines = [
        "| Benchmark | " + " | ".join(models) + " |",
        "| --- | " + " | ".join(["---:"] * len(models)) + " |",
    ]
    best = {row["model"]: row for row in best_deepswe(catalog)}
    for benchmark, label in catalog["benchmark_labels"].items():
        cells = []
        for model in models:
            if benchmark == "deepswe":
                row = best.get(catalog["deepswe_display_mapping"].get(model))
                effort = (row["reasoning_effort"] or "unspecified") if row else ""
                note = f" ({effort})" if row else ""
            else:
                row = next(
                    r
                    for r in catalog["results"]
                    if r["benchmark"] == benchmark and r["model"] == model
                )
                note = " " + row["display_note"] if row["display_note"] else ""
            if row is None or row["score_percent"] is None:
                cells.append("—")
            else:
                digits = 2 if benchmark == "deepswe" else 1
                cells.append(
                    f"[{row['score_percent']:.{digits}f}{note}][{row['source_id']}]"
                )
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def deepswe_table(catalog: dict[str, Any]) -> str:
    lines = [
        "| Model ID | Effort | pass@1 (%) | Success / scored trials | Scored tasks | pass@4 (%) | Passed / scored tasks | 95% CI (%) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in catalog["results"]:
        if row["benchmark"] != "deepswe":
            continue
        lines.append(
            f"| `{row['model']}` | {row['reasoning_effort'] or 'unspecified'} | "
            f"[{row['score_percent']:.2f}][{row['source_id']}] | {row['passed_attempts']}/{row['scored_attempts']} | "
            f"{row['task_count']}/113 | {row['pass_at_4_percent']:.2f} | "
            f"{row['tasks_passed_any']}/{row['task_count']} | "
            f"{row['ci_lo_percent']:.2f}–{row['ci_hi_percent']:.2f} |"
        )
    return "\n".join(lines)
