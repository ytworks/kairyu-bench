#!/usr/bin/env python3
"""Exploratory paired source-equal linear contrasts from approved score-only JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any

MAX_BYTES = 16 * 1024 * 1024
MAX_SAMPLED_ROWS = 20_000_000
TOKEN = re.compile(r"[A-Za-z0-9_./:+-]{1,512}")
SHA = re.compile(r"[0-9a-f]{64}")
SCHEMA_KEYS = {
    "schema_version",
    "kind",
    "data_kind",
    "source_artifact_sha256",
    "arms",
    "rows",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(value: Any) -> float:
    require(
        type(value) in (int, float),
        "scores and coefficients must be numeric, not bool/null",
    )
    result = float(value)
    require(math.isfinite(result), "scores and coefficients must be finite")
    return result


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def validate(data: Any) -> tuple[list[str], dict[str, list[dict[str, float]]]]:
    require(
        isinstance(data, dict) and set(data) == SCHEMA_KEYS,
        "score-only export schema fields required",
    )
    require(
        type(data["schema_version"]) is int and data["schema_version"] == 1,
        "unsupported schema",
    )
    require(
        data["kind"] == "paired-score-export", "approved score-only export required"
    )
    require(
        data["data_kind"] in ("recorded", "synthetic"),
        "explicit recorded/synthetic data_kind required",
    )
    require(
        isinstance(data["source_artifact_sha256"], str)
        and SHA.fullmatch(data["source_artifact_sha256"]) is not None,
        "source artifact SHA required",
    )
    arms = data["arms"]
    require(
        isinstance(arms, list) and 2 <= len(arms) <= 16,
        "two through sixteen arms required",
    )
    require(
        all(isinstance(a, str) and TOKEN.fullmatch(a) for a in arms),
        "invalid arm label",
    )
    require(len(set(arms)) == len(arms), "duplicate arm labels")
    rows = data["rows"]
    require(isinstance(rows, list) and bool(rows), "nonempty score rows required")
    seen: set[str] = set()
    groups: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        require(
            isinstance(row, dict) and set(row) == {"id", "source", "scores"},
            "only id/source/scores allowed",
        )
        require(
            all(
                isinstance(row[k], str) and TOKEN.fullmatch(row[k])
                for k in ("id", "source")
            ),
            "invalid item/source identifier",
        )
        require(
            row["id"] not in seen,
            "duplicate item ID (including cross-source duplicates)",
        )
        seen.add(row["id"])
        require(
            isinstance(row["scores"], dict) and set(row["scores"]) == set(arms),
            "every row must contain exactly every arm; missing/extra arms refused",
        )
        scores = {arm: finite(row["scores"][arm]) for arm in arms}
        groups.setdefault(row["source"], []).append(scores)
    return arms, groups


def summarize(
    groups: dict[str, list[dict[str, float]]],
    arms: list[str],
    coefficients: dict[str, float],
) -> tuple[float, dict[str, float], dict[str, float]]:
    by_arm = {
        arm: {
            source: sum(row[arm] for row in rows) / len(rows)
            for source, rows in groups.items()
        }
        for arm in arms
    }
    arm_means = {arm: sum(by_arm[arm].values()) / len(groups) for arm in arms}
    estimate = sum(coefficients[arm] * arm_means[arm] for arm in arms)
    source_estimates = {
        source: sum(coefficients[arm] * by_arm[arm][source] for arm in arms)
        for source in groups
    }
    require(
        all(
            math.isfinite(x)
            for x in [estimate, *arm_means.values(), *source_estimates.values()]
        ),
        "nonfinite derived arithmetic",
    )
    return estimate, source_estimates, arm_means


def interval(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [
        ordered[math.floor(0.025 * (len(ordered) - 1))],
        ordered[math.ceil(0.975 * (len(ordered) - 1))],
    ]


def contrast(
    data: Any,
    coefficients: dict[str, float],
    *,
    seed: int = 20261104,
    resamples: int = 2000,
) -> dict[str, Any]:
    arms, groups = validate(data)
    require(
        set(coefficients) == set(arms),
        "declare a coefficient for exactly every arm, including zeros",
    )
    coeff = {arm: finite(coefficients[arm]) for arm in arms}
    require(any(coeff.values()), "all-zero contrast refused")
    require(math.isfinite(sum(coeff.values())), "nonfinite coefficient sum")
    require(type(seed) is int, "seed must be an integer")
    require(
        type(resamples) is int and 2 <= resamples <= 10000,
        "resamples must be 2 through 10000",
    )
    count = sum(len(rows) for rows in groups.values())
    require(count * resamples <= MAX_SAMPLED_ROWS, "sampled-row CPU limit exceeded")
    point, source_points, arm_means = summarize(groups, arms, coeff)
    generator = random.Random(seed)
    draws: list[float] = []
    source_draws: dict[str, list[float]] = {source: [] for source in groups}
    for _ in range(resamples):
        # One index selects the complete row: never independently resample arms.
        selected = {
            source: [rows[generator.randrange(len(rows))] for _ in rows]
            for source, rows in sorted(groups.items())
        }
        estimate, source_estimates, _ = summarize(selected, arms, coeff)
        draws.append(estimate)
        for source, value in source_estimates.items():
            source_draws[source].append(value)
    return {
        "schema_version": 1,
        "status": "exploratory_linear_contrast",
        "data_kind": data["data_kind"],
        "source_artifact_sha256": data["source_artifact_sha256"],
        "source_artifact_identity": "caller-declared digest; upstream artifact is not opened",
        "arms": arms,
        "coefficients": coeff,
        "coefficient_sum": sum(coeff.values()),
        "estimate": point,
        "ci95": interval(draws),
        "arm_source_equal_means": arm_means,
        "paired_items": count,
        "sources": {
            source: {
                "items": len(rows),
                "estimate": source_points[source],
                "ci95": interval(source_draws[source]),
            }
            for source, rows in groups.items()
        },
        "bootstrap": {
            "seed": seed,
            "resamples": resamples,
            "sampling": "shared complete-row indices within each source, with replacement",
            "source_draw_order": sorted(groups),
            "row_order": "export order within each source",
            "aggregate": "equal source mean for each arm, then explicit linear combination",
            "interval": "sorted draws at floor(.025*(N-1)) and ceil(.975*(N-1))",
        },
        "limits": [
            "Exploratory conditional contrast, not official scoring, selection or causal-mechanism proof.",
            "All exported rows are jointly present; no missing-score imputation or intersection selection.",
            "Does not establish cross-problem independence, unseen data or true training-base identity.",
            "One generation per arm cannot separately estimate repeated-generation variance.",
            "Same input draws preserve paired covariance; no sum of independent pair variances.",
            "Source-equal aggregation matches a single WSP area's source weighting, not multi-area overall.",
            "Labels/data_kind/approval and source provenance are caller attestations, not independently established.",
        ],
        "model_calls": 0,
        "new_scoring": False,
        "raw_run_or_response_files_read": False,
    }


def file_stat(path: Path) -> tuple[int, int, int, int, int]:
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def execute(
    source: Path,
    expected_sha: str,
    output: Path,
    coefficients: dict[str, float],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    require(
        SHA.fullmatch(expected_sha) is not None,
        "explicit approved input SHA256 required",
    )
    require(
        source.is_file() and not source.is_symlink(),
        "input must be a regular non-symlink score export",
    )
    require(source.resolve() != output.resolve(), "input/output overlap")
    require(
        not os.path.lexists(output), "output must be fresh, including broken symlinks"
    )
    before = file_stat(source)
    require(before[2] <= MAX_BYTES, "score export exceeds 16 MiB")
    with source.open("rb") as handle:
        raw = handle.read(MAX_BYTES + 1)
    require(
        len(raw) <= MAX_BYTES and hashlib.sha256(raw).hexdigest() == expected_sha,
        "input hash/size mismatch",
    )
    data = json.loads(raw, object_pairs_hook=unique_object)
    result = contrast(data, coefficients, seed=seed, resamples=resamples)
    require(
        file_stat(source) == before
        and hashlib.sha256(source.read_bytes()).hexdigest() == expected_sha,
        "input changed during analysis",
    )
    result["input"] = {
        "path": str(source.resolve()),
        "sha256": expected_sha,
        "stat": dict(zip(("device", "inode", "size", "mtime_ns", "ctime_ns"), before)),
    }
    result["implementation_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    payload = (
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode()
    # Exclusive publication only after completed calculation and continuity check.
    fd, name = tempfile.mkstemp(
        prefix=".paired-score-", suffix=".tmp", dir=output.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.link(temporary, output)
    finally:
        temporary.unlink()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--coefficient", action="append", required=True, metavar="ARM=VALUE"
    )
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    try:
        coefficients = {}
        for text in args.coefficient:
            arm, value = text.split("=", 1)
            require(arm not in coefficients, "duplicate coefficient label")
            coefficients[arm] = finite(float(value))
        result = execute(
            args.input,
            args.input_sha256,
            args.output,
            coefficients,
            seed=args.seed,
            resamples=args.resamples,
        )
    except (OSError, ValueError, OverflowError) as error:
        # Do not echo unknown input fields or potentially unapproved text.
        print(json.dumps({"status": "refused", "error_type": type(error).__name__}))
        return 1
    print(
        json.dumps(
            {key: result[key] for key in ("status", "estimate", "ci95", "paired_items")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
