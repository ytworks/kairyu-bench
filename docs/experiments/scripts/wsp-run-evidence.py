#!/usr/bin/env python3
"""Supplement official WSP dev comparisons with item ranges and recorded usage.

Only run.json and whitelisted items.jsonl metadata are used. No model calls,
regrading, prompts, answers, tool arguments, or hidden reasoning are emitted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from kairyu_bench.wsp.report import compare


def number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def measurement(values: list[Any]) -> dict[str, Any]:
    known = [v for v in values if number(v) and v >= 0]
    return {
        "total": sum(known) if values and len(known) == len(values) else None,
        "sum_recorded": sum(known),
        "measured": len(known),
        "expected": len(values),
    }


def usage(outcome: Any) -> dict[str, Any]:
    samples = outcome.get("samples", []) if isinstance(outcome, dict) else []
    leaves: list[dict[str, Any] | None] = []
    for sample in samples:
        if not isinstance(sample, dict):
            leaves.append(None)
        elif isinstance(sample.get("calls"), list):
            leaves.extend(c if isinstance(c, dict) else None for c in sample["calls"])
        elif any(
            k in sample for k in ("completion_tokens", "prompt_tokens", "finish_reason")
        ):
            leaves.append(sample)
        else:
            leaves.append(None)
    return {
        "model_calls": measurement([1 if c is not None else None for c in leaves]),
        "input_tokens": measurement(
            [c.get("prompt_tokens") if c else None for c in leaves]
        ),
        "output_tokens": measurement(
            [c.get("completion_tokens") if c else None for c in leaves]
        ),
        "model_seconds_sum": measurement(
            [c.get("seconds") if c else None for c in leaves]
        ),
        "steps_recorded": measurement(
            [s.get("steps") if isinstance(s, dict) else None for s in samples]
        ),
        "sample_elapsed_seconds": measurement(
            [s.get("seconds") if isinstance(s, dict) else None for s in samples]
        ),
    }


def load_run(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    meta = json.loads((directory / "run.json").read_text())
    if meta.get("split") != "dev":
        raise ValueError("this experiment evidence tool accepts dev runs only")
    safe_meta = {
        k: meta.get(k)
        for k in (
            "schema_version",
            "kind",
            "status",
            "tier",
            "split",
            "manifest_sha256",
            "label",
            "model_id",
            "started_at",
            "finished_at",
        )
    }
    safe_meta["run_id"] = directory.name
    rows = {}
    with (directory / "items.jsonl").open() as stream:
        for line in stream:
            item = json.loads(line)
            key = item["id"]
            if key in rows:
                raise ValueError(f"duplicate item ID: {key}")
            score = item.get("score")
            if score is not None and not number(score):
                raise ValueError(f"invalid score for {key}")
            rows[key] = {k: item.get(k) for k in ("id", "source", "area", "score")}
            rows[key]["usage"] = usage(item.get("outcome"))
    return safe_meta, rows


def require_compatible(
    reference: dict[str, Any], other: dict[str, Any], *, same_weight: bool
) -> None:
    keys = ["manifest_sha256", "split", "tier", "model_id"]
    if same_weight:
        if not reference.get("label"):
            raise ValueError("repeat evidence needs a declared weight label")
        keys.append("label")
    if any(reference.get(k) != other.get(k) for k in keys):
        raise ValueError("repeat run identity/conditioning metadata differ")
    if other.get("status") != "completed":
        raise ValueError("repeat/reference run is not completed")


def write_fresh(path: Path, value: dict[str, Any], inputs: list[Path]) -> None:
    target = path.resolve()
    if any(target == p.resolve() or p.resolve() in target.parents for p in inputs):
        raise ValueError("output must be outside input run directories")
    if os.path.lexists(path):
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.link(temporary, path)  # exclusive publication; never replace another writer
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_evidence(
    base_path: Path,
    candidate_path: Path,
    *,
    baseline_repeats: tuple[Path, ...] = (),
    candidate_repeats: tuple[Path, ...] = (),
    historical_dev_refs: tuple[Path, ...] = (),
    partial: bool = False,
) -> dict[str, Any]:
    paths = [
        base_path,
        candidate_path,
        *baseline_repeats,
        *candidate_repeats,
        *historical_dev_refs,
    ]
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError(
            "duplicate input paths do not create independent repeat evidence"
        )
    base_meta, base = load_run(base_path)
    cand_meta, candidate = load_run(candidate_path)
    # Official weighted means, intervals, probe KL and score rules remain authoritative.
    official = compare(base_path, candidate_path, partial=partial)
    base_runs = [(base_meta, base)]
    cand_runs = [(cand_meta, candidate)]
    for path, destination, reference, cohort in [
        *[(p, base_runs, base_meta, base) for p in baseline_repeats],
        *[(p, cand_runs, cand_meta, candidate) for p in candidate_repeats],
    ]:
        meta, rows = load_run(path)
        require_compatible(reference, meta, same_weight=True)
        if set(rows) != set(cohort):
            raise ValueError("repeat cohort differs")
        for key in rows:
            if (rows[key]["source"], rows[key]["area"]) != (
                cohort[key]["source"],
                cohort[key]["area"],
            ):
                raise ValueError("repeat source/area differs")
        destination.append((meta, rows))
    history = []
    for path in historical_dev_refs:
        meta, rows = load_run(path)
        if (
            meta.get("status") != "completed"
            or meta.get("manifest_sha256") != base_meta.get("manifest_sha256")
            or meta.get("model_id") != base_meta.get("model_id")
            or meta.get("label") != base_meta.get("label")
        ):
            raise ValueError("historical base identity differs")
        for key in set(rows) & set(base):
            if (rows[key]["source"], rows[key]["area"]) != (
                base[key]["source"],
                base[key]["area"],
            ):
                raise ValueError("historical source/area differs")
        history.append((meta, rows))
    changes = []
    per_item = []
    for key in sorted(set(base) & set(candidate)):
        left, right = base[key], candidate[key]
        if (left["source"], left["area"]) != (right["source"], right["area"]):
            raise ValueError(f"source/area differs for {key}")
        if left["area"] is None:
            continue
        refs = [rows[key]["score"] for _, rows in base_runs]
        observed = [rows[key]["score"] for _, rows in cand_runs]
        hist = [
            {"run_id": m["run_id"], "tier": m["tier"], "score": rows[key]["score"]}
            for m, rows in history
            if key in rows
        ]
        known_refs = [v for v in refs if v is not None]
        known_cand = [v for v in observed if v is not None]
        hist_values = [v["score"] for v in hist if v["score"] is not None]
        paired = left["score"] is not None and right["score"] is not None
        delta = right["score"] - left["score"] if paired else None
        all_measured = len(known_refs) == len(refs) and len(known_cand) == len(observed)
        above = min(known_cand) > max(known_refs) if all_measured else None
        below = max(known_cand) < min(known_refs) if all_measured else None
        row = {
            "id": key,
            "source": left["source"],
            "area": left["area"],
            "base_score": left["score"],
            "candidate_score": right["score"],
            "delta": delta,
            "change": "unscored"
            if delta is None
            else "gain"
            if delta > 0
            else "loss"
            if delta < 0
            else "same",
            "crossed_0_5": ((left["score"] >= 0.5) != (right["score"] >= 0.5))
            if paired
            else None,
            "baseline_scores": refs,
            "candidate_scores": observed,
            "baseline_range": [min(known_refs), max(known_refs)]
            if known_refs
            else None,
            "candidate_range": [min(known_cand), max(known_cand)]
            if known_cand
            else None,
            "candidate_above_all_observed_baselines": above,
            "candidate_below_all_observed_baselines": below,
            "persistent_gain": above if len(observed) >= 2 else None,
            "persistent_loss": below if len(observed) >= 2 else None,
            "intermittent_gain": (max(known_cand) > min(known_refs) and not above)
            if all_measured and len(observed) >= 2
            else None,
            "intermittent_loss": (min(known_cand) < max(known_refs) and not below)
            if all_measured and len(observed) >= 2
            else None,
            "historical_dev_observations": hist,
            "above_all_including_historical": (
                min(known_cand) > max(known_refs + hist_values)
            )
            if all_measured and len(hist_values) == len(hist)
            else None,
            "below_all_including_historical": (
                max(known_cand) < min(known_refs + hist_values)
            )
            if all_measured and len(hist_values) == len(hist)
            else None,
            "base_usage": left["usage"],
            "candidate_usage": right["usage"],
        }
        per_item.append(row)
        if delta is not None and delta != 0:
            changes.append(key)
    return {
        "schema_version": 1,
        "official_comparison": official,
        "base": base_meta,
        "candidate": cand_meta,
        "baseline_repeats": [m for m, _ in base_runs[1:]],
        "candidate_repeats": [m for m, _ in cand_runs[1:]],
        "historical_dev_refs": [m for m, _ in history],
        "changed_item_ids": changes,
        "capability_items": per_item,
        "limits": [
            "Observed ranges are descriptive, not confidence bounds or causal effects.",
            "Historical dev references may have different tiers/cohorts/concurrent loads; range-only, excluded from official comparison and CI.",
            "Weight identity is declared run metadata, not a new checkpoint byte attestation.",
            "No adoption or no-regression gate is imposed; report gains and losses for user tradeoff decisions.",
            "Model-call counts, recorded agent steps and elapsed time are distinct. Missing totals remain null; sum_recorded is only the measured portion.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline-repeat", type=Path, action="append", default=[])
    parser.add_argument("--candidate-repeat", type=Path, action="append", default=[])
    parser.add_argument(
        "--historical-dev-reference", type=Path, action="append", default=[]
    )
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = [
        args.base,
        args.candidate,
        *args.baseline_repeat,
        *args.candidate_repeat,
        *args.historical_dev_reference,
    ]
    if os.path.lexists(args.output):
        raise FileExistsError(args.output)
    report = build_evidence(
        args.base,
        args.candidate,
        baseline_repeats=tuple(args.baseline_repeat),
        candidate_repeats=tuple(args.candidate_repeat),
        historical_dev_refs=tuple(args.historical_dev_reference),
        partial=args.partial,
    )
    write_fresh(args.output, report, inputs)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "paired_capability_items": len(report["capability_items"]),
                "changed_items": len(report["changed_item_ids"]),
            }
        )
    )


if __name__ == "__main__":
    main()
