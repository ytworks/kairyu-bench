#!/usr/bin/env python3
"""Prepare one normalized-target matrix plan; never solve or call a model."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np


def build(args):
    sys.path[:0] = [str(args.repo / "scripts/experiments/gemma4-purpose"), str(args.repo / "src")]
    import propose_native_objectives as adapter
    from kairyu_bench.wsp.weights.finite_response import bf16_realization

    started = time.time()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("fresh output directory required")
    checkpoint, original, collections, labels, pins = adapter.prepare(args.source_plan)
    if pins[str(args.source_plan.resolve())][0] != args.source_plan_sha256:
        raise ValueError("source plan SHA differs")
    adapter.check_output(args.output_dir.resolve(), checkpoint, args.source_plan, collections, pins)
    if original.get("eligible_blocks") != ["matrix"]:
        raise ValueError("explicit matrix-only input required")
    def numeric(path):
        return adapter.capture(path, pins, original["completed_collection_sha256"][str(path)])
    sample_path = collections["matrix:correct-logprob"]["path"] / "sample-plan.npz"
    with np.load(io.BytesIO(numeric(sample_path)), allow_pickle=False) as sample:
        base = sample["source_values"].astype(np.float64)
    entries = collections["matrix:correct-logprob"]["coverage"]["entries"]
    eligible = np.zeros(len(base), dtype=bool)
    for entry in entries:
        if entry["family"] in {"o", "gate", "up", "down"}:
            eligible[entry["start"]:entry["start"] + entry["count"]] = True
    base = base[eligible]
    costs = 1.0 / np.maximum(np.abs(base), 1e-6)
    radius = np.maximum(.025 * np.abs(base), 1e-6)
    def rounded(delta):
        result, _ = bf16_realization(base, np.column_stack((delta, np.zeros_like(delta))), [1., 0.])
        return result.astype(np.float64) - base
    low, high = rounded(-radius), rounded(radius)
    core = set(args.core_purposes)
    science = (set(args.science_purposes) if args.science_purposes is not None
               else {purpose for purpose, _ in labels if purpose.startswith("science_")})
    observed = {purpose for purpose, _ in labels}
    if not core or not science or core & science or not (core | science) <= observed:
        raise ValueError("nonempty disjoint observed core/science purposes required")
    rows, constraints = [], []
    for objective in adapter.OBJECTIVES:
        collection = collections["matrix:" + objective]
        for record in collection["summary"]["folds"]:
            purpose, fold = record["purpose"], record["fold"]
            gradient = np.load(io.BytesIO(numeric(collection["path"] / record["file"])), allow_pickle=False).astype(np.float64)
            if gradient.shape != eligible.shape or not np.isfinite(gradient).all():
                raise ValueError("finite matching gradient columns required")
            gradient = gradient[eligible]
            dual_norm = float(np.linalg.norm(gradient / costs))
            headroom = None
            category = "margin_guard" if objective == "margin" else "other_ll_guard"
            target = -.002
            multiplier = None
            if objective == "correct-logprob":
                mean = float(np.mean([row["correct_logprob"] for row in record["correct_logprobs"]]))
                if "mean_correct_logprob" in record and not np.isclose(mean, record["mean_correct_logprob"], rtol=0, atol=1e-12):
                    raise ValueError("recorded mean differs from per-case correct logprobs")
                if not np.isfinite(mean) or mean > 0:
                    raise ValueError("finite nonpositive baseline correct logprob required")
                headroom = -mean
                if purpose in core or purpose in science:
                    category = "core" if purpose in core else "science"
                    multiplier = .25 if purpose in core else .0625
                    target = min(multiplier * dual_norm, .1 * headroom)
            continuous_upper = float(np.sum(np.abs(gradient) * radius))
            rounded_upper = float(np.maximum(gradient * low, gradient * high).sum())
            upper = min(rounded_upper, dual_norm)
            rows.append(dict(objective=objective, purpose=purpose, fold=fold, category=category,
                             dual_norm=dual_norm, baseline_headroom=headroom,
                             norm_multiplier=multiplier, minimum_delta=target,
                             continuous_box_independent_upper=continuous_upper,
                             rounded_endpoint_independent_upper=rounded_upper,
                             weighted_norm_cap_independent_upper=dual_norm,
                             independent_upper=upper,
                             target_within_necessary_upper=target <= upper + 1e-12))
            constraints.append(dict(objective=objective, purpose=purpose, fold=fold, minimum_delta=target))
    expected = [(o, p, f) for o in adapter.OBJECTIVES for p, f in labels]
    if [(r["objective"], r["purpose"], r["fold"]) for r in rows] != expected:
        raise ValueError("complete ordered objective/fold contract differs")
    script_path = Path(__file__).resolve()
    adapter.capture(script_path, pins)
    adapter.stable(checkpoint, pins)
    plan = {k: copy.deepcopy(original[k]) for k in (
        "schema_version", "kind", "checkpoint", "cases", "parent_state", "collections",
        "baseline_tolerance", "observation_metadata_sha256",
        "completed_collection_sha256", "completed_collection_stat", "eligible_blocks") if k in original}
    plan["candidates"] = [dict(id=args.candidate_id, constraints=constraints,
        cost_metric=dict(type="relative", floor=1e-6, block_scales=dict(native=1., matrix=1.)),
        rounding=dict(relative_radius=.025, absolute_radius=1e-6, rounds=16))]
    policy = dict(protocol="normalized_coding_priority_matrix_v1",
                  core_purposes=sorted(core), science_purposes=sorted(science),
                  primary_ll_target="min(multiplier * norm(gradient / cost), 0.1 * baseline_headroom)",
                  core_multiplier=.25, science_multiplier=.0625, other_ll_lower=-.002,
                  margin_lower=-.002, cost_floor=1e-6, maximum_seconds=120,
                  actual_bf16_weighted_l2_postcheck_maximum=1.,
                  postcheck_enforced_by_existing_solver=False,
                  postcheck_failure="not eligible; no rescale, clipping, radius retry or model run",
                  reporting_utility_weights=dict(core=.75, science=.25),
                  utility_is_qp_objective=False)
    plan["normalized_target_policy"] = policy
    receipt = dict(schema_version=1, status="prepared_only", candidate_id=args.candidate_id,
        source_plan=dict(path=str(args.source_plan.resolve()), sha256=args.source_plan_sha256),
        source_checkpoint=plan["checkpoint"], parent_state=plan.get("parent_state"),
        method=policy, eligible_matrix_count=sum(e["family"] in {"o", "gate", "up", "down"} for e in entries),
        eligible_coordinates=int(eligible.sum()), rows=rows,
        inputs={path: dict(sha256=value[0], stat=list(value[1])) for path, value in sorted(pins.items())},
        limits=["Independent row upper bounds are necessary only, not joint feasibility or measured response.",
                "Rounded endpoints may lie outside the continuous box; both bounds are recorded.",
                "Headroom caps requested lower targets, not the whole linear prediction.",
                "Existing solver enforces no global norm cap; owner must check actual BF16 weighted L2 <= 1.",
                "Teacher-forced forced-empty-thought authored calibration; no actual benchmark gain is established."],
        solver_calls=0, model_calls=0, candidate_specs_written=0, elapsed_seconds=time.time()-started)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = args.output_dir / "preparation-receipt.json"
    raw = (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode()
    receipt_path.write_bytes(raw)
    plan["fixed_method_inputs"] = {path: value[0] for path, value in pins.items()
                                  if Path(path).suffix == ".py"}
    plan["fixed_method_inputs"][str(receipt_path)] = hashlib.sha256(raw).hexdigest()
    plan["normalized_target_preparation"] = dict(path=str(receipt_path), sha256=hashlib.sha256(raw).hexdigest())
    plan_path = args.output_dir / "plan.json"
    plan_raw = (json.dumps(plan, indent=2, allow_nan=False) + "\n").encode()
    plan_path.write_bytes(plan_raw)
    adapter.stable(checkpoint, pins)
    print(json.dumps(dict(status="prepared_only", plan=str(plan_path), plan_sha256=hashlib.sha256(plan_raw).hexdigest(),
        receipt=str(receipt_path), receipt_sha256=hashlib.sha256(raw).hexdigest(),
        rows=len(rows), independent_bound_failures=sum(not r["target_within_necessary_upper"] for r in rows),
        source_sha256=pins[str(script_path)][0], elapsed_seconds=time.time()-started)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--source-plan-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--core-purposes", nargs="+", default=["code", "implementation", "workflow", "algorithm"])
    parser.add_argument("--science-purposes", nargs="+")
    build(parser.parse_args(argv))


if __name__ == "__main__":
    main()
