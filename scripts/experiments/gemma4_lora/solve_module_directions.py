#!/usr/bin/env python3
"""Solve explicit donor-gate plans in continuous weight cost; never round gates to BF16."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemma4-purpose"))
import joint  # noqa: E402
import profile_donor_directions as profiling  # noqa: E402

PRIMARY = {"code", "implementation", "algorithm", "science_numerics", "science_contracts"}
RELATED = {"tools", "instruction", "reasoning", "workflow", "tool_call", "tool_no_call", "tool_arguments"}
MAX_COEFFICIENT, MAX_L2 = .25, 1.0
ROUNDING_MARGIN = 1e-7


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fresh_json(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def collection(directory):
    _, provenance = profiling.analyze(directory)
    summary = json.loads((directory / "summary.json").read_text())
    coverage = summary["coverage"]
    if coverage["matrices"] != 410 or len(coverage["entries"]) != 410 or len(summary["folds"]) != 34:
        raise ValueError("this fixed plan requires 410 directions and all 34 recorded folds")
    gradients = np.stack([np.load(directory / row["file"], allow_pickle=False) for row in summary["folds"]]).astype(np.float64)
    costs = np.load(directory / "unit-gate-weight-l2.npy", allow_pickle=False).astype(np.float64)
    return summary, provenance, gradients, costs


def make_plans(directory, donor_spec, unit_costs):
    summary, provenance, _, costs = collection(directory)
    donor_content = donor_spec.read_bytes()
    old = json.loads(donor_content)
    if old.get("kind") != "dense_lora_recipe_v1" or old.get("schema_version") != 2:
        raise ValueError("pinned original scalar donor spec required")
    donor = provenance["donor"]
    if any(old[key] != donor[other] for key, other in (("donor_sha256", "payload_sha256"), ("donor_config_sha256", "config_sha256"),
                                                    ("donor_repo", "repo"), ("donor_revision", "revision"))):
        raise ValueError("donor differs from gradient provenance")
    if sha(unit_costs) != donor["unit_costs_sha256"]:
        raise ValueError("unit-cost audit differs from gradient provenance")
    labels = [(row["purpose"], row["fold"]) for row in summary["folds"]]
    if not PRIMARY <= {label[0] for label in labels}:
        raise ValueError("primary calibration purposes missing")
    methods = [
        ("balanced", {"code": .005, "implementation": .005, "algorithm": .005, "science_numerics": .01, "science_contracts": .01}, None, 0),
        ("scientific-contracts", {"code": .002, "implementation": .005, "algorithm": .002, "science_numerics": .005, "science_contracts": .02}, None, 0),
        ("numerical-algorithms", {"code": .002, "implementation": .005, "algorithm": .01, "science_numerics": .02, "science_contracts": .005}, None, 0),
        ("mlp-implementation", {"code": .01, "implementation": .005, "algorithm": .002, "science_numerics": .005, "science_contracts": .005}, ["gate", "up", "down"], 0),
        ("attention-interfaces", {"code": .005, "implementation": .01, "algorithm": .002, "science_numerics": .005, "science_contracts": .005,
                                  "tool_arguments": .005, "tool_call": 0., "tool_no_call": 0.}, ["q", "k", "v", "o"], 0),
        ("late-compensation", {"code": .005, "implementation": .005, "algorithm": .005, "science_numerics": .01, "science_contracts": .01}, None, 30),
    ]
    plans = []
    for name, targets, families, minimum_layer in methods:
        constraints = [{"purpose": purpose, "fold": fold,
                        "minimum_delta": targets.get(purpose, -.005 if purpose in RELATED else -.01)} for purpose, fold in labels]
        plans.append({"id": "donor-" + name, "scope": {"families": families, "minimum_layer": minimum_layer, "maximum_layer": 59},
                      "constraints": constraints})
    files = [directory / name for name in ("summary.json", "coverage.json", "provenance.json", "frame-evidence.json", "unit-gate-weight-l2.npy")]
    files += [directory / row["file"] for row in summary["folds"]]
    return {"schema_version": 1, "kind": "module_donor_continuous_plans_v1", "collection": str(directory.resolve()),
            "input_sha256": {str(p.resolve()): sha(p) for p in files},
            "source_checkpoint": provenance["checkpoint"]["path"],
            "source_attestation_sha256": provenance["attestation_receipt_sha256"],
            "donor_spec": str(donor_spec.resolve()), "donor_spec_sha256": hashlib.sha256(donor_content).hexdigest(),
            "unit_costs_path": str(unit_costs.resolve()), "unit_costs_sha256": sha(unit_costs),
            "limits": {"maximum_absolute_coefficient": MAX_COEFFICIENT, "maximum_continuous_weight_l2": MAX_L2},
            "rounding_feasibility_margin": ROUNDING_MARGIN, "objective_units": "mean correct-minus-foil public-answer log probability",
            "coefficient_dtype": "FP32, not BF16", "continuous_unit_cost_l2": float(np.linalg.norm(costs)),
            "plans": plans, "actual_bf16_metrics_required": True, "candidate_adopted": False}


def solve_coefficients(gradient, lower, costs, mask):
    gradient, lower, costs = np.asarray(gradient), np.asarray(lower), np.asarray(costs)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != costs.shape or not mask.any() or np.any(costs[mask] <= 0):
        raise ValueError("nonempty explicit scope with positive unit costs required")
    active, dual = joint.minimum_step(gradient[:, mask], lower + ROUNDING_MARGIN, costs[mask])
    unbounded = {"continuous_delta_l2": float(np.linalg.norm(costs[mask] * active)),
                 "maximum_absolute_coefficient": float(np.max(np.abs(active))), "dual": dual}
    if unbounded["continuous_delta_l2"] <= MAX_L2 and unbounded["maximum_absolute_coefficient"] > MAX_COEFFICIENT:
        identity = np.eye(int(mask.sum()))
        bounded_g = np.vstack((gradient[:, mask], identity, -identity))
        bounded_b = np.concatenate((lower + ROUNDING_MARGIN, np.full(2 * len(identity), -MAX_COEFFICIENT + ROUNDING_MARGIN)))
        active, dual = joint.minimum_step(bounded_g, bounded_b, costs[mask])
    coefficients = np.zeros(len(costs), dtype=np.float64)
    coefficients[mask] = active.astype(np.float32).astype(np.float64)
    prediction = gradient @ coefficients
    norm = float(np.linalg.norm(costs * coefficients))
    maximum = float(np.max(np.abs(coefficients)))
    shortfall = float(np.max(np.maximum(lower - prediction, 0)))
    status = ("outside_trust_bounds" if maximum > MAX_COEFFICIENT or norm > MAX_L2 else
              "fp32_constraint_failure" if shortfall > 1e-10 else "continuous_proposal")
    return coefficients, {"status": status, "continuous_delta_l2": norm, "maximum_absolute_coefficient": maximum,
                           "unbounded_solution": unbounded,
                           "box_constraints_in_solver": bool(unbounded["continuous_delta_l2"] <= MAX_L2 and unbounded["maximum_absolute_coefficient"] > MAX_COEFFICIENT),
                           "nonzero_modules": int(np.count_nonzero(coefficients)), "predicted_response": prediction.tolist(),
                           "minimum_delta": lower.tolist(), "maximum_shortfall": shortfall, "dual": dual,
                           "coefficient_rounding": "F64 solution -> F32 once; original constraints rechecked; no BF16 gate rounding",
                           "actual_bf16_metrics": False, "candidate_adopted": False}


def solve(plan_path, output):
    plan_content = plan_path.read_bytes()
    plan = json.loads(plan_content)
    if plan.get("schema_version") != 1 or plan.get("kind") != "module_donor_continuous_plans_v1":
        raise ValueError("unsupported plan bundle")
    if plan["limits"] != {"maximum_absolute_coefficient": MAX_COEFFICIENT, "maximum_continuous_weight_l2": MAX_L2} or plan["rounding_feasibility_margin"] != ROUNDING_MARGIN:
        raise ValueError("fixed trust/rounding bounds differ")
    directory = Path(plan["collection"]).resolve()
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(directory):
        raise ValueError("output must be a fresh directory outside collection")
    for path, digest in plan["input_sha256"].items():
        if sha(path) != digest:
            raise ValueError("collection input hash changed")
    donor_spec, cost_path = Path(plan["donor_spec"]), Path(plan["unit_costs_path"])
    if sha(donor_spec) != plan["donor_spec_sha256"] or sha(cost_path) != plan["unit_costs_sha256"]:
        raise ValueError("pinned donor/cost input changed")
    summary, provenance, gradient, costs = collection(directory)
    if provenance["checkpoint"]["path"] != plan["source_checkpoint"] or provenance["attestation_receipt_sha256"] != plan["source_attestation_sha256"]:
        raise ValueError("gradient source identity changed")
    source = Path(plan["source_checkpoint"]).resolve()
    if output.resolve() == source or output.resolve().is_relative_to(source) or source.is_relative_to(output.resolve()):
        raise ValueError("output overlaps original source")
    expected_files = {str((directory / n).resolve()) for n in ("summary.json", "coverage.json", "provenance.json", "frame-evidence.json", "unit-gate-weight-l2.npy")}
    expected_files.update(str((directory / f["file"]).resolve()) for f in summary["folds"])
    if set(plan["input_sha256"]) != expected_files:
        raise ValueError("plan must pin every gradient/metadata artifact")
    original_bytes = donor_spec.read_bytes()
    if hashlib.sha256(original_bytes).hexdigest() != plan["donor_spec_sha256"]:
        raise ValueError("donor spec changed while parsing")
    original = json.loads(original_bytes)
    labels = [(f["purpose"], f["fold"]) for f in summary["folds"]]
    entries = summary["coverage"]["entries"]
    if len(plan["plans"]) != 6 or len({p["id"] for p in plan["plans"]}) != 6:
        raise ValueError("fixed bundle requires six distinct method scopes")
    code_paths = (Path(__file__), Path(joint.__file__), Path(joint.common.__file__), Path(joint.finite_response.__file__))
    code_hashes = {str(p): sha(p) for p in code_paths}
    output.mkdir(parents=True)
    results = []
    try:
        for method in plan["plans"]:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", method["id"]):
                raise ValueError("unsafe method id")
            constraints = method["constraints"]
            if [(row["purpose"], row["fold"]) for row in constraints] != labels:
                raise ValueError("each method must retain all 34 folds in the same order")
            if any(type(row["minimum_delta"]) not in (float, int) for row in constraints):
                raise ValueError("explicit numeric constraint required")
            lower = np.array([row["minimum_delta"] for row in constraints], dtype=np.float64)
            scope = method["scope"]
            if (type(scope["minimum_layer"]) is not int or type(scope["maximum_layer"]) is not int
                    or not 0 <= scope["minimum_layer"] <= scope["maximum_layer"] <= 59
                    or scope["families"] is not None and (not isinstance(scope["families"], list) or not set(scope["families"]) <= profiling.FAMILIES)):
                raise ValueError("explicit valid structural scope required")
            mask = np.array([(scope["families"] is None or e["family"] in scope["families"])
                             and scope["minimum_layer"] <= e["layer"] <= scope["maximum_layer"] for e in entries])
            try:
                coefficients, report = solve_coefficients(gradient, lower, costs, mask)
            except ValueError as error:
                results.append({"id": method["id"], "status": "continuous_unresolved", "reason": str(error)})
                continue
            report.update(id=method["id"], scope=scope, scope_modules=int(mask.sum()), constraints=constraints)
            if report["status"] == "continuous_proposal":
                spec = {k: v for k, v in original.items() if k not in {"coefficient", "native_edits"}}
                spec.update(id=method["id"], schema_version=3, kind="module_lora_recipe_v1",
                            hypothesis="Signed per-module continuous local prediction under explicit 34-fold constraints; actual BF16 response and benchmark benefit are unverified.",
                            module_coefficients={entry["tensor"]: float(c) for entry, c in zip(entries, coefficients, strict=True)},
                            unit_costs_path=str(cost_path.resolve()), unit_costs_sha256=plan["unit_costs_sha256"],
                            selection_plan_sha256=hashlib.sha256(plan_content).hexdigest(),
                            collection_inputs_sha256=plan["input_sha256"], qualification="continuous_prediction_only",
                            actual_bf16_metrics_required=True)
                path = output / (method["id"] + ".json")
                fresh_json(path, spec)
                report.update(spec_path=str(path.resolve()), spec_sha256=sha(path))
            results.append(report)
        if plan_path.read_bytes() != plan_content or any(sha(p) != digest for p, digest in plan["input_sha256"].items()) or sha(donor_spec) != plan["donor_spec_sha256"] or sha(cost_path) != plan["unit_costs_sha256"]:
            raise ValueError("solve inputs changed")
        if any(sha(p) != digest for p, digest in code_hashes.items()):
            raise ValueError("solver implementation changed")
        result = {"schema_version": 1, "status": "completed", "response_kind": "continuous_local_prediction_not_finite",
                  "plan_sha256": hashlib.sha256(plan_content).hexdigest(), "results": results,
                  "implementation_sha256": code_hashes, "numpy_version": np.__version__,
                  "actual_bf16_metrics_required": True, "new_checkpoints": 0, "model_calls": 0, "benchmark_calls": 0,
                  "limits": ["Outside trust bounds is not proof that the bounded problem is infeasible.",
                             "Continuous coefficient feasibility does not prove BF16 finite response or benchmark gains.",
                             "Gate coefficients are FP32; actual model weights alone are rounded to BF16."]}
        fresh_json(output / "report.json", result)
        return result
    except BaseException as error:
        fresh_json(output / "failure.json", {"status": "incomplete", "error_type": type(error).__name__})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    make = sub.add_parser("make-plans")
    make.add_argument("--collection", type=Path, required=True)
    make.add_argument("--donor-spec", type=Path, required=True)
    make.add_argument("--unit-costs", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("solve")
    run.add_argument("--plans", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "make-plans":
        if args.output.exists() or args.output.is_symlink() or args.output.resolve().is_relative_to(args.collection.resolve()):
            parser.error("plan output must be fresh and outside the collection")
        result = make_plans(args.collection.resolve(), args.donor_spec, args.unit_costs)
        source = Path(result["source_checkpoint"]).resolve()
        if args.output.resolve().is_relative_to(source):
            parser.error("plan output overlaps checkpoint")
        fresh_json(args.output, result)
        print(json.dumps({"plans": len(result["plans"]), "model_calls": 0}))
    else:
        result = solve(args.plans, args.output)
        print(json.dumps({"status": result["status"], "methods": [{"id": row["id"], "status": row["status"]} for row in result["results"]]}))
    return 0


if __name__ == "__main__":
    main()
