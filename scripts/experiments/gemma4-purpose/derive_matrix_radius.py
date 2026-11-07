#!/usr/bin/env python3
"""Derive one matrix-only BF16 box, then invoke the existing quantized proposal once."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import propose_native_objectives as adapter

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def publish(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")

def check_plan(path, expected):
    checkpoint, plan, collections, labels, pins = adapter.prepare(path)
    if pins[str(path.resolve())][0] != expected:
        raise ValueError("explicit plan SHA mismatch")
    if plan.get("eligible_blocks") != ["matrix"] or len(plan["candidates"]) != 1:
        raise ValueError("one strict matrix-only candidate required")
    policy = dict(plan["analytic_calibration"])
    metric_type = policy.pop("metric", "relative")
    if metric_type not in {"relative", "euclidean"} or policy != {"calibration_seconds": 120, "candidate_seconds": 180,
                  "fractional_buffer": 1 / 128, "floor": 1e-6,
                  "dual_tolerance": 2e-10, "maximum_iterations": 20000}:
        raise ValueError("fixed analytic stage contract differs")
    for filename, expected_sha in plan.get("fixed_method_inputs", {}).items():
        adapter.capture(Path(filename), pins, expected_sha)
    adapter.capture(Path(__file__), pins)
    reference = plan.get("reference_plan", plan.get("prior_aa_plan"))
    if not isinstance(reference, dict):
        raise ValueError("pinned reference plan required")
    prior = adapter.document(Path(reference["path"]), pins, reference["sha256"])
    for key in ("checkpoint", "cases", "parent_state", "collections", "baseline_tolerance", "observed_blocks"):
        if plan.get(key) != prior.get(key):
            raise ValueError("reference observation contract differs: " + key)
    if "observed_blocks" in plan and plan.get("eligible_blocks") != prior.get("eligible_blocks"):
        raise ValueError("reference eligible blocks differ")
    method = plan["candidates"][0]
    for key in ("constraints", "cost_metric"):
        if method[key] != prior["candidates"][0][key]:
            raise ValueError("predeclared bounds/cost differ")
    expected_metric = {"type": metric_type, "block_scales": {"native": 1.0, "matrix": 1.0}}
    if metric_type == "relative":
        expected_metric["floor"] = 1e-6
    if method["cost_metric"] != expected_metric:
        raise ValueError("analytic cost contract differs")
    observed = collections["matrix:correct-logprob"]["coverage"]["entries"]
    active, selected = adapter.matrix_column_indices(plan, observed)
    reference_active, _ = adapter.matrix_column_indices(prior, observed)
    if not np.array_equal(active, reference_active):
        raise ValueError("reference matrix mask differs")
    if any(adapter.matrix_recipe.PATTERN.fullmatch(row["tensor"])[3]
           not in {"q", "k", "o", "gate", "up", "down"} for row in selected):
        raise ValueError("analytic scope permits only Q/K/O/MLP matrices")
    for filename, old in plan["completed_collection_stat"].items():
        if adapter.stamp(Path(filename)) != tuple(old[k] for k in ("device", "inode", "size", "mtime_ns", "ctime_ns")):
            raise ValueError("completed observation stat changed")
    adapter.stable(checkpoint, pins)
    return checkpoint, plan, collections, labels, pins

def matrix_arrays(checkpoint, plan, collections, labels, pins):
    loaded = []
    constraints = [{"purpose": p, "fold": f, "minimum_delta": 0.0} for p, f in labels]
    for objective in adapter.OBJECTIVES:
        collection = collections["matrix:" + objective]
        base, gradient, _, layout, coordinates, _, hashes, _, _ = adapter.joint.read_collection(
            collection["path"], {"constraints": constraints})
        for filename, digest in hashes.items():
            if plan["completed_collection_sha256"].get(filename) != digest:
                raise ValueError("matrix payload differs from completed observation")
            adapter.capture(Path(filename), pins, digest)
        with np.load(collection["path"] / "sample-plan.npz", allow_pickle=False) as data:
            if set(data.files) != {"coordinates", "source_values"}:
                raise ValueError("unexpected sample arrays")
        expected = adapter.matrix_recipe.mlp_triplet_samples(checkpoint.tensors, collection["provenance"]["seed"])
        cursor = 0
        for entry in layout:
            count = entry["count"]
            name = entry["tensor"]
            if entry["shape"] != list(checkpoint.tensors[name].shape):
                raise ValueError("matrix header shape differs")
            if name in expected and not np.array_equal(coordinates[cursor:cursor+count], expected[name][0]):
                raise ValueError("coherent MLP coordinates differ")
            cursor += count
        active, selected = adapter.matrix_column_indices(plan, layout)
        selected = [dict(entry, family=adapter.matrix_recipe.PATTERN.fullmatch(entry["tensor"])[3])
                    for entry in selected]
        loaded.append((base, gradient, layout, coordinates, active, selected))
    left, right = loaded
    if (left[2] != right[2] or not np.array_equal(left[3], right[3])
            or not np.array_equal(left[0].astype(np.float32).view(np.uint32), right[0].astype(np.float32).view(np.uint32))):
        raise ValueError("matrix objective coordinates/source bytes differ")
    active = left[4]
    if not len(active):
        raise ValueError("nonempty sampled Q/K/O/MLP scope required")
    return left[0][active], np.vstack([row[1][:, active] for row in loaded]), left[5]

def calibration(checkpoint, plan, collections, labels, pins, destination):
    base, gradient, layout = matrix_arrays(checkpoint, plan, collections, labels, pins)
    method = plan["candidates"][0]
    lower = np.asarray([row["minimum_delta"] for row in method["constraints"]])
    scale = np.maximum(np.abs(base), 1e-6)
    cost = np.ones_like(base) if method["cost_metric"]["type"] == "euclidean" else 1 / scale
    step, dual = adapter.joint.minimum_step(gradient, lower, cost, tolerance=2e-10, maximum_iterations=20000)
    prediction = gradient @ step
    norms = np.linalg.norm(gradient / cost, axis=1)
    shortfall = np.maximum(lower - prediction, 0)
    scaled_shortfall = np.divide(shortfall, norms, out=np.zeros_like(shortfall), where=norms != 0)
    solver_ok = (np.isfinite(step).all() and np.isfinite(prediction).all()
                 and dual["dual_residual"] <= 2e-10 and scaled_shortfall.max() <= 2e-10
                 and not np.any((norms == 0) & (lower > 0)))
    if not solver_ok:
        publish(destination / "continuous-failure.json",
                {"dual": dual, "maximum_scaled_shortfall": float(scaled_shortfall.max()),
                 "maximum_absolute_shortfall": float(shortfall.max()), "continuous_feasible": False})
        raise ValueError("existing minimum_step numerical contract was not met; no infeasibility claim")
    maximum_ratio = float(np.max(np.abs(step) / scale))
    rho = maximum_ratio + 1 / 128
    absolute = max(1e-6, rho * 1e-6)
    radius = np.maximum(rho * np.abs(base), absolute)
    if not np.isfinite([rho, absolute]).all() or not np.all(np.abs(step) <= radius):
        raise ValueError("nonfinite derived box or continuous solution not contained")
    zero = np.zeros_like(base)
    endpoint_values = [adapter.joint.bf16_realization(base, np.column_stack((sign * radius, zero)), [1., 0.])[0]
                       for sign in (-1, 1)]
    endpoint_delta = [values.astype(np.float64) - base for values in endpoint_values]
    if not all(np.isfinite(values).all() for values in endpoint_values):
        raise ValueError("derived BF16 endpoint is nonfinite")
    upper = np.maximum(gradient * endpoint_delta[0], gradient * endpoint_delta[1]).sum(axis=1)
    rows = [dict(row, continuous_prediction=float(value), continuous_shortfall=float(miss),
                 rounded_box_upper_bound=float(bound), necessary_shortfall=float(max(target-bound, 0)))
            for row, value, miss, bound, target in zip(method["constraints"], prediction, shortfall, upper, lower, strict=True)]
    derived = json.loads(json.dumps(plan))
    derived["candidates"][0]["rounding"] = {"relative_radius": rho, "absolute_radius": absolute, "rounds": 16}
    derived["calibration_origin"] = {"continuous_calls": 1, "fractional_buffer": 1/128,
                                   "not_a_universal_ulp_guarantee": True}
    report = {"schema_version": 1, "status": "analytic_calibration_complete",
              "continuous_calls": 1, "quantized_proposals": 0, "model_calls": 0,
              "continuous_feasible_under_solver_contract": True, "dual": dual,
              "maximum_absolute_shortfall": float(shortfall.max()),
              "maximum_scaled_shortfall": float(scaled_shortfall.max()),
              "continuous_delta_l2": float(np.linalg.norm(step)),
              "continuous_weighted_delta_l2": float(np.linalg.norm(cost * step)),
              "maximum_normalized_continuous_step": maximum_ratio,
              "derived_relative_radius": rho, "derived_absolute_radius": absolute,
              "continuous_box_contains_solution": True, "fractional_buffer": 1/128,
              "fractional_buffer_is_ulp_guarantee": False,
              "rounded_box_endpoint_maximum_continuous_box_excess":
                  max(float(np.maximum(np.abs(delta)-radius, 0).max()) for delta in endpoint_delta),
              "rounded_box_necessary_conditions_all_met": bool(np.all(upper >= lower - 1e-10)),
              "rounded_box_maximum_necessary_shortfall": float(np.maximum(lower-upper, 0).max()),
              "row_bounds_are_necessary_not_joint_sufficient": True,
              "native_edit_columns": 0,
              "observed_blocks": adapter.observed_blocks(plan),
              "baseline_source": "matrix:correct-logprob",
              "qkv_edit_columns": sum(entry["count"] for entry in layout
                                      if entry["family"] in {"q", "k", "v"}),
              "matrix_columns": len(base),
              "observed_matrix_columns": sum(entry["count"] for entry in
                  collections["matrix:correct-logprob"]["coverage"]["entries"]),
              "cost_metric": method["cost_metric"],
              "eligible_matrix_tensors": [entry["tensor"] for entry in layout],
              "layout": layout, "rows": rows, "source_checkpoint": plan["checkpoint"],
              "parent_state": plan.get("parent_state"),
              "source_sha256": {p: record[0] for p, record in pins.items()},
              "no_fresh_checkpoint_payload_hash": True}
    adapter.stable(checkpoint, pins)
    publish(destination / "report.json", report)
    report_sha = sha(destination / "report.json")
    derived["analytic_report"] = {"path": str(destination / "report.json"), "sha256": report_sha}
    publish(destination / "candidate-plan.json", derived)
    return {"report_sha256": report_sha, "candidate_plan_sha256": sha(destination / "candidate-plan.json"),
            "rho": rho, "absolute_radius": absolute}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("calibration", "candidate"), default="calibration")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    started = time.monotonic()
    def timeout(*unused):
        raise TimeoutError("fixed stage wall budget exceeded; no automatic retry")
    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120 if args.phase == "calibration" else 180)
    destination = args.output.resolve()
    owned = False
    try:
        checkpoint, plan, collections, labels, pins = check_plan(args.plan, args.plan_sha256)
        if destination.exists() or args.output.is_symlink():
            raise ValueError("fresh output required")
        adapter.check_output(destination, checkpoint, args.plan, collections, pins)
        if not args.run:
            print(json.dumps({"status": "metadata_preview", "phase": args.phase,
                              "stage_seconds": 120 if args.phase == "calibration" else 180,
                              "gradient_reads": 0, "solves": 0, "model_calls": 0}))
            return
        destination.mkdir(parents=True)
        owned = True
        publish(destination / "started.json", {"phase": args.phase, "plan": str(args.plan),
                "plan_sha256": args.plan_sha256, "runtime_sha256": sha(__file__),
                "epoch": time.time(), "pid": os.getpid(), "uid": os.getuid()})
        if args.phase == "calibration":
            result = calibration(checkpoint, plan, collections, labels, pins, destination)
        else:
            origin = plan["analytic_report"]
            binding = {name: record[0] for name, record in pins.items()
                       if name != str(args.plan.resolve())}
            report = adapter.document(Path(origin["path"]), pins, origin["sha256"])
            if any(report["source_sha256"].get(name) != digest for name, digest in binding.items()):
                raise ValueError("calibration input/source binding differs")
            rows = [{k: row[k] for k in ("objective", "purpose", "fold", "minimum_delta")} for row in report["rows"]]
            if (report["status"] != "analytic_calibration_complete" or report["continuous_calls"] != 1
                    or report["source_checkpoint"] != plan["checkpoint"]
                    or rows != plan["candidates"][0]["constraints"]):
                raise ValueError("one successful analytic calibration required")
            _, selected = adapter.matrix_column_indices(
                plan, collections["matrix:correct-logprob"]["coverage"]["entries"])
            selected_names = [entry["tensor"] for entry in selected]
            legacy_metric = {"type": "relative", "floor": 1e-6,
                             "block_scales": {"native": 1.0, "matrix": 1.0}}
            if (report.get("cost_metric", legacy_metric) != plan["candidates"][0]["cost_metric"]
                    or [entry["tensor"] for entry in report["layout"]] != selected_names):
                raise ValueError("calibration metric or matrix mask differs")
            expected = {"relative_radius": report["derived_relative_radius"],
                        "absolute_radius": report["derived_absolute_radius"], "rounds": 16}
            if plan["candidates"][0]["rounding"] != expected:
                raise ValueError("candidate radius differs from calibration")
            remaining = 180 - (time.monotonic() - started)
            argv = [sys.executable, "-B", str(Path(adapter.__file__)), "--plan", str(args.plan),
                    "--output", str(destination / "proposals"), "--run", "--write-spec",
                    "--maximum-seconds", str(max(1, math.floor(remaining - 1)))]
            with (destination / "candidate.log").open("x") as log:
                run = subprocess.run(argv, timeout=max(0.1, remaining), check=True, stdout=log, stderr=subprocess.STDOUT)
            proposal_dir = destination / "proposals"
            proposal_report = json.loads((proposal_dir / "report.json").read_bytes())
            measured = proposal_report["methods"][0]
            sample = collections["matrix:correct-logprob"]["path"] / "sample-plan.npz"
            with np.load(sample, allow_pickle=False) as data:
                original = np.asarray(data["source_values"], dtype=np.float64)
                coordinates = data["coordinates"]
            coverage = collections["matrix:correct-logprob"]["coverage"]
            lookup = {}
            for entry in coverage["entries"]:
                start, count = entry["start"], entry["count"]
                lookup[entry["tensor"]] = {tuple(rc): float(value) for rc, value in
                    zip(coordinates[start:start+count], original[start:start+count], strict=True)}
            maximum_excess, outside, changed = 0.0, 0, 0
            if measured["spec_emitted"]:
                spec = json.loads((proposal_dir / (measured["id"] + ".json")).read_bytes())
                for edit in spec["edits"]:
                    match = adapter.matrix_recipe.PATTERN.fullmatch(edit["tensor"])
                    if ("indices" in edit or not match or match[3] not in {"q", "k", "o", "gate", "up", "down"}
                            or edit["tensor"] not in selected_names):
                        raise ValueError("candidate edit outside analytic matrix mask")
                    for rc, value in zip(edit["coordinates"], edit["set_values"], strict=True):
                        old = lookup[edit["tensor"]][tuple(rc)]
                        radius = max(abs(old) * expected["relative_radius"], expected["absolute_radius"])
                        excess = max(abs(value - old) - radius, 0.0)
                        maximum_excess = max(maximum_excess, excess)
                        outside += excess > 0
                        changed += 1
            if measured["realized_blocks"]["native"]["changed_elements"] != 0 or changed != measured["changed_elements"]:
                raise ValueError("realized matrix-only count differs")
            realization = {"changed_elements": changed, "native_changed_elements": 0, "qkv_changed_elements": 0,
                           "maximum_realized_continuous_box_excess": maximum_excess,
                           "coordinates_outside_continuous_box": outside,
                           "continuous_box_excess_is_reported_not_silently_clipped": True,
                           "candidate_report_sha256": sha(proposal_dir / "report.json")}
            publish(destination / "realized-box-audit.json", realization)
            result = {"returncode": run.returncode, "proposals": str(proposal_dir), "realization": realization}
        adapter.stable(checkpoint, pins)
        publish(destination / "completion.json", {"status": "completed", "phase": args.phase,
                "seconds": time.monotonic()-started, "result": result, "inputs_stable": True,
                "model_calls": 0, "checkpoint_writes": 0})
        print(json.dumps(result))
    except BaseException as exc:
        if owned and not (destination / "failure.json").exists():
            publish(destination / "failure.json", {"status": "failed", "phase": args.phase,
                    "exception_type": type(exc).__name__, "seconds": time.monotonic()-started,
                    "no_automatic_retry": True})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)

if __name__ == "__main__":
    main()
