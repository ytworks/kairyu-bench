#!/usr/bin/env python3
"""One joint native/donor local proposal; no model execution or checkpoint creation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np

import solve_positive_module_directions as positive

joint = positive.legacy.joint
sha = positive.legacy.sha
fresh_json = positive.legacy.fresh_json


def native_collection(path, objective, donor, labels):
    """Read authored calibration gradients, never benchmark items or responses."""
    path = path.resolve()
    summary_path, frame_path = path / "summary.json", path / "frame-evidence.json"
    summary_bytes, frame_bytes = summary_path.read_bytes(), frame_path.read_bytes()
    summary, frames = json.loads(summary_bytes), json.loads(frame_bytes)
    constraints = [{"purpose": p, "fold": f, "minimum_delta": 0.} for p, f in labels]
    base, gradient, _, layout, coordinates, actual_labels, hashes, mode, provenance = joint.read_collection(
        path, {"constraints": constraints})
    hashes[str(frame_path)] = hashlib.sha256(frame_bytes).hexdigest()
    if (hashes[str(summary_path)] != hashlib.sha256(summary_bytes).hexdigest()
            or summary.get("objective_kind") != objective or provenance.get("objective_kind") != objective
            or summary.get("answer_batching") != "serial" or provenance.get("answer_batching") != "serial"
            or mode != donor["summary"]["mode"] or coordinates is not None
            or provenance.get("virtual_unit_gains") is not False):
        raise ValueError("matched serial native-only objective collection required")
    coverage = json.loads((path / "coverage.json").read_text())
    if (any(t["kind"] != "native" for t in coverage["tensors"])
            or frames != donor["frames"]
            or [(r["purpose"], r["fold"]) for r in actual_labels] != labels):
        raise ValueError("native coverage, frames or measured folds differ")
    for key in ("checkpoint", "attestation_receipt_sha256", "threads", "seed", "max_cases_per_microbatch"):
        if provenance[key] != donor["provenance"][key]:
            raise ValueError("native/donor source/runtime differs: " + key)
    for key in ("torch", "transformers", "attention", "max_cases_per_microbatch"):
        if summary["runtime"][key] != donor["summary"]["runtime"][key]:
            raise ValueError("native/donor numerical runtime differs: " + key)
    if summary["runtime"].get("weight_dtype") != donor["summary"]["runtime"].get("source_dtype"):
        raise ValueError("native/donor source dtype differs")
    common_inputs = set(provenance["input_sha256"]) & set(donor["provenance"]["input_sha256"])
    if not common_inputs or any(provenance["input_sha256"][p] != donor["provenance"]["input_sha256"][p]
                                for p in common_inputs):
        raise ValueError("shared collector input provenance differs")
    key = "mean_margin" if objective == "margin" else "mean_correct_logprob"
    sample_key, value_key = ("margins", "margin") if objective == "margin" else ("correct_logprobs", "correct_logprob")
    records = {(row["purpose"], row["fold"]): row for row in summary["folds"]}
    for other in donor["summary"]["folds"]:
        row = records[other["purpose"], other["fold"]]
        first = [(c["id"], c[value_key]) for c in row[sample_key]]
        second = [(c["id"], c[value_key]) for c in other[sample_key]]
        if row[key] != other[key] or first != second:
            raise ValueError("native/donor unmodified per-case objective differs")
    if any(sha(p) != digest for p, digest in hashes.items()):
        raise ValueError("native collection changed while reading")
    return {"base": base, "gradient": gradient, "layout": layout, "hashes": hashes,
            "provenance": provenance, "frames": frames, "summary": summary}


def joint_proposal(base, native_gradient, donor_gradient, lower, donor_costs):
    """One unconstrained minimum-L2 relaxation, then explicit trust/rounding checks.

    No clipping, second solve, quantization repair or donor-gate BF16 rounding.
    A failed postcheck is a rejection of this proposal, not an infeasibility proof.
    """
    base, gn, gd, lower, costs = (np.asarray(x, dtype=np.float64) for x in
                                  (base, native_gradient, donor_gradient, lower, donor_costs))
    if (gn.ndim != 2 or gd.ndim != 2 or gn.shape != (len(lower), len(base))
            or gd.shape != (len(lower), len(costs)) or not np.isfinite(base).all()
            or not np.isfinite(costs).all() or np.any(costs <= 0)):
        raise ValueError("matching finite native/donor gradients and positive donor costs required")
    gradient = np.concatenate((gn, gd), axis=1)
    weight_cost = np.concatenate((np.ones(base.size), costs))
    proposed, dual = joint.minimum_step(gradient, lower, weight_cost)
    native_step, raw_gates = proposed[:base.size], proposed[base.size:]
    values, rounding = joint.bf16_realization(
        base, np.column_stack((native_step, np.zeros_like(native_step))), [1., 0.])
    # Unedited coordinates need not be emitted, and retain their original sign bits.
    values[native_step == 0] = base[native_step == 0].astype(np.float32)
    native_delta = values.astype(np.float64) - base
    gates = raw_gates.astype(np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        predicted = gn @ native_delta + gd @ gates.astype(np.float64)
    continuous = gradient @ proposed
    native_l2 = float(np.linalg.norm(native_delta))
    donor_l2 = float(np.linalg.norm(costs * gates.astype(np.float64)))
    norm = float(np.hypot(native_l2, donor_l2))
    shortfalls = np.maximum(lower - predicted, 0.)
    reasons = []
    if not np.isfinite(gates).all() or not np.isfinite(predicted).all() or not np.isfinite(norm):
        reasons.append("nonfinite_realization")
    if np.max(np.abs(gates)) > .25:
        reasons.append("donor_coefficient_outside_box")
    if norm > 1.:
        reasons.append("hybrid_norm_outside_radius")
    if not np.all(predicted >= lower - 1e-10):
        reasons.append("native_bf16_and_donor_fp32_constraint_failure")
    if not np.count_nonzero(native_delta) and not np.count_nonzero(gates):
        reasons.append("no_change")
    report = {
        "status": "hybrid_local_proposal" if not reasons else "post_realization_rejected",
        "eligible_for_joint_finite": not reasons, "rejection_reasons": reasons, "dual": dual,
        "continuous_weight_l2": float(np.linalg.norm(weight_cost * proposed)),
        "continuous_prediction": continuous.tolist(),
        "continuous_maximum_shortfall": float(np.maximum(lower - continuous, 0.).max()),
        "native_actual_delta_l2": native_l2, "native_changed_elements": int(np.count_nonzero(native_delta)),
        "native_rounding": rounding, "donor_continuous_delta_l2": donor_l2,
        "hybrid_weight_l2": norm, "maximum_absolute_donor_coefficient": float(np.max(np.abs(gates))),
        "donor_nonzero_coefficients": int(np.count_nonzero(gates)),
        "post_realization_prediction": predicted.tolist(), "maximum_shortfall": float(shortfalls.max()),
        "failed_constraint_indices": np.flatnonzero(shortfalls > 1e-10).tolist(),
        "minimum_delta": lower.tolist(),
        "norm_scope": "Exact selected native BF16 differences plus continuous donor matrix directions on disjoint tensors; donor BF16 weight rounding is not measured.",
        "actual_joint_model_response_measured": False, "donor_bf16_rounding_measured": False,
        "solver_calls": 1, "coefficient_rounding": "FP32 only; never BF16",
    }
    return values, gates, report


def repair_proposal(base, frozen_values, native_gradient, donor_gradient, lower, donor_costs):
    """One donor-only residual solve, keeping the previous native BF16 bytes fixed."""
    base, gn, gd, lower, costs = (np.asarray(x, dtype=np.float64) for x in
                                  (base, native_gradient, donor_gradient, lower, donor_costs))
    values = np.asarray(frozen_values)
    if (values.dtype != np.float32 or values.shape != base.shape or not np.isfinite(values).all()
            or np.any(values.view(np.uint32) & 0xFFFF)
            or gn.shape != (len(lower), len(base)) or gd.shape != (len(lower), len(costs))):
        raise ValueError("exact finite frozen native BF16 values and matching gradients required")
    native_delta = values.astype(np.float64) - base
    native_response = gn @ native_delta
    residual = lower - native_response
    raw_gates, dual = joint.minimum_step(gd, residual, costs)
    gates = raw_gates.astype(np.float32)
    prediction = native_response + gd @ gates.astype(np.float64)
    continuous = native_response + gd @ raw_gates
    native_l2 = float(np.linalg.norm(native_delta))
    donor_l2 = float(np.linalg.norm(costs * gates.astype(np.float64)))
    norm = float(np.hypot(native_l2, donor_l2))
    shortfall = np.maximum(lower - prediction, 0.)
    reasons = []
    if not np.isfinite(gates).all() or not np.isfinite(prediction).all() or not np.isfinite(norm):
        reasons.append("nonfinite_realization")
    if np.max(np.abs(gates)) > .25:
        reasons.append("donor_coefficient_outside_box")
    if norm > 1.:
        reasons.append("hybrid_norm_outside_radius")
    if not np.all(prediction >= lower - 1e-10):
        reasons.append("frozen_native_donor_fp32_constraint_failure")
    if not np.count_nonzero(native_delta) and not np.count_nonzero(gates):
        reasons.append("no_change")
    return values.copy(), gates, {
        "status": "hybrid_local_proposal" if not reasons else "post_realization_rejected",
        "eligible_for_joint_finite": not reasons, "rejection_reasons": reasons,
        "proposal_mode": "frozen_native_donor_residual_repair", "dual": dual,
        "continuous_weight_l2": float(np.hypot(native_l2, np.linalg.norm(costs * raw_gates))),
        "continuous_prediction": continuous.tolist(),
        "continuous_maximum_shortfall": float(np.maximum(lower - continuous, 0.).max()),
        "native_actual_delta_l2": native_l2, "native_changed_elements": int(np.count_nonzero(native_delta)),
        "frozen_native_f32_sha256": hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest(),
        "native_rescaled_or_rerounded": False, "native_predicted_response": native_response.tolist(),
        "donor_residual_targets": residual.tolist(), "donor_continuous_delta_l2": donor_l2,
        "hybrid_weight_l2": norm, "maximum_absolute_donor_coefficient": float(np.max(np.abs(gates))),
        "donor_nonzero_coefficients": int(np.count_nonzero(gates)),
        "post_realization_prediction": prediction.tolist(), "maximum_shortfall": float(shortfall.max()),
        "failed_constraint_indices": np.flatnonzero(shortfall > 1e-10).tolist(),
        "minimum_delta": lower.tolist(),
        "norm_scope": "Frozen native BF16 differences plus continuous donor directions on disjoint tensors; donor BF16 weight rounding unmeasured.",
        "actual_joint_model_response_measured": False, "donor_bf16_rounding_measured": False,
        "solver_calls": 1, "coefficient_rounding": "FP32 only; never BF16",
    }


def read_frozen_proposal(report_path, base, layout, gn, gd, hashes, code_hashes):
    content = report_path.read_bytes()
    previous = json.loads(content)
    arrays_path = report_path.parent / "proposal.npz"
    if (previous.get("status") != "post_realization_rejected" or previous.get("solver_calls") != 1
            or previous.get("repair_from_report") is not None
            or previous.get("layout") != layout or previous.get("source_sha256") != hashes
            or sha(arrays_path) != previous["proposal_sha256"]):
        raise ValueError("pinned first rejected joint proposal required; recursive repair is not allowed")
    # This script may add the explicit repair action; every reused numerical dependency must match.
    implementation = previous["implementation_sha256"]
    if set(implementation) != set(code_hashes) or any(
            implementation[p] != digest for p, digest in code_hashes.items() if p != str(Path(__file__).resolve())):
        raise ValueError("joint numerical dependencies changed since the frozen proposal")
    with np.load(arrays_path, allow_pickle=False) as data:
        if set(data.files) != {"native_bf16_values", "donor_fp32_coefficients"}:
            raise ValueError("unexpected frozen proposal array inventory")
        values, gates = data["native_bf16_values"].copy(), data["donor_fp32_coefficients"].copy()
    if values.shape != base.shape or gates.shape != (gd.shape[1],):
        raise ValueError("frozen proposal dimensions differ")
    actual_prediction = gn @ (values.astype(np.float64) - base) + gd @ gates.astype(np.float64)
    if (not np.allclose(actual_prediction, previous["post_realization_prediction"], rtol=0, atol=1e-12)
            or int(np.count_nonzero(values.astype(np.float64) - base)) != previous["native_changed_elements"]
            or report_path.read_bytes() != content or sha(arrays_path) != previous["proposal_sha256"]):
        raise ValueError("frozen native proposal no longer reproduces its recorded prediction")
    return values, {str(report_path.resolve()): hashlib.sha256(content).hexdigest(),
                    str(arrays_path.resolve()): previous["proposal_sha256"]}


def solve(plans_path, native_margin_path, native_correct_path, output, *, repair_report=None):
    plan_bytes = plans_path.read_bytes()
    plan = json.loads(plan_bytes)
    if plan.get("kind") != positive.HEADROOM_KIND:
        raise ValueError("verified forced-empty-thought headroom plan required")
    margin, correct, labels = positive.paired_collections(Path(plan["margin_collection"]), Path(plan["correct_collection"]))
    positive.validate_plan(plan, margin, correct, labels)
    destination = output.resolve()
    protected = [margin["path"], correct["path"], native_margin_path.resolve(), native_correct_path.resolve(),
                 Path(plan["source_checkpoint"]).resolve(), plans_path.resolve(), Path(__file__).resolve()]
    if repair_report is not None:
        protected.append(repair_report.parent.resolve())
    if output.exists() or output.is_symlink() or any(
            destination.is_relative_to(p) or p.is_relative_to(destination) for p in protected):
        raise ValueError("fresh output disjoint from inputs/checkpoint required")
    output.mkdir(parents=True)
    started = time.monotonic()
    previous = signal.getsignal(signal.SIGALRM)
    def deadline(*unused):
        raise TimeoutError("one joint proposal exceeded fixed 120 second CPU budget")
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(120)
    hashes = dict(plan["input_sha256"])
    code = [Path(__file__), Path(positive.__file__), Path(positive.legacy.__file__),
            Path(joint.__file__), Path(joint.common.__file__), Path(joint.finite_response.__file__)]
    code_hashes = {str(p.resolve()): sha(p) for p in code}
    try:
        native_margin = native_collection(native_margin_path, "margin", margin, labels)
        native_correct = native_collection(native_correct_path, "correct-logprob", correct, labels)
        if (native_margin["layout"] != native_correct["layout"]
                or not np.array_equal(native_margin["base"], native_correct["base"])):
            raise ValueError("native source inventory/values differ")
        native_names = {e["tensor"] for e in native_margin["layout"]}
        donor_names = {e["tensor"] for e in margin["summary"]["coverage"]["entries"]}
        if native_names & donor_names:
            raise ValueError("native/donor tensors overlap; diagonal joint cost invalid")
        hashes |= native_margin["hashes"] | native_correct["hashes"]
        method = next(m for m in plan["plans"] if m["id"] == "donor-positive-balanced")
        lower = np.array([c["minimum_delta"] for c in method["constraints"]], dtype=np.float64)
        gn = np.vstack((native_correct.pop("gradient"), native_margin.pop("gradient")))
        gd = np.vstack((correct["gradient"], margin["gradient"]))
        hashes[str(plans_path.resolve())] = hashlib.sha256(plan_bytes).hexdigest()
        if repair_report is None:
            values, gates, report = joint_proposal(native_margin["base"], gn, gd, lower, margin["costs"])
        else:
            frozen, repair_hashes = read_frozen_proposal(
                repair_report, native_margin["base"], native_margin["layout"], gn, gd, hashes, code_hashes)
            values, gates, report = repair_proposal(native_margin["base"], frozen, gn, gd, lower, margin["costs"])
            if not np.array_equal(values.view(np.uint32), frozen.view(np.uint32)):
                raise ValueError("repair changed frozen native bytes")
            report["repair_from_report"] = str(repair_report.resolve())
            hashes |= repair_hashes
        if any(sha(p) != digest for p, digest in hashes.items()) or any(sha(p) != v for p, v in code_hashes.items()):
            raise ValueError("input/code changed during joint proposal")
        report.update(schema_version=1, id="native-donor-positive-balanced",
                      frame_protocol=positive.FRAME_PROTOCOL, answer_batching="serial",
                      constraints=method["constraints"], source_checkpoint=plan["source_checkpoint"],
                      source_attestation_sha256=plan["source_attestation_sha256"],
                      source_sha256=hashes, implementation_sha256=code_hashes,
                      layout=native_margin["layout"], native_tensor_count=len(native_names),
                      native_coordinates=int(values.size), donor_tensor_count=len(donor_names),
                      disjoint_weight_tensors=True, matched_per_case_original_objectives=True,
                      limits=plan["limits"], seconds=time.monotonic() - started,
                      model_calls=0, new_checkpoints=0, candidate_adopted=False,
                      limitations=["All 84 constraints are fitted diagnostics, not holdout or actual capability.",
                                   "Forced-empty-thought authored answers differ from normal generated thinking.",
                                   "Native and donor functional effects need not add at finite changes.",
                                   "One relaxed solve with postchecks; rejection is not joint infeasibility.",
                                   "Proposal arrays are not a checkpoint or an automatically eligible editor spec."])
        with (output / "proposal.npz").open("xb") as stream:
            np.savez_compressed(stream, native_bf16_values=values, donor_fp32_coefficients=gates)
        report["proposal_sha256"] = sha(output / "proposal.npz")
        fresh_json(output / "report.json", report)
    except BaseException as error:
        fresh_json(output / "failure.json", {"status": "incomplete", "error_type": type(error).__name__,
                   "reason": str(error), "seconds": time.monotonic() - started,
                   "model_calls": 0, "new_checkpoints": 0})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    return report


def export_components(plans_path, report_path, output):
    """Export two required-together components from an eligible saved repair; no solve."""
    plan_bytes, report_bytes = plans_path.read_bytes(), report_path.read_bytes()
    plan, report = json.loads(plan_bytes), json.loads(report_bytes)
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    arrays = report_path.parent / "proposal.npz"
    hashes = report["source_sha256"]
    if (plan.get("kind") != positive.HEADROOM_KIND or report.get("eligible_for_joint_finite") is not True
            or report.get("proposal_mode") != "frozen_native_donor_residual_repair"
            or hashes.get(str(plans_path.resolve())) != hashlib.sha256(plan_bytes).hexdigest()
            or report["source_checkpoint"] != plan["source_checkpoint"]
            or report["source_attestation_sha256"] != plan["source_attestation_sha256"]
            or report["constraints"] != plan["plans"][0]["constraints"]
            or sha(arrays) != report["proposal_sha256"]):
        raise ValueError("eligible pinned balanced repair report required")
    protected = [Path(plan["source_checkpoint"]).resolve(), report_path.parent.resolve(), plans_path.resolve(),
                 Path(plan["margin_collection"]).resolve(), Path(plan["correct_collection"]).resolve()]
    if output.exists() or output.is_symlink() or any(
            output.resolve().is_relative_to(p) or p.is_relative_to(output.resolve()) for p in protected):
        raise ValueError("fresh component directory disjoint from inputs required")
    if any(sha(p) != digest for p, digest in hashes.items()):
        raise ValueError("recorded proposal input changed")
    native_paths = [Path(p) for p in hashes if Path(p).name == "source-native-values.npz"]
    if len(native_paths) != 2:
        raise ValueError("paired native source arrays required")
    with np.load(arrays, allow_pickle=False) as data:
        values, gates = data["native_bf16_values"], data["donor_fp32_coefficients"]
    sources = []
    for source in native_paths:
        with np.load(source, allow_pickle=False) as data:
            if set(data.files) != {r["tensor"] for r in report["layout"]}:
                raise ValueError("native component source inventory differs")
            sources.append(np.concatenate([data[r["tensor"]].reshape(-1) for r in report["layout"]]).astype(np.float64))
    if not np.array_equal(sources[0], sources[1]) or values.shape != sources[0].shape:
        raise ValueError("native component source bytes/shape differ")
    if (values.dtype != np.float32 or gates.dtype != np.float32
            or not np.isfinite(values).all() or not np.isfinite(gates).all()
            or np.any(values.view(np.uint32) & 0xFFFF) or np.max(np.abs(gates)) > .25):
        raise ValueError("finite BF16 native and bounded FP32 donor component arrays required")
    donor_path = Path(plan["donor_spec"])
    if sha(donor_path) != plan["donor_spec_sha256"]:
        raise ValueError("original scalar donor recipe changed")
    original = json.loads(donor_path.read_text())
    donor_dir = Path(plan["margin_collection"])
    entries = json.loads((donor_dir / "summary.json").read_text())["coverage"]["entries"]
    costs = np.load(donor_dir / "unit-gate-weight-l2.npy", allow_pickle=False)
    if gates.shape != (len(entries),) or len(entries) != 410:
        raise ValueError("all 410 explicit donor coefficients required")
    native_names = {r["tensor"] for r in report["layout"]}
    if native_names & {r["tensor"] for r in entries}:
        raise ValueError("component tensors overlap")
    actual_l2 = float(np.hypot(np.linalg.norm(values.astype(np.float64) - sources[0]),
                               np.linalg.norm(costs * gates.astype(np.float64))))
    if actual_l2 > 1. or not np.isclose(actual_l2, report["hybrid_weight_l2"], rtol=0, atol=1e-14):
        raise ValueError("component total norm differs")
    edits, cursor = [], 0
    for row in report["layout"]:
        count = row["count"]
        indices = np.flatnonzero(values[cursor:cursor+count].astype(np.float64) != sources[0][cursor:cursor+count])
        if indices.size:
            edits.append({"tensor": row["tensor"], "indices": indices.tolist(),
                          "set_values": values[cursor+indices].tolist()})
        cursor += count
    if cursor != len(values):
        raise ValueError("component native layout length differs")
    common = {"hypothesis": "Required-together native/donor residual proposal; fitted local predictions only.",
              "qualification": "joint_component_only_prediction", "joint_report_sha256": report_sha,
              "joint_report_path": str(report_path.resolve()), "companion_required": True,
              "actual_bf16_metrics_required": True}
    native = dict(common, id="native-donor-positive-balanced-native", edits=edits)
    donor = {k: v for k, v in original.items() if k not in {"coefficient", "native_edits"}}
    donor.update(common, id="native-donor-positive-balanced-donor", schema_version=3, kind="module_lora_recipe_v1",
                 module_coefficients={e["tensor"]: float(c) for e, c in zip(entries, gates, strict=True)},
                 unit_costs_path=plan["unit_costs_path"], unit_costs_sha256=plan["unit_costs_sha256"],
                 selection_plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
                 collection_inputs_sha256=hashes)
    if (plans_path.read_bytes() != plan_bytes or report_path.read_bytes() != report_bytes
            or sha(arrays) != report["proposal_sha256"]
            or any(sha(p) != digest for p, digest in hashes.items())):
        raise ValueError("component inputs changed while exporting")
    output.mkdir(parents=True)
    result = {"status": "component_specs_only", "joint_report_sha256": report_sha,
              "hybrid_weight_l2": actual_l2, "actual_joint_response_measured": False,
              "model_calls": 0, "new_checkpoints": 0, "candidate_adopted": False, "components": []}
    for spec in (native, donor):
        path = output / (spec["id"] + ".json")
        fresh_json(path, spec)
        result["components"].append({"path": str(path.resolve()), "sha256": sha(path),
                                     "companion_required": True})
    fresh_json(output / "components.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--native-margin", type=Path)
    parser.add_argument("--native-correct", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair-report", type=Path,
                        help="one explicit donor-only residual solve with a first rejected proposal's native BF16 values fixed")
    parser.add_argument("--export-report", type=Path, help="export required-together specs from a saved eligible repair; no solve")
    args = parser.parse_args(argv)
    if args.export_report is not None:
        if args.repair_report is not None or args.native_margin is not None or args.native_correct is not None:
            parser.error("--export-report is a separate metadata-only action")
        print(json.dumps(export_components(args.plans, args.export_report, args.output)))
        return 0
    if args.native_margin is None or args.native_correct is None:
        parser.error("both native collection paths required for a solve")
    report = solve(args.plans, args.native_margin, args.native_correct, args.output, repair_report=args.repair_report)
    print(json.dumps({k: report[k] for k in ("status", "eligible_for_joint_finite", "native_changed_elements",
          "native_actual_delta_l2", "donor_continuous_delta_l2", "hybrid_weight_l2", "maximum_shortfall", "seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
