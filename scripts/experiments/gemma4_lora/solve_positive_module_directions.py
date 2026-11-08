#!/usr/bin/env python3
"""Fit signed donor gates to explicit correct-LL gains and separate margin floors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np

import profile_donor_directions as profiling
import solve_module_directions as legacy

BASIC = {"code", "implementation", "algorithm", "science_numerics", "science_contracts"}
ADVANCED = {"science_advanced_state", "science_advanced_counting",
            "science_advanced_simulation", "science_advanced_probability"}
PRIMARY = BASIC | ADVANCED
RELATED = legacy.RELATED
KIND = "positive_correct_module_plans_v2"
HEADROOM_KIND = "positive_correct_module_plans_v3_forced_empty_thought"
FRAME_PROTOCOL = "gemma4-public-answer-v2-forced-empty-thought"
HEADROOM_POLICY = {"tau": .01, "emphasis_tau": .02, "rho": .25, "saturation_headroom": .004,
                   "saturated_target": 0., "primary_margin_floor": 0.}


def collection(path, objective):
    path = path.resolve()
    result, provenance = profiling.analyze(path, expected_objective=objective)
    summary = json.loads((path / "summary.json").read_text())
    frame_path = path / "frame-evidence.json"
    frame_bytes = frame_path.read_bytes()
    frames = json.loads(frame_bytes)
    gradients = np.stack([np.load(path / row["file"], allow_pickle=False) for row in summary["folds"]]).astype(np.float64)
    costs = np.load(path / "unit-gate-weight-l2.npy", allow_pickle=False).astype(np.float64)
    hashes = dict(result["input_sha256"])
    hashes[str(frame_path)] = hashlib.sha256(frame_bytes).hexdigest()
    if any(legacy.sha(p) != value for p, value in hashes.items()):
        raise ValueError("collection changed during loading")
    return {"path": path, "summary": summary, "provenance": provenance, "frames": frames,
            "gradient": gradients, "costs": costs, "hashes": hashes}


def paired_collections(margin_path, correct_path):
    margin = collection(margin_path, "margin")
    correct = collection(correct_path, "correct-logprob")
    if margin["summary"].get("answer_batching", "paired") != correct["summary"].get("answer_batching", "paired"):
        raise ValueError("paired collection answer batching mismatch")
    for key in ("coverage", "mode"):
        if margin["summary"][key] != correct["summary"][key]:
            raise ValueError("paired collection inventory/mode mismatch")
    labels = [(r["purpose"], r["fold"]) for r in margin["summary"]["folds"]]
    if labels != [(r["purpose"], r["fold"]) for r in correct["summary"]["folds"]]:
        raise ValueError("paired fold order differs")
    if margin["frames"] != correct["frames"] or not np.array_equal(margin["costs"], correct["costs"]):
        raise ValueError("paired frame or weight-cost mismatch")
    for key in ("checkpoint", "attestation_receipt_sha256", "donor", "input_sha256",
                "threads", "seed", "max_cases_per_microbatch"):
        if margin["provenance"][key] != correct["provenance"][key]:
            raise ValueError("paired source/input/runtime provenance mismatch: " + key)
    for key in ("torch", "transformers", "source_dtype", "factor_compute_dtype",
                "gates_dtype", "attention", "max_cases_per_microbatch", "arithmetic"):
        if margin["summary"]["runtime"][key] != correct["summary"]["runtime"][key]:
            raise ValueError("paired runtime differs: " + key)
    if len(labels) != 42 or len(margin["summary"]["coverage"]["entries"]) != 410:
        raise ValueError("this amendment requires 42 folds and all 410 directions")
    if any(sorted(f for p, f in labels if p == name) != [0, 1] for name in {p for p, _ in labels}):
        raise ValueError("each purpose requires both authored folds")
    if not PRIMARY | RELATED <= {p for p, _ in labels}:
        raise ValueError("required purpose families missing")
    return margin, correct, labels


def headroom_rows(margin, correct):
    for data in (margin, correct):
        frame = data["frames"]
        if (frame.get("schema_version") != 2 or frame.get("frame_protocol") != FRAME_PROTOCOL
                or frame.get("enable_thinking") is not True or frame.get("forced_empty_thought") is not True
                or frame.get("public_boundary") != "<|channel>thought\n<channel|>"
                or data["summary"].get("answer_batching") != "serial"
                or data["summary"]["runtime"].get("gradient_accumulation_dtype") != "FP32"):
            raise ValueError("headroom plans require the verified forced-empty-thought serial frame")
    rows = []
    for fold in correct["summary"]["folds"]:
        value = fold["mean_correct_logprob"]
        cases = fold["correct_logprobs"]
        values = [case["correct_logprob"] for case in cases]
        if (type(value) not in (float, int) or not np.isfinite(value) or value > 0
                or len(cases) != fold["examples"] or not cases
                or len({case["id"] for case in cases}) != len(cases)
                or not all(type(v) in (float, int) and np.isfinite(v) and v <= 0 for v in values)
                or not np.isclose(np.mean(values), value, rtol=0, atol=1e-7)):
            raise ValueError("finite nonpositive per-case/fold correct-LL metadata required")
        rows.append({"purpose": fold["purpose"], "fold": fold["fold"],
                     "baseline_correct_logprob": value, "headroom": -value,
                     "saturated": -value <= HEADROOM_POLICY["saturation_headroom"],
                     "cases": [{"id": case["id"], "headroom": -case["correct_logprob"]} for case in cases]})
    return rows


def methods(labels, headroom=None):
    scopes = [
        ("balanced", None, 0, {}),
        ("science", None, 0, {p: .02 for p in PRIMARY if p.startswith("science_")}),
        ("algorithm", None, 0, {"algorithm": .02, "implementation": .015}),
        ("mlp", ["gate", "up", "down"], 0, {}),
        ("attention", ["q", "k", "v", "o"], 0, {}),
        ("late", None, 30, {}),
    ]
    result = []
    headroom_by_label = {(r["purpose"], r["fold"]): r for r in headroom} if headroom is not None else {}
    for name, families, layer, overrides in scopes:
        if headroom is not None and name == "algorithm":
            overrides = {"algorithm": .02, "implementation": .02}
        rows = []
        for objective in ("correct-logprob", "margin"):
            for purpose, fold in labels:
                if objective == "correct-logprob":
                    lower = overrides.get(purpose, .01 if purpose in PRIMARY else -.005 if purpose in RELATED else -.01)
                    if headroom is not None and purpose in PRIMARY:
                        h = headroom_by_label[purpose, fold]["headroom"]
                        lower = 0. if h <= .004 else min(overrides.get(purpose, .01), .25 * h)
                else:
                    lower = (0. if headroom is not None else -.005) if purpose in PRIMARY else -.01 if purpose in RELATED else -.02
                rows.append({"objective": objective, "purpose": purpose, "fold": fold, "minimum_delta": lower})
        result.append({"id": "donor-positive-" + name,
                       "scope": {"families": families, "minimum_layer": layer, "maximum_layer": 59},
                       "constraints": rows})
    if headroom is not None:
        for index, item in enumerate(result):
            previous = next((other for other in result[:index] if other["scope"] == item["scope"]
                             and other["constraints"] == item["constraints"]), None)
            if previous is not None:
                item["duplicate_of"] = previous["id"]
    return result


def validate_donor(original, provenance, unit_costs):
    if original.get("kind") != "dense_lora_recipe_v1" or original.get("schema_version") != 2:
        raise ValueError("original scalar donor recipe required")
    for key, other in (("donor_sha256", "payload_sha256"), ("donor_config_sha256", "config_sha256"),
                       ("donor_repo", "repo"), ("donor_revision", "revision")):
        if original[key] != provenance["donor"][other]:
            raise ValueError("donor recipe identity differs")
    if legacy.sha(unit_costs) != provenance["donor"]["unit_costs_sha256"]:
        raise ValueError("unit costs differ")


def validate_oracle(path, expected_sha, provenance):
    if path is None or legacy.sha(path) != expected_sha:
        raise ValueError("pinned official-frame oracle receipt required")
    oracle = json.loads(path.read_text())
    p = oracle.get("provenance", {})
    rows = [r for r in oracle.get("rows", []) if r.get("frame") == "shared_public_final_boundary"]
    if (oracle.get("status") != "completed" or oracle.get("source_stat_continuity") is not True
            or p.get("attestation_receipt_sha256") != provenance["attestation_receipt_sha256"]
            or p.get("frame_evidence", {}).get("frame_protocol") != FRAME_PROTOCOL
            or p.get("frame_evidence", {}).get("forced_empty_thought") is not True
            or len({r.get("id") for r in rows}) < 3):
        raise ValueError("matched-source completed three-case public-frame oracle required")
    for row in rows:
        values = row["custom_native_softcap"]
        official = row["official_full_model"]["correct_mean_logprob"]
        if (values["max_abs_logit_difference_from_official"] != 0
                or values["mean_logprob_difference_from_official"] != 0
                or type(official) not in (float, int) or not np.isfinite(official) or official > 0):
            raise ValueError("oracle scorer disagrees with official full forward")
    if legacy.sha(path) != expected_sha:
        raise ValueError("oracle receipt changed")


def make_plans(margin_path, correct_path, donor_spec, unit_costs, review_path, review_sha, label, method_review, method_review_sha,
               *, headroom_aware=False, frame_oracle=None, frame_oracle_sha=None):
    margin, correct, labels = paired_collections(margin_path, correct_path)
    provenance = margin["provenance"]
    content = donor_spec.read_bytes()
    original = json.loads(content)
    validate_donor(original, provenance, unit_costs)
    if legacy.sha(review_path) != review_sha or not label.strip():
        raise ValueError("explicit pinned calibration review and label required")
    if legacy.sha(method_review) != method_review_sha:
        raise ValueError("pinned method review differs")
    hashes = margin["hashes"] | correct["hashes"]
    hashes[str(review_path.resolve())] = review_sha
    hashes[str(method_review.resolve())] = method_review_sha
    headroom = headroom_rows(margin, correct) if headroom_aware else None
    if headroom_aware:
        validate_oracle(frame_oracle, frame_oracle_sha, provenance)
        hashes[str(frame_oracle.resolve())] = frame_oracle_sha
    return {"schema_version": 3 if headroom_aware else 2, "kind": HEADROOM_KIND if headroom_aware else KIND,
            **({"headroom_policy": HEADROOM_POLICY, "baseline_headroom": headroom, "frame_protocol": FRAME_PROTOCOL,
                "frame_oracle": str(frame_oracle.resolve()), "frame_oracle_sha256": frame_oracle_sha,
                "oracle_scope": "Caller-reviewed oracle receipt pinned; its acceptance remains an owner decision"} if headroom_aware else {}),
            "amendment": "adaptive correct-answer likelihood objective after margin-only finite degradation; not a new holdout",
            "margin_collection": str(margin["path"]), "correct_collection": str(correct["path"]),
            "calibration_label": label, "calibration_review": str(review_path.resolve()),
            "calibration_review_sha256": review_sha,
            "method_review": str(method_review.resolve()), "method_review_sha256": method_review_sha,
            "method_review_scope": "Pinned anonymous design review, not an actual candidate or measurement endorsement",
            "calibration_review_scope": "Caller-supplied pinned review evidence, not an automatic quality or approval assertion",
            "source_checkpoint": provenance["checkpoint"]["path"],
            "source_attestation_sha256": provenance["attestation_receipt_sha256"],
            "donor_spec": str(donor_spec.resolve()), "donor_spec_sha256": hashlib.sha256(content).hexdigest(),
            "unit_costs_path": str(unit_costs.resolve()), "unit_costs_sha256": legacy.sha(unit_costs),
            "input_sha256": hashes,
            "limits": {"maximum_absolute_coefficient": .25, "maximum_continuous_weight_l2": 1.0,
                       "maximum_cpu_seconds": 120},
            "objective_order": ["correct-logprob", "margin"],
            "coefficient_dtype": "FP32; model weights alone are rounded to BF16",
            "plans": methods(labels, headroom), "actual_bf16_metrics_required": True,
            "candidate_adopted": False}


def validate_plan(plan, margin, correct, labels):
    headroom = None
    if plan.get("schema_version") == 3 and plan.get("kind") == HEADROOM_KIND:
        headroom = headroom_rows(margin, correct)
        if (plan["headroom_policy"] != HEADROOM_POLICY or plan["baseline_headroom"] != headroom
                or plan["frame_protocol"] != FRAME_PROTOCOL):
            raise ValueError("frozen headroom policy/baseline/frame differs")
    elif plan.get("schema_version") != 2 or plan.get("kind") != KIND:
        raise ValueError("unsupported amended plan")
    if plan["limits"] != {"maximum_absolute_coefficient": .25, "maximum_continuous_weight_l2": 1.0,
                          "maximum_cpu_seconds": 120}:
        raise ValueError("fixed trust/time limits differ")
    if plan["objective_order"] != ["correct-logprob", "margin"] or plan["plans"] != methods(labels, headroom):
        raise ValueError("six registered scopes and all 84 explicit constraints must remain unchanged")
    expected = margin["hashes"] | correct["hashes"]
    review = plan["calibration_review"]
    expected[review] = plan["calibration_review_sha256"]
    expected[plan["method_review"]] = plan["method_review_sha256"]
    if headroom is not None:
        expected[plan["frame_oracle"]] = plan["frame_oracle_sha256"]
        validate_oracle(Path(plan["frame_oracle"]), plan["frame_oracle_sha256"], margin["provenance"])
    if expected != plan["input_sha256"]:
        raise ValueError("plan does not pin the complete paired collections/review")
    if (margin["provenance"]["checkpoint"]["path"] != plan["source_checkpoint"]
            or margin["provenance"]["attestation_receipt_sha256"] != plan["source_attestation_sha256"]):
        raise ValueError("source provenance differs")
    if any(legacy.sha(p) != digest for p, digest in expected.items()):
        raise ValueError("pinned inputs changed")


def solve_headroom_coefficients(gradient, lower, costs, mask):
    # A zero preservation target must not become a positive gain target on an
    # exactly saturated/zero-gradient row. Keep legacy defaults untouched.
    adjusted = np.asarray(lower, dtype=np.float64).copy()
    zero = adjusted == 0
    adjusted[zero] -= legacy.ROUNDING_MARGIN
    coefficients, report = legacy.solve_coefficients(gradient, adjusted, costs, mask)
    prediction = np.asarray(gradient) @ coefficients
    shortfall = float(np.maximum(np.asarray(lower) - prediction, 0).max())
    report.update(minimum_delta=np.asarray(lower).tolist(), predicted_response=prediction.tolist(),
                  maximum_shortfall=shortfall,
                  numerical_safety_margin=np.where(zero, 0., legacy.ROUNDING_MARGIN).tolist())
    if report["status"] == "continuous_proposal" and shortfall > 1e-10:
        report["status"] = "fp32_constraint_failure"
    if report["status"] == "continuous_proposal" and not np.count_nonzero(coefficients):
        report["status"] = "no_change_solution"
    return coefficients, report


def solve(plan_path, output):
    content = plan_path.read_bytes()
    plan = json.loads(content)
    margin, correct, labels = paired_collections(Path(plan["margin_collection"]), Path(plan["correct_collection"]))
    validate_plan(plan, margin, correct, labels)
    donor_spec, cost_path = Path(plan["donor_spec"]), Path(plan["unit_costs_path"])
    original_content = donor_spec.read_bytes()
    if (hashlib.sha256(original_content).hexdigest() != plan["donor_spec_sha256"]
            or legacy.sha(cost_path) != plan["unit_costs_sha256"]):
        raise ValueError("recipe or unit cost input changed")
    original = json.loads(original_content)
    validate_donor(original, margin["provenance"], cost_path)
    protected = [margin["path"], correct["path"], Path(plan["source_checkpoint"]).resolve()]
    destination = output.resolve()
    if output.exists() or output.is_symlink() or any(destination.is_relative_to(p) or p.is_relative_to(destination) for p in protected):
        raise ValueError("fresh output outside collections/checkpoint required")
    gradient = np.vstack((correct["gradient"], margin["gradient"]))
    costs = margin["costs"]
    entries = margin["summary"]["coverage"]["entries"]
    code = [Path(__file__), Path(legacy.__file__), Path(profiling.__file__),
            Path(legacy.joint.__file__), Path(legacy.joint.common.__file__), Path(legacy.joint.finite_response.__file__)]
    code_hashes = {str(p.resolve()): legacy.sha(p) for p in code}
    output.mkdir(parents=True)
    results, vectors = [], {}
    start = time.monotonic()
    previous = signal.getsignal(signal.SIGALRM)
    def deadline(*unused):
        raise TimeoutError("fixed six-method CPU budget exhausted")
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(plan["limits"]["maximum_cpu_seconds"])
    status = "completed"
    try:
        for method in plan["plans"]:
            if "duplicate_of" in method:
                results.append({"id": method["id"], "status": "duplicate_constraints_pruned", "duplicate_of": method["duplicate_of"]})
                continue
            scope = method["scope"]
            mask = np.array([(scope["families"] is None or entry["family"] in scope["families"])
                             and scope["minimum_layer"] <= entry["layer"] <= scope["maximum_layer"] for entry in entries])
            lower = np.array([row["minimum_delta"] for row in method["constraints"]], dtype=np.float64)
            try:
                solve_function = solve_headroom_coefficients if plan["kind"] == HEADROOM_KIND else legacy.solve_coefficients
                coefficients, record = solve_function(gradient, lower, costs, mask)
            except ValueError as error:
                results.append({"id": method["id"], "status": "continuous_unresolved", "reason": str(error)})
                continue
            record.update(id=method["id"], scope=scope, constraints=method["constraints"])
            response = np.asarray(record["predicted_response"])
            record["binding_constraint_indices"] = np.flatnonzero(np.abs(response - lower) <= 1e-6).tolist()
            record["box_active_tensors"] = [e["tensor"] for e, c in zip(entries, coefficients, strict=True) if abs(c) >= .24999]
            active = int(np.count_nonzero(coefficients))
            record["negative_nonzero_fraction"] = float(np.count_nonzero(coefficients < 0) / active) if active else None
            if record["status"] == "continuous_proposal":
                spec = {key: value for key, value in original.items() if key not in {"coefficient", "native_edits"}}
                spec.update(id=method["id"], schema_version=3, kind="module_lora_recipe_v1",
                            hypothesis=("Forced-empty-thought baseline-headroom targets plus separate margin floors; fitted diagnostics, not normal thinking-enabled generation capability."
                                        if plan["kind"] == HEADROOM_KIND else
                                        "Adaptive positive correct-LL plus separate margin floors; all constraints are fit diagnostics, not actual capability."),
                            module_coefficients={e["tensor"]: float(c) for e, c in zip(entries, coefficients, strict=True)},
                            unit_costs_path=str(cost_path.resolve()), unit_costs_sha256=plan["unit_costs_sha256"],
                            selection_plan_sha256=hashlib.sha256(content).hexdigest(),
                            collection_inputs_sha256=plan["input_sha256"], qualification="continuous_prediction_only",
                            actual_bf16_metrics_required=True)
                path = output / (method["id"] + ".json")
                legacy.fresh_json(path, spec)
                record.update(spec_path=str(path.resolve()), spec_sha256=legacy.sha(path))
                vectors[method["id"]] = costs * coefficients
            results.append(record)
    except TimeoutError:
        status = "time_budget_exhausted"
    except BaseException as error:
        legacy.fresh_json(output / "failure.json", {"status": "incomplete", "error_type": type(error).__name__})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    if (plan_path.read_bytes() != content or donor_spec.read_bytes() != original_content
            or legacy.sha(cost_path) != plan["unit_costs_sha256"]
            or any(legacy.sha(p) != digest for p, digest in plan["input_sha256"].items())
            or any(legacy.sha(p) != digest for p, digest in code_hashes.items())):
        raise ValueError("inputs or implementation changed")
    def cosine(a, b):
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(a @ b / denominator) if denominator else None
    cosines = {first: {second: cosine(a, b) for second, b in vectors.items()} for first, a in vectors.items()}
    normalized = np.divide(correct["gradient"], costs[None, :], out=np.zeros_like(correct["gradient"]), where=costs[None, :] > 0)
    algorithm_code = [{"fold": fold, "cost_normalized_gradient_cosine":
                       cosine(normalized[labels.index(("algorithm", fold))], normalized[labels.index(("code", fold))])}
                      for fold in (0, 1)]
    report = {"schema_version": plan["schema_version"], "status": status,
              **({"frame_protocol": FRAME_PROTOCOL, "headroom_policy": plan["headroom_policy"],
                  "baseline_headroom": plan["baseline_headroom"]} if plan["kind"] == HEADROOM_KIND else {}), "elapsed_seconds": time.monotonic() - start,
              "results": results, "unattempted": [m["id"] for m in plan["plans"] if m["id"] not in {r["id"] for r in results}],
              "plan_sha256": hashlib.sha256(content).hexdigest(), "implementation_sha256": code_hashes,
              "weighted_coefficient_cosines": cosines, "correct_ll_algorithm_code_cosines": algorithm_code,
              "cosine_scope": "Descriptive local geometry; near-collinear proposals may share a finite test, never an automatic score or significance claim",
              "numpy_version": np.__version__, "actual_bf16_metrics_required": True,
              "model_calls": 0, "new_checkpoints": 0, "candidate_adopted": False,
              "limitations": ["All rows fit; no holdout claim.", "Finite BF16 correct-NLL, margin and KL must be measured separately.",
                              "Outside trust bounds or numerical nonconvergence is not a global infeasibility proof.",
                              "Allowed correct-LL and margin losses are explicit tradeoffs, not a blanket no-regression rule."]}
    legacy.fresh_json(output / "report.json", report)
    return report


def audit_old(margin_path, correct_path, old_path, actual_path, spec_directory):
    margin, correct, labels = paired_collections(margin_path, correct_path)
    old = collection(old_path, "margin")
    oldlabels = [(f["purpose"], f["fold"]) for f in old["summary"]["folds"]]
    if len(oldlabels) != 34 or old["summary"]["coverage"] != margin["summary"]["coverage"]:
        raise ValueError("exact old 34-fold coordinate inventory required")
    if any(margin["frames"]["frames"][key] != value for key, value in old["frames"]["frames"].items()):
        raise ValueError("old frames changed")
    for key in ("checkpoint", "attestation_receipt_sha256", "donor"):
        if old["provenance"][key] != margin["provenance"][key]:
            raise ValueError("old source/donor differs")
    indices = [labels.index(label) for label in oldlabels]
    if not np.array_equal(margin["gradient"][indices], old["gradient"]):
        raise ValueError("old margin derivatives do not reproduce exactly")
    actual_content = actual_path.read_bytes()
    actual = json.loads(actual_content)
    hashes = margin["hashes"] | correct["hashes"] | old["hashes"]
    hashes[str(actual_path.resolve())] = hashlib.sha256(actual_content).hexdigest()
    rows = []
    for candidate in actual["candidates"]:
        spec_path = spec_directory / (candidate["id"] + ".json")
        spec_content = spec_path.read_bytes()
        hashes[str(spec_path.resolve())] = hashlib.sha256(spec_content).hexdigest()
        if hashes[str(spec_path.resolve())] != candidate["spec_sha256"]:
            raise ValueError("actual candidate spec differs")
        spec = json.loads(spec_content)
        entries = old["summary"]["coverage"]["entries"]
        if set(spec["module_coefficients"]) != {e["tensor"] for e in entries}:
            raise ValueError("candidate coordinate set differs")
        coefficients = np.array([spec["module_coefficients"][e["tensor"]] for e in entries])
        actual_labels = [(purpose, f["fold"]) for purpose, group in candidate["groups"].items() for f in group["folds"]]
        if (len(actual_labels) != len(oldlabels) or set(actual_labels) != set(oldlabels)
                or not np.isfinite(coefficients).all()):
            raise ValueError("finite exact old-fold actual evidence required")
        prediction = correct["gradient"][indices] @ coefficients
        actual_ll = np.array([-next(f for f in candidate["groups"][purpose]["folds"] if f["fold"] == fold)["correct_nll_delta"]
                              for purpose, fold in oldlabels])
        if not np.isfinite(actual_ll).all():
            raise ValueError("actual likelihood changes must be finite")
        primary = np.array([p in BASIC for p, _ in oldlabels])
        rows.append({"id": candidate["id"], "folds": 34,
                     "predicted_correct_ll_negative": int((prediction < 0).sum()),
                     "actual_correct_ll_negative": int((actual_ll < 0).sum()),
                     "sign_agreement": int((np.sign(prediction) == np.sign(actual_ll)).sum()),
                     "pearson": float(np.corrcoef(prediction, actual_ll)[0, 1]) if np.std(prediction) and np.std(actual_ll) else None,
                     "predicted_correct_ll_mean": float(prediction.mean()), "actual_correct_ll_mean": float(actual_ll.mean()),
                     "primary_predicted_correct_ll_mean": float(prediction[primary].mean()),
                     "primary_actual_correct_ll_mean": float(actual_ll[primary].mean()),
                     "fold_values": [{"purpose": p, "fold": f, "predicted_correct_ll_delta": float(x), "actual_correct_ll_delta": float(y)}
                                     for (p, f), x, y in zip(oldlabels, prediction, actual_ll, strict=True)]})
    if any(legacy.sha(p) != digest for p, digest in hashes.items()):
        raise ValueError("audit inputs changed")
    return {"schema_version": 1, "status": "completed", "old_margin_reproduction": "all 34 arrays numerically identical",
            "advanced_v1_rows_used": 0, "candidates": rows, "input_sha256": hashes,
            "implementation_sha256": legacy.sha(Path(__file__)), "model_calls": 0,
            "limits": ["Compares local continuous predictions with recorded finite BF16 likelihood changes; sign agreement/correlation are reported, not assumed.",
                       "Existing actual results are partial 2/4 and remain so; no missing candidate is scored as zero."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    make = sub.add_parser("make-plans")
    audit = sub.add_parser("audit-old")
    for command in (make, audit):
        command.add_argument("--margin", type=Path, required=True)
        command.add_argument("--correct", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    make.add_argument("--donor-spec", type=Path, required=True)
    make.add_argument("--unit-costs", type=Path, required=True)
    make.add_argument("--calibration-review", type=Path, required=True)
    make.add_argument("--calibration-review-sha256", required=True)
    make.add_argument("--calibration-label", required=True)
    make.add_argument("--method-review", type=Path, required=True)
    make.add_argument("--method-review-sha256", required=True)
    make.add_argument("--headroom-aware", action="store_true", help="explicit v3 forced-empty-thought serial baseline-headroom protocol")
    make.add_argument("--frame-oracle", type=Path)
    make.add_argument("--frame-oracle-sha256")
    audit.add_argument("--old-margin", type=Path, required=True)
    audit.add_argument("--actual", type=Path, required=True)
    audit.add_argument("--spec-directory", type=Path, required=True)
    run = sub.add_parser("solve")
    run.add_argument("--plans", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "solve":
        result = solve(args.plans, args.output)
    else:
        if args.output.exists() or args.output.is_symlink():
            parser.error("fresh output required")
        if args.action == "make-plans":
            result = make_plans(args.margin, args.correct, args.donor_spec, args.unit_costs,
                                args.calibration_review, args.calibration_review_sha256, args.calibration_label,
                                args.method_review, args.method_review_sha256, headroom_aware=args.headroom_aware,
                                frame_oracle=args.frame_oracle, frame_oracle_sha=args.frame_oracle_sha256)
        else:
            result = audit_old(args.margin, args.correct, args.old_margin, args.actual, args.spec_directory)
        source = Path(json.loads((args.margin / "provenance.json").read_text())["checkpoint"]["path"]).resolve()
        protected = [source, args.margin, args.correct]
        if args.action == "audit-old":
            protected += [args.old_margin, args.spec_directory]
        if any(args.output.resolve().is_relative_to(p.resolve()) for p in protected):
            parser.error("output overlaps protected source/collection")
        legacy.fresh_json(args.output, result)
    print(json.dumps({"status": result.get("status", "plans_prepared"), "model_calls": 0}))
    return 0


if __name__ == "__main__":
    main()
