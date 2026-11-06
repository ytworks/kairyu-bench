#!/usr/bin/env python3
"""Saved-gradient sanity checks: uniform donor directions and one native BF16 correction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np

import propose_joint_native_donor as proposal

positive = proposal.positive
joint = proposal.joint


def uniform_donor(correct, margin, costs, labels, baseline_correct_ll):
    """Descriptive local predictions; no inherited radius-one eligibility gate."""
    gc, gm, costs, ll = (np.asarray(x, dtype=np.float64) for x in
                         (correct, margin, costs, baseline_correct_ll))
    if (gc.ndim != 2 or gm.shape != gc.shape or gc.shape != (len(labels), len(costs))
            or ll.shape != (len(labels),) or any(not np.isfinite(x).all() for x in (gc, gm, costs, ll))
            or np.any(costs < 0) or np.any(ll > 0)):
        raise ValueError("matching finite likelihood/margin gradients, costs and nonpositive baseline LL required")
    groups = {"primary": [i for i, (p, _) in enumerate(labels) if p in positive.PRIMARY],
              "related": [i for i, (p, _) in enumerate(labels) if p in positive.RELATED],
              "other": [i for i, (p, _) in enumerate(labels) if p not in positive.PRIMARY | positive.RELATED]}
    rows = []
    for requested in (.05, .10, .25):
        coefficient = float(np.float32(requested))
        dc, dm = gc.sum(axis=1) * coefficient, gm.sum(axis=1) * coefficient
        grouped = {}
        for group, indices in groups.items():
            a, b = dc[indices], dm[indices]
            grouped[group] = {"folds": len(indices), "correct_ll_positive": int(np.count_nonzero(a > 0)),
                              "correct_ll_negative": int(np.count_nonzero(a < 0)),
                              "correct_ll_mean": float(a.mean()) if len(a) else None,
                              "correct_ll_min": float(a.min()) if len(a) else None,
                              "margin_mean": float(b.mean()) if len(b) else None,
                              "margin_positive": int(np.count_nonzero(b > 0))}
        rows.append({"requested_coefficient": requested, "actual_fp32_coefficient": coefficient,
                     "continuous_weight_l2": float(np.linalg.norm(costs) * coefficient),
                     "groups": grouped, "correct_ll_prediction": dc.tolist(), "margin_prediction": dm.tolist(),
                     "linear_logprob_ceiling_exceed_folds": np.flatnonzero(ll + dc > 0).tolist()})
    return {"uniform_full_donor_l2": float(np.linalg.norm(costs)), "rows": rows,
            "labels": [{"purpose": p, "fold": f} for p, f in labels],
            "radius_one_applied": False, "actual_bf16_rounding_measured": False, "model_calls": 0,
            "limitations": ["Local directional extrapolation, not finite response or coding performance.",
                            "Coefficient amplitude is explicit; no old experiment radius is treated as a user requirement.",
                            "Positive predicted logprob is physically impossible and flags local-model extrapolation failure."]}


def native_residual_once(base, frozen_values, gradient, lower):
    """One native-only residual solve; round original base plus accumulated delta once."""
    base, g, lower = (np.asarray(x, dtype=np.float64) for x in (base, gradient, lower))
    frozen = np.asarray(frozen_values)
    if (frozen.dtype != np.float32 or frozen.shape != base.shape or g.shape != (len(lower), len(base))
            or not np.isfinite(frozen).all() or np.any(frozen.view(np.uint32) & 0xFFFF)):
        raise ValueError("exact frozen native BF16 values and matching gradients required")
    previous_delta = frozen.astype(np.float64) - base
    before = g @ previous_delta
    correction, dual = joint.minimum_step(g, lower - before, np.ones(len(base)))
    cumulative = previous_delta + correction
    values, rounding = joint.bf16_realization(
        base, np.column_stack((cumulative, np.zeros_like(cumulative))), [1., 0.])
    values[cumulative == 0] = base[cumulative == 0].astype(np.float32)
    delta = values.astype(np.float64) - base
    prediction = g @ delta
    shortfall = np.maximum(lower - prediction, 0.)
    report = {"status": "native_local_proposal" if bool(np.all(shortfall <= 1e-10)) and np.count_nonzero(delta)
              else "native_post_rounding_rejected",
              "eligible_for_native_finite": bool(np.all(shortfall <= 1e-10)) and bool(np.count_nonzero(delta)),
              "solver_calls": 1, "dual": dual, "donor_coefficients_used": False,
              "before_prediction": before.tolist(), "post_rounding_prediction": prediction.tolist(),
              "continuous_prediction": (g @ cumulative).tolist(), "minimum_delta": lower.tolist(),
              "before_maximum_shortfall": float(np.maximum(lower - before, 0.).max()),
              "continuous_maximum_shortfall": float(np.maximum(lower - g @ cumulative, 0.).max()),
              "maximum_shortfall": float(shortfall.max()),
              "failed_constraint_indices": np.flatnonzero(shortfall > 1e-10).tolist(),
              "before_native_l2": float(np.linalg.norm(previous_delta)),
              "correction_continuous_l2": float(np.linalg.norm(correction)),
              "cumulative_continuous_l2": float(np.linalg.norm(cumulative)),
              "actual_native_l2": float(np.linalg.norm(delta)),
              "changed_elements": int(np.count_nonzero(delta)),
              "values_changed_vs_previous": int(np.count_nonzero(values.view(np.uint32) != frozen.view(np.uint32))),
              "prior_radius_one_exceeded": bool(np.linalg.norm(delta) > 1.),
              "rounding": rounding, "actual_model_response_measured": False,
              "limitations": ["One correction only; no clipping, native reweighting, target change or iterative repair.",
                              "Minimum-step convergence and postcheck tolerance are numerical, not exact optimality proofs.",
                              "The local linear gradient may fail for these realized BF16 edits; official WSP remains decisive."]}
    return values, report


def analyze(plans_path, native_margin_path, native_correct_path, frozen_report, output):
    content = plans_path.read_bytes()
    plan = json.loads(content)
    if plan.get("kind") != positive.HEADROOM_KIND:
        raise ValueError("new forced-empty-thought headroom plan required")
    margin, correct, labels = positive.paired_collections(Path(plan["margin_collection"]), Path(plan["correct_collection"]))
    positive.validate_plan(plan, margin, correct, labels)
    protected = [margin["path"], correct["path"], native_margin_path.resolve(), native_correct_path.resolve(),
                 plans_path.resolve(), frozen_report.parent.resolve(), Path(plan["source_checkpoint"]).resolve()]
    if output.exists() or output.is_symlink() or any(output.resolve().is_relative_to(p) or
                                                     p.is_relative_to(output.resolve()) for p in protected):
        raise ValueError("fresh output disjoint from inputs/source checkpoint required")
    output.mkdir(parents=True)
    started = time.monotonic()
    previous = signal.getsignal(signal.SIGALRM)
    def deadline(*unused):
        raise TimeoutError("fixed 120 second CPU analysis budget exhausted")
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(120)
    hashes = dict(plan["input_sha256"])
    hashes[str(plans_path.resolve())] = hashlib.sha256(content).hexdigest()
    old_code = [Path(proposal.__file__), Path(positive.__file__), Path(positive.legacy.__file__),
                Path(joint.__file__), Path(joint.common.__file__), Path(joint.finite_response.__file__)]
    old_code_hashes = {str(p.resolve()): proposal.sha(p) for p in old_code}
    code_hashes = old_code_hashes | {str(Path(__file__).resolve()): proposal.sha(__file__)}
    try:
        uniform = uniform_donor(correct["gradient"], margin["gradient"], margin["costs"], labels,
                                [r["mean_correct_logprob"] for r in correct["summary"]["folds"]])
        nm = proposal.native_collection(native_margin_path, "margin", margin, labels)
        nc = proposal.native_collection(native_correct_path, "correct-logprob", correct, labels)
        if nm["layout"] != nc["layout"] or not np.array_equal(nm["base"], nc["base"]):
            raise ValueError("native source inventory/value mismatch")
        if {e["tensor"] for e in nm["layout"]} & {e["tensor"] for e in margin["summary"]["coverage"]["entries"]}:
            raise ValueError("native/donor tensors overlap")
        hashes |= nm["hashes"] | nc["hashes"]
        gn = np.vstack((nc.pop("gradient"), nm.pop("gradient")))
        gd = np.vstack((correct["gradient"], margin["gradient"]))
        frozen, frozen_hashes = proposal.read_frozen_proposal(
            frozen_report, nm["base"], nm["layout"], gn, gd, hashes, old_code_hashes)
        hashes |= frozen_hashes
        method = next(m for m in plan["plans"] if m["id"] == "donor-positive-balanced")
        lower = np.array([r["minimum_delta"] for r in method["constraints"]])
        values, native = native_residual_once(nm["base"], frozen, gn, lower)
        native.update(constraints=method["constraints"], layout=nm["layout"])
        if any(proposal.sha(p) != digest for p, digest in hashes.items()) or any(
                proposal.sha(p) != digest for p, digest in code_hashes.items()):
            raise ValueError("input or implementation changed during analysis")
        with (output / "native-proposal.npz").open("xb") as stream:
            np.savez_compressed(stream, native_bf16_values=values)
        native["proposal_sha256"] = proposal.sha(output / "native-proposal.npz")
        report = {"schema_version": 1, "status": "completed", "seconds": time.monotonic() - started,
                  "uniform_donor": uniform, "native_residual": native,
                  "frame_protocol": positive.FRAME_PROTOCOL, "answer_batching": "serial",
                  "source_checkpoint": plan["source_checkpoint"],
                  "source_attestation_sha256": plan["source_attestation_sha256"],
                  "source_sha256": hashes, "implementation_sha256": code_hashes,
                  "model_calls": 0, "new_checkpoints": 0, "candidate_adopted": False}
        proposal.fresh_json(output / "report.json", report)
    except BaseException as error:
        proposal.fresh_json(output / "failure.json", {"status": "incomplete", "error_type": type(error).__name__,
                            "reason": str(error), "seconds": time.monotonic() - started})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plans", required=True, type=Path)
    p.add_argument("--native-margin", required=True, type=Path)
    p.add_argument("--native-correct", required=True, type=Path)
    p.add_argument("--native-proposal-report", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    report = analyze(args.plans, args.native_margin, args.native_correct, args.native_proposal_report, args.output)
    print(json.dumps({"status": report["status"], "seconds": report["seconds"],
                      "native_status": report["native_residual"]["status"],
                      "native_maximum_shortfall": report["native_residual"]["maximum_shortfall"],
                      "native_actual_l2": report["native_residual"]["actual_native_l2"],
                      "uniform_primary": [r["groups"]["primary"] for r in report["uniform_donor"]["rows"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
