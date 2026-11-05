#!/usr/bin/env python3
"""Make one local-gradient BF16 proposal; no finite/model/WSP measurement or adoption."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

import recipe_common as common
from kairyu_bench.wsp.weights import finite_response
from kairyu_bench.wsp.weights.finite_response import bf16_realization


def minimum_step(gradient, lower, cost, *, tolerance=2e-10, maximum_iterations=20000):
    """Convex diagonal-cost L2 relaxation via the nonnegative Gram dual."""
    g, b, c = (np.asarray(v, dtype=np.float64) for v in (gradient, lower, cost))
    if (g.ndim != 2 or b.shape != (g.shape[0],) or c.shape != (g.shape[1],)
            or not g.size or not all(np.isfinite(v).all() for v in (g, b, c)) or np.any(c <= 0)):
        raise ValueError("finite matching gradient/lower and strictly positive diagonal costs required")
    scaled = g / c
    norms = np.linalg.norm(scaled, axis=1)
    zero = norms == 0
    if np.any(zero & (b > 0)):
        raise ValueError("positive target has a zero local derivative")
    if zero.all():
        return np.zeros(g.shape[1]), {"iterations": 0, "dual_residual": 0.}
    normalized = scaled[~zero] / norms[~zero, None]
    target = b[~zero] / norms[~zero]
    gram, dual = normalized @ normalized.T, np.zeros(len(target))
    for iteration in range(maximum_iterations):
        for i in range(len(target)):
            dual[i] = max(0., dual[i] + (target[i] - gram[i] @ dual) / max(gram[i, i], 1e-15))
        residual = gram @ dual - target
        violation = float(np.max(np.abs(np.where(dual > 1e-12, residual, np.minimum(residual, 0.)))))
        if violation <= tolerance:
            break
    else:
        raise ValueError("local convex dual did not converge; not a proof of infeasibility")
    step = (normalized.T @ dual) / c
    if not np.isfinite(step).all():
        raise ValueError("nonfinite convex solution")
    return step, {"iterations": iteration + 1, "dual_residual": violation,
                  "continuous_max_scaled_shortfall": float(np.maximum((b - g @ step)[~zero] / norms[~zero], 0).max())}


def quantized_proposal(base, gradient, lower, cost, *, relative_radius, absolute_radius, rounds):
    """Every trial rounds original BF16 base + accumulated F64 proposal once."""
    base, g, lower, cost = (np.asarray(v, dtype=np.float64) for v in (base, gradient, lower, cost))
    if (not np.isfinite([relative_radius, absolute_radius]).all()
            or relative_radius < 0 or absolute_radius < 0 or relative_radius + absolute_radius == 0
            or type(rounds) is not int or not 1 <= rounds <= 32):
        raise ValueError("explicit nonnegative radius and 1..32 rounds required")
    # Also verifies that base is already exact finite BF16, including subnormals.
    bf16_realization(base, np.zeros((base.size, 2)), [1., 0.])
    radius = np.maximum(np.abs(base) * relative_radius, absolute_radius)
    proposed, history, best = np.zeros_like(base), [], None
    for index in range(rounds):
        step, dual = minimum_step(g, lower - g @ (bf16_realization(
            base, np.column_stack((proposed, np.zeros_like(proposed))), [1., 0.])[0].astype(np.float64) - base), cost)
        proposed = np.clip(proposed + step, -radius, radius)
        values, rounding = bf16_realization(base, np.column_stack((proposed, np.zeros_like(proposed))), [1., 0.])
        delta = values.astype(np.float64) - base
        prediction = g @ delta
        # Explicit units; no implicit zero-loss requirement or fixed old guard.
        shortfall = np.maximum(lower - prediction, 0)
        objective = float(np.linalg.norm(cost * delta))
        row = {"round": index + 1, "maximum_prediction_shortfall": float(shortfall.max()),
               "weighted_delta_l2": objective, "actual_delta_l2": float(np.linalg.norm(delta)),
               "changed_elements": rounding["changed_elements"], "dual": dual}
        history.append(row)
        key = (float(np.linalg.norm(shortfall)), objective)
        if best is None or key < best[0]:
            best = (key, values.copy(), prediction.copy(), dict(rounding), index + 1)
        if not np.any(shortfall > 1e-10):
            break
    _, values, prediction, rounding, chosen = best
    delta = values.astype(np.float64) - base
    return values, {"history": history, "chosen_round": chosen, "rounding": rounding,
                    "prediction": prediction.tolist(), "minimum_delta": lower.tolist(),
                    "predicted_constraints_satisfied": bool(np.all(prediction >= lower - 1e-10)),
                    "weighted_delta_l2": float(np.linalg.norm(cost * delta)),
                    "actual_delta_l2": float(np.linalg.norm(delta)),
                    "radius_scope": "continuous F64 proposal before F32/BF16 rounding; realized values may exceed the continuous box",
                    "model_response_measured": False, "adopted": False}


def read_collection(directory, plan):
    paths = [directory / n for n in ("summary.json", "coverage.json", "provenance.json")]
    hashes = {str(p.resolve()): common.sha(p) for p in paths}
    summary, coverage, provenance = (json.loads(p.read_text()) for p in paths)
    if summary.get("status") != "completed" or summary.get("candidate_selection") is not False:
        raise ValueError("completed collector output required")
    mode = summary["mode"]
    if mode != provenance["mode"] or summary.get("source_values_unchanged") is not True:
        raise ValueError("collector mode or source continuity mismatch")
    constraints = plan["constraints"]
    if not isinstance(constraints, list) or not 1 <= len(constraints) <= 128:
        raise ValueError("one through 128 explicit constraints required")
    records = {(r["purpose"], r["fold"]): r for r in summary["folds"]}
    if len(records) != len(summary["folds"]):
        raise ValueError("duplicate measured fold")
    layout, coordinate_data = [], None
    matrix = "sampled_matrices" in coverage
    if matrix:
        sample = directory / "sample-plan.npz"
        hashes[str(sample.resolve())] = common.sha(sample)
        with np.load(sample, allow_pickle=False) as data:
            base = np.asarray(data["source_values"], dtype=np.float64)
            coordinate_data = data["coordinates"].copy()
        cursor = 0
        for entry in coverage["entries"]:
            count = entry["count"]
            if entry["start"] != cursor or count < 1:
                raise ValueError("matrix layout is not contiguous")
            rc = coordinate_data[cursor:cursor + count]
            shape = entry["shape"]
            if (rc.shape != (count, 2) or not np.issubdtype(rc.dtype, np.integer)
                    or np.any(rc < 0) or np.any(rc >= shape) or len(set(map(tuple, rc))) != count):
                raise ValueError("invalid sampled coordinates")
            layout.append({"tensor": entry["tensor"], "count": count, "shape": shape})
            cursor += count
        if base.shape != (cursor,) or coordinate_data.shape != (cursor, 2):
            raise ValueError("sample plan length mismatch")
    else:
        sample = directory / "source-native-values.npz"
        names = sorted(t["name"] for t in coverage["tensors"] if t["kind"] == "native")
        hashes[str(sample.resolve())] = common.sha(sample)
        with np.load(sample, allow_pickle=False) as data:
            if set(data.files) != set(names):
                raise ValueError("native source inventory mismatch")
            arrays = [np.asarray(data[n], dtype=np.float64).reshape(-1) for n in names]
        layout = [{"tensor": n, "count": len(a)} for n, a in zip(names, arrays, strict=True)]
        base = np.concatenate(arrays)
    paths.append(sample)
    if len({r["tensor"] for r in layout}) != len(layout):
        raise ValueError("duplicate source tensor")
    gradients, bounds, labels, seen = [], [], [], set()
    for constraint in constraints:
        key = (constraint["purpose"], constraint["fold"])
        target = constraint["minimum_delta"]
        if key in seen or key not in records or isinstance(target, bool) or not np.isfinite(target):
            raise ValueError("unique measured fold and explicit finite minimum_delta required")
        seen.add(key)
        record = records[key]
        filename = record["file"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("gradient must be a direct collector artifact")
        file = directory / filename
        hashes[str(file.resolve())] = common.sha(file)
        if matrix:
            row = np.load(file, allow_pickle=False)
        else:
            with np.load(file, allow_pickle=False) as data:
                row = np.concatenate([np.asarray(data[e["tensor"]]).reshape(-1) for e in layout])
        if row.shape != base.shape or not np.isfinite(row).all():
            raise ValueError("gradient shape/nonfinite mismatch")
        gradients.append(row.astype(np.float64))
        bounds.append(float(target))
        labels.append({"purpose": key[0], "fold": key[1], "minimum_delta": float(target)})
        paths.append(file)
    if any(common.sha(Path(p)) != digest for p, digest in hashes.items()):
        raise ValueError("collection changed while reading")
    return base, np.stack(gradients), np.asarray(bounds), layout, coordinate_data, labels, hashes, mode, provenance



def editor_compatibility(edits, *, allow_batched_matrix=False):
    counts = {e["tensor"]: len(e["coordinates"]) for e in edits if "coordinates" in e}
    total = sum(counts.values())
    requires_batch = total > 64 or max(counts.values(), default=0) > 64
    within_limit = total <= 65536 and max(counts.values(), default=0) <= 512
    allowed = within_limit and (not requires_batch or allow_batched_matrix)
    return {"matrix_coordinates_per_tensor": counts, "total_matrix_coordinates": total,
            "requires_matrix_batch": requires_batch, "within_batched_editor_limit": within_limit,
            "batch_opt_in": allow_batched_matrix, "spec_emission_allowed": allowed}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-spec", action="store_true", help="write the diagnostic proposal to a fresh directory")
    parser.add_argument("--allow-batched-matrix", action="store_true",
                        help="explicitly permit editor-compatible <=512/tensor, <=65536 total matrix coordinates")
    args = parser.parse_args(argv)
    plan_sha = common.sha(args.plan)
    plan = json.loads(args.plan.read_text())
    identifier = plan["id"]
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", identifier):
        raise ValueError("safe candidate id required")
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("output must be new")
    for source in (args.collection.resolve(), args.plan.resolve(), Path(__file__).resolve()):
        out = args.output.resolve()
        if out == source or out in source.parents or source in out.parents:
            raise ValueError("output must be disjoint from inputs")
    base, g, lower, layout, coordinates, labels, hashes, mode, source = read_collection(args.collection, plan)
    for path in (args.plan, Path(__file__), Path(common.__file__), Path(finite_response.__file__)):
        hashes[str(path.resolve())] = common.sha(path)
    if hashes[str(args.plan.resolve())] != plan_sha:
        raise ValueError("plan changed while reading")
    metric = plan["cost_metric"]
    if metric["type"] == "euclidean":
        cost = np.ones_like(base)
    elif metric["type"] == "relative" and np.isfinite(metric["floor"]) and metric["floor"] > 0:
        cost = 1 / np.maximum(np.abs(base), metric["floor"])
    else:
        raise ValueError("explicit euclidean or positive-floor relative diagonal cost required")
    values, report = quantized_proposal(base, g, lower, cost, **plan["rounding"])
    edits, cursor = [], 0
    for entry in layout:
        count = entry["count"]
        indices = np.flatnonzero(values[cursor:cursor + count].astype(np.float64) != base[cursor:cursor + count])
        if indices.size:
            edit = {"tensor": entry["tensor"], "set_values": values[cursor + indices].tolist()}
            edit["indices" if coordinates is None else "coordinates"] = (
                indices.tolist() if coordinates is None else coordinates[cursor + indices].tolist())
            edits.append(edit)
        cursor += count
    compatibility = editor_compatibility(edits, allow_batched_matrix=args.allow_batched_matrix)
    report["editor_compatibility"] = compatibility
    report["spec_emitted"] = bool(args.write_spec and edits and compatibility["spec_emission_allowed"])
    report.update(schema_version=1, id=identifier, mode=mode, constraints=labels, cost_metric=metric,
                  source_sha256=hashes, source_checkpoint=source["checkpoint"],
                  source_attestation_receipt_sha256=source.get("attestation_receipt_sha256"),
                  represented_source_l2=float(np.linalg.norm(base)),
                  represented_source_scope="selected native values" if coordinates is None else "sampled matrix coordinates only",
                  limits=["Local G predictions, not measured finite candidate responses or WSP improvements.",
                          "Discrete BF16 global optimum is not established; no automatic adoption.",
                          "Virtual gain coordinates are not materialized by this solver.",
                          "Output may require the editor's explicit batched-matrix limits."])
    if "reference_norm" in plan:
        reference = plan["reference_norm"]
        if not np.isfinite(reference["l2"]) or reference["l2"] <= 0 or not isinstance(reference["scope"], str) or not reference["scope"]:
            raise ValueError("reference_norm requires positive L2 and explicit scope")
        report["caller_declared_reference_norm"] = reference
        report["relative_delta_l2_to_declared_reference"] = report["actual_delta_l2"] / reference["l2"]
    if any(common.sha(Path(p)) != digest for p, digest in hashes.items()):
        raise ValueError("input changed during solution")
    if args.write_spec:
        args.output.mkdir(parents=True)
        common.save(args.output / "report.json", report)
        if report["spec_emitted"]:
            metadata = ({"matrix_batch": True, "matrix_sample_plan_sha256": hashes[str((args.collection / "sample-plan.npz").resolve())]}
                        if compatibility["requires_matrix_batch"] else {})
            common.save(args.output / (identifier + ".json"), {
                "id": identifier, "hypothesis": "Explicit local-purpose-gradient joint BF16 proposal; finite response unmeasured.",
                "edits": edits, **metadata})
    print(json.dumps({"id": identifier, "changed_elements": report["rounding"]["changed_elements"],
                      "actual_delta_l2": report["actual_delta_l2"],
                      "predicted_constraints_satisfied": report["predicted_constraints_satisfied"],
                      "model_response_measured": False, "written": args.write_spec,
                      "spec_emitted": report["spec_emitted"], "editor_compatibility": compatibility}))
    return 2 if args.write_spec and not compatibility["spec_emission_allowed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
