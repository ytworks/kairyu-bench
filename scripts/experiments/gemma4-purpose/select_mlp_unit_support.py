#!/usr/bin/env python3
"""Select bounded full MLP units from observed frozen-weight UP-output gain sensitivities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import propose_native_objectives as adapter


def select(scores, units, groups, maximum_per_layer):
    """Declared group order, distinct quota fills, deterministic layer/unit ties."""
    selected, counts, chosen = set(), {}, []
    for group in groups:
        ranking = sorted(range(len(units)), key=lambda i: (-scores[group["name"]][i], *units[i]))
        added = 0
        for index in ranking:
            layer, unit = units[index]
            if index in selected or counts.get(layer, 0) >= maximum_per_layer:
                continue
            if scores[group["name"]][index] <= 0:
                continue
            selected.add(index)
            counts[layer] = counts.get(layer, 0) + 1
            chosen.append(dict(index=index, layer=layer, unit=unit, quota_group=group["name"]))
            added += 1
            if added == group["quota"]:
                break
        if added != group["quota"]:
            raise ValueError("not enough distinct positive-salience units under declared layer bound")
    return chosen


def build(args):
    pins = {}
    plan = adapter.document(args.plan, pins, args.plan_sha256)
    for key in ("generated_baseline", "method_review"):
        if key in plan:
            evidence = plan[key]
            adapter.capture(Path(evidence["path"]), pins, evidence["sha256"])
    ref = plan["reference_plan"]
    # A is completed historical evidence, not a request to rerun its obsolete
    # source paths. Consume only its pinned immutable metadata/sample artifacts;
    # D's sensitivities below come exclusively from the new unit collections.
    reference = adapter.document(Path(ref["path"]), pins, ref["sha256"])
    checkpoint = adapter.Checkpoint(reference["checkpoint"]["path"])
    source = reference["checkpoint"]
    att = source["attestation"]
    receipt = adapter.document(Path(att["path"]), pins, att["sha256"])
    identity = receipt.get("checkpoint", receipt)
    current = checkpoint.identity(hash_files=False)
    if (identity.get("identity_level") != "content_sha256" or identity.get("hash_files") is not True
            or identity.get("path") != str(checkpoint.path) or identity.get("sha256") != source["tree_sha256"]
            or len(identity.get("files", [])) != len(current["files"])
            or any(any(old[k] != now[k] for k in ("path", "size", "stat"))
                   for old, now in zip(identity["files"], current["files"], strict=True))):
        raise ValueError("reference full checkpoint attestation differs")
    if reference.get("parent_state") != plan["parent_state"]:
        raise ValueError("selection and reference current parent differ")
    adapter.capture(Path(plan["parent_state"]["path"]), pins, plan["parent_state"]["sha256"])
    if adapter.observed_blocks(reference) != ["matrix"]:
        raise ValueError("explicit complete matrix-only reference required")
    case_ref = reference["cases"]
    raw = adapter.capture(Path(case_ref["path"]), pins, case_ref["sha256"])
    cases = adapter.joint.common.read_cases(Path(case_ref["path"]), content=raw)
    labels = [(purpose, fold) for purpose, folds in cases.items() for fold in range(len(folds))]
    matrices = {}
    for objective in adapter.OBJECTIVES:
        directory = Path(reference["collections"]["matrix"][objective]).resolve()
        documents = {}
        for key, basename in (("summary", "summary.json"), ("provenance", "provenance.json"),
                              ("coverage", "coverage.json"), ("frames", "frame-evidence.json")):
            path = directory / basename
            documents[key] = adapter.document(path, pins, reference["completed_collection_sha256"][str(path)])
        if (documents["summary"].get("status") != "completed"
                or documents["summary"].get("source_values_unchanged") is not True
                or documents["provenance"]["checkpoint"] != current
                or [(r["purpose"],r["fold"]) for r in documents["summary"]["folds"]] != labels):
            raise ValueError("historical matrix metadata/source cohort differs")
        matrices["matrix:"+objective] = dict(path=directory, **documents)
    policy = plan["selection"]
    groups = policy["groups"]
    if (policy["maximum_units"] != 32 or policy["maximum_per_layer"] != 2
            or policy["score"] != "row_normalized_gain_per_up_row_l2_rms"
            or not groups or len({g["name"] for g in groups}) != len(groups)
            or sum(g["quota"] for g in groups) != 32
            or any(type(g["quota"]) is not int or g["quota"] < 1 for g in groups)):
        raise ValueError("explicit 32-unit/two-per-layer quota policy required")
    purposes = {p for p, _ in labels}
    for group in groups:
        if (not group["purposes"] or len(set(group["purposes"])) != len(group["purposes"])
                or not set(group["purposes"]) <= purposes):
            raise ValueError("selection purpose names differ from observed cohort")
    expected = {name: tensor for name, tensor in checkpoint.tensors.items()
                if name.endswith(".mlp.up_proj.weight")}
    names = sorted(expected)
    units = [(int(adapter.matrix_recipe.PATTERN.fullmatch(name)[1]), unit)
             for name in names for unit in range(expected[name].shape[0])]
    collections, loaded = {}, []
    for objective in adapter.OBJECTIVES:
        directory = Path(plan["unit_collections"][objective]).resolve()
        collections[objective] = dict(path=directory)
        summary, provenance, coverage, frames = [adapter.document(directory / filename, pins)
            for filename in ("summary.json", "provenance.json", "coverage.json", "frame-evidence.json")]
        baseline = matrices["matrix:" + objective]
        entries = coverage["entries"]
        if (summary.get("status") != "completed" or summary.get("source_values_unchanged") is not True
                or summary.get("candidate_selection") is not False
                or summary.get("objective_kind") != objective or provenance.get("objective_kind") != objective
                or coverage.get("units_method") != "up_output_gain"
                or {e["tensor"] for e in entries} != set(names) or len(entries) != len(names)
                or coverage.get("observed_units") != len(units)
                or any(e["shape"] != list(expected[e["tensor"]].shape)
                       or e["count"] != expected[e["tensor"]].shape[0] for e in entries)):
            raise ValueError("complete full-unit gain collection required")
        if (frames != baseline["frames"] or provenance["checkpoint"] != baseline["provenance"]["checkpoint"]
                or provenance["attestation_receipt_sha256"] != baseline["provenance"]["attestation_receipt_sha256"]
                or provenance.get("observation_policy") != baseline["provenance"].get("observation_policy")
                or summary["runtime"].get("observation_policy") != provenance.get("observation_policy")
                or summary.get("answer_batching") != "serial"):
            raise ValueError("unit and matrix source/frame/numerical policy differ")
        if [(r["purpose"], r["fold"]) for r in summary["folds"]] != labels:
            raise ValueError("full ordered unit objective cohort differs")
        for filename, digest in provenance["input_sha256"].items():
            adapter.capture(Path(filename), pins, digest)
        norm_path = directory / "source-up-row-norms.npz"
        adapter.capture(norm_path, pins)
        with np.load(norm_path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(names) or any(arrays[n].shape != (expected[n].shape[0],) for n in names):
                raise ValueError("UP row norm source layout differs")
            norm = np.concatenate([np.asarray(arrays[n], dtype=np.float64) for n in names])
        if not np.isfinite(norm).all() or np.any(norm < 0):
            raise ValueError("finite nonnegative UP row norms required")
        loaded.append((summary, provenance, frames, norm))
    if not np.array_equal(loaded[0][3], loaded[1][3]):
        raise ValueError("unit objective source row norms differ")
    destination = adapter.check_output(args.output, checkpoint, args.plan,
                                      {**matrices, **collections}, pins)
    if not args.run:
        return dict(status="metadata_preview", gradient_reads=0, model_calls=0)
    scores = {g["name"]: np.zeros(len(units)) for g in groups}
    signed = {g["name"]: np.zeros(len(units)) for g in groups}
    norm = loaded[0][3]
    tolerance = plan["baseline_tolerance"]
    if type(tolerance) not in (int, float) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("declared finite nonnegative baseline tolerance required")
    maximum_baseline_error = 0.
    for objective, data in zip(adapter.OBJECTIVES, loaded, strict=True):
        summary, _, _, _ = data
        baseline = matrices["matrix:" + objective]["summary"]["folds"]
        field, value = ("correct_logprobs", "correct_logprob") if objective == "correct-logprob" else ("margins", "margin")
        for row, original in zip(summary["folds"], baseline, strict=True):
            if row["examples"] != original["examples"] or [v["id"] for v in row[field]] != [v["id"] for v in original[field]]:
                raise ValueError("unit objective case cohort differs")
            errors = [abs(v[value] - b[value]) for v, b in zip(row[field], original[field], strict=True)]
            if not np.isfinite(errors).all() or max(errors, default=0.) > tolerance:
                raise ValueError("unit gain forward changed the reference objective")
            maximum_baseline_error = max(maximum_baseline_error, max(errors, default=0.))
            if Path(row["file"]).name != row["file"]:
                raise ValueError("unit gradients must be direct artifacts")
            path = collections[objective]["path"] / row["file"]
            adapter.capture(path, pins)
            with np.load(path, allow_pickle=False) as arrays:
                if set(arrays.files) != set(names) or any(arrays[n].shape != (expected[n].shape[0],) for n in names):
                    raise ValueError("unit gradient layout differs")
                gradient = np.concatenate([np.asarray(arrays[n], dtype=np.float64) for n in names])
            if not np.isfinite(gradient).all() or np.any((norm == 0) & (gradient != 0)):
                raise ValueError("nonfinite or nonzero zero-cost gain derivative")
            scaled = np.divide(gradient, norm, out=np.zeros_like(gradient), where=norm > 0)
            length = np.linalg.norm(scaled)
            if length:
                scaled /= length
            for group in groups:
                if row["purpose"] in group["purposes"]:
                    count = 2 * sum(p in group["purposes"] for p, _ in labels)
                    scores[group["name"]] += scaled * scaled / count
                    signed[group["name"]] += scaled / count
    scores = {name: np.sqrt(value) for name, value in scores.items()}
    chosen = select(scores, units, groups, policy["maximum_per_layer"])
    selected, geometry = {}, []
    for row in chosen:
        layer, unit = row["layer"], row["unit"]
        selected.setdefault(layer, []).append(unit)
        prefix = f"model.language_model.layers.{layer}.mlp."
        gate, up, down = [checkpoint.tensors[prefix + family + "_proj.weight"] for family in ("gate", "up", "down")]
        g = checkpoint.read(gate.name, unit * gate.shape[1], gate.shape[1]).astype(np.float64)
        u = checkpoint.read(up.name, unit * up.shape[1], up.shape[1]).astype(np.float64)
        d = checkpoint.read_indices(down.name, [r * down.shape[1] + unit for r in range(down.shape[0])]).astype(np.float64)
        gn, un, dn = (float(np.linalg.norm(x)) for x in (g, u, d))
        if not np.isclose(un, norm[row["index"]], rtol=1e-10, atol=1e-12):
            raise ValueError("selected UP row norm differs from exact source weights")
        geometry.append(dict(layer=layer, unit=unit, quota_group=row["quota_group"],
            gate_row_l2=gn, up_row_l2=un, down_column_l2=dn,
            gate_up_cosine=float(g @ u / (gn * un)) if gn * un else None,
            rms_salience={name:float(v[row["index"]]) for name,v in scores.items()},
            signed_mean={name:float(v[row["index"]]) for name,v in signed.items()}))
    directory = matrices["matrix:correct-logprob"]["path"]
    attention = {}
    for key, filename in (("coverage", "coverage.json"), ("sample_plan", "sample-plan.npz")):
        path = directory / filename
        digest = reference["completed_collection_sha256"][str(path)]
        adapter.capture(path, pins, digest)
        attention[key] = dict(path=str(path), sha256=digest)
    support = dict(schema_version=1, kind="mlp_full_unit_support_v1",
        layer_prefix="model.language_model.layers.", reference_attention=attention,
        source_checkpoint=loaded[0][1]["checkpoint"], parent_state=plan["parent_state"],
        attestation_receipt_sha256=loaded[0][1]["attestation_receipt_sha256"],
        units=[dict(layer=layer, indices=sorted(values)) for layer, values in sorted(selected.items())],
        reference_matrix_plan=ref, selection_plan=dict(path=str(args.plan.resolve()),sha256=args.plan_sha256),
        selected_unit_count=len(chosen), native_parameters="frozen_not_observed")
    adapter.capture(Path(__file__), pins)
    adapter.stable(checkpoint, pins)
    destination.mkdir(parents=True)
    for filename, value in (("support.json",support), ("selection-report.json",dict(status="selected_support_only",
            selection=policy, geometry=geometry, source_sha256={p:r[0] for p,r in pins.items()},
            maximum_baseline_error=maximum_baseline_error, model_calls=0, candidate_solves=0,
            full_row_derivatives_measured=False, salience_is_not_circuit_attribution=True))):
        with (destination/filename).open("x") as stream:
            json.dump(value,stream,indent=2,allow_nan=False)
            stream.write("\n")
    return dict(status="selected_support_only", units=len(chosen), layers=len(selected), model_calls=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    print(json.dumps(build(parser.parse_args())))
