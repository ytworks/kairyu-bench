#!/usr/bin/env python3
"""CPU-only, at most three explicit local-G cost comparisons from saved chat derivatives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import numpy as np

import joint
import recipe_common as common


def role_name(name):
    return re.sub(r"\.layers\.\d+\.", ".layers.*.", name)


def diagonal_cost(base, names, offsets, metric):
    if metric["type"] == "euclidean":
        return np.ones_like(base), {}
    if metric["type"] != "role_relative":
        raise ValueError("cost type must be euclidean or role_relative")
    ratio, absolute = metric["floor_ratio"], metric["absolute_floor"]
    if not np.isfinite([ratio, absolute]).all() or not 0 < ratio <= 1 or absolute <= 0:
        raise ValueError("explicit positive role floor required")
    roles = {}
    for name, start, stop in zip(names, offsets, [*offsets[1:], len(base)], strict=True):
        roles.setdefault(role_name(name), []).append((start, stop))
    cost, details = np.empty_like(base), {}
    for role, slices in roles.items():
        values = np.concatenate([np.abs(base[start:stop]) for start, stop in slices])
        median = float(np.median(values))
        floor = max(absolute, ratio * median)
        details[role] = {"coordinates": len(values), "median_absolute_weight": median, "floor": floor}
        for start, stop in slices:
            cost[start:stop] = 1 / np.maximum(np.abs(base[start:stop]), floor)
    return cost, details


def load_inputs(args):
    import torch

    hashes = {}
    def take(path):
        hashes[str(path.resolve())] = common.sha(path)
        return path
    directory = args.legacy_chat
    coverage = json.loads(take(directory / "scan-coverage.json").read_text())
    observation = json.loads(take(directory / "observations.json").read_text())
    method = "official-thinking-on-public-answer-v1"
    if coverage.get("method_id") != method or observation.get("method_id") != method or coverage.get("raw_gradients_reused") is not False:
        raise ValueError("only the explicit saved chat-native derivative family is accepted")
    cases_sha = common.sha(take(directory / "cases.json"))
    frame_sha = common.sha(take(directory / "frame-evidence.json"))
    basis_frame = take(args.basis.parent / "frame-evidence.json")
    if json.loads(basis_frame.read_text()) != json.loads((directory / "frame-evidence.json").read_text()):
        raise ValueError("basis and local G chat frames differ")
    provenance = json.loads(take(args.basis_provenance).read_text())
    if (observation["cases_sha256"] != cases_sha
            or provenance["source_sha256"]["cases.json"] != cases_sha
            or provenance["expected_base_tree_sha256"] != observation["base_tree_sha256"]):
        raise ValueError("chat cases or source checkpoint identity mismatch")
    with np.load(take(args.basis), allow_pickle=False) as data:
        base = data["base"].astype(np.float64)
        names, offsets = data["tensor_names"].tolist(), data["tensor_offsets"].tolist()
    expected = sorted(n for n, kind in coverage["kinds"].items() if kind == "native")
    if names != expected or len(offsets) != len(names) or not offsets or offsets[0] != 0:
        raise ValueError("basis-native inventory mismatch")
    gradients, groups, fold_counts = [], [], {}
    for name in names:
        if not name.startswith("model.language_model."):
            raise ValueError("text-only native basis required")
    for group, folds in json.loads((directory / "cases.json").read_text()).items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", group):
            raise ValueError("unsafe purpose name")
        fold_counts[group] = len(folds)
        for fold in range(len(folds)):
            file = take(directory / f"{group}-{fold}.pt")
            data = torch.load(file, map_location="cpu", weights_only=True)
            if set(data) != set(names):
                raise ValueError("saved derivative tensor inventory mismatch")
            pieces = []
            for name, start, stop in zip(names, offsets, [*offsets[1:], len(base)], strict=True):
                value = data[name].detach().float().cpu().numpy().reshape(-1)
                if value.size != stop - start or not np.isfinite(value).all():
                    raise ValueError("saved derivative shape/nonfinite mismatch")
                pieces.append(value)
            gradients.append(np.concatenate(pieces))
            groups.append((group, fold))
    gradient = np.stack(gradients).astype(np.float64)
    if not all(np.isfinite(v).all() for v in (base, gradient)):
        raise ValueError("nonfinite source")
    return base, gradient, names, offsets, groups, hashes, {
        "method": method, "cases_sha256": cases_sha, "frame_sha256": frame_sha,
        "source_tree_sha256": observation["base_tree_sha256"],
        "fold_counts": fold_counts, "raw_G_mixed": False,
        "scope": "saved CPU BF16 official chat public-answer mean-logp derivatives; no new forward",
        "basis_trust_boundary": "saved basis values and matching source identity/inventory are trusted artifacts; original checkpoint payload bytes are not reread or reattested here"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-chat", type=Path, required=True)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--basis-provenance", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    plan_sha = common.sha(args.plan)
    plan = json.loads(args.plan.read_text())
    profiles = plan["profiles"]
    if not isinstance(profiles, list) or not 1 <= len(profiles) <= 3:
        raise ValueError("at most three prespecified cost profiles")
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("fresh output required")
    inputs = [args.legacy_chat, args.basis, args.basis_provenance, args.plan, Path(__file__), Path(joint.__file__), Path(common.__file__), Path(joint.finite_response.__file__)]
    output = args.output.resolve()
    if any(output == p.resolve() or output in p.resolve().parents or p.resolve() in output.parents for p in inputs):
        raise ValueError("output must be disjoint from inputs")
    for profile in profiles[1:]:
        if any(profile[field] != profiles[0][field] for field in ("group_mean_lower", "fold_lower")):
            raise ValueError("cost comparison requires identical explicit constraints across profiles")
    ids = [p["id"] for p in profiles]
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", i) for i in ids):
        raise ValueError("safe unique profile identifiers required")
    filenames = ["input-evidence.json", "progress.json", "summary.json", "failure.json"]
    filenames.extend(filename for identifier in ids
                     for filename in (identifier + ".json", identifier + "-report.json"))
    if len(set(filenames)) != len(filenames):
        raise ValueError("profile IDs collide with generated output filenames")
    diagnostic_floor = plan.get("diagnostic_relative_floor")
    if (type(diagnostic_floor) not in (int, float)
            or not np.isfinite(diagnostic_floor) or diagnostic_floor <= 0):
        raise ValueError("diagnostic_relative_floor must be finite and positive")
    if not args.run:
        print(json.dumps({"profiles": ids, "mode": "CPU saved derivatives only", "new_forwards": 0, "writes": 0}))
        return 0
    started = time.monotonic()
    input_hashes = {str(path.resolve()): common.sha(path) for path in inputs[1:]}
    base, gradient, names, offsets, labels, hashes, evidence = load_inputs(args)
    if any(name in hashes and hashes[name] != digest for name, digest in input_hashes.items()):
        raise ValueError("input changed while loading")
    hashes.update(input_hashes)
    if hashes[str(args.plan.resolve())] != plan_sha:
        raise ValueError("plan changed during input loading")
    args.output.mkdir(parents=True)
    common.save(args.output / "input-evidence.json", {**evidence, "sha256": hashes, "plan": plan})
    results = []
    try:
        for profile in profiles:
            rows, lower, constraint_labels = [], [], []
            groups = list(evidence["fold_counts"])
            if set(profile["group_mean_lower"]) != set(groups) or set(profile["fold_lower"]) != set(groups):
                raise ValueError("every measured purpose needs explicit mean and fold lower bounds")
            for group in groups:
                indices = [i for i, (purpose, _) in enumerate(labels) if purpose == group]
                rows.append(gradient[indices].mean(0))
                lower.append(profile["group_mean_lower"][group])
                constraint_labels.append(f"{group}:mean")
                for index in indices:
                    rows.append(gradient[index])
                    lower.append(profile["fold_lower"][group])
                    constraint_labels.append(f"{group}:fold{labels[index][1]}")
            g = np.stack(rows)
            cost, role_floors = diagonal_cost(base, names, offsets, profile["cost_metric"])
            values, report = joint.quantized_proposal(base, g, lower, cost, **plan["rounding"])
            delta = values.astype(np.float64) - base
            per_role, edits = {}, []
            for name, first, last in zip(names, offsets, [*offsets[1:], len(base)], strict=True):
                part = delta[first:last]
                ix = np.flatnonzero(part)
                role = per_role.setdefault(role_name(name), {"changed": 0, "delta_l2_squared": 0., "relative_cost_squared": 0.})
                role["changed"] += len(ix)
                role["delta_l2_squared"] += float(part @ part)
                role["relative_cost_squared"] += float(np.sum((part / np.maximum(np.abs(base[first:last]), plan["diagnostic_relative_floor"])) ** 2))
                if ix.size:
                    edits.append({"tensor": name, "indices": ix.tolist(), "set_values": values[first + ix].tolist()})
            sensitivity = []
            for ratio in plan.get("fixed_delta_floor_ratios", []):
                metric = {"type": "role_relative", "floor_ratio": ratio,
                          "absolute_floor": plan["diagnostic_relative_floor"]}
                alternative_cost, _ = diagonal_cost(base, names, offsets, metric)
                sensitivity.append({"floor_ratio": ratio, "fixed_delta_weighted_l2": float(np.linalg.norm(alternative_cost * delta)),
                                    "reoptimized": False})
            report["fixed_delta_cost_sensitivity"] = sensitivity
            if "reference_norm" in plan:
                reference = plan["reference_norm"]
                if not np.isfinite(reference["l2"]) or reference["l2"] <= 0:
                    raise ValueError("invalid declared reference norm")
                report["declared_reference_norm"] = reference
                report["relative_delta_l2_to_declared_reference"] = report["actual_delta_l2"] / reference["l2"]
            report.update(id=profile["id"], cost_metric=profile["cost_metric"], role_floors=role_floors,
                          constraint_labels=constraint_labels, fold_labels=labels,
                          fold_predictions=(gradient @ delta).tolist(), per_role=per_role,
                          changed_tensors=len(edits), changed_elements=int(np.count_nonzero(delta)),
                          spec_emitted=bool(edits), new_forwards=0, new_checkpoints=0)
            common.save(args.output / (profile["id"] + "-report.json"), report)
            if edits:
                common.save(args.output / (profile["id"] + ".json"),
                            {"id": profile["id"], "hypothesis": "Local chat-gradient cost comparison only; actual finite response unmeasured.", "edits": edits})
            results.append({k: report[k] for k in ("id", "actual_delta_l2", "weighted_delta_l2", "changed_elements", "changed_tensors", "predicted_constraints_satisfied", "rounding")})
            common.save(args.output / "progress.json", {"profiles": results})
        if any(common.sha(Path(p)) != digest for p, digest in hashes.items()):
            raise ValueError("input changed during CPU comparison")
        common.save(args.output / "summary.json", {"status": "completed", "profiles": results, "source": evidence,
            "sha256": hashes, "seconds": time.monotonic() - started, "new_forwards": 0, "new_checkpoints": 0,
            "limitations": ["Local CPU-BF16 chat G only; no actual interior response measured.",
                           "Predicted margin units are not accuracy percentage points or WSP scores.",
                           "Role floor is a declared regularizer, not Fisher or measured activation protection."]})
    except BaseException as error:
        common.save(args.output / "failure.json", {"status": "incomplete", "error_type": type(error).__name__})
        raise
    print(json.dumps({"profiles": results, "new_forwards": 0, "new_checkpoints": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
