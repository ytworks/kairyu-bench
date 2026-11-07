#!/usr/bin/env python3
"""Prepare native-frozen science/code matrix sets from current-parent observations."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import propose_native_objectives as adapter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def method_metric(method):
    """The explicit metric sets both selection geometry and proposal cost."""
    default = dict(type="relative", floor=method["relative_floor"],
                   block_scales=dict(native=1., matrix=1.))
    metric = method.get("cost_metric", default)
    if (metric != default and metric != dict(type="euclidean",
                                            block_scales=dict(native=1., matrix=1.))):
        raise ValueError("unsupported declared matrix cost")
    return metric


def dual_scale(base, method):
    metric = method_metric(method)
    return (np.ones_like(base) if metric["type"] == "euclidean"
            else np.maximum(np.abs(base), metric["floor"]))


def agreement(plan, base, gradients, layout, labels, method):
    """Layer groups, using cost-scaled fold gradients; no circuit attribution."""
    active, chosen = adapter.matrix_column_indices(plan, layout)
    scale = dual_scale(base[active], method)
    folds = sorted({fold for _, fold in labels})
    positions = {pair: index for index, pair in enumerate(labels)}
    vectors = []
    for objective in adapter.OBJECTIVES:
        g = gradients[objective][:, active] * scale
        norms = np.linalg.norm(g, axis=1)
        g = np.divide(g, norms[:, None], out=np.zeros_like(g), where=norms[:, None] > 0)
        for fold in folds:
            sides = []
            for key in ("science_purposes", "code_purposes"):
                indices = [positions[(purpose, fold)] for purpose in method[key]]
                sides.append(np.mean(g[indices], axis=0))
            vectors.append((objective, fold, *sides))
    groups = {}
    cursor = 0
    for entry in chosen:
        layer = int(adapter.matrix_recipe.PATTERN.fullmatch(entry["tensor"])[1])
        group = groups.setdefault(layer, {"tensors": [], "indices": []})
        group["tensors"].append(entry["tensor"])
        group["indices"].extend(range(cursor, cursor + entry["count"]))
        cursor += entry["count"]
    report, selected = [], []
    for layer, group in sorted(groups.items()):
        ix = np.asarray(group["indices"], dtype=np.int64)
        rows = []
        for objective, fold, science, code in vectors:
            left, right = science[ix], code[ix]
            dot = float(np.dot(left, right))
            norm = float(np.linalg.norm(left) * np.linalg.norm(right))
            rows.append(dict(objective=objective, fold=fold, dot=dot,
                             cosine=dot / norm if norm else None))
        keep = all(row["dot"] >= 0 for row in rows) and any(row["dot"] > 0 for row in rows)
        if keep:
            selected.extend(group["tensors"])
        report.append(dict(layer=layer, tensors=group["tensors"], selected=keep, agreement=rows))
    return selected, report


def build(args):
    checkpoint, template, collections, labels, pins = adapter.prepare(args.template)
    if template["schema_version"] != 2:
        raise ValueError("complete v2 native/matrix template required")
    policy = adapter.document(args.policy, pins, args.policy_sha)
    parent = template.get("parent_state")
    if not parent or policy["parent_state"] != parent:
        raise ValueError("policy must bind the current template parent")
    review = policy["review"]
    opinion = adapter.document(Path(review["path"]), pins, review["sha256"])
    if not (opinion["verified_requested_model"] and opinion["review_usable"]
            and opinion["assistant_models"] == ["claude-opus-5-5"] and opinion["tool_uses"] == 0):
        raise ValueError("actual method review required")
    adapter.capture(Path(__file__), pins)
    output = args.output.resolve()
    inputs = [checkpoint.path.resolve(), args.template.resolve(), args.policy.resolve(),
              Path(review["path"]).resolve(),
              *(row["path"].resolve() for row in collections.values())]
    if output.exists() or args.output.is_symlink() or any(
            output == path or path in output.parents or output in path.parents for path in inputs):
        raise ValueError("fresh nonoverlapping output required")
    expected = template.get("completed_collection_sha256")
    if not expected:
        raise ValueError("completed collection content pins required")
    constraints = [dict(purpose=purpose, fold=fold, minimum_delta=0.) for purpose, fold in labels]
    loaded = {}
    for objective in adapter.OBJECTIVES:
        collection = collections["matrix:" + objective]
        base, g, _, layout, coordinates, _, found, _, _ = adapter.joint.read_collection(
            collection["path"], dict(constraints=constraints))
        for path, digest in found.items():
            if expected.get(path) != digest:
                raise ValueError("completed matrix content pin differs")
            adapter.capture(Path(path), pins, digest)
        loaded[objective] = (base, g, layout, coordinates)
    first, second = (loaded[objective] for objective in adapter.OBJECTIVES)
    if (first[2] != second[2] or not np.array_equal(first[3], second[3])
            or not np.array_equal(first[0].astype(np.float32).view(np.uint32),
                                  second[0].astype(np.float32).view(np.uint32))):
        raise ValueError("matrix objective columns or source words differ")
    base, _, layout, _ = first
    gradients = {objective: loaded[objective][1] for objective in adapter.OBJECTIVES}
    ll = collections["matrix:correct-logprob"]["summary"]["folds"]
    headroom = np.asarray([-np.mean([x["correct_logprob"] for x in row["correct_logprobs"]]) for row in ll])
    if np.any(~np.isfinite(headroom)) or np.any(headroom < 0):
        raise ValueError("invalid correct-LL headroom")
    purposes = {purpose for purpose, _ in labels}
    ids = [method["id"] for method in policy["methods"]]
    if (not ids or len(ids) != len(set(ids))
            or any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", value) for value in ids)):
        raise ValueError("unique method IDs required")
    plans, reports = [], []
    for method in policy["methods"]:
        for key in ("science_purposes", "code_purposes", "additional_primary_purposes"):
            values = method[key]
            if (not isinstance(values, list) or len(set(values)) != len(values)
                    or any(purpose not in purposes for purpose in values)
                    or (key != "additional_primary_purposes" and not values)):
                raise ValueError("method purpose selection differs")
        if (method["selection"] not in ("all", "fold_agreement")
                or not np.isfinite(method["relative_floor"]) or method["relative_floor"] <= 0):
            raise ValueError("method selection or cost differs")
        mode = method.get("target_mode", "dual_norm")
        if mode not in ("dual_norm", "functional"):
            raise ValueError("unsupported target mode")
        method_metric(method)
        target_cap = method.get("ll_target_cap")
        if target_cap is not None and (
                mode != "functional" or type(target_cap) not in (int, float)
                or not np.isfinite(target_cap) or target_cap <= 0):
            raise ValueError("LL target cap requires a positive finite functional target")
        target_keys = (("ll_multiplier", "margin_multiplier", "ll_headroom_fraction")
                       if mode == "dual_norm" else ("ll_headroom_fraction", "margin_target"))
        for key in target_keys:
            if not np.isfinite(method[key]) or method[key] <= 0:
                raise ValueError("positive finite target scale required")
        if (not np.isfinite(method["retention_lower"]) or method["retention_lower"] > 0
                or method["ll_headroom_fraction"] > 1):
            raise ValueError("retention or headroom policy differs")
        plan = copy.deepcopy(template)
        plan["eligible_blocks"] = ["matrix"]
        plan.pop("native_tensors", None)
        plan.pop("matrix_tensors", None)
        plan["matrix_families"] = method["matrix_families"]
        selection = []
        if method["selection"] == "fold_agreement":
            names, selection = agreement(plan, base, gradients, layout, labels, method)
            if not names:
                reports.append(dict(id=method["id"], status="no_joint_agreement", groups=selection))
                continue
            plan["matrix_tensors"] = names
        active, selected = adapter.matrix_column_indices(plan, layout)
        scale = dual_scale(base[active], method)
        primary = set(method["science_purposes"] + method["code_purposes"]
                      + method["additional_primary_purposes"])
        rows, target_report = [], []
        for objective in adapter.OBJECTIVES:
            norms = np.linalg.norm(gradients[objective][:, active] * scale, axis=1)
            for index, (purpose, fold) in enumerate(labels):
                target = method["retention_lower"]
                if purpose in primary:
                    if mode == "functional":
                        target = (method["ll_headroom_fraction"] * headroom[index]
                                  if objective == "correct-logprob" else method["margin_target"])
                        if objective == "correct-logprob" and target_cap is not None:
                            target = min(target, target_cap)
                    else:
                        multiplier = method["ll_multiplier" if objective == "correct-logprob"
                                            else "margin_multiplier"]
                        target = multiplier * float(norms[index])
                        if objective == "correct-logprob":
                            target = min(target, method["ll_headroom_fraction"] * headroom[index])
                row = dict(objective=objective, purpose=purpose, fold=fold, minimum_delta=float(target))
                rows.append(row)
                target_report.append(dict(**row, dual_norm=float(norms[index]), primary=purpose in primary))
        plan["candidates"] = [dict(id=method["id"], constraints=rows,
                                   cost_metric=method_metric(method),
                                   rounding=method["rounding"])]
        plan["prospective_method"] = dict(path=str(args.policy.resolve()), sha256=args.policy_sha)
        plan["fixed_method_inputs"] = {
            **template.get("fixed_method_inputs", {}),
            str(args.template.resolve()): sha(args.template),
            str(args.policy.resolve()): args.policy_sha,
            str(Path(review["path"]).resolve()): review["sha256"],
            str(Path(__file__).resolve()): sha(__file__),
        }
        plans.append((method["id"], plan))
        reports.append(dict(id=method["id"], status="prepared_no_solve",
                            selected_tensors=[entry["tensor"] for entry in selected],
                            selected_coordinates=len(active), groups=selection, targets=target_report,
                            native_coordinates=0, method=method))
    adapter.stable(checkpoint, pins)
    output.mkdir()
    roster = []
    for identifier, plan in plans:
        path = output / (identifier + ".json")
        write(path, plan)
        adapter.prepare(path)
        roster.append(dict(id=identifier, plan=str(path), plan_sha256=sha(path)))
    write(output / "preparation.json", dict(status="prepared_no_solve", parent=parent,
          policy_sha256=args.policy_sha, reports=reports, plans=roster, model_calls=0, solver_calls=0,
          observations_are_proxies=True, gradient_agreement_is_not_causal_attribution=True))
    adapter.stable(checkpoint, pins)
    return dict(status="prepared_no_solve", plans=roster, skipped=len(ids)-len(roster))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(build(parser.parse_args()), allow_nan=False))
