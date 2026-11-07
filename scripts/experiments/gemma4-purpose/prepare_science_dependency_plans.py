#!/usr/bin/env python3
"""Bind a pinned science dependency policy to fresh parent observations; no solve."""

import argparse
import copy
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def stamp(path):
    s = Path(path).stat()
    return dict(
        device=s.st_dev,
        inode=s.st_ino,
        size=s.st_size,
        mtime_ns=s.st_mtime_ns,
        ctime_ns=s.st_ctime_ns,
    )


def build(args):
    sys.path[:0] = [
        str(args.repo / "scripts/experiments/gemma4-purpose"),
        str(args.repo / "src"),
    ]
    import propose_native_objectives as adapter
    import joint

    started = time.monotonic()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("fresh output required")
    obs = read(args.observation / "plan.json")
    hashes = read(args.observation / "collector-output-hashes.json")
    parent = obs["parent"]
    if parent["sha256"] != args.parent_sha or sha(parent["path"]) != args.parent_sha:
        raise ValueError("parent changed")
    closure = read(args.observation / "completion.json")
    if closure["status"] != "completed" or not closure["same_container_restored"]:
        raise ValueError("observation not closed")
    review = read(args.review)
    if not (
        review["verified_requested_model"]
        and review["review_usable"]
        and [row["label"] for row in review["response"]["reviews"]] == list("ABCDEFGH")
        and review["tool_uses"] == 0
        and review["assistant_models"] == ["claude-opus-5-5"]
    ):
        raise ValueError("review identity differs")
    if sha(args.review) != args.review_sha:
        raise ValueError("review pin differs")
    ext = {
        str(p): sha(p)
        for p in [
            args.observation / n
            for n in (
                "plan.json",
                "completion.json",
                "postchecks.json",
                "collector-output-hashes.json",
                "independent-numeric-review.json",
            )
        ]
    }
    ext.update(
        {
            str(args.method): args.method_sha,
            str(args.review): args.review_sha,
            str(args.review.with_name("method-disposition.json")): sha(
                args.review.with_name("method-disposition.json")
            ),
            str(Path(__file__).resolve()): sha(__file__),
        }
    )
    digest_map = {}
    stat_map = {}
    dirs = {}
    for name, record in hashes["outputs"].items():
        dirs[name] = record["path"]
        for filename, digest in record["files"].items():
            path = Path(record["path"]) / filename
            digest_map[str(path)] = digest
            stat_map[str(path)] = stamp(path)
    att = obs["checkpoint_attestation"]
    identity = read(att["path"])["checkpoint"]
    output = args.output.resolve()
    source_directories = [
        Path(identity["path"]).resolve(),
        args.observation.resolve(),
        *(Path(path).resolve() for path in dirs.values()),
    ]
    source_files = [
        args.method.resolve(),
        args.review.resolve(),
        args.review.with_name("method-disposition.json").resolve(),
        Path(parent["path"]).resolve(),
        Path(att["path"]).resolve(),
    ]
    if any(
        output == path or path in output.parents or output in path.parents
        for path in source_directories
    ) or any(output == path or output in path.parents for path in source_files):
        raise ValueError("output overlaps input")
    ll = read(Path(dirs["native-correct-logprob"]) / "summary.json")
    labels = [(r["purpose"], r["fold"]) for r in ll["folds"]]
    science = {p for p, _ in labels if p.startswith("science_")}
    if len(labels) != 42 or len(science) != 6:
        raise ValueError("purpose layout differs")
    headroom = []
    for row in ll["folds"]:
        mean = math.fsum(x["correct_logprob"] for x in row["correct_logprobs"]) / len(
            row["correct_logprobs"]
        )
        if (
            not math.isfinite(mean)
            or mean > 0
            or abs(mean - row["mean_correct_logprob"]) > 1e-12
        ):
            raise ValueError("headroom differs")
        headroom.append(-mean)
    stub = [
        dict(objective=o, purpose=p, fold=f, minimum_delta=-0.002)
        for o in adapter.OBJECTIVES
        for p, f in labels
    ]
    policy = read(args.method)
    if (
        sha(args.method) != args.method_sha
        or policy["parent_state"]["sha256"] != args.parent_sha
    ):
        raise ValueError("prospective method or parent changed")
    if list(policy["methods"]) != list("ABCDEFGH"):
        raise ValueError("fixed A-H method roster required")
    args.output.mkdir()
    write(
        args.output / "method-decision.json",
        dict(policy=policy, inputs=ext, review_scope="anonymous public method only"),
    )
    baseplan = dict(
        schema_version=2,
        kind="native_matrix_coupled_objectives_v2",
        checkpoint=dict(
            path=identity["path"], tree_sha256=parent["tree"], attestation=att
        ),
        cases={k: obs["cases"][k] for k in ("path", "sha256")},
        parent_state={k: parent[k] for k in ("path", "sha256")},
        collections={
            family: {o: dirs[family + "-" + o] for o in adapter.OBJECTIVES}
            for family in ("native", "matrix")
        },
        baseline_tolerance=1e-6,
        observation_metadata_sha256={
            p: h for p, h in digest_map.items() if p.endswith(".json")
        },
        completed_collection_sha256=digest_map,
        completed_collection_stat=stat_map,
        eligible_blocks=["native", "matrix"],
        matrix_families=["q", "k", "v", "o", "gate", "up", "down"],
        candidates=[
            dict(
                id="metadata-only",
                constraints=stub,
                cost_metric=dict(
                    type="relative",
                    floor=1e-6,
                    block_scales=dict(native=1.0, matrix=1.0),
                ),
                rounding=dict(relative_radius=0.025, absolute_radius=1e-6, rounds=16),
            )
        ],
    )
    bootstrap = args.output / "metadata-input-plan.json"
    write(bootstrap, baseplan)
    checkpoint, _, collections, actual_labels, pins = adapter.prepare(bootstrap)
    if labels != actual_labels:
        raise ValueError("ordered cohort differs")
    for path, digest in ext.items():
        adapter.capture(Path(path), pins, digest)
    method_path = args.output / "method-decision.json"
    adapter.capture(method_path, pins)
    constraints = [dict(purpose=p, fold=f, minimum_delta=0.0) for p, f in labels]
    observed_native_names = [
        e["name"] for e in collections["correct-logprob"]["coverage"]["tensors"]
    ]
    layers = [
        int(name.split(".layers.")[1].split(".")[0])
        for name in observed_native_names
        if ".layers." in name
    ]
    final_layer = max(layers)
    protected_native_names = [
        name
        for name in observed_native_names
        if name != "model.language_model.norm.weight"
        and f".layers.{final_layer}." not in name
    ]
    if not protected_native_names:
        raise ValueError("protected native eligibility is empty")
    normsq = {}
    counts = {}
    source_equal = {}
    for family in ("native", "matrix"):
        previous = None
        for objective in adapter.OBJECTIVES:
            key = objective if family == "native" else "matrix:" + objective
            collection = collections[key]
            base, g, _, layout, coords, _, found, _, _ = joint.read_collection(
                collection["path"], dict(constraints=constraints)
            )
            for path, digest in found.items():
                if digest_map.get(path) != digest:
                    raise ValueError("completed collection digest differs")
                adapter.capture(Path(path), pins, digest)
            current = (
                base.astype(np.float32).view(np.uint32).copy(),
                layout,
                None if coords is None else coords.copy(),
            )
            if previous is not None:
                if (
                    not np.array_equal(previous[0], current[0])
                    or previous[1] != layout
                    or not np.array_equal(previous[2], coords)
                ):
                    raise ValueError("objective base/layout mismatch")
                source_equal[family] = True
            previous = current
            if family == "native":
                base, g, _ = adapter.select_native_columns(baseplan, base, g, layout)
            scale = np.maximum(np.abs(base), 1e-6)
            if family == "native":
                normsq[(family, objective)] = np.array(
                    [float(np.dot(row * scale, row * scale)) for row in g]
                )
                counts["native"] = len(base)
                protected_base, protected_g, _ = adapter.select_native_columns(
                    {"native_tensors": protected_native_names}, base, g, layout
                )
                protected_scale = np.maximum(np.abs(protected_base), 1e-6)
                normsq[("protected-native", objective)] = np.array(
                    [
                        float(np.dot(row * protected_scale, row * protected_scale))
                        for row in protected_g
                    ]
                )
                counts["protected-native"] = len(protected_base)
                del protected_base, protected_g
            else:
                offset = 0
                masks = {f: [] for f in ("q", "k", "v", "o", "gate", "up", "down")}
                for entry in layout:
                    f = adapter.matrix_recipe.PATTERN.fullmatch(entry["tensor"])[3]
                    masks[f].extend(range(offset, offset + entry["count"]))
                    offset += entry["count"]
                for f, indices in masks.items():
                    index = np.asarray(indices, dtype=np.int64)
                    counts[f] = len(index)
                    normsq[(f, objective)] = np.array(
                        [
                            float(
                                np.dot(
                                    row[index] * scale[index], row[index] * scale[index]
                                )
                            )
                            for row in g
                        ]
                    )
            del g, base, coords
            gc.collect()
    science_ix = [i for i, (purpose, _) in enumerate(labels) if purpose in science]
    ns = math.sqrt(float(np.mean(normsq[("native", "correct-logprob")][science_ix])))
    ms = math.sqrt(
        float(
            np.mean(
                sum(
                    normsq[(f, "correct-logprob")]
                    for f in ("q", "k", "v", "o", "gate", "up", "down")
                )[science_ix]
            )
        )
    )
    if not all(math.isfinite(value) and value > 0 for value in (ns, ms)):
        raise ValueError("H science block scale undefined")
    plans, roster, report = [], [], []
    for label in "BCDEFGH":
        method = policy["methods"][label]
        families = method["matrix_families"]
        use_native = "native" in method["eligible_blocks"]
        native_key = (
            "protected-native"
            if method["freeze_final_norm_and_last_layer_native"]
            else "native"
        )
        sn, sm = (ns, ms) if label == "H" else (1.0, 1.0)
        rows, targets = [], []
        for objective in adapter.OBJECTIVES:
            squared = np.zeros(len(labels))
            if use_native:
                squared += normsq[(native_key, objective)] / (sn * sn)
            for family in families:
                squared += normsq[(family, objective)] / (sm * sm)
            norms = np.sqrt(squared)
            for i, (purpose, fold) in enumerate(labels):
                multiplier = None
                category = "retention"
                target = policy["common"]["unselected_LL_and_margin_lower"]
                if purpose in method["primary_purposes"]:
                    multiplier = method[
                        "primary_ll_multiplier"
                        if objective == "correct-logprob"
                        else "primary_margin_multiplier"
                    ]
                    category = "primary"
                elif (
                    objective == "correct-logprob"
                    and purpose in method["secondary_ll_purposes"]
                ):
                    multiplier = method["secondary_ll_multiplier"]
                    category = "secondary_LL"
                if multiplier is not None:
                    target = multiplier * float(norms[i])
                    if objective == "correct-logprob":
                        target = min(target, 0.1 * headroom[i])
                row = dict(
                    objective=objective,
                    purpose=purpose,
                    fold=fold,
                    minimum_delta=target,
                )
                targets.append(row)
                rows.append(
                    dict(
                        **row,
                        category=category,
                        dual_norm=float(norms[i]),
                        multiplier=multiplier,
                        headroom=headroom[i]
                        if objective == "correct-logprob"
                        else None,
                    )
                )
        plan = copy.deepcopy(baseplan)
        ident = "ad-science-" + label.lower()
        if label == "G":
            plan["schema_version"] = 1
            plan["kind"] = "native_coupled_objectives_v1"
            plan["collections"] = plan["collections"]["native"]
            for key in ("eligible_blocks", "matrix_families", "baseline_tolerance"):
                plan.pop(key, None)
            metric = dict(type="relative", floor=1e-6)
        else:
            plan["eligible_blocks"] = method["eligible_blocks"]
            plan["matrix_families"] = families
            metric = dict(
                type="relative", floor=1e-6, block_scales=dict(native=sn, matrix=sm)
            )
        if use_native and method["freeze_final_norm_and_last_layer_native"]:
            plan["native_tensors"] = protected_native_names
        plan["candidates"] = [
            dict(
                id=ident,
                constraints=targets,
                cost_metric=metric,
                rounding=dict(relative_radius=0.025, absolute_radius=1e-6, rounds=16),
            )
        ]
        plan["science_method_label"] = label
        plan["prospective_method"] = dict(path=str(args.method), sha256=args.method_sha)
        report.append(
            dict(
                label=label,
                id=ident,
                native_coordinates=counts[native_key] if use_native else 0,
                matrix_coordinates=sum(counts[f] for f in families),
                matrix_families=families,
                block_scales=dict(native=sn, matrix=sm),
                rows=rows,
                native_frozen_tensors=sorted(
                    set(observed_native_names) - set(protected_native_names)
                )
                if use_native and method["freeze_final_norm_and_last_layer_native"]
                else [],
            )
        )
        plans.append((label, ident, plan))
    adapter.stable(checkpoint, pins)
    for path, old in stat_map.items():
        if stamp(path) != old:
            raise ValueError("collection stat changed")
    prep = args.output / "preparation.json"
    write(
        prep,
        dict(
            status="prepared_no_solve",
            method=policy,
            input_pins={p: dict(sha256=h, stat=list(s)) for p, (h, s) in pins.items()},
            source_values_equal_across_objectives=source_equal,
            H_scales=dict(native=ns, matrix=ms),
            candidates=report,
            gradient_arrays_read=168,
            solver_calls=0,
            model_calls=0,
            candidate_specs_written=0,
            seconds=time.monotonic() - started,
        ),
    )
    fixed = {p: h for p, (h, _) in pins.items() if p.endswith(".py")}
    fixed.update(ext)
    fixed[str(method_path)] = sha(method_path)
    fixed[str(prep)] = sha(prep)
    (args.output / "plans").mkdir()
    for label, ident, plan in plans:
        plan["fixed_method_inputs"] = fixed
        path = args.output / "plans" / (label + ".json")
        write(path, plan)
        adapter.prepare(path)
        roster.append(
            dict(
                label=label,
                plan=str(path),
                plan_sha256=sha(path),
                output=str(args.output / "proposals" / label),
                ids=[ident],
            )
        )
    adapter.stable(checkpoint, pins)
    write(
        args.output / "plan-roster.json",
        dict(
            status="prepared_no_solve",
            parent=parent,
            preparation=dict(path=str(prep), sha256=sha(prep)),
            method=dict(path=str(method_path), sha256=sha(method_path)),
            source_sha256=sha(__file__),
            rows=roster,
        ),
    )
    print(
        json.dumps(
            dict(
                status="prepared_no_solve",
                roster=str(args.output / "plan-roster.json"),
                roster_sha256=sha(args.output / "plan-roster.json"),
                preparation_sha256=sha(prep),
                method_sha256=sha(method_path),
                seconds=time.monotonic() - started,
                H_scales=dict(native=ns, matrix=ms),
            )
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--observation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--parent-sha", required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--review-sha", required=True)
    p.add_argument("--method", type=Path, required=True)
    p.add_argument("--method-sha", required=True)
    build(p.parse_args())
