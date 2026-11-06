#!/usr/bin/env python3
"""Prepare eight current-parent dependency-set plans; no solver or model call."""

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
        and review["labels_match"]
        and review["labels"] == "ABCDEFGH"
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
    ll = read(Path(dirs["native-correct-logprob"]) / "summary.json")
    labels = [(r["purpose"], r["fold"]) for r in ll["folds"]]
    core = {"code", "implementation", "workflow", "algorithm"}
    science = {p for p, _ in labels if p.startswith("science_")}
    repair = {"instruction", "tools", "lookup"}
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
    policy = dict(
        protocol="eight_dependency_sets_v1",
        parent_state=parent,
        core_purposes=sorted(core),
        science_purposes=sorted(science),
        repair_purposes=sorted(repair),
        standard_core_multiplier=0.25,
        standard_science_multiplier=0.0625,
        repair_multiplier=0.125,
        H_core_margin_multiplier=0.25,
        H_core_correct_ll_multiplier=0.0625,
        ll_headroom_fraction=0.1,
        all_other_LL_and_margin_lower=-0.002,
        cost_type="raw_stored_parameter_relative",
        cost_floor=1e-6,
        relative_radius=0.025,
        absolute_radius=1e-6,
        rounds=16,
        maximum_seconds_per_solve=120,
        actual_declared_weighted_l2_cap=1.0,
        cap_is_owner_postcheck=True,
        failure_policy="no rescale, retry, radius increase or amplitude grid",
        F_partition=["native", "matrix"],
        F_scale="sqrt(mean over eight core LL folds of block base-cost dual norm squared)",
        comparison_limits=[
            "Eligible-scope targets differ across family plans; subset min-cost dominance does not follow.",
            "F uses a different metric; report raw and unscaled relative norm and block collapse.",
            "Finite science LL/margin and related-purpose tradeoffs must be reported; not core LL alone.",
            "All are alternatives from one accepted parent, not additive components.",
        ],
        methods=dict(
            A="native",
            B="QK",
            C="VO",
            D="MLP",
            E="all-matrix",
            F="two-block-native-allmatrix",
            G="O-MLP-repair",
            H="O-MLP-margin",
        ),
    )
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
            scale = np.maximum(np.abs(base), 1e-6)
            if family == "native":
                normsq[(family, objective)] = np.array(
                    [float(np.dot(row * scale, row * scale)) for row in g]
                )
                counts["native"] = len(base)
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
    core_ix = [i for i, (p, _) in enumerate(labels) if p in core]
    ns = math.sqrt(float(np.mean(normsq[("native", "correct-logprob")][core_ix])))
    ms = math.sqrt(
        float(
            np.mean(
                sum(
                    normsq[(f, "correct-logprob")]
                    for f in ("q", "k", "v", "o", "gate", "up", "down")
                )[core_ix]
            )
        )
    )
    if not all(math.isfinite(s) and s > 0 for s in (ns, ms)):
        raise ValueError("F block scale undefined")
    scopes = dict(
        A=[],
        B=["q", "k"],
        C=["v", "o"],
        D=["gate", "up", "down"],
        E=["q", "k", "v", "o", "gate", "up", "down"],
        F=["q", "k", "v", "o", "gate", "up", "down"],
        G=["o", "gate", "up", "down"],
        H=["o", "gate", "up", "down"],
    )
    suffix = dict(
        A="native",
        B="qk",
        C="vo",
        D="mlp",
        E="all-matrix",
        F="balanced-blocks",
        G="repair",
        H="margin",
    )
    plans = []
    roster = []
    report = []
    for label, families in scopes.items():
        use_native = label in ("A", "F")
        sn, sm = (ns, ms) if label == "F" else (1.0, 1.0)
        rows = []
        targets = []
        for objective in adapter.OBJECTIVES:
            squared = np.zeros(42)
            if use_native:
                squared += normsq[("native", objective)] / (sn * sn)
            for f in families:
                squared += normsq[(f, objective)] / (sm * sm)
            norms = np.sqrt(squared)
            for i, (purpose, fold) in enumerate(labels):
                multiplier = None
                target = -0.002
                category = "guard"
                if objective == "correct-logprob":
                    if purpose in core:
                        multiplier = 0.0625 if label == "H" else 0.25
                        category = "core"
                    elif label == "G" and purpose in repair:
                        multiplier = 0.125
                        category = "repair"
                    elif label not in ("G", "H") and purpose in science:
                        multiplier = 0.0625
                        category = "science"
                    if multiplier is not None:
                        target = min(multiplier * float(norms[i]), 0.1 * headroom[i])
                elif label == "H" and purpose in core:
                    multiplier = 0.25
                    category = "core_margin"
                    target = multiplier * float(norms[i])
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
        ident = "ac-eight-" + label.lower() + "-" + suffix[label]
        if label == "A":
            plan["schema_version"] = 1
            plan["kind"] = "native_coupled_objectives_v1"
            plan["collections"] = plan["collections"]["native"]
            for key in ("eligible_blocks", "matrix_families", "baseline_tolerance"):
                plan.pop(key, None)
            metric = dict(type="relative", floor=1e-6)
        else:
            plan["eligible_blocks"] = ["native", "matrix"] if use_native else ["matrix"]
            plan["matrix_families"] = families
            metric = dict(
                type="relative", floor=1e-6, block_scales=dict(native=sn, matrix=sm)
            )
        plan["candidates"] = [
            dict(
                id=ident,
                constraints=targets,
                cost_metric=metric,
                rounding=dict(relative_radius=0.025, absolute_radius=1e-6, rounds=16),
            )
        ]
        plan["eight_method_label"] = label
        plan["eight_method_policy"] = policy
        report.append(
            dict(
                label=label,
                id=ident,
                native_coordinates=counts["native"] if use_native else 0,
                matrix_coordinates=sum(counts[f] for f in families),
                matrix_families=families,
                block_scales=dict(native=sn, matrix=sm),
                rows=rows,
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
            F_scales=dict(native=ns, matrix=ms),
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
                F_scales=dict(native=ns, matrix=ms),
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
    build(p.parse_args())
