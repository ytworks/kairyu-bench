#!/usr/bin/env python3
"""Compose completed disjoint-purpose observations without a model or solver."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import propose_native_objectives as adapter

FAMILIES = ("native", "matrix")
OBJECTIVES = adapter.OBJECTIVES
RUNTIME = (
    "torch",
    "transformers",
    "weight_dtype",
    "attention",
    "observation_policy",
    "gradients",
    "gradient_accumulation",
    "gradient_checkpointing",
    "max_cases_per_microbatch",
)
SETTINGS = (
    "threads",
    "seed",
    "max_cases_per_microbatch",
    "calibration_scope",
    "mode",
    "answer_batching",
    "objective_kind",
    "objective_units",
    "objective_forward_layout",
    "virtual_unit_gains",
    "sampling_mode",
    "observation_policy",
)


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def ref(path, pins, expected=None):
    raw = adapter.capture(Path(path), pins, expected)
    return json.loads(raw)


def array_equal_bytes(left, right):
    with (
        np.load(io.BytesIO(left), allow_pickle=False) as a,
        np.load(io.BytesIO(right), allow_pickle=False) as b,
    ):
        return set(a.files) == set(b.files) and all(
            a[k].dtype == b[k].dtype
            and a[k].shape == b[k].shape
            and a[k].tobytes() == b[k].tobytes()
            for k in a.files
        )


def validation_plan(plan, cases, collections, labels):
    return {
        "schema_version": 2,
        "kind": "native_matrix_coupled_objectives_v2",
        "checkpoint": plan["checkpoint"],
        "parent_state": plan["parent_state"],
        "cases": cases,
        "collections": collections,
        "baseline_tolerance": plan["baseline_tolerance"],
        "candidates": [
            {
                "id": "composition-validation-only",
                "constraints": [
                    {
                        "objective": objective,
                        "purpose": purpose,
                        "fold": fold,
                        "minimum_delta": 0.0,
                    }
                    for objective in OBJECTIVES
                    for purpose, fold in labels
                ],
                "cost_metric": {
                    "type": "euclidean",
                    "block_scales": {"native": 1.0, "matrix": 1.0},
                },
                "rounding": {
                    "relative_radius": 0.0,
                    "absolute_radius": 1e-6,
                    "rounds": 1,
                },
            }
        ],
    }


def checked_segment(plan, segment, pins):
    receipt = ref(segment["receipt"]["path"], pins, segment["receipt"]["sha256"])
    if receipt.get("all_stat_stable") is not True:
        raise ValueError("completed stable collection receipt required")
    completion = ref(
        receipt["completion"]["path"], pins, receipt["completion"]["sha256"]
    )
    if completion.get("status") != "completed":
        raise ValueError("collection owner has not completed")
    parent = receipt["source_parent"]
    if (
        parent["path"] != plan["parent_state"]["path"]
        or parent["sha256"] != plan["parent_state"]["sha256"]
        or parent["tree"] != plan["checkpoint"]["tree_sha256"]
    ):
        raise ValueError("segment parent identity differs")
    cases_ref = segment["cases"]
    case_path = Path(cases_ref["path"]).resolve()
    raw = adapter.capture(case_path, pins, cases_ref["sha256"])
    cases = adapter.joint.common.read_cases(case_path, content=raw)
    labels = [
        (purpose, fold)
        for purpose, folds in cases.items()
        for fold in range(len(folds))
    ]
    collections = {family: {} for family in FAMILIES}
    for family in FAMILIES:
        for objective in OBJECTIVES:
            row = receipt["outputs"][family + "-" + objective]
            directory = Path(row["path"]).resolve()
            collections[family][objective] = str(directory)
            if not directory.is_dir() or directory.is_symlink():
                raise ValueError("ordinary collection directory required")
            for name in row["files"]:
                relative = Path(name)
                path = directory / relative
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or path.is_symlink()
                ):
                    raise ValueError("unsafe collection relative file")
                absolute_stat = row["stat"].get(str(path))
                relative_stat = row["stat"].get(name)
                if (
                    absolute_stat is not None
                    and relative_stat is not None
                    and absolute_stat != relative_stat
                ):
                    raise ValueError("conflicting absolute/relative collection stat")
                frozen = absolute_stat if absolute_stat is not None else relative_stat
                if frozen is None:
                    raise ValueError("missing completed collection stat")
                if isinstance(frozen, dict):
                    frozen = [
                        frozen[k]
                        for k in ("device", "inode", "size", "mtime_ns", "ctime_ns")
                    ]
                if adapter.stamp(path) != tuple(frozen):
                    raise ValueError("completed collection stat changed")
            for name in (
                "summary.json",
                "provenance.json",
                "coverage.json",
                "frame-evidence.json",
            ):
                adapter.capture(directory / name, pins, row["files"][name])
    vplan = validation_plan(plan, cases_ref, collections, labels)
    with tempfile.TemporaryDirectory(prefix="purpose-compose-validation-") as tmp:
        path = Path(tmp) / "plan.json"
        write_json(path, vplan)
        checkpoint, _, loaded, actual_labels, validation_pins = adapter.prepare(path)
        assert actual_labels == labels
        for name, value in validation_pins.items():
            if name != str(path):
                if name in pins and pins[name] != value:
                    raise ValueError("validation input pins changed")
                pins[name] = value
    return {
        "receipt": receipt,
        "reference": segment,
        "cases": cases,
        "labels": labels,
        "collections": loaded,
        "checkpoint": checkpoint,
    }


def compose(plan_path, output, *, write=False, expected_plan_sha=None):
    pins = {}
    adapter.capture(Path(__file__).resolve(), pins)
    plan_path = Path(plan_path).resolve()
    plan = ref(plan_path, pins, expected_plan_sha)
    if (
        plan.get("schema_version") != 1
        or plan.get("kind") != "disjoint_purpose_collections_v1"
    ):
        raise ValueError("explicit disjoint-purpose composition plan required")
    if not isinstance(plan.get("segments"), list) or len(plan["segments"]) < 2:
        raise ValueError("at least two completed segments required")
    cases_path = Path(plan["cases"]["path"]).resolve()
    cases_raw = adapter.capture(cases_path, pins, plan["cases"]["sha256"])
    cases = adapter.joint.common.read_cases(cases_path, content=cases_raw)
    segments = [checked_segment(plan, segment, pins) for segment in plan["segments"]]
    union, ids = {}, set()
    for segment in segments:
        if set(union) & set(segment["cases"]):
            raise ValueError("segment purposes must be disjoint")
        for purpose, folds in segment["cases"].items():
            for fold in folds:
                for case in fold:
                    if case["id"] in ids:
                        raise ValueError("segment case IDs must be unique")
                    ids.add(case["id"])
            union[purpose] = folds
    if list(cases) != list(union) or cases != union:
        raise ValueError("combined cases must preserve exact ordered segment cases")
    full_cases = cases
    selection = plan.get("purpose_selection")
    if selection is not None:
        names = selection.get("selected_purposes")
        reasons = selection.get("omitted_reasons")
        if (
            not isinstance(names, list)
            or not names
            or any(type(n) is not str or n not in cases for n in names)
            or len(set(names)) != len(names)
            or names != [name for name in cases if name in names]
            or not isinstance(reasons, dict)
            or set(reasons) != set(cases) - set(names)
            or any(type(v) is not str or not v.strip() for v in reasons.values())
        ):
            raise ValueError(
                "explicit ordered unique purpose subset and every omission reason required"
            )
        cases = {name: cases[name] for name in names}
        ids = {
            case["id"] for folds in cases.values() for fold in folds for case in fold
        }
    labels = [
        label
        for segment in segments
        for label in segment["labels"]
        if label[0] in cases
    ]
    if len(labels) * 2 > 128:
        raise ValueError("composed cohort exceeds adapter constraint limit")
    checkpoint = segments[0]["checkpoint"]
    if any(
        s["checkpoint"].identity(False) != checkpoint.identity(False) for s in segments
    ):
        raise ValueError("segment checkpoint identities differ")
    for segment in segments[1:]:
        for key, original in segments[0]["collections"].items():
            current = segment["collections"][key]
            if original["coverage"] != current["coverage"]:
                raise ValueError("coverage/sample layout differs")
            if {k: v for k, v in original["frames"].items() if k != "frames"} != {
                k: v for k, v in current["frames"].items() if k != "frames"
            }:
                raise ValueError("frame protocol differs")
            if any(
                original["provenance"].get(k) != current["provenance"].get(k)
                for k in SETTINGS
            ):
                raise ValueError("collector settings differ")
            if any(
                original["summary"]["runtime"].get(k)
                != current["summary"]["runtime"].get(k)
                for k in RUNTIME
            ):
                raise ValueError("collector numerical runtime differs")
    shared_inputs = {}
    for segment in segments:
        for collection in segment["collections"].values():
            for path, digest in collection["provenance"]["input_sha256"].items():
                if path in shared_inputs and shared_inputs[path] != digest:
                    raise ValueError("shared implementation/input SHA differs")
                shared_inputs[path] = digest
    for path, digest in shared_inputs.items():
        adapter.capture(Path(path), pins, digest)
    for family in FAMILIES:
        filename = (
            "source-native-values.npz" if family == "native" else "sample-plan.npz"
        )
        first = None
        for segment in segments:
            for objective in OBJECTIVES:
                key = objective if family == "native" else "matrix:" + objective
                row = segment["receipt"]["outputs"][family + "-" + objective]
                raw = adapter.capture(
                    segment["collections"][key]["path"] / filename,
                    pins,
                    row["files"][filename],
                )
                if first is None:
                    first = raw
                elif not array_equal_bytes(first, raw):
                    raise ValueError("source array/sample bytes differ")
    protected = [
        plan_path,
        cases_path,
        checkpoint.path,
        *[Path(p) for p in pins],
        *[c["path"] for s in segments for c in s["collections"].values()],
    ]
    output = Path(output)
    destination = output.resolve()
    if (
        output.exists()
        or output.is_symlink()
        or any(
            destination == p or destination in p.parents or p in destination.parents
            for p in protected
        )
    ):
        raise ValueError("fresh output disjoint from all inputs required")
    # Completed gradients are immutable receipt-pinned files. Their bytes are copied,
    # never decoded, averaged, rescaled, reordered within a fold, or regenerated.
    copies = {}
    for family in FAMILIES:
        for objective in OBJECTIVES:
            key = objective if family == "native" else "matrix:" + objective
            entries = []
            for index, segment in enumerate(segments):
                collection = segment["collections"][key]
                files = segment["receipt"]["outputs"][family + "-" + objective]["files"]
                for record in collection["summary"]["folds"]:
                    if record["purpose"] not in cases:
                        continue
                    relative = Path(record["file"])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("unsafe gradient relative file")
                    source = collection["path"] / relative
                    if record["file"] not in files:
                        raise ValueError("gradient is absent from completed receipt")
                    entries.append(
                        (
                            source,
                            Path("segment-" + str(index) + "-" + relative.name),
                            files[record["file"]],
                            copy.deepcopy(record),
                        )
                    )
            copies[family + "-" + objective] = entries
    adapter.stable(checkpoint, pins)
    preview = {
        "status": "prepared_no_write",
        "purposes": len(cases),
        "folds": len(labels),
        "cases": len(ids),
        "segments": len(segments),
        "source_purposes": len(full_cases),
        "source_folds": sum(len(v) for v in full_cases.values()),
        "purpose_selection": selection,
        "model_calls": 0,
        "solver_calls": 0,
        "checkpoint_payload_hashes": 0,
        "output": str(destination),
    }
    if not write:
        return preview
    owned = False
    try:
        destination.mkdir(parents=False, mode=0o700)
        owned = True
        selected_cases_ref = plan["cases"]
        if selection is not None:
            original_document = json.loads(cases_raw)
            original_groups = original_document.get("cases", original_document)
            selected_path = destination / "cases.json"
            write_json(
                selected_path,
                {
                    "schema_version": 1,
                    "split": "train",
                    "gradient_allowed": True,
                    "cases": {name: original_groups[name] for name in cases},
                    "purpose_selection": selection,
                    "full_cases": plan["cases"],
                },
            )
            selected_cases_ref = {
                "path": str(selected_path),
                "sha256": adapter.joint.common.sha(selected_path),
            }
        composition = {
            "schema_version": 1,
            "kind": "disjoint_purpose_composition",
            "not_a_single_model_run": True,
            "segments": plan["segments"],
            "input_plan": {"path": str(plan_path), "sha256": pins[str(plan_path)][0]},
            "cases": selected_cases_ref,
            "full_cases": plan["cases"],
            "purpose_selection": selection,
            "parent_state": plan["parent_state"],
            "checkpoint": plan["checkpoint"],
            "model_calls": 0,
            "solver_calls": 0,
            "gradient_operation": "exact source-file byte copies; ordered disjoint fold concatenation",
        }
        shared_inputs.update(
            {
                str(cases_path): plan["cases"]["sha256"],
                str(plan_path): pins[str(plan_path)][0],
                str(Path(__file__).resolve()): pins[str(Path(__file__).resolve())][0],
            }
        )
        shared_inputs[selected_cases_ref["path"]] = selected_cases_ref["sha256"]
        for segment in segments:
            for reference in (
                segment["reference"]["receipt"],
                segment["receipt"]["completion"],
            ):
                shared_inputs[reference["path"]] = reference["sha256"]
        common_stat = {path: adapter.stamp(Path(path)) for path in shared_inputs}
        frames = copy.deepcopy(segments[0]["collections"][OBJECTIVES[0]]["frames"])
        frames["frames"] = {
            purpose: copy.deepcopy(
                segment["collections"][OBJECTIVES[0]]["frames"]["frames"][purpose]
            )
            for segment in segments
            for purpose in segment["cases"]
            if purpose in cases
        }
        collections, output_hashes = {f: {} for f in FAMILIES}, {}
        for family in FAMILIES:
            for objective in OBJECTIVES:
                label = family + "-" + objective
                key = objective if family == "native" else "matrix:" + objective
                first = segments[0]["collections"][key]
                target = destination / label
                target.mkdir(mode=0o700)
                collections[family][objective] = str(target)
                rows = []
                for source, relative, digest, record in copies[label]:
                    raw = adapter.capture(source, pins, digest)
                    path = target / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as stream:
                        stream.write(raw)
                    output_hashes[str(path.relative_to(destination))] = digest
                    record["file"] = str(relative)
                    rows.append(record)
                filename = (
                    "source-native-values.npz"
                    if family == "native"
                    else "sample-plan.npz"
                )
                raw = adapter.capture(first["path"] / filename, pins)
                with (target / filename).open("xb") as stream:
                    stream.write(raw)
                output_hashes[str((target / filename).relative_to(destination))] = (
                    hashlib.sha256(raw).hexdigest()
                )
                summary = copy.deepcopy(first["summary"])
                summary.update(folds=rows, seconds=None, composition=composition)
                summary["runtime"] = {
                    k: v
                    for k, v in summary["runtime"].items()
                    if not k.startswith("gpu_peak")
                }
                summary["runtime"]["execution_kind"] = "composed_completed_segments"
                provenance = copy.deepcopy(first["provenance"])
                provenance.update(
                    input_sha256=shared_inputs,
                    input_stat=common_stat,
                    device="none-composition",
                    model_forwards=0,
                    new_checkpoints=0,
                    wsp_calls=0,
                    full_tree_rehashed_here=False,
                    purposes=len(cases),
                    folds=len(labels),
                    cases=len(ids),
                    composition=composition,
                    original_provenance=[
                        {
                            "path": str(
                                s["collections"][key]["path"] / "provenance.json"
                            ),
                            "sha256": pins[
                                str(s["collections"][key]["path"] / "provenance.json")
                            ][0],
                        }
                        for s in segments
                    ],
                )
                for name, value in (
                    ("summary.json", summary),
                    ("provenance.json", provenance),
                    ("coverage.json", first["coverage"]),
                    ("frame-evidence.json", frames),
                ):
                    write_json(target / name, value)
                    output_hashes[str((target / name).relative_to(destination))] = (
                        adapter.joint.common.sha(target / name)
                    )
        output_plan = validation_plan(plan, selected_cases_ref, collections, labels)
        output_plan["composition"] = composition
        output_plan_path = destination / "adapter-input-template.json"
        write_json(output_plan_path, output_plan)
        # Exercise the unchanged adapter metadata boundary; no gradient solve.
        adapter.prepare(output_plan_path)
        adapter.stable(checkpoint, pins)
        if any(
            adapter.stamp(Path(path)) != stamp for path, stamp in common_stat.items()
        ):
            raise ValueError("composition input changed before publication")
        result = {
            **preview,
            "status": "completed",
            "collections": collections,
            "composition": composition,
            "files": output_hashes,
            "input_sha256": {p: v[0] for p, v in pins.items()},
            "adapter_input_template": {
                "path": str(output_plan_path),
                "sha256": adapter.joint.common.sha(output_plan_path),
            },
        }
        write_json(destination / "composition.json", result)
        return result
    except BaseException:
        if owned:
            shutil.rmtree(destination)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    result = compose(
        args.plan, args.output, write=args.write, expected_plan_sha=args.plan_sha256
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "purposes",
                    "folds",
                    "cases",
                    "model_calls",
                    "solver_calls",
                    "output",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
