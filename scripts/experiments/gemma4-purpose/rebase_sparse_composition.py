#!/usr/bin/env python3
"""Rebase a complete sparse BF16 edit only onto identical current-parent words."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from kairyu_bench.wsp.weights.checkpoint import Checkpoint


def captured(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("input digest differs")
    return json.loads(raw)


def frozen_reader(path, digest):
    freeze = captured(path, digest)
    reader = Checkpoint(freeze["checkpoint_path"])
    current = reader.identity(hash_files=False)["files"]
    if [(r["path"], r["size"]) for r in current] != [
        (r["path"], r["size"]) for r in freeze["all_checkpoint_files"]
    ]:
        raise ValueError("frozen inventory differs")
    for row in current:
        old = freeze["candidate_inventory_stat"][row["path"]]
        if row["size"] != old["size"] or any(
            row["stat"][key] != old[key] for key in row["stat"]
        ):
            raise ValueError("frozen file identity differs")
    return reader, freeze


def words(values):
    # Checkpoint.read_indices expands BF16 bits using an integer shift, with no
    # floating-point arithmetic. The inverse shift preserves even signed zero.
    if values.dtype != np.float32:
        raise ValueError("BF16 reader must return exact float32 expansion")
    bits = values.view(np.uint32)
    if np.any(bits & 65535) or not np.all(np.isfinite(values)):
        raise ValueError("values must be finite exact BF16")
    return (bits >> 16).astype(np.uint16)


def rebase(spec, old, current, *, candidate_id, floor, cap):
    if not math.isfinite(floor) or floor <= 0 or not math.isfinite(cap) or cap <= 0:
        raise ValueError("positive finite cost floor and cap required")
    rows, count, raw_sq, weighted_sq, seen = [], 0, 0.0, 0.0, set()
    for edit in spec["edits"]:
        name = edit["tensor"]
        if name in seen:
            raise ValueError("duplicate edited tensor")
        seen.add(name)
        before, after = old.tensors[name], current.tensors[name]
        if before.dtype != "BF16" or after.dtype != "BF16" or before.shape != after.shape:
            raise ValueError("BF16 tensor layout differs")
        selector = "coordinates" if "coordinates" in edit else "indices"
        if set(edit) != {"tensor", selector, "set_values"}:
            raise ValueError("only explicit sparse set-values edits supported")
        selected = edit[selector]
        if not isinstance(selected, list) or not 1 <= len(selected) <= 1000000:
            raise ValueError("bounded nonempty coordinate list required")
        if selector == "coordinates":
            if len(before.shape) != 2 or len(selected) > 512:
                raise ValueError("matrix edit must retain bounded 512-cell contract")
            indices = []
            for cell in selected:
                if (not isinstance(cell, list) or len(cell) != 2
                        or any(type(x) is not int for x in cell)
                        or not 0 <= cell[0] < before.shape[0]
                        or not 0 <= cell[1] < before.shape[1]):
                    raise ValueError("invalid matrix coordinate")
                indices.append(cell[0] * before.shape[1] + cell[1])
        else:
            indices = selected
        if len(set(indices)) != len(indices):
            raise ValueError("duplicate coordinate")
        source = old.read_indices(name, indices)
        base = current.read_indices(name, indices)
        if not np.array_equal(words(source), words(base)):
            raise ValueError("current parent conflicts with original source words; entire edit rejected")
        declared_target = np.asarray(edit["set_values"], dtype=np.float64)
        target = declared_target.astype(np.float32)
        if target.shape != base.shape:
            raise ValueError("target coordinate count differs")
        if not np.array_equal(declared_target, target.astype(np.float64)):
            raise ValueError("declared target is not exact float32")
        target_words = words(target)
        changed = int(np.count_nonzero(target_words != words(base)))
        delta = target.astype(np.float64) - base.astype(np.float64)
        raw = float(np.dot(delta, delta))
        weighted = delta / np.maximum(np.abs(base.astype(np.float64)), floor)
        weighted_norm = float(np.dot(weighted, weighted))
        rows.append({"tensor": name, "selected": len(indices), "changed": changed,
                     "delta_l2": math.sqrt(raw), "weighted_delta_l2": math.sqrt(weighted_norm)})
        count += changed
        raw_sq += raw
        weighted_sq += weighted_norm
    if not rows or not count or math.sqrt(weighted_sq) > cap:
        raise ValueError("empty edit or realized weighted cap exceeded")
    old.assert_stable()
    current.assert_stable()
    output = deepcopy(spec)
    output["id"] = candidate_id
    output["source_checkpoint"] = str(current.path)
    output["hypothesis"] = "Complete earlier sparse edit rebased onto exact matching current-parent BF16 words; functional interaction unmeasured."
    return output, {
        "changed_elements": count, "edited_tensors": len(rows),
        "raw_delta_l2": math.sqrt(raw_sq),
        "weighted_delta_l2": math.sqrt(weighted_sq),
        "cost_metric": {"type": "relative", "floor": floor, "realized_cap": cap},
        "composition": "all selected source words match; full original edit carried or rejected, no partial dropping",
        "per_tensor": rows, "model_calls": 0, "checkpoint_writes": 0,
        "finite_response_measured": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "source-freeze", "parent-freeze", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("spec-sha", "source-freeze-sha", "parent-freeze-sha", "candidate-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--relative-floor", type=float, required=True)
    parser.add_argument("--weighted-cap", type=float, required=True)
    parser.add_argument("--write-spec", action="store_true")
    args = parser.parse_args()
    old, old_freeze = frozen_reader(args.source_freeze, args.source_freeze_sha)
    current, current_freeze = frozen_reader(args.parent_freeze, args.parent_freeze_sha)
    spec = captured(args.spec, args.spec_sha)
    if (spec["base_tree_sha256"] != old_freeze["control_tree_sha256"]
            or Path(spec["source_checkpoint"]).resolve() != old.path):
        raise ValueError("original edit source differs from frozen source")
    result, report = rebase(spec, old, current, candidate_id=args.candidate_id,
                            floor=args.relative_floor, cap=args.weighted_cap)
    result["base_tree_sha256"] = current_freeze["control_tree_sha256"]
    report.update(source_spec_sha256=args.spec_sha,
                  source_freeze_sha256=args.source_freeze_sha,
                  parent_freeze_sha256=args.parent_freeze_sha,
                  parent_tree_sha256=result["base_tree_sha256"],
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_identity="previous full hash plus exact current stat continuity; no new full payload hash")
    # Recheck captured evidence after all bounded payload reads, before publication.
    captured(args.spec, args.spec_sha)
    frozen_reader(args.source_freeze, args.source_freeze_sha)
    frozen_reader(args.parent_freeze, args.parent_freeze_sha)
    if args.write_spec:
        output = args.output.resolve()
        if args.output.is_symlink() or any(
            output == root or output.is_relative_to(root) or root.is_relative_to(output)
            for root in (old.path, current.path)
        ):
            raise ValueError("output overlaps an input checkpoint or is a symlink")
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "spec.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_tensor"}))


if __name__ == "__main__":
    main()
