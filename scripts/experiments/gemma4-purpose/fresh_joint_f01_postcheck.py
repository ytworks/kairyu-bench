#!/usr/bin/env python3
"""Independently reconstruct F01's sampled BF16 spec and current source words."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from kairyu_bench.wsp.weights.checkpoint import Checkpoint


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate-id", default="fresh-code-joint-f01")
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "independent-postcheck.json"
    if output.exists():
        raise ValueError("fresh postcheck output required")
    started = time.monotonic()
    plan_path = root / "prepared/plan.json"
    report_path = root / "proposal/report.json"
    spec_path = root / "proposal" / (args.candidate_id + ".json")
    plan, report, spec = [json.loads(p.read_text()) for p in (plan_path, report_path, spec_path)]
    method = report["methods"][0]
    collection = Path(plan["collections"]["matrix"]["correct-logprob"])
    coverage = json.loads((collection / "coverage.json").read_text())
    sample_path = collection / "sample-plan.npz"
    if sha(sample_path) != report["source_sha256"][str(sample_path)]:
        raise ValueError("sample input hash changed")
    with np.load(sample_path, allow_pickle=False) as data:
        base = data["source_values"].copy()
        coordinates = data["coordinates"].copy()
    values = base.copy()
    entries = {e["tensor"]: e for e in coverage["entries"]}
    checkpoint = Checkpoint(plan["checkpoint"]["path"])
    now = checkpoint.identity(hash_files=False)
    attestation = json.loads(Path(plan["checkpoint"]["attestation"]["path"]).read_text())
    old = attestation.get("checkpoint", attestation)
    if len(old["files"]) != len(now["files"]):
        raise ValueError("checkpoint inventory changed")
    for a, b in zip(old["files"], now["files"], strict=True):
        if any(a[k] != b[k] for k in ("path", "size", "stat")):
            raise ValueError("checkpoint stat changed")
    if spec["base_tree_sha256"] != plan["checkpoint"]["tree_sha256"]:
        raise ValueError("spec source identity changed")
    seen = set()
    for edit in spec["edits"]:
        name = edit["tensor"]
        if name in seen or name not in entries or "indices" in edit:
            raise ValueError("duplicate or ineligible tensor")
        seen.add(name)
        entry = entries[name]
        start, count = entry["start"], entry["count"]
        index = {tuple(c): start + i for i, c in enumerate(coordinates[start:start + count])}
        cells = [tuple(c) for c in edit["coordinates"]]
        targets = np.asarray(edit["set_values"], dtype=np.float32)
        if (len(cells) != len(set(cells)) or len(cells) != len(targets)
                or np.any(targets.view(np.uint32) & 0xFFFF) or not np.all(np.isfinite(targets))):
            raise ValueError("invalid exact BF16 set_values")
        for cell, value in zip(cells, targets, strict=True):
            values[index[cell]] = value
    tensors = []
    family = {}
    for entry in coverage["entries"]:
        start, count = entry["start"], entry["count"]
        sl = slice(start, start + count)
        current = checkpoint.read_indices(entry["tensor"], [
            int(row) * entry["shape"][1] + int(col) for row, col in coordinates[sl]
        ]).astype(np.float32)
        if not np.array_equal(current.view(np.uint32), base[sl].view(np.uint32)):
            raise ValueError("sample source words differ from current checkpoint")
        delta_part = values[sl].astype(np.float64) - base[sl].astype(np.float64)
        row = dict(tensor=entry["tensor"], family=entry["family"], layer=entry["layer"],
                   sampled=count, changed=int(np.count_nonzero(delta_part)),
                   l2=float(np.linalg.norm(delta_part)), max_abs_delta=float(np.max(np.abs(delta_part))))
        tensors.append(row)
        f = family.setdefault(entry["family"], dict(sampled=0, changed=0, delta_squared=0.0, tensors=0))
        f["sampled"] += count
        f["changed"] += row["changed"]
        f["delta_squared"] += float(delta_part @ delta_part)
        f["tensors"] += int(row["changed"] > 0)
    delta = values.astype(np.float64) - base.astype(np.float64)
    predictions = []
    for row in method["constraints"]:
        directory = Path(plan["collections"]["matrix"][row["objective"]])
        path = directory / f"gradient-{row['purpose']}-{row['fold']}.npy"
        if sha(path) != report["source_sha256"][str(path)]:
            raise ValueError("gradient hash changed")
        gradient = np.load(path, allow_pickle=False).astype(np.float64)
        predicted = float(gradient @ delta)
        predictions.append(dict(**row, predicted_delta=predicted,
                                shortfall=max(0.0, row["minimum_delta"] - predicted)))
    prediction_error = float(np.max(np.abs(np.array([r["predicted_delta"] for r in predictions]) - method["prediction"])))
    target_hash = hashlib.sha256(values.tobytes()).hexdigest()
    l2 = float(np.linalg.norm(delta))
    if (prediction_error > 1e-12 or not np.isclose(l2, method["actual_delta_l2"], atol=1e-12, rtol=0)
            or int(np.count_nonzero(delta)) != method["changed_elements"]
            or target_hash != method["rounding"]["stored_f32_sha256"]):
        raise ValueError("independent reconstruction differs from solver")
    radius = np.maximum(0.25 * np.abs(base.astype(np.float64)), 0.0001)
    excess = np.maximum(np.abs(delta) - radius, 0)
    checkpoint.assert_stable()
    result = dict(
        status="independent_sample_word_and_spec_checks_pass", scope="209920 sampled matrix cells only",
        changed_elements=int(np.count_nonzero(delta)), changed_tensors=len(seen), native_changes=0,
        actual_delta_l2=l2, source_sample_l2=float(np.linalg.norm(base.astype(np.float64))),
        relative_sample_l2=l2 / float(np.linalg.norm(base.astype(np.float64))),
        max_abs_delta=float(np.max(np.abs(delta))),
        continuous_box_exceeded_by_bf16_cells=int(np.count_nonzero(excess)),
        maximum_bf16_box_excess=float(np.max(excess)),
        box_limit_note="Owner radius bounds continuous F64 steps; BF16 rounded values can exceed it; no clamp or retry performed.",
        stored_f32_sha256=target_hash, report_prediction_max_abs_error=prediction_error,
        current_source_words_exact=True, current_source_word_count=len(base),
        missed_constraints=sum(r["shortfall"] > 1e-10 for r in predictions),
        maximum_prediction_shortfall=max(r["shortfall"] for r in predictions),
        rows=predictions, family_support={k: {**v, "delta_l2": float(v["delta_squared"] ** 0.5)} for k, v in family.items()},
        tensor_support=tensors,
        inputs_sha256={p.name: sha(p) for p in (plan_path, report_path, spec_path, Path(__file__))},
        seconds=time.monotonic() - started, model_calls=0, checkpoint_writes=0, wsp_calls=0,
    )
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "tensor_support", "inputs_sha256")}))


if __name__ == "__main__":
    main()
