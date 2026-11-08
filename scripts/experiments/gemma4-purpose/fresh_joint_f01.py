#!/usr/bin/env python3
"""Prepare the fixed F01 purpose plan and complete sampled-gradient geometry.

CPU only. This does not solve, change checkpoints, launch services, or score WSP.
The targets are fixed by the experiment owner, not derived from historical ranks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import propose_native_objectives as coupled


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def cosine(gradient):
    norms = np.linalg.norm(gradient, axis=1)
    product = norms[:, None] * norms[None, :]
    result = np.divide(
        gradient @ gradient.T, product,
        out=np.zeros_like(product), where=product != 0,
    )
    # Zero norm is unknown, not evidence of orthogonality.
    return [
        [float(result[i, j]) if product[i, j] else None for j in range(len(norms))]
        for i in range(len(norms))
    ]


def prepare(args):
    started = time.monotonic()
    cases = json.loads(args.cases.read_text())
    receipt = json.loads(args.attestation.read_text())
    identity = receipt.get("checkpoint", receipt)
    constraints = []
    for objective in coupled.OBJECTIVES:
        for purpose, folds in cases.items():
            target = (
                0.03 if purpose in set(args.target_purposes)
                else (0.01 if objective == "correct-logprob" else 0.0)
                if purpose == "code" else -0.03
            )
            for fold in range(len(folds)):
                constraints.append(dict(
                    objective=objective, purpose=purpose, fold=fold,
                    minimum_delta=target,
                ))
    plan = dict(
        schema_version=2, kind="native_matrix_coupled_objectives_v2",
        observed_blocks=["matrix"], eligible_blocks=["matrix"],
        matrix_families=["q", "k", "v", "o", "gate", "up", "down"],
        baseline_tolerance=1e-6,
        checkpoint=dict(
            path=str(args.checkpoint.resolve()), tree_sha256=identity["sha256"],
            attestation=dict(path=str(args.attestation.resolve()), sha256=digest(args.attestation)),
        ),
        cases=dict(path=str(args.cases.resolve()), sha256=digest(args.cases)),
        collections={"matrix": {
            objective: str((args.collection_root / ("matrix-" + objective)).resolve())
            for objective in coupled.OBJECTIVES
        }},
        candidates=[dict(
            id=args.candidate_id, constraints=constraints,
            cost_metric=dict(type="euclidean", block_scales=dict(native=1, matrix=1)),
            rounding=dict(relative_radius=0.25, absolute_radius=0.0001, rounds=16),
        )],
    )
    args.output.mkdir(parents=True, exist_ok=False)
    plan_path = args.output / "plan.json"
    write(plan_path, plan)
    checkpoint, _, collections, labels, pins = coupled.prepare(plan_path)
    loaded = []
    source = coordinates = layout = None
    for objective in coupled.OBJECTIVES:
        collection = collections["matrix:" + objective]
        zero = [{"purpose": p, "fold": f, "minimum_delta": 0.0} for p, f in labels]
        data = coupled.joint.read_collection(collection["path"], {"constraints": zero})
        base, gradient, _, current_layout, coords, _, hashes, _, _ = data
        for name, sha in hashes.items():
            coupled.capture(Path(name), pins, sha)
        if source is not None:
            if (layout != current_layout or not np.array_equal(coordinates, coords)
                    or not np.array_equal(source.astype(np.float32).view(np.uint32),
                                          base.astype(np.float32).view(np.uint32))):
                raise ValueError("source words or coordinate order differ between objectives")
        source, coordinates, layout = base, coords, current_layout
        loaded.append(gradient)
    gradient = np.vstack(loaded)
    if gradient.shape != (len(constraints), 209920) or len(layout) != 410:
        raise ValueError("F01 requires all 410 matrices and 209920 stratified cells")
    if not np.all(np.isfinite(gradient)):
        raise ValueError("nonfinite sampled gradient")
    tensor_norms = np.empty((len(constraints), len(layout)))
    family_columns = {family: [] for family in plan["matrix_families"]}
    layer_columns = {}
    cursor = 0
    for j, entry in enumerate(layout):
        match = coupled.matrix_recipe.PATTERN.fullmatch(entry["tensor"])
        count = entry["count"]
        indices = list(range(cursor, cursor + count))
        family_columns[match[3]].extend(indices)
        layer_columns.setdefault(int(match[1]), []).extend(indices)
        tensor_norms[:, j] = np.linalg.norm(gradient[:, cursor:cursor + count], axis=1)
        cursor += count
    all_cosines = {"all": cosine(gradient)}
    family_norms = {}
    for family, indices in family_columns.items():
        part = gradient[:, indices]
        all_cosines[family] = cosine(part)
        family_norms[family] = np.linalg.norm(part, axis=1).tolist()
    layer_norms = {
        str(layer): np.linalg.norm(gradient[:, indices], axis=1).tolist()
        for layer, indices in sorted(layer_columns.items())
    }
    conflicts = {}
    for family, matrix in all_cosines.items():
        values = [matrix[i][j] for i in range(len(matrix)) for j in range(i)
                  if matrix[i][j] is not None]
        conflicts[family] = dict(
            pair_count=len(values), positive=sum(v > 0 for v in values),
            negative=sum(v < 0 for v in values), zero=sum(v == 0 for v in values),
            minimum=min(values), maximum=max(values),
        )
    coupled.stable(checkpoint, pins)
    row_labels = [{k: row[k] for k in ("objective", "purpose", "fold")} for row in constraints]
    write(args.output / "geometry.json", dict(
        scope="209920 stratified sampled matrix cells only, not whole-model gradients",
        labels=row_labels, cosine=all_cosines, cosine_sign_counts=conflicts,
        source_sample_l2=float(np.linalg.norm(source)),
        gradient_sample_l2=np.linalg.norm(gradient, axis=1).tolist(),
        family_gradient_sample_l2=family_norms,
        layer_gradient_sample_l2=layer_norms,
        tensor_layout=layout, tensor_gradient_sample_l2=tensor_norms.tolist(),
        note="Positive/negative cosine describes shared/conflicting local derivatives; no causal capability localization.",
    ))
    write(args.output / "preparation.json", dict(
        status="metadata_and_sample_geometry_valid", plan_sha256=digest(plan_path),
        implementation_sha256=digest(Path(__file__)),
        input_sha256={p: h for p, (h, _) in pins.items()},
        constraints=len(constraints), purposes=len(cases), folds=len(labels),
        sample_coordinates=len(source), matrices=len(layout),
        native="frozen_not_observed", cosine_sign_counts=conflicts,
        seconds=time.monotonic() - started,
        model_calls=0, checkpoint_writes=0, wsp_calls=0,
    ))
    print(json.dumps(dict(status="prepared", constraints=len(constraints),
                          sample_coordinates=len(source), seconds=time.monotonic() - started)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("collection-root", "cases", "checkpoint", "attestation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--candidate-id", default="fresh-code-joint-f01")
    parser.add_argument("--target-purposes", nargs="+", default=["algorithm", "implementation"])
    args = parser.parse_args()
    if not set(args.target_purposes).issubset(json.loads(args.cases.read_text())):
        parser.error("target purposes must exist in the unchanged all-case set")
    prepare(args)


if __name__ == "__main__":
    main()
