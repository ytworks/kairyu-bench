#!/usr/bin/env python3
"""Read-only F01 derivative/finite-error analysis for a future joint objective."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    root = args.root
    paths = {k:root/v for k,v in dict(plan="prepared/plan.json", proposal="proposal/report.json",
             spec="proposal/fresh-code-joint-f01.json", geometry="prepared/geometry.json",
             finite="finite-gpu6-v4/numeric-analysis.json").items()}
    data = {k:json.loads(p.read_text()) for k,p in paths.items()}
    plan, report, spec = (data[k] for k in ("plan","proposal","spec"))
    collection = Path(plan["collections"]["matrix"]["correct-logprob"])
    coverage_path = collection/"coverage.json"
    sample_path = collection/"sample-plan.npz"
    for path in (coverage_path,sample_path):
        assert sha(path) == report["source_sha256"][str(path)]
    entries = json.loads(coverage_path.read_text())["entries"]
    with np.load(sample_path, allow_pickle=False) as sample:
        base, coords = sample["source_values"].copy(), sample["coordinates"].copy()
    targets = base.copy()
    by_name = {e["tensor"]:e for e in entries}
    for edit in spec["edits"]:
        entry = by_name[edit["tensor"]]
        start,count = entry["start"],entry["count"]
        coordinate_index = {tuple(c):start+i for i,c in enumerate(coords[start:start+count])}
        for coordinate,value in zip(edit["coordinates"],edit["set_values"],strict=True):
            targets[coordinate_index[tuple(coordinate)]] = value
    delta = targets.astype(np.float64)-base.astype(np.float64)
    assert sha(paths["spec"]) == "12f4e111a9d31539c6e2d3a94e8270c34b4dbc8a1c4f145fcdcf21ef936eb9d5"
    assert hashlib.sha256(targets.tobytes()).hexdigest() == report["methods"][0]["rounding"]["stored_f32_sha256"]
    labels = data["geometry"]["labels"]
    gradients = []
    hashes = {str(p):sha(p) for p in paths.values()}
    for label in labels:
        path = Path(plan["collections"]["matrix"][label["objective"]])/f"gradient-{label['purpose']}-{label['fold']}.npy"
        digest = sha(path)
        assert digest == report["source_sha256"][str(path)]
        hashes[str(path)] = digest
        gradients.append(np.load(path,allow_pickle=False).astype(np.float64))
    gradients = np.stack(gradients)
    prediction = gradients@delta
    assert np.max(np.abs(prediction-report["methods"][0]["prediction"])) < 1e-12
    groups = {"all":np.arange(len(base))}
    for kind in ("family","layer"):
        keys = sorted({entry[kind] for entry in entries},key=str)
        for key in keys:
            groups[f"{kind}:{key}"] = np.concatenate([np.arange(e["start"],e["start"]+e["count"]) for e in entries if e[kind]==key])
    left = [i for i,label in enumerate(labels) if label["purpose"]=="referent"]
    right = [i for i,label in enumerate(labels) if label["purpose"] in ("referent","algorithm","implementation","code")]
    pair_rows, contributions = [],[]
    for group,indices in groups.items():
        block = gradients[:,indices]
        norms = np.linalg.norm(block,axis=1)
        dots = block[left]@block[right].T
        cosines = dots/(norms[left,None]*norms[None,right])
        for a,i in enumerate(left):
            for b,j in enumerate(right):
                pair_rows.append(dict(group=group,left=labels[i],right=labels[j],
                                      cosine=float(cosines[a,b]),dot=float(dots[a,b]),
                                      left_l2=float(norms[i]),right_l2=float(norms[j])))
        products = block*delta[indices]
        for i,label in enumerate(labels):
            contributions.append(dict(group=group,**label,
                                      predicted_delta=float(products[i].sum()),
                                      positive_coordinate_sum=float(products[i][products[i]>0].sum()),
                                      negative_coordinate_sum=float(products[i][products[i]<0].sum()),
                                      gradient_l2=float(norms[i]), edit_l2=float(np.linalg.norm(delta[indices]))))
    for kind in ("family:","layer:"):
        totals=np.array([sum(r["predicted_delta"] for r in contributions if r["group"].startswith(kind) and
                            all(r[k]==label[k] for k in ("objective","purpose","fold"))) for label in labels])
        assert np.max(np.abs(totals-prediction)) < 1e-12
    errors = data["finite"]["rows"]
    error_values = np.array([r["prediction_error"] for r in errors])
    result = dict(scope="All 76 original gradients, 209920 stratified sampled matrix coordinates; derivatives only, no causal family/layer attribution.",
                  model_calls=0, checkpoint_writes=0, candidate_generation=0,
                  source_sample_l2=float(np.linalg.norm(base.astype(np.float64))),
                  f01_edit_l2=float(np.linalg.norm(delta)), labels=labels,
                  cosine_rows=pair_rows, f01_contribution_rows=contributions,
                  all_76_finite_errors=errors,
                  error_summary=dict(mean_absolute=float(np.abs(error_values).mean()),
                                     median_absolute=float(np.median(np.abs(error_values))),
                                     maximum_absolute=float(np.abs(error_values).max()),
                                     predicted_positive_finite_negative=[r for r in errors if r["predicted_delta"]>0 and r["finite_delta"]<0]),
                  inputs_sha256=hashes, script_sha256=sha(Path(__file__)),
                  seconds=time.monotonic()-started)
    with args.output.open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write(chr(10))
    print(json.dumps(dict(cosine_rows=len(pair_rows),contribution_rows=len(contributions),
                         seconds=result["seconds"],error_summary=result["error_summary"])))


if __name__ == "__main__":
    main()
