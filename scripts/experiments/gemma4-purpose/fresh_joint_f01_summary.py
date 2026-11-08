#!/usr/bin/env python3
"""Summarize all fixed F01 finite constraints and all 120 numeric case margins."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--revision", required=True, type=int)
    parser.add_argument("--candidate-id", default="fresh-code-joint-f01")
    args = parser.parse_args()
    root = args.root
    replay = root / f"finite-gpu6-v{args.revision}" / "work/replay"
    paths = {name: replay / f"{name}.json" for name in ("base", args.candidate_id, "restored_base", "summary")}
    paths.update(plan=root / "prepared/plan.json", proposal=root / "proposal/report.json")
    data = {name: read(path) for name, path in paths.items()}
    base, candidate, restored, summary = (data[k] for k in ("base", args.candidate_id, "restored_base", "summary"))
    assert summary["status"] == "completed" and len(summary["candidates"]) == 1
    assert summary["base_restoration"]["matched"] and summary["base_restoration"]["storage_bytes_exact"]
    assert base["values"] == restored["values"] and base["correct_nll"] == restored["correct_nll"]
    finite_labels = [f"{purpose}-{fold}" for purpose, folds in base["values"].items() for fold in range(len(folds))]
    assert finite_labels == summary["labels"]
    deltas = summary["candidates"][0]
    prediction = data["proposal"]["methods"][0]["prediction"]
    constraints = data["plan"]["candidates"][0]["constraints"]
    assert len(prediction) == len(constraints) == 76
    rows = []
    for label, constraint, predicted in zip(data["proposal"]["labels"], constraints, prediction, strict=True):
        objective, purpose, fold = label
        assert label == [constraint[k] for k in ("objective", "purpose", "fold")]
        index = finite_labels.index(f"{purpose}-{fold}")
        actual = (-deltas["correct_nll_delta"][index] if objective == "correct-logprob"
                  else deltas["margin_delta"][index])
        rows.append(dict(objective=objective, purpose=purpose, fold=fold,
                         minimum_delta=constraint["minimum_delta"], predicted_delta=predicted,
                         finite_delta=actual, prediction_error=actual-predicted,
                         shortfall=max(0.0, constraint["minimum_delta"]-actual),
                         met=actual >= constraint["minimum_delta"]))
    cases = []
    for purpose, folds in base["values"].items():
        for fold, values in enumerate(folds):
            candidate_values = candidate["values"][purpose][fold]
            assert len(values) == len(candidate_values)
            index = finite_labels.index(f"{purpose}-{fold}")
            differences = [b-a for a, b in zip(values, candidate_values, strict=True)]
            assert abs(sum(differences)/len(values)-deltas["margin_delta"][index]) < 1e-12
            for case_index, (before, after) in enumerate(zip(values, candidate_values, strict=True)):
                cases.append(dict(purpose=purpose, fold=fold, case_index=case_index,
                                  base_margin=before, candidate_margin=after,
                                  margin_delta=after-before))
    assert len(cases) == 120
    retention_misses = [r for r in rows if r["minimum_delta"] < 0 and not r["met"]]
    target_purposes = {r["purpose"] for r in rows if r["minimum_delta"] > 0}
    primary = [r for r in rows if r["purpose"] in target_purposes]
    target_failures = [r for r in primary if (r["finite_delta"] < 0 if r["purpose"] == "code" and r["objective"] == "margin" else r["finite_delta"] <= 0)]
    output = dict(status="completed", candidate=args.candidate_id,
                  advance_gate=dict(target_purposes=sorted(target_purposes), target_sign_failures=target_failures,
                                    retention_failures=retention_misses,
                                    base_restoration_exact=summary["base_restoration"]["matched"],
                                    passed=not target_failures and not retention_misses),
                  scope="Authored proxy measurements, not WSP ability scores; case-level values are margins only.",
                  constraints=len(rows), met=sum(r["met"] for r in rows),
                  missed=sum(not r["met"] for r in rows),
                  retention_misses=retention_misses,
                  primary_sign_reversals=[r for r in primary if r["finite_delta"] <= 0],
                  case_count=len(cases), case_margin_gains=sum(c["margin_delta"]>0 for c in cases),
                  case_margin_losses=sum(c["margin_delta"]<0 for c in cases),
                  case_margin_ties=sum(c["margin_delta"]==0 for c in cases),
                  seconds={k:data[k]["seconds"] for k in ("base",args.candidate_id,"restored_base")},
                  base_restoration=summary["base_restoration"], rows=rows, cases=cases,
                  inputs={k:dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()},
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target = root / f"finite-gpu6-v{args.revision}" / "numeric-analysis.json"
    with target.open("x") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k:output[k] for k in ("constraints","met","missed","retention_misses","primary_sign_reversals","case_count","case_margin_gains","case_margin_losses","seconds")}))


if __name__ == "__main__":
    main()
