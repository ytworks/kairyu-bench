#!/usr/bin/env python3
"""Profile saved continuous donor gate gradients; no model or benchmark calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

FAMILIES = {"q", "k", "v", "o", "gate", "up", "down"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(summary, costs, gradients):
    entries, folds = summary["coverage"]["entries"], summary["folds"]
    n = len(entries)
    if (not 1 <= n <= 4096 or summary["coverage"]["direction_coordinates"] != n
            or len({e["tensor"] for e in entries}) != n
            or [e["coordinate"] for e in entries] != list(range(n))
            or any(e["family"] not in FAMILIES or type(e["layer"]) is not int or e["layer"] < 0 for e in entries)):
        raise ValueError("invalid coordinate inventory")
    if (costs.shape != (n,) or gradients.shape != (len(folds), n) or not 1 <= len(folds) <= 128
            or not np.isfinite(costs).all() or not np.isfinite(gradients).all() or np.any(costs < 0)
            or len({(f["purpose"], f["fold"]) for f in folds}) != len(folds)):
        raise ValueError("invalid cost/gradient arrays or duplicate folds")
    if np.any(gradients[:, costs == 0] != 0):
        raise ValueError("nonzero derivative at a zero-cost direction")
    expected_costs = np.array([e["unit_gate_weight_l2"] for e in entries])
    if not np.array_equal(costs, expected_costs):
        raise ValueError("stored costs differ from coverage")
    normalized = np.divide(gradients, costs[None, :], out=np.zeros_like(gradients), where=costs[None, :] > 0)
    groups = {}
    for i, entry in enumerate(entries):
        for key in (f"layer:{entry['layer']}", f"family:{entry['family']}",
                    "mechanism:MLP" if entry["family"] in {"gate", "up", "down"} else "mechanism:attention"):
            groups.setdefault(key, []).append(i)
    fold_rows = []
    for index, fold in enumerate(folds):
        g = gradients[index]
        profiles = {}
        for name, indices in groups.items():
            cost = float(np.linalg.norm(costs[indices]))
            derivative = float(g[indices].sum())
            profiles[name] = {"directions": len(indices), "uniform_gate_derivative": derivative,
                              "uniform_direction_weight_l2": cost,
                              "uniform_derivative_per_unit_weight_l2": derivative / cost if cost else None,
                              "positive": int((g[indices] > 0).sum()), "negative": int((g[indices] < 0).sum())}
        fold_rows.append({"purpose": fold["purpose"], "fold": fold["fold"],
                          "uniform_all_gate_derivative": float(g.sum()),
                          "positive": int((g > 0).sum()), "negative": int((g < 0).sum()),
                          "zero": int((g == 0).sum()), "gradient_l2": float(np.linalg.norm(g)),
                          "weight_cost_normalized_gradient_l2": float(np.linalg.norm(normalized[index])),
                          "groups": profiles})
    def sign_counts(values):
        positive, negative = np.all(values > 0, axis=0), np.all(values < 0, axis=0)
        mixed = np.any(values > 0, axis=0) & np.any(values < 0, axis=0)
        return {"all_positive": int(positive.sum()), "all_negative": int(negative.sum()),
                "mixed_sign": int(mixed.sum()), "all_zero": int(np.all(values == 0, axis=0).sum()),
                "zero_or_one_sided": int((~(positive | negative | mixed | np.all(values == 0, axis=0))).sum())}
    purposes = {}
    for name in sorted({fold["purpose"] for fold in folds}):
        indices = [i for i, fold in enumerate(folds) if fold["purpose"] == name]
        purposes[name] = {"fold_count": len(indices), **sign_counts(gradients[indices])}
    cosine = []
    norms = np.linalg.norm(normalized, axis=1)
    for i in range(len(folds)):
        cosine.append([float(normalized[i] @ normalized[j] / (norms[i] * norms[j])) if norms[i] and norms[j] else None
                       for j in range(len(folds))])
    common_positive = np.all(gradients > 0, axis=0)
    order = sorted(np.flatnonzero(common_positive), key=lambda i: float(normalized[:, i].min()), reverse=True)
    strongest = [{"tensor": entries[i]["tensor"], "unit_weight_l2": float(costs[i]),
                  "derivatives": gradients[:, i].tolist(), "per_unit_weight_l2": normalized[:, i].tolist(),
                  "worst_fold_per_unit_weight_l2": float(normalized[:, i].min())} for i in order[:20]]
    return {"schema_version": 1, "status": "completed", "mode": summary["mode"],
            "objective": summary.get("objective", "margin"),
            "directions": n, "folds": fold_rows, "purposes": purposes,
            "uniform_all_direction_weight_l2": float(np.linalg.norm(costs)),
            "all_fold_signs": sign_counts(gradients), "all_positive_ranked_by_worst_normalized_fold": strongest,
            "fold_order": [{"purpose": f["purpose"], "fold": f["fold"]} for f in folds],
            "weight_cost_normalized_gradient_cosine": cosine,
            "formulas": {"normalized_coordinate": "g_i / ||2 B_i A_i||_F",
                         "uniform_group": "sum(g_i) / sqrt(sum(cost_i^2)); projection tensors are disjoint",
                         "cosine": "cosine between g/cost vectors, not between activation values",
                         "ranking": "descriptive minimum over all recorded fold derivatives per unit weight norm; no uncertainty estimate"},
            "candidate_selected": False, "model_calls": 0, "benchmark_calls": 0,
            "limits": ["Continuous zero-gate local sensitivities, not BF16 finite outcomes or capability scores.",
                       "Common signs across authored folds do not establish statistical significance or generalization.",
                       "Group magnitude depends on objective and direction cost; it is not a causal ability localization.",
                       "Do not mix raw and chat modes, different frames, sources or computation conventions."]}


def analyze(collection, *, expected_objective="margin"):
    collection = collection.resolve()
    files = [collection / name for name in ("summary.json", "coverage.json", "provenance.json", "unit-gate-weight-l2.npy")]
    if any(p.stat().st_size > 16 * 1024**2 for p in files):
        raise ValueError("collection metadata/costs exceed bounded input size")
    hashes = {str(p): sha(p) for p in files}
    summary, coverage, provenance = [json.loads((collection / name).read_text())
                                    for name in ("summary.json", "coverage.json", "provenance.json")]
    objective = summary.get("objective", "margin")
    if (expected_objective not in {"margin", "correct-logprob"} or objective != expected_objective
            or provenance.get("objective", "margin") != objective):
        raise ValueError("collection objective differs from explicit expected objective")
    checks = summary.get("zero_alpha_checks", {})
    difference = checks.get("max_absolute_objective_difference", checks.get("max_absolute_margin_difference", float("inf")))
    if (not np.isfinite(difference)
            or not np.isfinite(checks.get("tolerance", float("nan")))
            or checks.get("tolerance", -1) < 0
            or summary.get("status") != "completed" or not summary.get("source_values_unchanged")
            or summary.get("mode") != provenance.get("mode") or summary.get("coverage") != coverage
            or not checks.get("every_microbatch_compared")
            or difference > checks.get("tolerance", 0)):
        raise ValueError("completed source-preserving, mode-matched zero-gate collection required")
    n = len(coverage["entries"])
    if not 1 <= n <= 4096 or not 1 <= len(summary["folds"]) <= 128:
        raise ValueError("unsupported coordinate/fold count")
    arrays = []
    for fold in summary["folds"]:
        path = (collection / fold["file"]).resolve()
        if not path.is_relative_to(collection) or path.stat().st_size > 16 * 1024**2:
            raise ValueError("gradient must be a bounded local array")
        hashes[str(path)] = sha(path)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != (n,) or array.dtype.kind != "f":
            raise ValueError("gradient must match coordinate inventory")
        arrays.append(array.astype(np.float64))
    costs = np.load(collection / "unit-gate-weight-l2.npy", mmap_mode="r", allow_pickle=False)
    if costs.shape != (n,) or costs.dtype.kind != "f":
        raise ValueError("costs must match coordinate inventory")
    costs = costs.astype(np.float64)
    result = profile(summary, costs, np.stack(arrays))
    if any(sha(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError("collection changed during profiling")
    result["input_sha256"] = hashes
    result["implementation_sha256"] = sha(Path(__file__))
    return result, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=("margin", "correct-logprob"), default="margin")
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink() or args.output.resolve().is_relative_to(args.collection.resolve()):
            raise ValueError("output must be fresh and outside the collection")
        result, provenance = analyze(args.collection, expected_objective=args.objective)
        protected = [provenance.get("checkpoint", {}).get("path"), provenance.get("donor", {}).get("identity", {}).get("path")]
        if any(args.output.resolve().is_relative_to(Path(p).resolve()) for p in protected if p):
            raise ValueError("output overlaps a checkpoint input")
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"status": result["status"], "directions": result["directions"], "folds": len(result["folds"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
