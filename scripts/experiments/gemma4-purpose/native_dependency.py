"""Read-only full-native gradient geometry after a completed, restored collector owner.

Reuses joint.read_collection only as a loader. No optimizer, model, edit or API call.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

import joint
from kairyu_bench.wsp.weights.checkpoint import Checkpoint


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bf16_gaps(values):
    """Distances to adjacent BF16 values, preserving exponent-boundary asymmetry."""
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    bits = x.view(np.uint32)
    require(np.isfinite(x).all() and not np.any(bits & 65535), "finite exact BF16 values required")
    word = bits >> 16
    up = np.where(x > 0, word + 1, word - 1)
    down = np.where(x > 0, word - 1, word + 1)
    up = np.where(x == 0, 1, up).astype(np.uint32)
    down = np.where(x == 0, 0x8001, down).astype(np.uint32)
    upper = (up << 16).view(np.float32).astype(np.float64)
    lower = (down << 16).view(np.float32).astype(np.float64)
    return upper - x, x - lower


def ratios(numerator, denominator):
    result = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                       where=denominator != 0)
    return np.where(denominator != 0, result, np.nan)


def nullable(value):
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array) if np.isfinite(array) else None
    return [nullable(v) for v in array]


def geometry(gram, global_gram):
    norm = np.sqrt(np.maximum(np.diag(gram), 0))
    global_norm = np.sqrt(np.maximum(np.diag(global_gram), 0))
    return {
        "gradient_l2": norm.tolist(),
        "gradient_energy_fraction": nullable(ratios(np.diag(gram), np.diag(global_gram))),
        "gram": gram.tolist(),
        "cosine": nullable(ratios(gram, norm[:, None] * norm[None, :])),
        "contribution_to_global_cosine": nullable(
            ratios(gram, global_norm[:, None] * global_norm[None, :])),
    }


def statistics(values):
    v = np.asarray(values, dtype=np.float64)
    return {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())}


def source_stats(values):
    up, down = bf16_gaps(values)
    return {
        "coordinates": int(values.size), "zero_coordinates": int(np.count_nonzero(values == 0)),
        "l2": float(np.linalg.norm(values)), "rms": float(np.sqrt(np.mean(values ** 2))),
        "absolute_value": statistics(np.abs(values)),
        "bf16_next_up_gap": statistics(up), "bf16_next_down_gap": statistics(down),
        "spacing_is_a_representation_fact_not_an_edit_or_safe_radius": True,
    }


def analyze(root, closure_sha):
    pins = {}

    def capture(path, expected=None):
        path = Path(path).resolve()
        digest = joint.common.sha(path)
        require(expected is None or digest == expected, f"SHA mismatch: {path}")
        require(str(path) not in pins or pins[str(path)] == digest, f"changed input: {path}")
        pins[str(path)] = digest
        return path

    def document(path, expected=None):
        return json.loads(capture(path, expected).read_text())

    closure = document(root / "measurement-closure.json", closure_sha)
    complete = document(closure["completion"]["path"], closure["completion"]["sha256"])
    plan = document(closure["plan"]["path"], closure["plan"]["sha256"])
    require(closure["status"] == complete["status"] == "completed", "complete owner required")
    require(all(complete.get(k) is True for k in [
        "restored", "protected_exact", "collections_completed", "collector_client_finished"
    ]), "owner restoration incomplete")
    after = closure["after"]
    require(after["owner_pid_absent"] and not after["global_lock_rows"]
            and after["protected_exact"], "owner closure incomplete")
    require(after["target"]["id"] == plan["target"]["id"]
            and after["target"]["running"] and after["target"]["health"] == "healthy",
            "same target healthy restoration required")
    require(all(v == 0 for vs in after["target_queues"]["queues"].values() for v in vs),
            "nonidle closure")
    require(not closure["failure_files"], "collector failure recorded; do not analyze as complete")
    for item in closure["raw_inventory"]:
        capture(item["path"], item["sha256"])
    for path, digest in plan["source_pins"].items():
        capture(path, digest)
    for key in ["cases", "frame", "attestation", "native_behavior_result"]:
        capture(plan[key]["path"], plan[key]["sha256"])
    cases = joint.common.read_cases(Path(plan["cases"]["path"]))
    labels = [(p, f, rows[0]["id"]) for p, folds in cases.items() for f, rows in enumerate(folds)]
    require(len(labels) == 12 and all(len(rows) == 1 for folds in cases.values() for rows in folds),
            "all 12 singleton training folds required")
    expected_ids = [f"behavior-{p}{f + 1}" for p in "ABCDEF" for f in range(2)]
    require([i for _, _, i in labels] == expected_ids, "full behavior training inventory mismatch")
    behavior = document(plan["native_behavior_result"]["path"])
    for item in behavior["artifacts"]:
        capture(item["path"], item["sha256"])
    observed = {row["id"]: row["scored_result"] for row in behavior["rows"]}
    require(set(observed) == set(expected_ids), "behavior join mismatch")
    checkpoint = Checkpoint(plan["checkpoint"])
    current = checkpoint.identity(hash_files=False)
    attestation = document(plan["attestation"]["path"])
    identity = attestation.get("checkpoint", attestation)
    require(identity["identity_level"] == "content_sha256" and identity["hash_files"] is True
            and identity["sha256"] == closure["checkpoint_tree_sha256"]
            and identity["path"] == current["path"], "checkpoint attestation mismatch")
    require(len(identity["files"]) == len(current["files"]), "checkpoint inventory mismatch")
    for old, now in zip(identity["files"], current["files"], strict=True):
        require(all(old[k] == now[k] for k in ["path", "size", "stat"]), "checkpoint stat changed")
    native = {n: t for n, t in checkpoint.tensors.items()
              if n.startswith("model.language_model.") and len(t.shape) <= 1 and t.floating}
    require(len(native) == 421 and all(t.dtype == "BF16" for t in native.values()),
            "all 421 BF16 native tensors required")
    expected = sorted((n, list(t.shape), "torch.bfloat16", "native") for n, t in native.items())
    read_plan = {"constraints": [{"purpose": p, "fold": f, "minimum_delta": 0.0}
                                 for p, f, _ in labels]}
    collections, base, layout, row_labels, baselines = [], None, None, [], {}
    unused_rows = []
    policies = []
    for objective in ["correct-logprob", "margin"]:
        directory = root / "work" / objective
        summary = document(directory / "summary.json", closure["objectives"][objective]["summary"]["sha256"])
        coverage = document(directory / "coverage.json")
        provenance = document(directory / "provenance.json")
        capture(directory / "frame-evidence.json", plan["frame"]["sha256"])
        require(summary["objective_kind"] == provenance["objective_kind"] == objective,
                "objective label mismatch")
        require(summary["answer_batching"] == provenance["answer_batching"] == "serial"
                and provenance["max_cases_per_microbatch"] == 1
                and provenance["virtual_unit_gains"] is False, "collection policy mismatch")
        require(provenance["checkpoint"] == current
                and provenance["attestation_receipt_sha256"] == plan["attestation"]["sha256"],
                "collection checkpoint differs")
        for path, digest in provenance["input_sha256"].items():
            capture(path, digest)
        require(provenance["input_sha256"][plan["cases"]["path"]] == plan["cases"]["sha256"],
                "collection cases differ")
        actual = sorted((t["name"], t["shape"], t["dtype"], t["kind"]) for t in coverage["tensors"])
        require(actual == expected and coverage["native_tensors"] == 421
                and coverage["virtual_tensors"] == coverage["virtual_coordinates"] == 0,
                "complete native coverage mismatch")
        policies.append(provenance["observation_policy"])
        data = joint.read_collection(directory, read_plan)
        source, gradient, _, this_layout, coordinates, _, hashes, mode, _ = data
        require(coordinates is None and mode == "chat-public-answer", "native chat observations required")
        for path, digest in hashes.items():
            capture(path, digest)
        require(source.size == coverage["native_coordinates"], "coordinate count mismatch")
        if base is None:
            base, layout = source, this_layout
        else:
            require(np.array_equal(source.view(np.uint64), base.view(np.uint64))
                    and this_layout == layout, "objective source values/layout differ")
        records = {(r["purpose"], r["fold"]): r for r in summary["folds"]}
        require(len(records) == len(summary["folds"]) == 12, "all folds required")
        for p, f, case_id in labels:
            record = records[p, f]
            require(record["examples"] == 1, "singleton per-case collection required")
            unused_rows.append(list(record["unused_tensors"]))
            with np.load(directory / record["file"], allow_pickle=False) as arrays:
                require(set(arrays.files) == set(native), "gradient contains extra/missing tensors")
                require(all(arrays[n].size == native[n].numel for n in native), "gradient shape mismatch")
            field, key = ("correct_logprobs", "correct_logprob") if objective == "correct-logprob" else ("margins", "margin")
            require(len(record[field]) == 1 and record[field][0]["id"] == case_id, "case ID mismatch")
            value = record[field][0][key]
            require(np.isfinite(value), "nonfinite baseline")
            baselines.setdefault(case_id, {})[key] = value
            row_labels.append({"objective": objective, "purpose": p, "fold": f, "id": case_id})
        collections.append(gradient)
    require(policies[0] == policies[1], "objective runtime policies differ")
    gradients = np.concatenate(collections)
    gram = gradients @ gradients.T
    family_slices, layer_slices = defaultdict(list), defaultdict(list)
    family_grams, layer_grams = {}, {}
    tensors, offset = [], 0
    for entry in layout:
        name, count = entry["tensor"], entry["count"]
        sl = slice(offset, offset + count)
        values = base[sl]
        actual = checkpoint.read(name).reshape(-1).astype(np.float64)
        require(np.array_equal(actual.view(np.uint64), values.view(np.uint64)),
                f"source native words differ: {name}")
        match = re.fullmatch(r"model.language_model.layers.(\d+).(.+)", name)
        layer, family = (match[1], match[2]) if match else ("global", name.removeprefix("model.language_model."))
        part = gradients[:, sl]
        tensor_gram = part @ part.T
        for key, slices, grams in [(family, family_slices, family_grams), (layer, layer_slices, layer_grams)]:
            slices[key].append(sl)
            if key not in grams:
                grams[key] = np.zeros_like(gram)
            grams[key] += tensor_gram
        tensors.append({"tensor": name, "shape": list(native[name].shape), "layer": layer, "family": family,
                        "source": source_stats(values),
                        "gradient_l2": np.sqrt(np.maximum(np.diag(tensor_gram), 0)).tolist(),
                        "gradient_zero_coordinates": np.count_nonzero(part == 0, axis=1).tolist(),
                        "gradient_energy_fraction": nullable(ratios(np.diag(tensor_gram), np.diag(gram)))})
        offset += count
    require(offset == base.size and np.allclose(sum(family_grams.values()), gram, rtol=1e-12, atol=1e-12)
            and np.allclose(sum(layer_grams.values()), gram, rtol=1e-12, atol=1e-12),
            "group Gram decomposition does not match whole native set")
    groups = {}
    for kind, slices, grams in [("families", family_slices, family_grams), ("layers", layer_slices, layer_grams)]:
        groups[kind] = {key: {"source": source_stats(np.concatenate([base[sl] for sl in blocks])),
                              **geometry(grams[key], gram),
                              "gradient_zero_coordinates": sum(np.count_nonzero(gradients[:, sl] == 0, axis=1)
                                                               for sl in blocks).tolist()}
                       for key, blocks in slices.items()}
    checkpoint.assert_stable()
    # Include the imported CPU reader/package code, independent of the collector source snapshot.
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).suffix == ".py" and str(Path(__file__).resolve().parents[3]) in path:
            capture(path)
    for path, digest in pins.items():
        require(joint.common.sha(Path(path)) == digest, f"input changed during analysis: {path}")
    rows = []
    for _, _, case_id in labels:
        scored = observed[case_id]
        rows.append({"id": case_id, **baselines[case_id],
                     "native_status": scored["status"], "native_passed": scored["passed"],
                     "native_tests_passed": scored["tests_passed"], "native_tests_total": scored["tests_total"]})
    return {
        "schema_version": 1, "status": "completed", "collector_root": str(root), "input_sha256": pins,
        "checkpoint_tree_sha256": identity["sha256"], "checkpoint_stat_continuity": current,
        "payload_rehashed_here": False, "all_native_source_words_checked": True,
        "source_scope": "All 421 language-model native scalar/vector tensors; not all 31B matrix weights",
        "native_tensors": len(layout), "native_coordinates": len(base), "gradient_rows": len(gradients),
        "case_rows": rows, "row_labels": row_labels, "source": source_stats(base),
        "global": {**geometry(gram, gram),
                   "gradient_zero_coordinates": np.count_nonzero(gradients == 0, axis=1).tolist()},
        "unused_tensors_by_row": unused_rows, **groups, "tensors": tensors,
        "collection_policy": policies[0], "frame_evidence": document(plan["frame"]["path"]),
        "owner_seconds": complete["elapsed_seconds"],
        "collector_forward_counts": {o: closure["objectives"][o]["forward_count"] for o in closure["objectives"]},
        "collector_seconds_excluding_load": {o: closure["objectives"][o]["seconds_excluding_model_load"]
                                            for o in closure["objectives"]},
        "limits": [
            "Euclidean derivatives use raw parameter coordinates; cross-family units and parameterization differ.",
            "Gradient norm divided by source norm is not an edit size or safety bound; no such ratio is inferred.",
            "Source norms and adjacent BF16 gaps do not prove behavioral importance or finite response.",
            "Unused tensors and exact zero gradients are separate; neither demonstrates an absent ability.",
            "Forced-empty-thought teacher forcing differs from ordinary thinking generation.",
            "Positive correct-versus-authored-foil margin is only pair ranking, not argmax/free-generation correctness.",
            "F2 actual generated failure is off-by-one; the fixed foil instead removes zeros.",
            "All group/case values retained without support, layer, candidate, objective-target or radius selection.",
        ],
        "solver_calls": 0, "model_calls": 0, "weight_edits": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector-root", type=Path, required=True)
    parser.add_argument("--closure-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        report = analyze(args.collector_root.resolve(), args.closure_sha256)
        report["analysis_seconds"] = time.time() - started
        joint.common.save(args.output / "report.json", report)
        print(json.dumps({"status": report["status"], "native_tensors": report["native_tensors"],
                          "native_coordinates": report["native_coordinates"], "analysis_seconds": report["analysis_seconds"]}))
    except Exception as error:
        joint.common.save(args.output / "failure.json", {"type": type(error).__name__, "message": str(error),
                                                       "analysis_seconds": time.time() - started})
        raise


if __name__ == "__main__":
    main()
