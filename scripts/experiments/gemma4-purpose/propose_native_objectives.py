#!/usr/bin/env python3
"""Couple two matched native objectives; preview is metadata-only, never a model call."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import signal
import time
import sys

import numpy as np

import joint
from kairyu_bench.wsp.weights.checkpoint import Checkpoint

OBJECTIVES = ("correct-logprob", "margin")


def stamp(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def capture(path, pins, expected=None):
    path = path.resolve()
    before = stamp(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if stamp(path) != before or expected is not None and digest != expected:
        raise ValueError("input changed or SHA mismatch: " + str(path))
    record = (digest, before)
    if str(path) in pins and pins[str(path)] != record:
        raise ValueError("conflicting input pins")
    pins[str(path)] = record
    return raw


def document(path, pins, expected=None):
    return json.loads(capture(path, pins, expected))


def stable(checkpoint, pins):
    checkpoint.assert_stable()
    if any(stamp(Path(p)) != state or joint.common.sha(Path(p)) != digest
           for p, (digest, state) in pins.items()):
        raise ValueError("coupled proposal input changed")


def prepare(path):
    """Validate the whole checkpoint, cohorts and columns before reading gradients."""
    pins = {}
    plan = document(path, pins)
    if plan.get("schema_version") != 1 or plan.get("kind") != "native_coupled_objectives_v1":
        raise ValueError("explicit native coupled-objectives plan required")
    source = plan["checkpoint"]
    checkpoint = Checkpoint(source["path"])
    current = checkpoint.identity(hash_files=False)
    ref = source["attestation"]
    receipt = document(Path(ref["path"]), pins, ref["sha256"])
    identity = receipt.get("checkpoint", receipt)
    if (identity.get("identity_level") != "content_sha256" or identity.get("hash_files") is not True
            or identity.get("path") != str(checkpoint.path) or identity.get("sha256") != source["tree_sha256"]
            or len(identity.get("files", [])) != len(current["files"])):
        raise ValueError("complete checkpoint attestation differs")
    for old, now in zip(identity["files"], current["files"], strict=True):
        if (not re.fullmatch(r"[0-9a-f]{64}", str(old.get("sha256", "")))
                or any(old[k] != now[k] for k in ("path", "size", "stat"))):
            raise ValueError("checkpoint attested file inventory/stat changed")
    # Optional owner state is an opaque, explicitly pinned dependency, not a policy parser.
    if "parent_state" in plan:
        state = plan["parent_state"]
        capture(Path(state["path"]), pins, state["sha256"])
    case_ref = plan["cases"]
    cases_path = Path(case_ref["path"]).resolve()
    cases_raw = capture(cases_path, pins, case_ref["sha256"])
    cases = joint.common.read_cases(cases_path, content=cases_raw)
    labels = [(purpose, fold) for purpose, folds in cases.items() for fold in range(len(folds))]
    if len(labels) * 2 > 128:
        raise ValueError("at most 128 coupled constraints supported")
    native = {n: t for n, t in checkpoint.tensors.items()
              if n.startswith("model.language_model.") and len(t.shape) <= 1 and t.floating}
    if not native or any(t.dtype != "BF16" for t in native.values()):
        raise ValueError("complete finite BF16 native layout required")
    collections = {}
    for objective in OBJECTIVES:
        directory = Path(plan["collections"][objective]).resolve()
        summary = document(directory / "summary.json", pins)
        provenance = document(directory / "provenance.json", pins)
        coverage = document(directory / "coverage.json", pins)
        frames = document(directory / "frame-evidence.json", pins)
        if (summary.get("status") != "completed" or summary.get("candidate_selection") is not False
                or summary.get("source_values_unchanged") is not True
                or summary.get("mode") != "chat-public-answer" or provenance.get("mode") != summary["mode"]
                or summary.get("objective_kind") != objective or provenance.get("objective_kind") != objective
                or summary.get("answer_batching") != "serial" or provenance.get("answer_batching") != "serial"
                or provenance.get("virtual_unit_gains") is not False
                or provenance.get("max_cases_per_microbatch") != 1):
            raise ValueError("completed matched serial native objective required")
        if (provenance.get("checkpoint") != current
                or provenance.get("attestation_receipt_sha256") != ref["sha256"]
                or provenance["input_sha256"].get(str(cases_path)) != case_ref["sha256"]):
            raise ValueError("collection full checkpoint or cases identity differs")
        entries = coverage["tensors"]
        expected = sorted((n, list(t.shape), "torch.bfloat16", "native") for n, t in native.items())
        actual = sorted((e["name"], e["shape"], e["dtype"], e["kind"]) for e in entries)
        if (actual != expected or coverage.get("native_tensors") != len(native)
                or coverage.get("native_coordinates") != sum(t.numel for t in native.values())
                or coverage.get("virtual_coordinates") != 0 or coverage.get("virtual_tensors") != 0):
            raise ValueError("complete native tensor columns differ")
        if (frames.get("schema_version") != 2
                or frames.get("frame_protocol") != "gemma4-public-answer-v2-forced-empty-thought"
                or frames.get("forced_empty_thought") is not True):
            raise ValueError("explicit public-answer frame v2 required")
        records = summary["folds"]
        if [(r["purpose"], r["fold"]) for r in records] != labels:
            raise ValueError("complete ordered fold cohort differs")
        field, value = ("correct_logprobs", "correct_logprob") if objective == "correct-logprob" else ("margins", "margin")
        for record, (purpose, fold) in zip(records, labels, strict=True):
            ids = [row["id"] for row in cases[purpose][fold]]
            if (record["examples"] != len(ids) or [r["id"] for r in record[field]] != ids
                    or [r["id"] for r in frames["frames"][purpose][fold]] != ids
                    or not np.isfinite([r[value] for r in record[field]]).all()):
                raise ValueError("per-case cohort or finite objective differs")
        collections[objective] = {"path": directory, "summary": summary, "provenance": provenance,
                                  "frames": frames, "coverage": coverage}
    left, right = (collections[o] for o in OBJECTIVES)
    if left["frames"] != right["frames"] or left["coverage"] != right["coverage"]:
        raise ValueError("objective frame/columns differ")
    for key in ("threads", "seed", "max_cases_per_microbatch", "input_sha256", "input_stat", "calibration_scope"):
        if left["provenance"][key] != right["provenance"][key]:
            raise ValueError("objective collector input/runtime differs: " + key)
    for key in ("torch", "transformers", "weight_dtype", "attention", "gradients", "max_cases_per_microbatch"):
        if left["summary"]["runtime"].get(key) != right["summary"]["runtime"].get(key):
            raise ValueError("objective numerical runtime differs: " + key)
    methods = plan["candidates"]
    if not isinstance(methods, list) or not 1 <= len(methods) <= 2:
        raise ValueError("one or two fixed purpose sets required")
    names = set()
    expected_rows = [(o, p, f) for o in OBJECTIVES for p, f in labels]
    for method in methods:
        identifier = method["id"]
        if (not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", identifier)
                or identifier in names or identifier in {"report", "preview", "failure"}):
            raise ValueError("unique safe proposal IDs required")
        names.add(identifier)
        rows = method["constraints"]
        if ([(r["objective"], r["purpose"], r["fold"]) for r in rows] != expected_rows
                or any(type(r["minimum_delta"]) not in (int, float)
                       or not np.isfinite(r["minimum_delta"]) for r in rows)):
            raise ValueError("explicit complete ordered coupled bounds required")
        metric = method["cost_metric"]
        if (metric["type"] not in {"euclidean", "relative"}
                or metric["type"] == "relative" and (type(metric.get("floor")) not in (int, float)
                    or not np.isfinite(metric["floor"]) or metric["floor"] <= 0)):
            raise ValueError("explicit positive cost metric required")
        rounding = method["rounding"]
        if (set(rounding) != {"relative_radius", "absolute_radius", "rounds"}
                or any(type(rounding[k]) not in (int, float) or not np.isfinite(rounding[k]) or rounding[k] < 0
                       for k in ("relative_radius", "absolute_radius"))
                or rounding["relative_radius"] + rounding["absolute_radius"] <= 0
                or type(rounding["rounds"]) is not int or not 1 <= rounding["rounds"] <= 32):
            raise ValueError("explicit bounded rounding parameters required")
    for source_file in (Path(__file__), Path(joint.__file__), Path(joint.common.__file__),
                        Path(joint.finite_response.__file__), Path(joint.common.finite_replay.__file__),
                        Path(sys.modules[Checkpoint.__module__].__file__)):
        capture(source_file, pins)
    stable(checkpoint, pins)
    return checkpoint, plan, collections, labels, pins


def check_output(output, checkpoint, path, collections, pins):
    protected = [checkpoint.path, path.resolve(), *[c["path"] for c in collections.values()],
                 *[Path(p) for p in pins]]
    destination = output.resolve()
    if output.exists() or output.is_symlink() or any(
            destination == p or destination in p.parents or p in destination.parents for p in protected):
        raise ValueError("fresh output disjoint from all inputs required")
    return destination


def solve(path, output, *, write_spec=False, maximum_seconds=120):
    if type(maximum_seconds) is not int or not 1 <= maximum_seconds <= 600:
        raise ValueError("CPU time budget must be 1..600 seconds")
    checkpoint, plan, collections, labels, pins = prepare(path)
    destination = check_output(output, checkpoint, path, collections, pins)
    old_signal = signal.getsignal(signal.SIGALRM)
    def timeout(*unused):
        raise TimeoutError("native coupled CPU budget exceeded")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(maximum_seconds)
    started = time.monotonic()
    try:
        loaded = []
        for objective in OBJECTIVES:
            constraints = [{"purpose": p, "fold": f, "minimum_delta": 0.} for p, f in labels]
            data = joint.read_collection(collections[objective]["path"], {"constraints": constraints})
            base, g, _, layout, coords, _, hashes, _, _ = data
            if coords is not None:
                raise ValueError("native-only source required")
            for p, digest in hashes.items():
                capture(Path(p), pins, digest)
            # The original NPZ tensor shapes/keys must agree, not merely flattened lengths.
            with np.load(collections[objective]["path"] / "source-native-values.npz", allow_pickle=False) as arrays:
                if set(arrays.files) != {e["tensor"] for e in layout} or any(
                        arrays[e["tensor"]].shape != checkpoint.tensors[e["tensor"]].shape for e in layout):
                    raise ValueError("native source array shape mismatch")
            for record in collections[objective]["summary"]["folds"]:
                with np.load(collections[objective]["path"] / record["file"], allow_pickle=False) as arrays:
                    if set(arrays.files) != {e["tensor"] for e in layout} or any(
                            arrays[e["tensor"]].shape != checkpoint.tensors[e["tensor"]].shape for e in layout):
                        raise ValueError("native gradient array columns/shape mismatch")
            loaded.append((base, g, layout))
        base, _, layout = loaded[0]
        if loaded[1][2] != layout or not np.array_equal(
                base.astype(np.float32).view(np.uint32), loaded[1][0].astype(np.float32).view(np.uint32)):
            raise ValueError("objective native source bytes/columns differ")
        gradient = np.vstack([data[1] for data in loaded])
        methods, specs = [], []
        for method in plan["candidates"]:
            lower = np.asarray([r["minimum_delta"] for r in method["constraints"]], dtype=np.float64)
            metric = method["cost_metric"]
            cost = np.ones_like(base) if metric["type"] == "euclidean" else 1 / np.maximum(np.abs(base), metric["floor"])
            values, report = joint.quantized_proposal(base, gradient, lower, cost, **method["rounding"])
            edits, cursor = [], 0
            for entry in layout:
                count = entry["count"]
                indices = np.flatnonzero(values[cursor:cursor+count].astype(np.float64) != base[cursor:cursor+count])
                if indices.size:
                    edits.append({"tensor": entry["tensor"], "indices": indices.tolist(),
                                  "set_values": values[cursor+indices].tolist()})
                cursor += count
            changed = sum(len(e["indices"]) for e in edits)
            if changed > 1_000_000:
                raise ValueError("actual changed coordinates exceed one million; no silent trimming")
            report.update(id=method["id"], constraints=method["constraints"], cost_metric=metric,
                          changed_elements=changed, changed_tensors=len(edits),
                          represented_source_l2=float(np.linalg.norm(base)),
                          norm_scope="all selected native tensors; not the whole model",
                          spec_emitted=bool(write_spec and edits))
            methods.append(report)
            if write_spec and edits:
                specs.append({"id": method["id"], "edits": edits,
                              "base_tree_sha256": plan["checkpoint"]["tree_sha256"],
                              "source_checkpoint": str(checkpoint.path),
                              "hypothesis": "Coupled authored public-answer local-gradient proposal; finite response unmeasured."})
        stable(checkpoint, pins)
        result = {"schema_version": 1, "status": "local_proposals", "methods": methods,
                  "source_checkpoint": plan["checkpoint"], "parent_state": plan.get("parent_state"),
                  "source_sha256": {p: digest for p, (digest, _) in pins.items()},
                  "source_stat": {p: state for p, (_, state) in pins.items()},
                  "labels": [(o, p, f) for o in OBJECTIVES for p, f in labels],
                  "seconds": time.monotonic() - started, "fresh_checkpoint_payload_hash": False,
                  "model_calls": 0, "adopted": False,
                  "limits": ["Local BF16 predictions are not measured response or capability.",
                             "Explicit loss bounds are search inputs, not an adoption rule.",
                             "Forced empty thought under ON differs from free-thinking generation."]}
        destination.mkdir(parents=True)
        for name, document_value in [("report.json", result), *[(s["id"] + ".json", s) for s in specs]]:
            with (destination / name).open("x") as stream:
                json.dump(document_value, stream, indent=2, allow_nan=False)
                stream.write("\n")
        return result
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_signal)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="run the bounded CPU solver; default metadata-only preview")
    parser.add_argument("--write-spec", action="store_true", help="with --run, publish exact BF16 native specs")
    parser.add_argument("--maximum-seconds", type=int, default=120)
    args = parser.parse_args(argv)
    if not 1 <= args.maximum_seconds <= 600 or args.write_spec and not args.run:
        parser.error("require 1..600 seconds; --write-spec requires --run")
    if args.run:
        result = solve(args.plan, args.output, write_spec=args.write_spec, maximum_seconds=args.maximum_seconds)
        print(json.dumps({"status": result["status"], "methods": len(result["methods"]), "model_calls": 0}))
    else:
        checkpoint, plan, collections, labels, pins = prepare(args.plan)
        check_output(args.output, checkpoint, args.plan, collections, pins)
        print(json.dumps({"status": "metadata_preview", "checkpoint": str(checkpoint.path),
                          "folds_per_objective": len(labels), "candidates": len(plan["candidates"]),
                          "gradient_payload_reads": 0, "model_calls": 0}))


if __name__ == "__main__":
    main()
