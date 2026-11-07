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
import matrix as matrix_recipe
import matrix_full_units as full_matrix_recipe
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



def matrix_families(plan):
    """Explicit v2 scope; omission preserves historical O/MLP behavior."""
    values = plan.get("matrix_families", ["o", "gate", "up", "down"])
    if (not isinstance(values, list) or not values
            or any(type(value) is not str or value not in {"q", "k", "v", "o", "gate", "up", "down"}
                   for value in values) or len(set(values)) != len(values)):
        raise ValueError("matrix_families must be a nonempty unique supported-family allowlist")
    return values




def matrix_names(plan, available):
    """Optional exact observed scope; selected MLP groups remain complete."""
    allowed = set(matrix_families(plan))
    available = set(available)
    eligible = {name for name in available if matrix_recipe.PATTERN.fullmatch(name)[3] in allowed}
    if "matrix_tensors" not in plan:
        return eligible
    values = plan["matrix_tensors"]
    if (not isinstance(values, list) or not values
            or any(type(value) is not str or value not in eligible for value in values)
            or len(set(values)) != len(values)):
        raise ValueError("matrix_tensors must be a nonempty unique observed eligible-matrix allowlist")
    names = set(values)
    for name in names:
        match = matrix_recipe.PATTERN.fullmatch(name)
        if match[3] in {"gate", "up", "down"}:
            stem = name.rsplit(".", 2)[0]
            group = {stem + "." + kind + "_proj.weight" for kind in ("gate", "up", "down")}
            if not group <= names:
                raise ValueError("matrix_tensors requires complete gate/up/down tensor groups")
    return names


def matrix_column_indices(plan, layout):
    """Share one scope between the solve and normalized target construction."""
    names = matrix_names(plan, {entry["tensor"] for entry in layout})
    active, selected, cursor = [], [], 0
    for entry in layout:
        count = entry["count"]
        if entry["tensor"] in names:
            active.extend(range(cursor, cursor + count))
            selected.append(entry)
        cursor += count
    if not active:
        raise ValueError("explicit eligible matrix scope has no observed columns")
    return np.asarray(active, dtype=np.int64), selected

def native_names(plan, available):
    """Exact-name opt-in; whole native observation validation remains mandatory."""
    if "native_tensors" not in plan:
        return set(available)
    values = plan["native_tensors"]
    if (not isinstance(values, list) or not values
            or any(type(value) is not str or value not in available for value in values)
            or len(set(values)) != len(values)):
        raise ValueError("native_tensors must be a nonempty unique observed-native allowlist")
    if plan.get("eligible_blocks") == ["matrix"]:
        raise ValueError("native_tensors requires eligible native columns")
    return set(values)


def select_native_columns(plan, base, gradient, layout):
    """Shared column mask for solving and normalized-target preparation."""
    names = native_names(plan, {entry["tensor"] for entry in layout})
    if "native_tensors" not in plan:
        return base, gradient, layout
    active, selected, cursor = [], [], 0
    for entry in layout:
        count = entry["count"]
        if entry["tensor"] in names:
            active.extend(range(cursor, cursor + count))
            selected.append(entry)
        cursor += count
    indices = np.asarray(active, dtype=np.int64)
    return base[indices], gradient[:, indices], selected


def observed_blocks(plan):
    """An omitted field preserves the historical complete observation contract."""
    mixed = plan.get("schema_version") == 2
    default = ["native", "matrix"] if mixed else ["native"]
    blocks = plan.get("observed_blocks", default)
    if "observed_blocks" in plan and (not mixed or blocks not in (["native", "matrix"], ["matrix"])):
        raise ValueError("observed_blocks requires explicit v2 native+matrix or matrix only")
    if blocks == ["matrix"] and (plan.get("eligible_blocks") != ["matrix"]
            or "native_tensors" in plan or set(plan["collections"]) != {"matrix"}):
        raise ValueError("matrix-only observations require frozen native and only matrix collections")
    return blocks


def expected_mlp_samples(checkpoint, collection):
    if collection["coverage"].get("sampling_mode") == "mlp-full-units":
        return full_matrix_recipe.full_unit_samples(checkpoint.tensors, collection["coverage"]["unit_support"])
    return matrix_recipe.mlp_triplet_samples(checkpoint.tensors, collection["provenance"]["seed"])


def full_unit_declaration(collections):
    collection = collections.get("matrix:" + OBJECTIVES[0])
    if collection is None or collection["coverage"].get("sampling_mode") != "mlp-full-units":
        return None
    coverage = collection["coverage"]
    return dict(schema_version=1, support_plan_sha256=coverage["unit_support_sha256"],
                layer_prefix="model.language_model.layers.", units=coverage["unit_support"]["units"])


def prepare(path):
    """Validate the whole checkpoint, cohorts and columns before reading gradients."""
    pins = {}
    plan = document(path, pins)
    mixed = plan.get("schema_version") == 2 and plan.get("kind") == "native_matrix_coupled_objectives_v2"
    if not mixed and (plan.get("schema_version") != 1 or plan.get("kind") != "native_coupled_objectives_v1"):
        raise ValueError("explicit supported coupled-objectives plan required")
    if ("matrix_families" in plan or "matrix_tensors" in plan) and not mixed:
        raise ValueError("matrix family eligibility requires the v2 plan")
    if mixed:
        matrix_families(plan)
    if "eligible_blocks" in plan and not mixed:
        raise ValueError("explicit block eligibility requires the v2 plan")
    if mixed and plan.get("eligible_blocks", ["native", "matrix"]) not in (["native", "matrix"], ["matrix"]):
        raise ValueError("eligible_blocks must be native+matrix or matrix only")
    families = observed_blocks(plan)
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
    native_names(plan, native)
    matrices = {n: t for n, t in checkpoint.tensors.items() if matrix_recipe.PATTERN.fullmatch(n)}
    if mixed:
        matrix_names(plan, matrices)
    collections = {}
    for family in families:
        for objective in OBJECTIVES:
            key = objective if family == "native" else "matrix:" + objective
            refs = plan["collections"][family] if mixed else plan["collections"]
            directory = Path(refs[objective]).resolve()
            summary = document(directory / "summary.json", pins)
            provenance = document(directory / "provenance.json", pins)
            coverage = document(directory / "coverage.json", pins)
            frames = document(directory / "frame-evidence.json", pins)
            if (summary.get("status") != "completed" or summary.get("candidate_selection") is not False
                    or summary.get("source_values_unchanged") is not True
                    or summary.get("mode") != "chat-public-answer" or provenance.get("mode") != summary["mode"]
                    or summary.get("objective_kind") != objective or provenance.get("objective_kind") != objective
                    or summary.get("answer_batching") != "serial" or provenance.get("answer_batching") != "serial"
                    or provenance.get("max_cases_per_microbatch") != 1):
                raise ValueError("completed matched serial objective required")
            runtime = summary["runtime"]
            if ("observation_policy" in provenance) != ("observation_policy" in runtime):
                raise ValueError("observation policy presence differs")
            if "observation_policy" in provenance:
                policy = joint.common.validate_observation_policy(provenance["observation_policy"])
                if (runtime["observation_policy"] != policy
                        or runtime.get("attention") != policy["attention_dispatch"]["text_config"]):
                    raise ValueError("observation policy/runtime differs")
            if (provenance.get("checkpoint") != current
                    or provenance.get("attestation_receipt_sha256") != ref["sha256"]
                    or provenance["input_sha256"].get(str(cases_path)) != case_ref["sha256"]):
                raise ValueError("collection full checkpoint or cases identity differs")
            if family == "native":
                expected = sorted((n, list(t.shape), "torch.bfloat16", "native") for n, t in native.items())
                actual = sorted((e["name"], e["shape"], e["dtype"], e["kind"]) for e in coverage["tensors"])
                if (actual != expected or coverage.get("native_tensors") != len(native)
                        or coverage.get("native_coordinates") != sum(t.numel for t in native.values())
                        or coverage.get("virtual_coordinates") != 0 or coverage.get("virtual_tensors") != 0
                        or provenance.get("virtual_unit_gains") is not False):
                    raise ValueError("complete native tensor columns differ")
            else:
                entries = coverage["entries"]
                sampling = coverage.get("sampling_mode")
                expected_names, full_samples = set(matrices), {}
                if sampling == "mlp-full-units":
                    if families != ["matrix"] or "v" in matrix_families(plan):
                        raise ValueError("full-unit mode requires observed matrix only and explicit V frozen")
                    unit_ref = provenance["unit_support"]
                    support = document(Path(unit_ref["path"]), pins, unit_ref["sha256"])
                    if (coverage.get("unit_support") != support
                            or coverage.get("unit_support_sha256") != unit_ref["sha256"]
                            or support["source_checkpoint"] != current
                            or support["attestation_receipt_sha256"] != ref["sha256"]):
                        raise ValueError("full-unit support/source binding differs")
                    full_samples = full_matrix_recipe.full_unit_samples(checkpoint.tensors, support)
                    expected_names = {n for n in matrices if matrix_recipe.PATTERN.fullmatch(n)[2] == "self_attn"} | set(full_samples)
                    for evidence in support["reference_attention"].values():
                        capture(Path(evidence["path"]), pins, evidence["sha256"])
                if (not matrices or any(t.dtype != "BF16" or len(t.shape) != 2 for t in matrices.values())
                        or len(entries) != len(expected_names) or {e["tensor"] for e in entries} != expected_names
                        or coverage.get("sampled_matrices") != len(entries)
                        or coverage.get("sampled_coordinates") != sum(e["count"] for e in entries)
                        or sampling not in {"mlp-triplets", "mlp-full-units"}
                        or provenance.get("sampling_mode") != sampling):
                    raise ValueError("complete declared coherent matrix columns required")
                for entry in entries:
                    match = matrix_recipe.PATTERN.fullmatch(entry["tensor"])
                    if (entry["shape"] != list(matrices[entry["tensor"]].shape)
                            or type(entry["count"]) is not int or entry["count"] < 1
                            or (entry["tensor"] in full_samples and entry["count"] != len(full_samples[entry["tensor"]][0]))
                            or (entry["tensor"] not in full_samples and entry["count"] > 512)
                            or entry["family"] != match[3] or entry["layer"] != int(match[1])):
                        raise ValueError("matrix header/column layout differs")
                if summary["runtime"].get("gradient_checkpointing") is not False:
                    raise ValueError("matrix recomputation is unsupported")
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
            collections[key] = {"path": directory, "summary": summary, "provenance": provenance,
                                "frames": frames, "coverage": coverage, "family": family}
        left, right = (collections[o if family == "native" else "matrix:" + o] for o in OBJECTIVES)
        if left["frames"] != right["frames"] or left["coverage"] != right["coverage"]:
            raise ValueError("objective frame/columns differ")
        for key in ("threads", "seed", "max_cases_per_microbatch", "input_sha256", "input_stat", "calibration_scope"):
            if left["provenance"][key] != right["provenance"][key]:
                raise ValueError("objective collector input/runtime differs: " + key)
        for key in ("torch", "transformers", "weight_dtype", "attention", "gradients",
                    "gradient_accumulation", "gradient_checkpointing", "max_cases_per_microbatch", "observation_policy"):
            if left["summary"]["runtime"].get(key) != right["summary"]["runtime"].get(key):
                raise ValueError("objective numerical runtime differs: " + key)
    if mixed and "native" in families:
        tolerance = plan.get("baseline_tolerance")
        if type(tolerance) not in (int, float) or not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("explicit finite nonnegative cross-collector baseline tolerance required")
        for objective in OBJECTIVES:
            n, m = collections[objective], collections["matrix:" + objective]
            if n["frames"] != m["frames"]:
                raise ValueError("native/matrix frames differ")
            for key in ("threads", "seed", "max_cases_per_microbatch", "calibration_scope"):
                if n["provenance"][key] != m["provenance"][key]:
                    raise ValueError("native/matrix input/runtime differs: " + key)
            for key in ("torch", "transformers", "weight_dtype", "attention", "max_cases_per_microbatch", "observation_policy"):
                if n["summary"]["runtime"].get(key) != m["summary"]["runtime"].get(key):
                    raise ValueError("native/matrix numerical runtime differs: " + key)
            # Different entry scripts/gradient accumulation are explicit, not falsely equal.
            for collection in (n, m):
                for filename, expected_sha in collection["provenance"]["input_sha256"].items():
                    capture(Path(filename), pins, expected_sha)
            shared = set(n["provenance"]["input_sha256"]) & set(m["provenance"]["input_sha256"])
            if any(n["provenance"]["input_sha256"][key] != m["provenance"]["input_sha256"][key] for key in shared):
                raise ValueError("shared collector implementation/input pins differ")
            field, value = ("correct_logprobs", "correct_logprob") if objective == "correct-logprob" else ("margins", "margin")
            errors = [abs(a[value] - b[value])
                      for nr, mr in zip(n["summary"]["folds"], m["summary"]["folds"], strict=True)
                      for a, b in zip(nr[field], mr[field], strict=True)]
            m["baseline_max_abs_difference"] = max(errors, default=0.)
            if m["baseline_max_abs_difference"] > tolerance:
                raise ValueError("native/matrix baseline objective differs")
    if families == ["matrix"]:
        # No cross-family parity was measured. Bind the actual matrix collector
        # inputs directly; native remains protected by the full checkpoint identity.
        for collection in collections.values():
            for filename, expected_sha in collection["provenance"]["input_sha256"].items():
                capture(Path(filename), pins, expected_sha)
            collection["baseline_max_abs_difference"] = None
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
        if mixed:
            scales = metric.get("block_scales", {})
            if (set(scales) != {"native", "matrix"} or any(type(v) not in (int, float)
                    or not np.isfinite(v) or v <= 0 for v in scales.values())):
                raise ValueError("explicit positive native/matrix block cost scales required")
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
    if mixed:
        capture(Path(matrix_recipe.__file__), pins)
        if full_unit_declaration(collections) is not None:
            capture(Path(full_matrix_recipe.__file__), pins)
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
        mixed = plan["schema_version"] == 2
        loaded = {}
        for key, collection in collections.items():
            constraints = [{"purpose": p, "fold": f, "minimum_delta": 0.} for p, f in labels]
            data = joint.read_collection(collection["path"], {"constraints": constraints})
            base, g, _, layout, coords, _, hashes, _, _ = data
            for p, digest in hashes.items():
                capture(Path(p), pins, digest)
            if collection["family"] == "native":
                if coords is not None:
                    raise ValueError("native-only source required")
                with np.load(collection["path"] / "source-native-values.npz", allow_pickle=False) as arrays:
                    if set(arrays.files) != {e["tensor"] for e in layout} or any(
                            arrays[e["tensor"]].shape != checkpoint.tensors[e["tensor"]].shape for e in layout):
                        raise ValueError("native source array shape mismatch")
                for record in collection["summary"]["folds"]:
                    with np.load(collection["path"] / record["file"], allow_pickle=False) as arrays:
                        if set(arrays.files) != {e["tensor"] for e in layout} or any(
                                arrays[e["tensor"]].shape != checkpoint.tensors[e["tensor"]].shape for e in layout):
                            raise ValueError("native gradient array columns/shape mismatch")
            else:
                with np.load(collection["path"] / "sample-plan.npz", allow_pickle=False) as arrays:
                    if set(arrays.files) != {"coordinates", "source_values"}:
                        raise ValueError("unexpected matrix sample-plan arrays")
                expected = expected_mlp_samples(checkpoint, collection)
                cursor = 0
                for entry in layout:
                    count = entry["count"]
                    if entry["tensor"] in expected and not np.array_equal(coords[cursor:cursor + count], expected[entry["tensor"]][0]):
                        raise ValueError("MLP coordinates do not share the declared hidden-unit sample")
                    cursor += count
                if collection["coverage"].get("sampling_mode") == "mlp-full-units":
                    full_matrix_recipe.verify_attention(
                        [coords[e["start"]:e["start"]+e["count"]] for e in collection["coverage"]["entries"]],
                        collection["coverage"]["entries"], collection["coverage"]["unit_support"])
            loaded[key] = (base, g, layout, coords)
        for family in observed_blocks(plan):
            first, second = (loaded[o if family == "native" else "matrix:" + o] for o in OBJECTIVES)
            if (first[2] != second[2] or not np.array_equal(first[3], second[3])
                    or not np.array_equal(first[0].astype(np.float32).view(np.uint32),
                                          second[0].astype(np.float32).view(np.uint32))):
                raise ValueError("objective source bytes/columns differ")
        if "native" in observed_blocks(plan):
            base, _, layout, _ = loaded[OBJECTIVES[0]]
            observed_native_count = len(base)
            gradient = np.vstack([loaded[o][1] for o in OBJECTIVES])
            base, gradient, layout = select_native_columns(plan, base, gradient, layout)
        else:
            observed_native_count = 0
            base, layout = np.empty(0, dtype=np.float64), []
            gradient = np.empty((len(labels) * len(OBJECTIVES), 0), dtype=np.float64)
        selected_native_names = {entry["tensor"] for entry in layout}
        layout = [dict(e, coordinates=None, block="native") for e in layout]
        native_count = len(base)
        scope = None
        if mixed:
            matrix_base, _, matrix_layout, coords = loaded["matrix:" + OBJECTIVES[0]]
            active, selected_matrix_layout = matrix_column_indices(plan, matrix_layout)
            selected_matrix_names = {entry["tensor"] for entry in selected_matrix_layout}
            allowed_families = matrix_families(plan)
            cursor = 0
            for entry in matrix_layout:
                count = entry["count"]
                if entry["tensor"] in selected_matrix_names:
                    layout.append(dict(entry, coordinates=coords[cursor:cursor + count], block="matrix"))
                cursor += count
            if len({e["tensor"] for e in layout}) != len(layout):
                raise ValueError("native and matrix tensor columns overlap")
            active = np.asarray(active, dtype=np.int64)
            base = np.concatenate([base, matrix_base[active]])
            gradient = np.hstack([gradient, np.vstack([loaded["matrix:" + o][1][:, active] for o in OBJECTIVES])])
            scope = {"native_columns": native_count, "observed_matrix_columns": len(matrix_base),
                     "active_matrix_columns": len(active), "excluded_matrix_columns": len(matrix_base) - len(active),
                     "excluded_qkv_columns": sum(e["count"] for e in matrix_layout
                                                 if matrix_recipe.PATTERN.fullmatch(e["tensor"])[3] in {"q", "k", "v"}
                                                 and e["tensor"] not in selected_matrix_names),
                     "active_matrix_families": allowed_families,
                     "eligible_matrix_tensors": sorted(selected_matrix_names),
                     "baseline_max_abs_difference": {o: collections["matrix:" + o]["baseline_max_abs_difference"] for o in OBJECTIVES},
                     "cross_family_baseline_parity": "measured" if "native" in observed_blocks(plan) else "not_measured",
                     "gradient_methods": {"native": (collections[OBJECTIVES[0]]["summary"]["runtime"].get("gradients")
                                                      if "native" in observed_blocks(plan) else None),
                                          "matrix": collections["matrix:" + OBJECTIVES[0]]["summary"]["runtime"].get("gradient_accumulation")}}
        eligible_blocks = plan.get("eligible_blocks", ["native", "matrix"]) if mixed else ["native"]
        if mixed and eligible_blocks == ["matrix"]:
            # Exact exclusion: these variables never enter optimization or BF16 realization.
            base = base[native_count:]
            gradient = gradient[:, native_count:]
            layout = [entry for entry in layout if entry["block"] == "matrix"]
            native_count = 0
            scope.update(observed_native_columns=observed_native_count, native_columns=0,
                         excluded_native_columns=observed_native_count, eligible_blocks=eligible_blocks)
        methods, specs = [], []
        for method in plan["candidates"]:
            lower = np.asarray([r["minimum_delta"] for r in method["constraints"]], dtype=np.float64)
            metric = method["cost_metric"]
            cost = np.ones_like(base) if metric["type"] == "euclidean" else 1 / np.maximum(np.abs(base), metric["floor"])
            if mixed:
                cost[:native_count] *= metric["block_scales"]["native"]
                cost[native_count:] *= metric["block_scales"]["matrix"]
            values, report = joint.quantized_proposal(base, gradient, lower, cost, **method["rounding"])
            edits, realized_support, cursor = [], [], 0
            for entry in layout:
                count = entry["count"]
                indices = np.flatnonzero(values[cursor:cursor+count].astype(np.float64) != base[cursor:cursor+count])
                if indices.size:
                    coordinates = entry["coordinates"]
                    edits.append({"tensor": entry["tensor"],
                                  ("indices" if coordinates is None else "coordinates"):
                                      indices.tolist() if coordinates is None else coordinates[indices].tolist(),
                                  "set_values": values[cursor+indices].tolist()})
                if entry["block"] == "matrix":
                    delta_part = values[cursor:cursor + count].astype(np.float64) - base[cursor:cursor + count]
                    realized_support.append({"tensor": entry["tensor"], "sampled": count, "changed": len(indices),
                                             "unchanged_after_rounding": count - len(indices),
                                             "delta_l2": float(np.linalg.norm(delta_part)),
                                             "weighted_delta_l2": float(np.linalg.norm(cost[cursor:cursor + count] * delta_part))})
                cursor += count
            if any("indices" in edit and edit["tensor"] not in selected_native_names for edit in edits):
                raise ValueError("ineligible native tensor escaped selected columns")
            changed = sum(len(e["set_values"]) for e in edits)
            if changed > 1_000_000:
                raise ValueError("actual changed coordinates exceed one million; no silent trimming")
            report.update(id=method["id"], constraints=method["constraints"], cost_metric=metric,
                          changed_elements=changed, changed_tensors=len(edits),
                          native_scope={"observed_columns": observed_native_count,
                                        "eligible_columns": native_count,
                                        "eligible_tensors": sorted(selected_native_names) if native_count else []},
                          represented_source_l2=float(np.linalg.norm(base)),
                          norm_scope=("sampled eligible matrix coordinates only; native excluded" if mixed and native_count == 0 else
                                      "eligible native plus sampled eligible matrix coordinates; not whole matrix norms" if mixed else
                                      "all selected native tensors; not the whole model"),
                          spec_emitted=bool(write_spec and edits))
            if mixed:
                if any(("indices" in edit and eligible_blocks == ["matrix"]) or
                       ("coordinates" in edit and (not matrix_recipe.PATTERN.fullmatch(edit["tensor"]) or
                        edit["tensor"] not in selected_matrix_names))
                       for edit in edits):
                    raise ValueError("ineligible native/matrix family escaped selected columns")
                declaration = full_unit_declaration(collections)
                compatibility = joint.editor_compatibility(edits, allow_batched_matrix=True,
                    **({"matrix_full_units": declaration} if declaration is not None else {}))
                if not compatibility["spec_emission_allowed"]:
                    raise ValueError("matrix proposal exceeds existing editor bounds")
                delta = values.astype(np.float64) - base
                report.update(column_scope=scope, editor_compatibility=compatibility,
                              cost_convention="minimize ||diag(existing coordinate cost * explicit block scale) delta||_2 squared",
                              realized_blocks={name: {"changed_elements": int(np.count_nonzero(delta[section])),
                                                       "delta_l2": float(np.linalg.norm(delta[section])),
                                                       "weighted_delta_l2": float(np.linalg.norm(cost[section] * delta[section]))}
                                               for name, section in (("native", slice(None, native_count)),
                                                                     ("matrix", slice(native_count, None)))},
                              matrix_any_changed=bool(np.any(delta[native_count:] != 0)),
                              realized_matrix_support=realized_support)
            methods.append(report)
            if write_spec and edits:
                specs.append({"id": method["id"], "edits": edits,
                              "base_tree_sha256": plan["checkpoint"]["tree_sha256"],
                              "source_checkpoint": str(checkpoint.path),
                              "hypothesis": "Coupled authored public-answer local-gradient proposal; finite response unmeasured."})
        if mixed:
            sample_sha = pins[str((collections["matrix:" + OBJECTIVES[0]]["path"] / "sample-plan.npz").resolve())][0]
            for spec in specs:
                spec.update(matrix_batch=True, matrix_sample_plan_sha256=sample_sha)
                declaration = full_unit_declaration(collections)
                if declaration is not None:
                    spec["matrix_full_units"] = declaration
        stable(checkpoint, pins)
        result = {"schema_version": plan["schema_version"], "status": "local_proposals", "methods": methods,
                  "source_checkpoint": plan["checkpoint"], "parent_state": plan.get("parent_state"),
                  "observed_blocks": observed_blocks(plan),
                  "native_parameters": "frozen_not_observed" if observed_blocks(plan) == ["matrix"] else "observed",
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
    parser.add_argument("--write-spec", action="store_true", help="with --run, publish exact BF16 specs in the explicitly selected plan scope")
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
