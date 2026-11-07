#!/usr/bin/env python3
"""Preview metadata, or own native/finite/train-lora GPU4 work and restore exact gemma-2."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
TARGET = "gemma-4-31b-8gpu-gemma-2-1"
CID = "fd4ec2eb9d7deba2ce3e790add6b50c0ee79a6390495320d8697873b32957bcf"
IMAGE = "sha256:af9e746b913088cbf6c4464a31d6904a6b7e1d3ba62f0895362ddf13595f7abe"
TREE = "5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf"
OBJECTIVES = ("correct-logprob", "margin")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, data):
    with Path(path).open("x") as out:
        json.dump(data, out, indent=2, allow_nan=False)
        out.write("\n")


def pinned(pin):
    require(sha(pin["path"]) == pin["sha256"], "pinned file changed")
    return read(pin["path"])


def native_args(plan, objective, run=False):
    args = [str(Path(plan["source_root"]) / "scripts/experiments/gemma4-purpose/native.py"),
            "--checkpoint", plan["checkpoint"], "--cases", plan["cases"]["path"],
            "--output", str(Path(plan["output_root"]) / "work" / objective),
            "--checkpoint-attestation", plan["attestation"]["path"],
            "--attestation-sha256", plan["attestation"]["sha256"],
            "--mode", "chat-public-answer", "--objective", objective,
            "--answer-batching", "serial", "--max-cases-per-microbatch", "1",
            "--saved-activation-storage", "cpu-fp32", "--max-frame-tokens", "1024",
            "--text-attention-backend", "sdpa", "--cuda-memory-cap-gib", "88",
            "--execution-order", "longest_frame_first", "--device", "cuda:0",
            "--threads", "4", "--seed", "0", "--gpu-exclusive-approved"]
    return args + (["--run"] if run else [])


def stages(plan):
    mode = plan.get("mode", "native")
    require(mode in ("native", "finite", "train-lora"), "explicit supported owner mode required")
    return OBJECTIVES if mode == "native" else (mode,)


def training_args(plan, run=False):
    args = [str(Path(plan["source_root"]) / "scripts/experiments/gemma4-purpose/train_lora.py"),
            "--plan", plan["training"]["plan"]["path"],
            "--output", str(Path(plan["output_root"]) / "work/train-lora")]
    return args + (["--run", "--device", "cuda:0", "--gpu-exclusive-approved"] if run else [])


def training_metadata(plan):
    """Training identity only; trainer preview owns its frame/math validation."""
    training = plan["training"]
    require(set(training) == {"plan", "cpu_preview", "input_quality", "holdout", "restoration_proof"},
            "explicit training identity pins required")
    task = pinned(training["plan"])
    require(task["kind"] == "gemma4_lora_sft_v1", "LoRA trainer kind required")
    require(task["checkpoint"] == dict(path=plan["checkpoint"], tree_sha256=plan["checkpoint_tree_sha256"],
                                     attestation=plan["attestation"]), "training parent differs")
    deadline = plan["maximum_training_seconds"]
    require(type(task["maximum_seconds"]) is int and task["maximum_seconds"] > 0
            and type(task["continuation_wait_seconds"]) is int and task["continuation_wait_seconds"] > 0
            and type(plan["training_startup_allowance_seconds"]) is int
            and 30 <= plan["training_startup_allowance_seconds"] <= 300
            and deadline == task["maximum_seconds"] + task["continuation_wait_seconds"]
                + plan["training_startup_allowance_seconds"], "explicit training/pause/startup deadline required")
    data = task["data"]
    require(data["split"] == "train", "trainer must use the training split")
    split = pinned(data["split_summary"])
    frame = task["frame_evidence"]
    pinned(frame)
    ids_by_split = {}
    inputs = [data, data["split_summary"], frame, training["holdout"]]
    work = Path(plan["output_root"]).resolve() / "work"
    require(all(not Path(v["path"]).resolve().is_relative_to(work) for v in inputs),
            "training inputs must stay outside writable work")
    for label, pin in (("train", data), ("holdout", training["holdout"])):
        require(sha(pin["path"]) == pin["sha256"], "training/holdout file changed")
        rows = [json.loads(line) for line in Path(pin["path"]).read_text().splitlines() if line.strip()]
        ids = [row["task_id"] for row in rows]
        expected = split["training_ids" if label == "train" else "holdout_ids"]
        require(0 < len(ids) <= 512 and len(set(ids)) == len(ids) and ids == expected
                and len(ids) == split[label + "_rows"], "exact training/holdout IDs required")
        ids_by_split[label] = ids
    require(task["data"]["ids"] == ids_by_split["train"], "trainer selected IDs differ")
    require(not set(ids_by_split["train"]) & set(ids_by_split["holdout"]), "holdout overlaps training")
    quality = pinned(training["input_quality"])
    require(task["input_quality"] == quality["reference_quality_summary"],
            "trainer and owner reference-quality evidence differs")
    pinned(task["input_quality"])
    require(quality["status"] == "completed" and quality["training_data_sha256"] == data["sha256"]
            and quality["holdout_data_sha256"] == training["holdout"]["sha256"]
            and quality["train_ids"] == quality["passed_train_ids"] == ids_by_split["train"]
            and quality["gradient_holdout_ids"] == [], "all training references must pass; holdout gradients prohibited")
    preview = pinned(training["cpu_preview"])
    require(preview["returncode"] == 0 and preview["model_calls"] == 0
            and preview["trainer_arguments"] == training_args(plan), "CPU trainer metadata preview mismatch")
    restored = pinned(training["restoration_proof"])
    replica = restored["replicas"]["gemma-2"]
    require(restored["current_model"] == restored["best_parent"] == "F04"
            and restored["checkpoint_all9_fullstat_matched_manifest"] is True
            and replica["cid"] == plan["target"]["id"] and replica["tree"] == plan["checkpoint_tree_sha256"]
            and replica["checkpoint_mount"] == plan["checkpoint"] and replica["checkpoint_readonly"] is True,
            "training F04 restoration proof differs")
    return dict(ids=ids_by_split["train"], holdout_ids=ids_by_split["holdout"], task=task)


def training_complete(plan, result):
    metadata = training_metadata(plan)
    summary = read(result / "summary.json")
    require(summary["status"] in ("completed", "pilot_only")
            and summary["completed_steps"] == (64 if summary["status"] == "completed" else 4),
            "training must explicitly finish pilot4 or total64")
    require(summary["trainer_plan_sha256"] == plan["training"]["plan"]["sha256"]
            and summary["base_unchanged"] is True and summary["target_tensors"] == 410
            and summary["data_sha256"] == metadata["task"]["data"]["sha256"]
            and summary["training_ids"] == metadata["ids"] and summary["holdout_gradient_ids"] == [],
            "training completion parent/data/scope differs")
    require(summary["maximum_seconds_excludes_approval_wait"] is True and summary["error"] is None,
            "trainer timing/failure claim differs")
    require(summary["exports"], "training export absent")
    for export in summary["exports"]:
        receipt_path = Path(export["path"]).resolve()
        require(receipt_path.is_relative_to(result.resolve()), "training export outside owned output")
        receipt = pinned(export)
        require(receipt["status"] == "exported" and receipt["base_unchanged"] is True
                and receipt["parent_path"] == plan["checkpoint"]
                and receipt["parent_tree_sha256"] == plan["checkpoint_tree_sha256"]
                and receipt["data_sha256"] == metadata["task"]["data"]["sha256"]
                and receipt["modules"] == 410 and receipt["steps"] in (4, 64), "training export identity differs")
    require(receipt["steps"] == summary["completed_steps"], "final export step differs")
    for path, digest in summary["source_sha256"].items():
        require(sha(path) == digest and (path not in plan["source_pins"] or plan["source_pins"][path] == digest),
                "training input/source changed")
    return {k: summary[k] for k in ("status", "completed_steps", "trainer_plan_sha256", "target_tensors", "data_sha256")}


def finite_args(plan):
    return ["-m", "kairyu_bench.wsp.weights.finite_replay", "--checkpoint", plan["checkpoint"],
            "--plan", plan["finite"]["plan"]["path"],
            "--output", str(Path(plan["output_root"]) / "work/finite"),
            "--device", "cuda:0", "--maximum-seconds", "600", "--answer-batching", "serial",
            "--threads", "4", "--seed", "0", "--max-frame-tokens", "1024",
            "--text-attention-backend", "sdpa", "--cuda-memory-cap-gib", "88",
            "--execution-order", "longest_frame_first", "--gpu-exclusive-approved"]


def finite_policy():
    return dict(max_frame_tokens=1024, cuda_memory_cap_gib=88.0, answer_batching="serial",
                max_cases_per_microbatch=1, execution_order="longest_frame_first",
                attention_dispatch={"": "eager", "text_config": "sdpa", "vision_config": "eager", "audio_config": "eager"})


def finite_modules(plan):
    # CPU readers only. Never invoke finite_replay.main/run as a preview.
    source = Path(plan["source_root"]) / "src"
    sys.path.insert(0, str(source))
    from kairyu_bench.wsp.weights import checkpoint, editor, finite_replay
    for module in (checkpoint, editor, finite_replay):
        require(Path(module.__file__).resolve().is_relative_to(source.resolve()), "metadata reader source differs")
    return checkpoint, editor, finite_replay


def full_identity(identity, path):
    require(identity["path"] == str(path) and identity["hash_files"] is True
            and identity["identity_level"] == "content_sha256" and identity.get("source_stable") is True
            and len(identity["files"]) == 9,
            "full nine-file identity required")
    rows = identity["files"]
    require(len({r["path"] for r in rows}) == 9, "duplicate identity file")
    for row in rows:
        file = path / row["path"]
        require(file.resolve().is_relative_to(path.resolve()) and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]),
                "invalid identity path/hash")
        st = file.stat()
        require(row["size"] == st.st_size and row["stat"] == dict(device=st.st_dev, inode=st.st_ino,
                mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns), "attested checkpoint stat changed")
    canonical = [{k: row[k] for k in ("path", "size", "sha256")} for row in rows]
    require(identity["sha256"] == hashlib.sha256(json.dumps(canonical, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(), "content tree receipt inconsistent")
    return {row["path"]: row for row in rows}


def finite_metadata(plan):
    """Bounded CPU header/selected-coordinate reads; reuse pinned full-file receipts."""
    import numpy as np
    finite = plan["finite"]
    require(set(finite) == {"plan", "spec", "manifest", "verify", "candidate_identity"}, "finite identity pins required")
    records = {key: pinned(pin) for key, pin in finite.items()}
    replay, spec, manifest, verified = (records[key] for key in ("plan", "spec", "manifest", "verify"))
    require(replay["architecture"] == "gemma4_native_bf16" and replay["base_tree_sha256"] == TREE,
            "native-only AV finite replay required")
    require(replay["cases"] == plan["cases"] and replay["frame_evidence"] == plan["frame"]
            and replay["checkpoint_attestation"] == plan["attestation"]
            and replay["observation_policy"] == finite_policy(), "finite inputs/policy differ")
    require(list(pinned(plan["cases"])) == list("ABCDEF"), "finite requires raw train mapping")
    require(replay["candidates"] == [dict(id=spec["id"], spec=finite["spec"]["path"], sha256=finite["spec"]["sha256"])],
            "one fixed exact candidate required")
    candidate = Path(manifest["candidate"])
    require(candidate.is_absolute() and candidate.resolve().is_relative_to(ROOT / "checkpoints")
            and candidate.resolve() != Path(plan["checkpoint"]).resolve(), "separate saved candidate required")
    require(manifest["source"] == plan["checkpoint"] and manifest["spec"] == spec
            and manifest["spec_sha256"] == finite["spec"]["sha256"], "manifest/spec/source mismatch")
    require(verified["verified"] is True and verified["legacy_full_hashes_missing"] is False,
            "full editor verification required")
    base_files = full_identity(pinned(plan["attestation"])["checkpoint"], Path(plan["checkpoint"]))
    identity = records["candidate_identity"].get("checkpoint", records["candidate_identity"])
    candidate_files = full_identity(identity, candidate)
    manifest_files = {r["path"]: r for r in manifest["files"]}
    verified_files = {r["path"]: r for r in verified["files"]}
    require(len(manifest["files"]) == len(verified["files"]) == 9 and
            base_files.keys() == candidate_files.keys() == manifest_files.keys() == verified_files.keys(), "nine-file binding differs")
    for rel, row in manifest_files.items():
        require(row["source_sha256"] == base_files[rel]["sha256"]
                and row["candidate_sha256"] == candidate_files[rel]["sha256"]
                and row["size"] == base_files[rel]["size"] == candidate_files[rel]["size"]
                and verified_files[rel] == {k: row[k] for k in ("path", "source_sha256", "candidate_sha256")},
                "manifest/verify/full-identity file hashes differ")
    reader, editor, replay_module = finite_modules(plan)
    base, saved = reader.Checkpoint(plan["checkpoint"]), reader.Checkpoint(candidate)
    require(saved.identity(hash_files=False)["files"] == [dict(row, sha256=None) for row in identity["files"]],
            "candidate current inventory differs")
    _, specs, input_hashes = replay_module.load_plan(Path(finite["plan"]["path"]), base)
    metrics = specs[0]["_verified_metrics"]
    for report in (manifest, verified, editor.validate_manifest(manifest)):
        require(all(report[key] == metrics[key] for key in ("changed_elements", "changed_tensors"))
                and math.isclose(report["delta_l2"], metrics["delta_l2"], rel_tol=1e-12, abs_tol=1e-15),
                "actual BF16 edit metrics differ")
    proofs = {label: [] for label in ("base", spec["id"], "restored_base")}
    for edit in sorted(specs[0]["edits"], key=lambda e: e["tensor"]):
        name, indices = edit["tensor"], sorted(edit["_flat_indices"])
        values = dict(zip(edit["_flat_indices"], edit["set_values"], strict=True))
        original = base.read_indices(name, indices).astype(np.float32)
        actual = saved.read_indices(name, indices).astype(np.float32)
        expected = np.asarray([values[i] for i in indices], dtype=np.float32)
        require(np.array_equal(actual.view(np.uint32), expected.view(np.uint32)), "saved candidate BF16 values differ")
        def digest(values):
            return hashlib.sha256((values.view(np.uint32) >> 16).astype("<u2").tobytes()).hexdigest()
        for label in proofs:
            proofs[label].append(dict(tensor=name, selected_count=len(indices),
                indices_sha256=hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest(),
                original_bf16_sha256=digest(original), actual_bf16_sha256=digest(actual if label == spec["id"] else original)))
    base.assert_stable()
    saved.assert_stable()
    return dict(id=spec["id"], metrics=metrics, proofs=proofs, source_hashes=input_hashes,
                candidate_tree_sha256=identity["sha256"],
                full_tree_rehashed_here=False, scope="pinned full hashes plus fresh stats and selected exact BF16 reads")


def finite_complete(plan, result):
    metadata = finite_metadata(plan)
    summary = read(result / "summary.json")
    labels = [g + "-" + str(i) for g in "ABCDEF" for i in range(2)]
    require(summary["status"] == "completed" and summary["source_stable"] is True
            and summary["error_type"] is None and summary["labels"] == labels
            and summary["answer_batching"] == "serial" and summary["observation_policy"] == finite_policy()
            and summary["new_gradients"] == summary["generated_tokens"] == 0, "incomplete finite result")
    require(read(result / "frame-evidence.json") == pinned(plan["frame"]), "finite frame mismatch")
    provenance = read(result / "provenance.json")
    require(provenance["plan_sha256"] == plan["finite"]["plan"]["sha256"]
            and provenance["expected_base_tree_sha256"] == TREE and provenance["source_hashes"] == metadata["source_hashes"]
            and provenance["runner_sha256"] == sha(Path(plan["source_root"]) / "src/kairyu_bench/wsp/weights/finite_replay.py"),
            "finite provenance mismatch")
    execution = provenance["observation_execution"]
    require(execution == summary["observation_execution"]
            and execution["resolved_attention_dispatch"]["text_config"] == "sdpa"
            and execution["cuda_allocator"]["requested_gib"] == 88
            and execution["automatic_attention_api_or_length_retry"] is False, "finite actual execution differs")
    sweeps = {}
    for label in ("base", metadata["id"], "restored_base"):
        sweep = read(result / (label + ".json"))
        require(list(sweep["values"]) == list("ABCDEF") and all(len(fs) == 2 and all(len(row) == 1 for row in fs)
                for fs in sweep["values"].values()), "partial finite sweep")
        for field in ("means", "correct_nll", "kl"):
            require(len(sweep[field]) == 12 and all(type(v) in (int, float) and math.isfinite(v) for v in sweep[field]),
                    "partial/nonfinite finite metrics")
        require(sweep["means"] == [row[0] for fs in sweep["values"].values() for row in fs], "finite fold mapping differs")
        sweeps[label] = sweep
    diagnostics = summary["candidates"]
    require(len(diagnostics) == 1 and diagnostics[0]["id"] == metadata["id"], "fixed finite candidate absent")
    diag = diagnostics[0]
    require(all(diag[k] == v for k, v in metadata["metrics"].items()), "finite BF16 metrics differ")
    base, candidate, restored = (sweeps[label] for label in ("base", metadata["id"], "restored_base"))
    for field, raw in (("margin_delta", "means"), ("correct_nll_delta", "correct_nll")):
        require(diag[field] == [b - a for a, b in zip(base[raw], candidate[raw], strict=True)], "finite delta mismatch")
    require(diag["kl_base_to_candidate"] == candidate["kl"], "finite KL mismatch")
    restore = summary["base_restoration"]
    require(restore == read(result / "base-restoration.json") and all(restore[k] is True for k in
            ("attempted", "forward_completed", "matched", "storage_bytes_exact")), "finite restoration incomplete")
    require(max(abs(a-b) for a, b in zip(base["means"], restored["means"], strict=True)) <= 1e-4
            and max(abs(a-b) for a, b in zip(base["correct_nll"], restored["correct_nll"], strict=True)) <= 1e-4
            and max(restored["kl"]) <= 1e-6, "restored finite numerical mismatch")
    proofs = read(result / "selected-byte-proofs.json")
    require(proofs == summary["selected_byte_proofs"] and set(proofs) == set(metadata["proofs"]), "finite byte proof stages differ")
    for label, expected in metadata["proofs"].items():
        require(proofs[label]["method"] == "selected_bf16_cpu_readback" and proofs[label]["storage_bytes_exact"] is True
                and proofs[label]["tensors"] == expected, "finite selected BF16 proof differs from saved candidate")
    return metadata


def command(plan, objective, plan_sha):
    require(re.fullmatch(r"[0-9a-f]{64}", plan_sha), "plan SHA label required")
    work = Path(plan["output_root"]) / "work"
    args = ["docker", "run", "--rm", "--pull=never", "--network", "none",
            "--user", "1003:1003", "--gpus", "device=4", "--cpuset-cpus", "32-39",
            "--memory", "128g", "--memory-swap", "128g", "--pids-limit", "512",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only",
            "--shm-size", "1g", "--entrypoint", "/usr/bin/python3",
            "--name", plan["container_prefix"] + "-" + objective,
            "--cidfile", str(work / (objective + ".cid")),
            "--label", "kairyu.native.owner-plan-sha256=" + plan_sha,
            "--label", "kairyu.native.owner-root=" + plan["output_root"],
            "--mount", f"type=bind,src={ROOT},dst={ROOT},readonly",
            "--mount", f"type=bind,src={work},dst={work}",
            "--mount", f"type=bind,src={work / 'tmp'},dst=/tmp", "--workdir", str(work)]
    env = dict(PYTHONPATH=str(Path(plan["source_root"]) / "src"), PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HOME=str(work / "cache/hf"),
               TORCH_HOME=str(work / "cache/torch"), TRITON_CACHE_DIR=str(work / "cache/triton"),
               TORCHINDUCTOR_CACHE_DIR=str(work / "cache/inductor"), CUDA_CACHE_PATH=str(work / "cache/cuda"),
               XDG_CACHE_HOME=str(work / "cache"), TMPDIR="/tmp", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    for key, value in env.items():
        args.extend(["--env", key + "=" + value])
    require(objective in stages(plan), "job differs from selected owner mode")
    payload = (training_args(plan, run=True) if objective == "train-lora" else
               finite_args(plan) if objective == "finite" else native_args(plan, objective, run=True))
    return args + [IMAGE, *payload]


def expected_cohort(plan):
    legacy = [(g, i, "behavior-" + g + str(i + 1)) for g in "ABCDEF" for i in range(2)]
    if "native_cohort" not in plan:
        return legacy
    require(stages(plan) == OBJECTIVES, "explicit cohort is native only")
    cohort = plan["native_cohort"]
    require(set(cohort) == {"count", "rows"} and type(cohort["count"]) is int
            and 1 <= cohort["count"] <= 64 and len(cohort["rows"]) == cohort["count"], "explicit bounded native cohort required")
    rows = cohort["rows"]
    require(all(set(r) == {"purpose", "fold", "id"} and isinstance(r["purpose"], str)
                and r["purpose"] and type(r["fold"]) is int and r["fold"] >= 0
                and isinstance(r["id"], str) and r["id"] for r in rows), "invalid native cohort row")
    expected = [(r["purpose"], r["fold"], r["id"]) for r in rows]
    require(len({r[2] for r in expected}) == len({r[:2] for r in expected}) == len(expected),
            "duplicate native cohort ID/fold")
    return expected


def validate(plan, for_run=False):
    jobs = stages(plan)
    require(plan["schema_version"] == 1, "owner schema required")
    if jobs == ("train-lora",):
        require(type(plan["maximum_training_seconds"]) is int and 1 <= plan["maximum_training_seconds"] <= 3600,
                "bounded training deadline required")
        require(not any(k in plan for k in ("cases", "frame", "native_cohort", "native_behavior_result", "cpu_previews", "maximum_collection_seconds")),
                "training must not masquerade as native observation")
    else:
        require(plan["maximum_collection_seconds"] == 900, "fixed collection scope")
    require(re.fullmatch(r"[0-9a-f]{64}", plan["target"]["id"]) and plan["target"]["image"] == IMAGE, "exact target/image required")
    if jobs == ("finite",):
        require(plan["target"]["id"] == CID, "legacy finite target required")
    protected_ids = {v["id"] for v in plan["protected"].values()}
    require(len(plan["protected"]) == len(protected_ids) == 26 and plan["target"]["id"] not in protected_ids, "other26 unique required")
    require(re.fullmatch(r"[a-z0-9][a-z0-9_.-]{1,90}", plan["container_prefix"]), "safe owned-container prefix required")
    parent_tree = plan.get("checkpoint_tree_sha256", TREE)
    checkpoint = Path(plan["checkpoint"])
    require(checkpoint.is_absolute() and checkpoint.resolve() == checkpoint
            and checkpoint.is_relative_to(ROOT / "checkpoints") and checkpoint.is_dir()
            and re.fullmatch(r"[0-9a-f]{64}", parent_tree), "explicit NVMe checkpoint/tree required")
    if jobs == ("finite",) or "checkpoint_tree_sha256" not in plan:
        require(plan["checkpoint"] == str(ROOT / "checkpoints/code90-current-ad-helper-av-v1")
                and parent_tree == TREE, "legacy AV checkpoint required")
    for key in ("source_root", "output_root"):
        path = Path(plan[key])
        require(path.is_absolute() and path.resolve().is_relative_to(ROOT), "NVMe task path required")
    require(Path(plan["output_root"]).resolve().is_relative_to(ROOT / "analysis"), "analysis output required")
    require(str(Path(__file__).resolve()) in plan["source_pins"], "owner source must be pinned")
    work = Path(plan["output_root"]).resolve() / "work"
    input_keys = ("attestation",) if jobs == ("train-lora",) else ("cases", "frame", "attestation", "native_behavior_result")
    inputs = [plan[k]["path"] for k in input_keys]
    if jobs == ("train-lora",):
        inputs += [v["path"] for v in plan["training"].values()]
    inputs += list(plan["source_pins"]) + [v["path"] for v in plan.get("cpu_previews", {}).values()]
    if jobs == ("finite",):
        inputs += [v["path"] for v in plan["finite"].values()]
    require(all(not Path(p).resolve().is_relative_to(work) for p in inputs), "inputs must stay outside writable work")
    source = Path(plan["source_root"])
    recipe = source / "scripts/experiments/gemma4-purpose"
    required = {str((recipe / name).resolve()) for name in ("native.py", "recipe_common.py", "objective_variants.py")}
    if jobs == ("train-lora",):
        required.add(str((recipe / "train_lora.py").resolve()))
    required |= {str(p.resolve()) for p in (source / "src").rglob("*.py")}
    required.add(str(Path(__file__).with_name("fresh-f01-finite.py").resolve()))
    require(len(required) > 4 and required <= set(plan["source_pins"]), "complete executable source pins required")
    for path, expected in plan["source_pins"].items():
        require(sha(path) == expected, "source changed: " + path)
    att = pinned(plan["attestation"])["checkpoint"]
    require(att["path"] == plan["checkpoint"] and att["sha256"] == parent_tree and att["hash_files"] is True
            and att["identity_level"] == "content_sha256" and len(att["files"]) == 9, "full parent attestation required")
    if "checkpoint_tree_sha256" in plan:
        full_identity(att, checkpoint)
    for row in att["files"]:
        st = (Path(plan["checkpoint"]) / row["path"]).stat()
        actual = dict(device=st.st_dev, inode=st.st_ino, mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)
        require(st.st_size == row["size"] and actual == row["stat"], "checkpoint stat changed")
    if jobs == ("train-lora",):
        ids = training_metadata(plan)["ids"]
        if for_run:
            require(plan.get("approved_for_execution") is True, "root-reviewed approval required")
        return ids
    cases = pinned(plan["cases"])
    require(cases.get("split", "train") == "train" and cases.get("gradient_allowed") is not False,
            "transfer/evaluation envelope prohibited")
    groups = cases.get("cases", cases)
    expected = expected_cohort(plan)
    require(list(groups) == list(dict.fromkeys(row[0] for row in expected)), "exact declared train groups required")
    folds = [(group, i, rows) for group, fs in groups.items() for i, rows in enumerate(fs)]
    require(len(folds) == len(expected) and all(len(rows) == 1 for _, _, rows in folds), "declared singleton train folds required")
    require([(g, i, rows[0]["id"]) for g, i, rows in folds] == expected
            and all(rows[0].get("split") == "train" and rows[0].get("gradient_allowed") is not False
                    for _, _, rows in folds), "exact declared train IDs/row split required; transfer/duplicates prohibited")
    ids = [row[2] for row in expected]
    frames = pinned(plan["frame"])
    observed = {(g, i, row["id"]) for g, fs in frames["frames"].items() for i, rows in enumerate(fs) for row in rows}
    require(observed == {(g, i, rows[0]["id"]) for g, i, rows in folds}
            and sum(len(rows) for fs in frames["frames"].values() for rows in fs) == len(expected), "exact train frame join required")
    require(all(row["prefix_tokens"] + max(row["correct_tokens"], row["foil_tokens"]) <= 1024
                for fs in frames["frames"].values() for rows in fs for row in rows), "complete frame exceeds1024")
    for objective in (() if jobs == ("finite",) else OBJECTIVES):
        preview = pinned(plan["cpu_previews"][objective])
        require(preview["returncode"] == 0 and preview["model_calls"] == 0
                and preview["collector_arguments"] == native_args(plan, objective), "CPU metadata preview mismatch")
    if jobs == ("finite",):
        finite_metadata(plan)
    pinned(plan["native_behavior_result"])
    if for_run:
        require(plan.get("approved_for_execution") is True, "root-reviewed approval required")
    return ids


def helper(plan):
    path = Path(__file__).with_name("fresh-f01-finite.py")
    require(plan["source_pins"].get(str(path.resolve())) == sha(path), "helper must be pinned")
    spec = importlib.util.spec_from_file_location("native_owner_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def services(plan, guard, serving=True):
    for name, expected in plan["protected"].items():
        require(guard.snapshot(name) == expected, "protected container changed: " + name)
    if serving:
        require(guard.snapshot(TARGET) == plan["target"], "target changed before stop")


def idle(guard, plan):
    result = guard.target_metrics(plan["target"]["id"])
    require(all(v == 0 for values in result["queues"].values() for v in values), "target not idle")
    return result


def cleanup_collectors(plan, guard, status, plan_sha, phase):
    """Fresh cidfile/daemon checks on every pass; no cached absence proof."""
    safe = True
    status.setdefault("collector_cleanup", [])
    for objective in stages(plan):
        name = plan["container_prefix"] + "-" + objective
        cidfile = Path(plan["output_root"]) / "work" / (objective + ".cid")
        cid = cidfile.read_text().strip() if cidfile.exists() else None
        record = dict(objective=objective, cid=cid, phase=phase)
        status["collector_cleanup"].append(record)
        def lookup():
            try:
                owned = guard.inspect(cid or name)
            except subprocess.CalledProcessError:
                # A failed inspect alone cannot prove absence (daemon might be down).
                all_ids = guard.docker("ps", "-aq", "--no-trunc").split()
                names = guard.docker("ps", "-aq", "--no-trunc", "--filter", "name=^/" + name + "$").split()
                require(not names and (cid is None or cid not in all_ids), "collector absence not proven")
                return None
            labels = owned["Config"].get("Labels", {}) or {}
            require((cid is None or owned["Id"] == cid) and owned["Image"] == IMAGE
                    and owned["Name"] == "/" + name
                    and labels.get("kairyu.native.owner-plan-sha256") == plan_sha
                    and labels.get("kairyu.native.owner-root") == plan["output_root"], "collector ownership differs")
            return owned
        try:
            owned = lookup()
            if owned is not None and owned["State"]["Running"]:
                try:
                    guard.docker("stop", "--time", "30", owned["Id"])
                except subprocess.CalledProcessError:
                    record["stop_failed"] = True
                owned = lookup()  # Recheck ownership immediately before any kill.
                if owned is not None and owned["State"]["Running"]:
                    guard.docker("kill", owned["Id"])
                    record["owned_kill"] = True
            owned = lookup()
            require(owned is None or not owned["State"]["Running"], "owned collector remains active")
            record["proven_inactive"] = True
        except BaseException as error:
            safe = False
            record["error_type"] = type(error).__name__
    return safe


def finish(plan, guard, process, stopped, status, plan_sha):
    """Client exit must precede a fresh final collector inactivity proof."""
    initial_safe = cleanup_collectors(plan, guard, status, plan_sha, "before_client_wait")
    client_finished = False
    try:
        if process is not None and process.poll() is None:
            process.wait(timeout=45)
        client_finished = process is None or process.poll() is not None
        require(client_finished, "collector client still running")
    except BaseException as error:
        status["cleanup_error_type"] = type(error).__name__
    final_safe = cleanup_collectors(plan, guard, status, plan_sha, "after_client_wait")
    safe = initial_safe and client_finished and final_safe
    status["collector_client_finished"] = client_finished
    if stopped and safe:
        try:
            guard.docker("start", plan["target"]["id"])
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                actual = guard.snapshot(TARGET)
                require(all(actual[k] == v for k, v in plan["target"].items() if k not in
                            ("started_at", "running", "health")), "restored identity differs")
                if actual["running"] and actual["health"] == "healthy":
                    status["after_queues"] = idle(guard, plan)
                    status["restored"] = True
                    break
                time.sleep(2)
            require(status["restored"], "restoration health deadline")
        except BaseException as error:
            status["restoration_error_type"] = type(error).__name__
    elif stopped:
        status["restoration_blocked"] = "collector inactivity/ownership not proven; original remains stopped"
    try:
        services(plan, guard, serving=False)
        status["protected_exact"] = True
    except BaseException as error:
        status["protected_error_type"] = type(error).__name__


def run(plan, expected_sha, plan_path):
    require(expected_sha == sha(plan_path), "explicit exact plan SHA required")
    require(plan.get("approved_for_execution") is True, "root-reviewed approval required")
    validate(plan, for_run=True)
    require(os.getuid() == 1003, "ordinary owner UID required")
    guard = helper(plan)
    out = Path(plan["output_root"])
    require(not (out / "work").exists() and not (out / "launch.json").exists(), "attempt already started")
    status = dict(pid=os.getpid(), started_epoch=time.time(), jobs=[], restored=False, protected_exact=False)
    with ExitStack() as stack:
        for path in (ROOT / "candidate-evaluation.lock", ROOT / "gpu4-native.lock"):
            lock = stack.enter_context(path.open("a+"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status["lock_proof"] = guard.lock_owner_proof(ROOT / "candidate-evaluation.lock", os.getpid(), 1003)
        validate(plan, for_run=True)
        services(plan, guard)
        status["before_queues"] = idle(guard, plan)
        require(set(guard.docker("ps", "-aq", "--no-trunc").split()) ==
                {plan["target"]["id"], *(v["id"] for v in plan["protected"].values())}, "container inventory changed")
        save(out / "launch.json", status)
        for path in (out / "work", out / "work/tmp", out / "work/cache"):
            path.mkdir()
        stopped, process = False, None
        def interrupted(signum, frame):
            raise InterruptedError("owner interrupted")
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, interrupted)
        try:
            stopped = True  # Stop may succeed even if its client is interrupted.
            guard.docker("stop", "--time", "90", plan["target"]["id"])
            require(not guard.inspect(plan["target"]["id"])["State"]["Running"], "target still running")
            free = int(subprocess.check_output(["nvidia-smi", "-i", "4", "--query-gpu=memory.free",
                                                "--format=csv,noheader,nounits"], text=True).strip())
            require(free >= 92 * 1024, "need88GiB allocator plus4GiB reserve")
            status["gpu4_free_mib"] = free
            deadline = time.monotonic() + (plan["maximum_training_seconds"] if stages(plan) == ("train-lora",) else 900)
            for objective in stages(plan):
                validate(plan, for_run=True)
                services(plan, guard, serving=False)
                require(time.monotonic() < deadline, "collection deadline")
                entry = dict(objective=objective, started_epoch=time.time(), command=command(plan, objective, expected_sha))
                status["jobs"].append(entry)
                save(out / (objective + "-start.json"), entry)
                with (out / (objective + ".log")).open("x") as log:
                    process = subprocess.Popen(entry["command"], stdout=log, stderr=subprocess.STDOUT)
                    entry["returncode"] = process.wait(timeout=max(.01, deadline - time.monotonic()))
                entry["finished_epoch"] = time.time()
                require(entry["returncode"] == 0, "collector failed")
                result = out / "work" / objective
                if objective == "train-lora":
                    entry["training_result"] = training_complete(plan, result)
                    status["training_result"] = entry["training_result"]
                elif objective == "finite":
                    entry["finite_identity"] = finite_complete(plan, result)
                else:
                    summary = read(result / "summary.json")
                    require(summary["status"] == "completed" and summary["objective_kind"] == objective
                            and summary["source_values_unchanged"] is True and len(summary["folds"]) == len(expected_cohort(plan))
                            and sum(v["examples"] for v in summary["folds"]) == len(expected_cohort(plan)), "partial collector result")
                    require(summary["coverage"]["native_tensors"] == 421
                            and summary["coverage"]["native_coordinates"] == 1331516, "native inventory differs")
                    require(read(result / "frame-evidence.json") == pinned(plan["frame"]), "measured frame mismatch")
                entry["summary_sha256"] = sha(result / "summary.json")
                save(out / (objective + "-complete.json"), entry)
            validate(plan, for_run=True)
            status["jobs_completed"] = True
            if stages(plan) != ("train-lora",):
                status["collections_completed"] = True
        except BaseException as error:
            status["error_type"] = type(error).__name__
        finally:
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                signal.signal(sig, signal.SIG_IGN)
            finish(plan, guard, process, stopped, status, expected_sha)
            status["finished_epoch"] = time.time()
            status["elapsed_seconds"] = status["finished_epoch"] - status["started_epoch"]
            success = status.get("training_result", {}).get("status", "completed")
            status["status"] = success if (status.get("jobs_completed") and status["restored"]
                and status["protected_exact"] and not any(k.endswith("error_type") for k in status)) else "failed"
            save(out / "completion.json", status)
        require(status["status"] in ("completed", "pilot_only"), "preserved incomplete attempt; inspect completion")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--approved-plan-sha256")
    args = parser.parse_args()
    plan = read(args.plan)
    if args.run:
        try:
            result = run(plan, args.approved_plan_sha256, args.plan)
        except BaseException as error:
            out = Path(plan["output_root"])
            if out.is_dir() and out.resolve().is_relative_to(ROOT / "analysis"):
                save(out / f"failed-launch-{time.time_ns()}.json", dict(error_type=type(error).__name__,
                     plan_sha256=sha(args.plan), model_calls_claim="see launch/completion; preflight may have rejected"))
            raise
        print(json.dumps({k: result[k] for k in ("status", "elapsed_seconds", "restored", "protected_exact")}))
    else:
        ids = validate(plan)
        result = dict(status="preview_no_execution", plan_sha256=sha(args.plan),
                      commands=[command(plan, o, sha(args.plan)) for o in stages(plan)],
                      mode=plan.get("mode", "native"), model_calls=0, service_changes=0)
        result["training_task_ids" if stages(plan) == ("train-lora",) else "case_ids"] = ids
        print(json.dumps(result))


if __name__ == "__main__":
    main()
