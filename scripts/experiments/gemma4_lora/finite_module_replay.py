#!/usr/bin/env python3
"""Replay fixed signed-module LoRA recipes in actual BF16 model RAM.

Preview is the default. No checkpoint payloads are written. CPU caches contain
one original BF16 matrix set and one FP32 2BA set, never six candidate copies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import time

import numpy as np

from kairyu_bench.wsp.weights.checkpoint import Checkpoint
from kairyu_bench.wsp.weights.finite_replay import (
    PublicAnswerEvaluator, prepare_frames, replay_sequence,
)

GIB = 2**30
RESERVED = {"base", "restored_base", "summary", "status", "provenance", "frame-evidence",
            "base-restoration", "ram-weight-receipts"}


def state(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 2**20):
            h.update(block)
    return h.hexdigest()


def captured(path, expected=None):
    path = path.resolve()
    before = state(path)
    if before[2] > 32 * 2**20:
        raise ValueError("bounded JSON input required")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if state(path) != before or expected is not None and digest != expected:
        raise ValueError("JSON hash/stat mismatch")
    return json.loads(raw), digest


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def preflight(args, module):
    checkpoint = Checkpoint(args.checkpoint)
    plan, plan_sha = captured(args.plan)
    if plan.get("schema_version") != 1 or plan.get("architecture") != "gemma4_module_lora_bf16":
        raise ValueError("unsupported module replay plan")
    items = plan.get("candidates")
    if not isinstance(items, list) or not 1 <= len(items) <= 8:
        raise ValueError("one through eight fixed candidates required")
    hashes = {str(args.plan.resolve()): plan_sha}
    documents = {}
    for field in ("cases", "frame_evidence", "checkpoint_attestation"):
        row = plan[field]
        path = Path(row["path"]).resolve()
        documents[field], hashes[str(path)] = captured(path, row["sha256"])
    attested = documents["checkpoint_attestation"]
    identity = attested.get("checkpoint", attested)
    current = checkpoint.identity(hash_files=False)
    if (identity.get("identity_level") != "content_sha256" or not identity.get("hash_files")
            or Path(identity["path"]).resolve() != checkpoint.path.resolve()
            or len(identity["files"]) != len(current["files"])):
        raise ValueError("complete exact source content attestation required")
    for old, new in zip(identity["files"], current["files"], strict=True):
        if (not re.fullmatch("[0-9a-f]{64}", str(old.get("sha256")))
                or any(old[k] != new[k] for k in ("path", "size", "stat"))):
            raise ValueError("source no longer matches prior full-content identity")
    index = checkpoint.path / "model.safetensors.index.json"
    if sha(index) != plan["checkpoint_index_sha256"]:
        raise ValueError("source index hash mismatch")
    candidates, identifiers, common_mapping, common_recipe = [], set(), None, None
    all_states = {}
    header = offset = None
    for row in items:
        identifier = row.get("id")
        if (not isinstance(identifier, str) or not re.fullmatch("[A-Za-z0-9][A-Za-z0-9._-]{0,127}", identifier)
                or identifier in identifiers or identifier in RESERVED):
            raise ValueError("invalid or duplicate candidate ID")
        identifiers.add(identifier)
        path = Path(row["spec"]).resolve()
        spec, hashes[str(path)] = captured(path, row["sha256"])
        if spec.get("id") != identifier:
            raise ValueError("spec ID differs")
        prepared = module.prepared(checkpoint.path, path)
        actual_spec, content, input_states, recipe, mapping, h, off = prepared
        if actual_spec != spec or hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError("spec changed during recipe validation")
        if not re.fullmatch("[0-9a-f]{64}", str(spec.get("base_tree_sha256"))):
            raise ValueError("declared original control-tree SHA required")
        common = {k: recipe[k] for k in (
            "donor_path", "donor_config_path", "donor_sha256", "donor_config_sha256",
            "donor_repo", "donor_revision", "mapping_sha256", "block_rows",
            "implementation_sha256", "numpy_version", "numpy_build_sha256",
        )}
        common["declared_base_tree_sha256"] = spec["base_tree_sha256"]
        if common_mapping is None:
            common_mapping, common_recipe, header, offset = mapping, common, h, off
        elif mapping != common_mapping or common != common_recipe:
            raise ValueError("all fixed candidates must share source/donor/numerical implementation")
        all_states.update(input_states)
        candidates.append({"id": identifier, "spec": spec, "recipe": recipe,
                           "spec_path": str(path), "spec_sha256": row["sha256"],
                           "prepared": prepared})
    assert common_mapping is not None and common_recipe is not None
    code_paths = [Path(__file__).resolve(), Path(module.__file__).resolve(),
                  Path(__import__("kairyu_bench.wsp.weights.finite_replay", fromlist=["x"]).__file__).resolve(),
                  Path(__import__("kairyu_bench.wsp.weights.checkpoint", fromlist=["x"]).__file__).resolve()]
    for name, digest in candidates[0]["recipe"]["implementation_files"].items():
        path = module.dense.LIBRARY if name == "compiled_kernel" else Path(module.__file__).parent / name
        if sha(path) != digest:
            raise ValueError("recipe implementation fingerprint differs")
        code_paths.append(path.resolve())
    for path in code_paths:
        hashes[str(path)] = sha(path)
        all_states[str(path)] = state(path)
    for path in hashes:
        all_states[path] = state(Path(path))
    protected = [checkpoint.path.resolve(), *map(Path, all_states)]
    output = args.output.resolve()
    if args.output.is_symlink() or args.output.exists() or any(
            output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("fresh output disjoint from all inputs required")
    elements = sum(math.prod(row["shape"]) for row in common_mapping)
    if max(math.prod(row["shape"]) * 2 for row in common_mapping) * 4 > GIB:
        raise ValueError("whole-matrix transfer temporaries exceed 1 GiB")
    estimate = elements * 6 + GIB
    if estimate > args.maximum_host_cache_gib * GIB:
        raise ValueError("original plus delta cache exceeds explicit host memory bound")
    checkpoint.assert_stable()
    return checkpoint, plan, candidates, common_mapping, header, offset, documents, hashes, all_states, estimate


class StreamedModuleEdits:
    """One original/delta cache with an explicit all-element storage proof."""
    def __init__(self, checkpoint, params, mapping, recipes, module, header, offset,
                 *, maximum_bytes, deadline=lambda: None, progress=lambda **fields: None, device_proof="roundtrip"):
        import torch
        if device_proof not in {"roundtrip", "staged-int16"}:
            raise ValueError("unsupported device storage proof")
        self.device_proof = device_proof
        self.torch, self.checkpoint, self.params = torch, checkpoint, params
        self.mapping = json.loads(json.dumps(mapping))
        self.recipes = json.loads(json.dumps(recipes))
        self.module, self.header, self.offset = module, header, offset
        self.deadline, self.progress = deadline, progress
        required = sum(math.prod(e["shape"]) for e in mapping) * 6
        if required > maximum_bytes:
            raise ValueError("host cache bound exceeded")
        self.originals, self.deltas, self.original_sha, self.original_ss = {}, {}, {}, {}
        self.ready = False
        self.receipts = {}
        self.restoration = {"storage_bytes_exact": False, "proof_method": device_proof, "matrices": 0}
        if device_proof == "roundtrip":
            self.restoration["readback_exact"] = False
        for item in mapping:
            name, shape = item["tensor"], tuple(item["shape"])
            param = params[name]
            if (param.dtype != torch.bfloat16 or tuple(param.shape) != shape or not param.is_contiguous()
                    or param.requires_grad):
                raise ValueError("exact frozen contiguous BF16 matrices required")

    def raw(self, tensor):
        return tensor.detach().view(self.torch.int16).cpu().numpy().view(np.uint16)

    def prepare(self):
        if self.ready:
            return
        donor = Path(next(iter(self.recipes.values()))["donor_path"])
        for index, item in enumerate(self.mapping):
            name = item["tensor"]
            rows, columns = item["shape"]
            self.deadline()
            # One device transfer per matrix; retain a private host copy even
            # when the CPU fixture's .cpu() would otherwise alias its parameter.
            original = (self.raw(self.params[name]).copy() if self.device_proof == "roundtrip"
                        else np.empty((rows, columns), dtype=np.uint16))
            delta = np.empty((rows, columns), np.float32)
            a = self.module.dense.tensor(donor, self.header, self.offset, item["lora_A"])
            b = self.module.dense.tensor(donor, self.header, self.offset, item["lora_B"])
            hasher, squares = hashlib.sha256(), []
            for first in range(0, rows, self.module.BLOCK_ROWS):
                self.deadline()
                stop = min(rows, first + self.module.BLOCK_ROWS)
                raw = self.checkpoint.read_raw(name, first * columns, (stop - first) * columns)
                block = np.frombuffer(raw, "<u2").reshape(stop - first, columns)
                if self.device_proof == "staged-int16":
                    original[first:stop] = block
                elif not np.array_equal(original[first:stop], block):
                    raise ValueError("loaded source matrix bytes differ from attested checkpoint")
                delta[first:stop] = self.module.delta_block(a, b[first:stop])
                if not np.isfinite(delta[first:stop]).all():
                    raise ValueError("nonfinite cached direction")
                values = (block.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
                if not np.isfinite(values).all():
                    raise ValueError("nonfinite original matrix")
                squares.append(float(np.sum(values * values)))
                hasher.update(raw)
            if self.device_proof == "staged-int16":
                self.staged_check(self.params[name], original, write=False)
            # Publish only fully initialized originals; a failed cache build has
            # not changed model weights and can safely restore this partial set.
            original.flags.writeable = delta.flags.writeable = False
            self.originals[name], self.deltas[name] = original, delta
            self.original_sha[name], self.original_ss[name] = hasher.hexdigest(), math.fsum(squares)
            self.progress(phase="cache", matrices=index + 1, total=len(self.mapping))
        self.checkpoint.assert_stable()
        self.ready = True

    def staged_check(self, target, bits, *, write):
        torch = self.torch
        staging = torch.from_numpy(np.array(bits, dtype=np.uint16, copy=True)).view(torch.bfloat16).to(device=target.device)
        if write:
            target.copy_(staging, non_blocking=False)
        if not torch.equal(target.view(torch.int16), staging.view(torch.int16)):
            raise ValueError("device int16 storage bytes differ from deterministic CPU target")
        # CPU target SHA plus equality of every device storage element. This
        # intentionally does not describe a D2H readback hash.
        return memoryview(bits).cast("B")

    def copy_and_check(self, target, bits):
        if self.device_proof == "staged-int16":
            return self.staged_check(target, bits, write=True)
        torch = self.torch
        host = torch.from_numpy(np.array(bits, dtype=np.uint16, copy=True)).view(torch.bfloat16)
        target.copy_(host, non_blocking=False)
        observed = self.raw(target)
        if not np.array_equal(observed, bits):
            raise ValueError("actual RAM BF16 bytes differ from deterministic merge")
        return observed.tobytes()

    def apply(self, candidate):
        torch = self.torch
        if candidate is None:
            with torch.no_grad():
                for name, original in self.originals.items():
                    self.copy_and_check(self.params[name], original)
            self.restoration = {"storage_bytes_exact": True, "proof_method": self.device_proof,
                                "matrices": len(self.originals), "all_target_matrices": self.ready}
            if self.device_proof == "roundtrip":
                self.restoration["readback_exact"] = True
            return
        identifier = candidate["id"]
        if candidate["recipe"] != self.recipes.get(identifier):
            raise ValueError("RAM application requires unchanged preverified recipe")
        self.prepare()
        recipe, records = self.recipes[identifier], []
        started = time.monotonic()
        with torch.no_grad():
            for item in self.mapping:
                name = item["tensor"]
                original, delta = self.originals[name], self.deltas[name]
                value = recipe["module_coefficients"][name]
                hasher, squares, changed, bytechanged, maximum = hashlib.sha256(), [], 0, 0, 0.
                target_bits = original if value == 0 else np.empty_like(original)
                for first in range(0, len(original), self.module.BLOCK_ROWS):
                    self.deadline()
                    stop = min(len(original), first + self.module.BLOCK_ROWS)
                    if value == 0:
                        merged, ds, nc, bc, peak = original[first:stop], 0., 0, 0, 0.
                    else:
                        merged, stats, counts = self.module.merge_block(original[first:stop], delta[first:stop], value)
                        ds, nc, bc, peak = float(stats[1]), int(counts[0]), int(counts[1]), float(stats[2])
                    if value != 0:
                        target_bits[first:stop] = merged
                    squares.append(ds)
                    changed += nc
                    bytechanged += bc
                    maximum = max(maximum, peak)
                self.deadline()
                # CPU arithmetic remains fixed at 128 rows; only transfer and
                # bitwise verification use the full matrix.
                hasher.update(self.copy_and_check(self.params[name], target_bits))
                del target_bits
                records.append(dict(item, coefficient=value, entry_kind="lora_matrix",
                                    source_tensor_sha256=self.original_sha[name],
                                    candidate_tensor_sha256=hasher.hexdigest(),
                                    candidate_hash_origin=("cpu_target_verified_by_device_int16_equality" if self.device_proof == "staged-int16" else "device_to_host_readback"),
                                    changed_count=changed,
                                    changed_byte_count=bytechanged, source_l2=math.sqrt(self.original_ss[name]),
                                    delta_l2=math.sqrt(math.fsum(squares)), maximum_absolute_delta=maximum))
        metrics = self.module.metrics(records)
        self.receipts[identifier] = {"id": identifier, "edits": records, **metrics,
                                    "actual_ram_readback_verified": self.device_proof == "roundtrip",
                                    "actual_ram_storage_bytes_verified": True, "device_proof": self.device_proof,
                                    "seconds": time.monotonic() - started}
        self.progress(phase="applied", candidate=identifier, **metrics)


def run(args, module, prepared):
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    checkpoint, plan, candidates, mapping, header, offset, documents, hashes, states, estimate = prepared
    if not args.gpu_exclusive_approved or args.device != "cuda:0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("owner approval and exactly one exposed GPU required")
    if torch.cuda.mem_get_info()[0] < 70 * GIB:
        raise ValueError("at least 70 GiB GPU memory required")
    if any(os.environ.get(k) != "4" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")):
        raise ValueError("launch with OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=4")
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    if available < estimate + 32 * GIB:
        raise ValueError("host cache plus 32 GiB free headroom unavailable")
    cgroup = Path("/sys/fs/cgroup")
    memory_budget = {"cache_estimate_bytes": estimate, "available_host_bytes": available,
                     "cgroup_limit_bytes": None, "cgroup_current_bytes": None}
    if (cgroup / "memory.max").exists():
        maximum = (cgroup / "memory.max").read_text().strip()
        if maximum != "max":
            limit = int(maximum)
            current_memory = int((cgroup / "memory.current").read_text())
            # The checkpoint's file-backed pages can count against the same
            # cgroup while the original/delta cache remains live.
            required = estimate + sum(p.stat().st_size for p in checkpoint.files) + 16 * GIB
            memory_budget.update(cgroup_limit_bytes=limit, cgroup_current_bytes=current_memory,
                                 required_additional_bytes=required)
            if limit - current_memory < required:
                raise ValueError("container memory cap cannot cover cache, checkpoint pages and 16 GiB headroom")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    def note(**fields):
        save(args.output / "status.json", {"elapsed_seconds": time.monotonic() - started, **fields})
    def deadline():
        if time.monotonic() - started > args.maximum_seconds - 120:
            raise TimeoutError("module replay deadline; reserve final restoration")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261105)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.path, local_files_only=True)
    frames, frame_evidence = prepare_frames(tokenizer, documents["cases"])
    if frame_evidence != documents["frame_evidence"]:
        raise ValueError("fixed official chat frame differs")
    save(args.output / "frame-evidence.json", frame_evidence)
    provenance = {"plan_sha256": hashes[str(args.plan.resolve())], "input_sha256": hashes,
                  "checkpoint_identity": checkpoint.identity(hash_files=False),
                  "prior_source_attestation": plan["checkpoint_attestation"],
                  "full_source_rehashed_here": False, "host_cache_estimate_bytes": estimate,
                  "memory_budget": memory_budget,
                  "device_proof": args.device_proof,
                  "device_proof_contract": ("CPU target SHA plus all-element device int16 equality; no D2H hash"
                                            if args.device_proof == "staged-int16" else "full device-to-host bytes compared and hashed"),
                  "transfer_granularity": "one complete BF16 matrix; all CPU GEMM/merge blocks remain 128 rows",
                  "maximum_transient_transfer_bytes": max(math.prod(row["shape"]) * 2 for row in mapping) * 4,
                  "numeric_recipe": candidates[0]["recipe"],
                  "declared_control_tree_sha256": candidates[0]["spec"]["base_tree_sha256"],
                  "model_loads": 1,
                  "scope": "Actual BF16 merged RAM weights; public-answer teacher-forced diagnostics, not generation or WSP."}
    save(args.output / "provenance.json", provenance)
    note(phase="loading")
    model = Gemma4ForConditionalGeneration.from_pretrained(
        checkpoint.path, dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="eager", device_map={"": args.device}).eval()
    model.requires_grad_(False)
    net, head = model.model.language_model, model.get_output_embeddings()
    params = {"model.language_model." + name: value for name, value in net.named_parameters()}
    replay = StreamedModuleEdits(checkpoint, params, mapping, {c["id"]: c["recipe"] for c in candidates},
                                module, header, offset, maximum_bytes=args.maximum_host_cache_gib * GIB,
                                deadline=deadline, progress=note, device_proof=args.device_proof)
    evaluator = PublicAnswerEvaluator(net, head, tokenizer, frames, args.device, deadline,
                                      answer_batching=args.answer_batching)
    def on_result(identifier, result):
        save(args.output / (identifier + ".json"), result)
        save(args.output / "ram-weight-receipts.json", replay.receipts)
        note(phase="sweep_complete", candidate=identifier, seconds=result["seconds"])
    def before_restore():
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, signal.SIG_IGN)
        torch.cuda.empty_cache()
    base, results, repeat, failure = replay_sequence(candidates, replay.apply, evaluator, on_result, before_restore)
    restoration = dict(replay.restoration, forward_completed=repeat is not None, matched=False)
    if base is not None and repeat is not None:
        margin_error = max(abs(x - y) for name, folds in base["values"].items()
                           for xs, ys in zip(folds, repeat["values"][name], strict=True)
                           for x, y in zip(xs, ys, strict=True))
        nll_error = float(np.max(np.abs(np.array(repeat["correct_nll"]) - base["correct_nll"])))
        kl_error = max(repeat["kl"])
        restoration.update(max_margin_error=margin_error, max_correct_nll_error=nll_error, max_kl=kl_error,
                           matched=restoration["storage_bytes_exact"] and margin_error <= 1e-4
                           and nll_error <= 1e-4 and kl_error <= 1e-6)
    stable = True
    try:
        checkpoint.assert_stable()
        if any(state(Path(p)) != tuple(s) for p, s in states.items()) or any(sha(Path(p)) != h for p, h in hashes.items()):
            raise ValueError("source/provenance/implementation changed")
        for candidate in candidates:
            spec, content, before, recipe, mapped, _, _ = candidate["prepared"]
            module.final_inputs(checkpoint.path, Path(candidate["spec_path"]), spec, content, before, recipe, mapped)
    except BaseException as error:
        stable = False
        failure = failure or error
    diagnostics = []
    if base is not None:
        for candidate, result in zip(candidates, results):
            receipt = replay.receipts[candidate["id"]]
            diagnostics.append({"id": candidate["id"], "spec_sha256": candidate["spec_sha256"],
                                "changed_elements": receipt["changed_elements"], "delta_l2": receipt["delta_l2"],
                                "continuous_delta_l2": candidate["recipe"]["continuous_delta_l2"],
                                "margin_delta": (np.array(result["means"]) - base["means"]).tolist(),
                                "correct_nll_delta": (np.array(result["correct_nll"]) - base["correct_nll"]).tolist(),
                                "kl_base_to_candidate": result["kl"]})
    status = "completed" if failure is None and restoration["matched"] and len(results) == len(candidates) else "incomplete"
    summary = {"status": status, "device_proof": args.device_proof, "answer_batching": args.answer_batching, "labels": [f"{g}-{i}" for g, fs in frames.items() for i in range(len(fs))],
               "candidates": diagnostics, "base_restoration": restoration, "source_stable": stable,
               "error_type": None if failure is None else type(failure).__name__,
               "elapsed_seconds": time.monotonic() - started,
               "gpu_peak_allocated_GiB": torch.cuda.max_memory_allocated() / GIB,
               "gpu_peak_reserved_GiB": torch.cuda.max_memory_reserved() / GIB,
               "model_loads": 1, "checkpoint_writes": 0, "generated_tokens": 0, "wsp_calls": 0,
               "limits": ["Local continuous predictions are separate from these actual BF16 finite responses.",
                          "KL is full-vocabulary only at correct-answer first/last token positions, not all generated tokens.",
                          "No universal nonregression or capability acceptance is inferred.",
                          "Future persistent candidate must match all recorded per-matrix BF16 SHA values; numerical backend changes are not presumed identical.",
                          "Final restoration may exceed the time budget; process/host loss requires owner recovery of serving."]}
    save(args.output / "ram-weight-receipts.json", replay.receipts)
    save(args.output / "base-restoration.json", restoration)
    save(args.output / "summary.json", summary)
    note(phase=status)
    if failure is not None:
        raise failure
    if status != "completed":
        raise RuntimeError("restored-base verification failed")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-seconds", type=int, default=1200)
    parser.add_argument("--maximum-host-cache-gib", type=float, default=192)
    parser.add_argument("--device", choices=["cuda:0"], default="cuda:0")
    parser.add_argument("--answer-batching", choices=("paired", "serial"), default="paired",
                        help="serial: one public answer per forward, with each correct/foil pair's shared padding width")
    parser.add_argument("--device-proof", choices=("roundtrip", "staged-int16"), default="roundtrip",
                        help="staged-int16 proves CPU target SHA by all-element device equality; default retains full D2H readback")
    parser.add_argument("--gpu-exclusive-approved", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if not 180 <= args.maximum_seconds <= 1200 or not math.isfinite(args.maximum_host_cache_gib) or args.maximum_host_cache_gib <= 0:
        parser.error("bounded positive memory and 180..1200 second deadline required")
    import module_lora_recipe as module
    prepared = preflight(args, module)
    if not args.run:
        print(json.dumps({"status": "preview", "candidates": len(prepared[2]), "matrices": len(prepared[3]),
                          "host_cache_estimate_bytes": prepared[-1], "device_proof": args.device_proof,
                          "answer_batching": args.answer_batching,
                          "model_calls": 0, "checkpoint_writes": 0}))
        return 0
    def interrupted(signum, frame):
        raise InterruptedError(f"module replay interrupted by signal {signum}")
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    run(args, module, prepared)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
