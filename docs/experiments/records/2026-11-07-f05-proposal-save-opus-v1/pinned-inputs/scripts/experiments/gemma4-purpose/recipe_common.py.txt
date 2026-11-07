"""Shared I/O and objective definitions for the two explicit Gemma 4 recipes."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from kairyu_bench.wsp.weights.checkpoint import Checkpoint
from kairyu_bench.wsp.weights import finite_replay
from kairyu_bench.wsp.weights.finite_replay import batches as batches
from kairyu_bench.wsp.weights.finite_replay import prepare_frames
from kairyu_bench.wsp.weights.finite_replay import validate_observation_policy as validate_observation_policy


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def objective_batches(rows: list, mode: str, max_cases_per_microbatch: int | None = None, execution_order: str = "original") -> list[list]:
    """Retain existing packing/order, optionally split each pack into bounded case groups."""
    if max_cases_per_microbatch is not None and (type(max_cases_per_microbatch) is not int or max_cases_per_microbatch < 1):
        raise ValueError("max cases per microbatch must be a positive integer")
    if execution_order == "longest_frame_first":
        if mode != "chat-public-answer" or max_cases_per_microbatch != 1:
            raise ValueError("longest-first requires one complete public-answer case")
        return [[row] for row in sorted(rows, key=lambda r: len(r[1]) + max(len(r[2]), len(r[3])), reverse=True)]
    if execution_order != "original":
        raise ValueError("invalid observation execution order")
    packed = batches(rows) if mode == "chat-public-answer" else [rows]
    if max_cases_per_microbatch is None:
        return packed
    return [batch[start:start + max_cases_per_microbatch] for batch in packed
            for start in range(0, len(batch), max_cases_per_microbatch)]


def parser(description: str, *, chat: bool) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--cases", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--checkpoint-attestation", type=Path)
    result.add_argument("--attestation-sha256")
    result.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--max-cases-per-microbatch", type=positive_int,
                        help="split each existing pack into at most this many correct/foil pairs")
    result.add_argument("--reference-tolerance", type=float, default=1e-5)
    result.add_argument("--run", action="store_true", help="otherwise validate inputs and print a preview only")
    result.add_argument("--gpu-exclusive-approved", action="store_true")
    finite_replay.add_observation_arguments(result)
    if chat:
        result.add_argument("--mode", choices=("raw-next-token", "chat-public-answer"), default="chat-public-answer")
    else:
        result.set_defaults(mode="raw-next-token")
    return result


def read_cases(path: Path, *, content: bytes | None = None) -> dict[str, list[list[dict[str, Any]]]]:
    """Accept explicit correct/foil cases or the old raw observation cases."""
    document = json.loads(path.read_bytes() if content is None else content)
    groups = document.get("cases", document) if isinstance(document, dict) else None
    if not isinstance(groups, dict) or not groups:
        raise ValueError("cases must map purpose names to nonempty folds")
    if document.get("gradient_allowed") is False or document.get("split") in ("transfer", "test", "holdout"):
        raise ValueError("declared transfer/evaluation cases cannot be gradient input")
    normalized: dict[str, list[list[dict[str, Any]]]] = {}
    count = 0
    for name, folds in groups.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", name):
            raise ValueError("purpose name must be a safe file component")
        if not isinstance(folds, list) or not folds:
            raise ValueError("each purpose needs nonempty folds")
        normalized[name] = []
        for index, fold in enumerate(folds):
            if not isinstance(fold, list) or not fold:
                raise ValueError("fold must be nonempty")
            rows, identifiers = [], set()
            for position, row in enumerate(fold):
                if not isinstance(row, dict):
                    raise ValueError("each case must be an object")
                if row.get("gradient_allowed") is False or row.get("split") in ("transfer", "test", "holdout"):
                    raise ValueError("declared transfer/evaluation cases cannot be gradient input")
                case = {"id": row.get("id", f"{name}:{index}:{position}"),
                        "prompt": row.get("prompt"),
                        "correct": row.get("correct", row.get("positive")),
                        "foil": row.get("foil", row.get("negative"))}
                if any(not isinstance(v, str) or not v or len(v) > 65536 for v in case.values()):
                    raise ValueError("case id/prompt/correct/foil must be nonempty bounded strings")
                if case["id"] in identifiers:
                    raise ValueError("duplicate case id within a fold")
                identifiers.add(case["id"])
                reference = row.get("expected_margin", row.get("margin"))
                if reference is not None:
                    if isinstance(reference, bool) or not isinstance(reference, (int, float)) or not np.isfinite(reference):
                        raise ValueError("expected_margin must be finite")
                    case["expected_margin"] = reference
                rows.append(case)
                count += 1
            normalized[name].append(rows)
    if count > 4096:
        raise ValueError("at most 4096 explicitly supplied calibration cases")
    return normalized


def preflight(args: argparse.Namespace, script: Path) -> tuple[Checkpoint, dict, dict]:
    if not 1 <= args.threads <= 32 or not np.isfinite(args.reference_tolerance) or args.reference_tolerance < 0:
        raise ValueError("invalid thread count or reference tolerance")
    if args.max_cases_per_microbatch is not None and args.max_cases_per_microbatch < 1:
        raise ValueError("max cases per microbatch must be positive")
    if args.output.is_symlink() or args.output.exists():
        raise ValueError("output must be a new, non-symlink directory")
    inputs = [args.checkpoint.resolve(), args.cases.resolve(), script.resolve(), Path(__file__).resolve(), Path(finite_replay.__file__).resolve()]
    if args.checkpoint_attestation:
        inputs.append(args.checkpoint_attestation.resolve())
    output = args.output.resolve()
    if any(output == p or output in p.parents or p in output.parents for p in inputs):
        raise ValueError("output must be disjoint from checkpoint and input files")
    file_inputs = [p for p in inputs if p.is_file()]
    input_states = {str(p): input_state(p) for p in file_inputs}
    checkpoint = Checkpoint(args.checkpoint)
    case_content = inputs[1].read_bytes()
    cases = read_cases(inputs[1], content=case_content)
    hashes = {str(p): (hashlib.sha256(case_content).hexdigest() if p == inputs[1] else sha(p)) for p in file_inputs}
    current = checkpoint.identity(hash_files=False)
    receipt_sha = None
    if args.checkpoint_attestation:
        receipt_sha = sha(args.checkpoint_attestation)
        if receipt_sha != args.attestation_sha256:
            raise ValueError("checkpoint attestation receipt SHA mismatch")
        receipt = json.loads(args.checkpoint_attestation.read_text())
        identity = receipt.get("checkpoint", receipt)
        if identity.get("hash_files") is not True or identity.get("identity_level") != "content_sha256":
            raise ValueError("a prior full-content checkpoint attestation is required")
        previous = identity["files"]
        if len(previous) != len(current["files"]):
            raise ValueError("checkpoint inventory differs from attestation")
        for old, new in zip(previous, current["files"]):
            if (not re.fullmatch(r"[0-9a-f]{64}", str(old.get("sha256", "")))
                    or any(old[k] != new[k] for k in ("path", "size", "stat"))):
                raise ValueError("checkpoint stat no longer matches content attestation")
    elif args.run:
        raise ValueError("--run requires checkpoint-attestation and its reviewed SHA256")
    if args.device != "cpu" and not args.gpu_exclusive_approved:
        raise ValueError("CUDA requires an explicit exclusive-owner handoff")
    policy = finite_replay.observation_policy(args, intrinsic_serial=getattr(args, "_intrinsic_serial", False))
    provenance = {"checkpoint": current, "full_tree_rehashed_here": False,
                  "attestation_receipt_sha256": receipt_sha, "input_sha256": hashes, "input_stat": input_states,
                  "calibration_scope": "Explicit caller cases only; no evaluation dataset/result lookup or independent holdout claim",
                  "mode": args.mode, "device": args.device, "threads": args.threads, "seed": args.seed,
                  "max_cases_per_microbatch": args.max_cases_per_microbatch,
                  "purposes": len(cases), "folds": sum(len(v) for v in cases.values()),
                  "cases": sum(len(f) for v in cases.values() for f in v),
                  "model_forwards": "not_run" if not args.run else "explicitly_requested",
                  "new_checkpoints": 0, "wsp_calls": 0}
    if policy is not None:
        provenance["observation_policy"] = policy
    assert_inputs(checkpoint, provenance)
    return checkpoint, cases, provenance


def input_state(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"device": stat.st_dev, "inode": stat.st_ino, "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}


def assert_inputs(checkpoint: Checkpoint, provenance: dict) -> None:
    checkpoint.assert_stable()
    if any(input_state(Path(name)) != state for name, state in provenance.get("input_stat", {}).items()):
        raise ValueError("recipe input stat changed during collection")
    for name, digest in provenance["input_sha256"].items():
        if sha(Path(name)) != digest:
            raise ValueError("recipe input changed during collection")


def transformers_version() -> str:
    try:
        return version("transformers")
    except PackageNotFoundError:
        return "unknown"


def load_model(args: argparse.Namespace, *, cases: dict | None = None, legacy_limit: int | None = None):
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    policy = finite_replay.observation_policy(args, intrinsic_serial=getattr(args, "_intrinsic_serial", False))
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    if args.device != "cpu":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.mem_get_info()[0] < 70 * 2**30:
            raise ValueError("exactly one exposed GPU with at least 70 GiB free is required")
        torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    if cases is not None:
        prepare(tokenizer, cases, args.mode, max_frame_tokens=frame_limit(args, legacy_limit))
    allocator = finite_replay.configure_observation_cuda(torch, args.device, policy)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, local_files_only=True,
        attn_implementation=policy["attention_dispatch"] if policy else "eager", device_map={"": args.device}).eval()
    model.requires_grad_(False)
    if policy is not None:
        args._observation_execution = finite_replay.resolved_observation_execution(model, policy, allocator)
    return torch, tokenizer, model


def raw_frames(tokenizer: Any, cases: dict) -> tuple[dict, dict]:
    prepared: dict[str, list] = {}
    evidence: dict[str, list] = {}
    for group, folds in cases.items():
        prepared[group], evidence[group] = [], []
        for fold in folds:
            rows, proof = [], []
            for case in fold:
                prefix = tokenizer.encode(case["prompt"], add_special_tokens=True)
                targets = []
                for key in ("correct", "foil"):
                    full = tokenizer.encode(case["prompt"] + case[key], add_special_tokens=True)
                    if not prefix or full[:len(prefix)] != prefix or len(full) <= len(prefix):
                        raise ValueError("raw continuation token boundary mismatch")
                    targets.append(full[len(prefix)])
                if targets[0] == targets[1]:
                    raise ValueError("raw next-token objective requires distinct first continuation tokens")
                rows.append((case["id"], prefix, [targets[0]], [targets[1]]))
                proof.append({"id": case["id"], "prefix_tokens": len(prefix),
                              "prefix_sha256": hashlib.sha256(case["prompt"].encode()).hexdigest(),
                              "correct_token": targets[0], "foil_token": targets[1]})
            prepared[group].append(rows)
            evidence[group].append(proof)
    return prepared, {"scope": "raw prompt, first continuation token only; no chat/thinking frame",
                      "frames": evidence}


def execution_order(args: argparse.Namespace) -> str:
    return getattr(args, "execution_order", "original")


def frame_limit(args: argparse.Namespace, legacy: int | None = None) -> int | None:
    explicit = getattr(args, "max_frame_tokens", None)
    return legacy if explicit is None else explicit


def observation_runtime(args: argparse.Namespace, provenance: dict) -> dict:
    policy = provenance.get("observation_policy")
    if policy is None:
        return {"attention": "eager"}
    validate_observation_policy(policy)
    return {"attention": policy["attention_dispatch"]["text_config"],
            "observation_policy": policy,
            "observation_execution": args._observation_execution}


def prepare(tokenizer: Any, cases: dict, mode: str, *, max_frame_tokens: int | None = None) -> tuple[dict, dict]:
    return (prepare_frames(tokenizer, cases, max_frame_tokens=max_frame_tokens)
            if mode == "chat-public-answer" else raw_frames(tokenizer, cases))


def objective(torch: Any, net: Any, head: Any, tokenizer: Any, batch: list, mode: str, device: str):
    cap = float(net.config.final_logit_softcapping)
    if mode == "raw-next-token":
        seq = [row[1] for row in batch]
        width = max(map(len, seq))
        ids = torch.tensor([row + [tokenizer.pad_token_id] * (width - len(row)) for row in seq], device=device)
        mask = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in seq], device=device)
        hidden = net(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
        last = hidden[torch.arange(len(seq), device=device), torch.tensor([len(row) - 1 for row in seq], device=device)].float()
        scores = []
        for target in (2, 3):
            indices = torch.tensor([row[target][0] for row in batch], device=device)
            raw = (last * head.weight[indices].detach().float()).sum(1)
            scores.append(cap * torch.tanh(raw / cap) if cap else raw)
        return scores[0] - scores[1]
    seq, lengths, targets = [], [], []
    for _, prefix, good, bad in batch:
        for answer in (good, bad):
            seq.append(prefix + answer)
            lengths.append(len(prefix))
            targets.append(answer)
    width = max(map(len, seq))
    ids = torch.tensor([row + [tokenizer.pad_token_id] * (width - len(row)) for row in seq], device=device)
    mask = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in seq], device=device)
    hidden = net(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
    selected = torch.cat([hidden[i, n - 1:len(seq[i]) - 1] for i, n in enumerate(lengths)])
    logits = head(selected).float()
    if cap:
        logits = cap * torch.tanh(logits / cap)
    answer = torch.tensor([token for row in targets for token in row], device=device)
    logp = torch.log_softmax(logits, -1).gather(1, answer[:, None]).flatten()
    means, at = [], 0
    for row in targets:
        means.append(logp[at:at + len(row)].mean())
        at += len(row)
    means = torch.stack(means)
    return means[::2] - means[1::2]


def check_references(cases: list, measured: dict, tolerance: float) -> dict:
    errors = [abs(measured[row["id"]] - row["expected_margin"]) for row in cases if "expected_margin" in row]
    maximum = max(errors, default=None)
    if maximum is not None and maximum > tolerance:
        raise ValueError("stored reference margin does not match these replay conditions")
    return {"reference_cases": len(errors), "max_reference_error": maximum}
