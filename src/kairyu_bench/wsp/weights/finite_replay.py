"""Finite-only Gemma 4 native/sparse-matrix edit replay: original, <=8 sets, original.

Optional torch/transformers dependencies are imported only when explicitly run.
No checkpoint writes, gradients, free generation, solver, or WSP acceptance gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import signal
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from kairyu_bench.wsp.weights import checkpoint as checkpoint_reader
from kairyu_bench.wsp.weights.checkpoint import Checkpoint


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)



def add_observation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-frame-tokens", type=int, choices=(1024, 2048),
                        help="opt-in complete prefix+answer limit; reject oversize, never truncate")
    parser.add_argument("--text-attention-backend", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--execution-order", choices=("original", "longest_frame_first"), default="original")
    parser.add_argument("--cuda-memory-cap-gib", type=float,
                        help="optional CUDA caching-allocator limit, not total device memory")


def validate_observation_policy(policy: Any) -> dict[str, Any]:
    keys = {"max_frame_tokens", "attention_dispatch", "cuda_memory_cap_gib",
            "answer_batching", "max_cases_per_microbatch", "execution_order"}
    if not isinstance(policy, dict) or set(policy) != keys:
        raise ValueError("invalid explicit observation policy")
    if policy["execution_order"] not in ("original", "longest_frame_first"):
        raise ValueError("invalid observation execution order")
    limit, cap = policy["max_frame_tokens"], policy["cuda_memory_cap_gib"]
    dispatch = policy["attention_dispatch"]
    if (limit is not None and (type(limit) is not int or limit not in (1024, 2048))):
        raise ValueError("explicit frame limit must be 1024 or 2048")
    if (not isinstance(dispatch, dict) or set(dispatch) != {"", "text_config", "vision_config", "audio_config"}
            or dispatch["text_config"] not in ("eager", "sdpa")
            or any(dispatch[key] != "eager" for key in ("", "vision_config", "audio_config"))):
        raise ValueError("only explicit text-only eager/SDPA dispatch is supported")
    if dispatch["text_config"] == "sdpa" and limit is None:
        raise ValueError("SDPA requires an explicit complete-frame token limit")
    if (cap is not None and (isinstance(cap, bool) or not isinstance(cap, (int, float))
                             or not np.isfinite(cap) or cap <= 0)):
        raise ValueError("CUDA allocator cap must be positive and finite")
    if policy["answer_batching"] != "serial" or type(policy["max_cases_per_microbatch"]) is not int or policy["max_cases_per_microbatch"] != 1:
        raise ValueError("explicit observation policy requires one serial case/answer")
    return policy


def observation_policy(args: argparse.Namespace, *, intrinsic_serial: bool = False) -> dict[str, Any] | None:
    limit = getattr(args, "max_frame_tokens", None)
    backend = getattr(args, "text_attention_backend", "eager")
    cap = getattr(args, "cuda_memory_cap_gib", None)
    order = getattr(args, "execution_order", "original")
    if limit is None and backend == "eager" and cap is None and order == "original":
        return None  # Preserve historical absent-policy behavior and artifacts.
    if getattr(args, "mode", "chat-public-answer") != "chat-public-answer":
        raise ValueError("explicit observation policy requires complete public-answer frames")
    if cap is not None and args.device != "cuda:0":
        raise ValueError("CUDA allocator cap requires cuda:0")
    return validate_observation_policy({
        "max_frame_tokens": limit, "execution_order": order,
        "attention_dispatch": {"": "eager", "text_config": backend,
                               "vision_config": "eager", "audio_config": "eager"},
        "cuda_memory_cap_gib": cap,
        "answer_batching": "serial" if intrinsic_serial else getattr(args, "answer_batching", "paired"),
        "max_cases_per_microbatch": 1 if intrinsic_serial else getattr(args, "max_cases_per_microbatch", None),
    })


def validate_frame_rows(rows: list[Any], maximum: int | None) -> None:
    if maximum is not None and any(len(row[1]) + max(len(row[2]), len(row[3])) > maximum for row in rows):
        raise ValueError(f"complete prefix plus answer exceeds {maximum} tokens; no truncation")



def ordered_folds(prepared: dict[str, Any], execution_order: str = "original") -> list[tuple[str, int, list[Any]]]:
    """Order execution only; callers retain original fold/case labels in artifacts."""
    result = [(group, fold, rows) for group, folds in prepared.items() for fold, rows in enumerate(folds)]
    if execution_order == "original":
        return result
    if execution_order != "longest_frame_first":
        raise ValueError("invalid observation execution order")
    def width(row: Any) -> int:
        return len(row[1]) + max(len(row[2]), len(row[3]))
    result.sort(key=lambda item: max(map(width, item[2])), reverse=True)
    return [(group, fold, sorted(rows, key=width, reverse=True)) for group, fold, rows in result]


def configure_observation_cuda(torch: Any, device: str, policy: dict[str, Any] | None) -> dict[str, Any] | None:
    if policy is None or policy["cuda_memory_cap_gib"] is None:
        return None
    validate_observation_policy(policy)
    if device != "cuda:0":
        raise ValueError("CUDA allocator cap requires cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    requested = policy["cuda_memory_cap_gib"] * 2**30
    if requested > free or requested > total:
        raise ValueError("CUDA allocator cap exceeds available/total device memory")
    fraction = requested / total
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    return {"requested_gib": policy["cuda_memory_cap_gib"], "fraction_of_total": fraction,
            "free_before_load_bytes": free, "total_device_bytes": total,
            "scope": "PyTorch caching allocator only; not a cap on all device allocations"}


def resolved_observation_execution(model: Any, policy: dict[str, Any], allocator: Any) -> dict[str, Any]:
    requested = validate_observation_policy(policy)["attention_dispatch"]
    config = model.config
    resolved = {key: getattr(config if key == "" else getattr(config, key, None),
                             "_attn_implementation", None) for key in requested}
    absent = [key for key in ("vision_config", "audio_config") if getattr(config, key, None) is None]
    if any(resolved[key] != expected for key, expected in requested.items() if key not in absent):
        raise ValueError("loaded attention dispatch differs from explicit observation policy")
    return {"resolved_attention_dispatch": resolved, "absent_subconfigs": absent, "cuda_allocator": allocator,
            "backend_policy": ("Transformers text SDPA dispatch uses PyTorch automatic kernels, including math; fused/GQA memory savings not asserted"
                               if requested["text_config"] == "sdpa" else "Transformers eager dispatch for all subconfigs"),
            "automatic_attention_api_or_length_retry": False}

def batches(rows: list[Any]) -> list[list[Any]]:
    """Preserve the original calibration's deterministic 1600-token packing."""
    ordered = sorted(rows, key=lambda row: max(len(row[1]) + len(row[2]), len(row[1]) + len(row[3])))
    result: list[list[Any]] = []
    current: list[Any] = []
    for row in ordered:
        trial = current + [row]
        width = max(len(x[1]) + max(len(x[2]), len(x[3])) for x in trial)
        if current and len(trial) * 2 * width > 1600:
            result.append(current)
            current = []
        current.append(row)
    if current:
        result.append(current)
    return result


def load_plan(path: Path, checkpoint: Checkpoint) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    plan = json.loads(path.read_text())
    if plan.get("schema_version") != 1 or plan.get("architecture") not in {"gemma4_native_bf16", "gemma4_sparse_bf16"}:
        raise ValueError("unsupported finite replay plan")
    if "observation_policy" in plan:
        validate_observation_policy(plan["observation_policy"])
    entries = plan.get("candidates")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 8:
        raise ValueError("one through eight fixed candidate specs required")
    hashes = {str(path.resolve()): _sha(path), str(Path(__file__).resolve()): _sha(Path(__file__)),
              str(Path(checkpoint_reader.__file__).resolve()): _sha(Path(checkpoint_reader.__file__))}
    for field in ("cases", "frame_evidence"):
        item = plan[field]
        source = Path(item["path"])
        if _sha(source) != item["sha256"]:
            raise ValueError(f"{field} SHA mismatch")
        hashes[str(source.resolve())] = item["sha256"]
    attestation = plan["checkpoint_attestation"]
    receipt_path = Path(attestation["path"])
    if _sha(receipt_path) != attestation["sha256"]:
        raise ValueError("checkpoint attestation receipt SHA mismatch")
    receipt = json.loads(receipt_path.read_text())
    identity = receipt.get("checkpoint", receipt)
    if not identity.get("hash_files") or identity.get("identity_level") != "content_sha256":
        raise ValueError("prior complete file-content attestation required")
    current = checkpoint.identity(hash_files=False)
    expected_files = identity["files"]
    if len(expected_files) != len(current["files"]):
        raise ValueError("checkpoint attested inventory changed")
    for old, new in zip(expected_files, current["files"]):
        if (not isinstance(old.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", old["sha256"]) is None
                or any(old[key] != new[key] for key in ("path", "size", "stat"))):
            raise ValueError("checkpoint stat no longer matches content attestation")
    hashes[str(receipt_path.resolve())] = attestation["sha256"]
    if _sha(checkpoint.path / "model.safetensors.index.json") != plan["checkpoint_index_sha256"]:
        raise ValueError("checkpoint index SHA mismatch")
    specs, identifiers = [], set()
    for item in entries:
        spec_path = Path(item["spec"])
        if _sha(spec_path) != item["sha256"]:
            raise ValueError("candidate spec SHA mismatch")
        hashes[str(spec_path.resolve())] = item["sha256"]
        spec = json.loads(spec_path.read_text())
        if (spec.get("id") != item["id"] or not isinstance(spec.get("id"), str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", spec["id"]) is None
                or spec["id"] in {"base", "restored_base", "summary", "status", "provenance", "frame-evidence", "base-restoration"}
                or spec["id"] in identifiers):
            raise ValueError("candidate ID mismatch or duplicate")
        identifiers.add(spec["id"])
        edits = spec.get("edits")
        if not isinstance(edits, list) or not edits:
            raise ValueError("nonempty edits required")
        seen, count, square = set(), 0, 0.0
        for edit in edits:
            name = edit.get("tensor")
            tensor = checkpoint.tensors.get(name) if isinstance(name, str) else None
            matrix = "coordinates" in edit
            if (tensor is None or not isinstance(name, str) or not name.startswith("model.language_model.")
                    or name in seen or tensor.dtype != "BF16"):
                raise ValueError("unique existing text BF16 tensors required")
            if matrix:
                if plan["architecture"] != "gemma4_sparse_bf16" or len(tensor.shape) != 2:
                    raise ValueError("rank-2 edits require an explicit sparse BF16 plan")
                if set(edit) != {"tensor", "coordinates", "set_values"}:
                    raise ValueError("matrix replay accepts exact coordinate set_values only")
                coordinates = edit["coordinates"]
                if (not isinstance(coordinates, list) or not coordinates or len(coordinates) > 512
                        or any(not isinstance(cell, list) or len(cell) != 2
                               or any(type(x) is not int for x in cell)
                               or not 0 <= cell[0] < tensor.shape[0]
                               or not 0 <= cell[1] < tensor.shape[1] for cell in coordinates)):
                    raise ValueError("invalid matrix coordinates")
                indices = [row * tensor.shape[1] + column for row, column in coordinates]
            else:
                if set(edit) != {"tensor", "indices", "set_values"} or len(tensor.shape) > 1:
                    raise ValueError("native replay accepts exact vector/scalar set_values only")
                indices = edit["indices"]
            seen.add(name)
            values = edit["set_values"]
            if (not isinstance(indices, list) or not indices or not isinstance(values, list)
                    or len(indices) != len(values)
                    or any(type(i) is not int or i < 0 or i >= tensor.numel for i in indices)
                    or len(set(indices)) != len(indices)
                    or any(type(v) not in (int, float) for v in values)):
                raise ValueError("invalid exact coordinates/values")
            v64 = np.asarray(values, dtype=np.float64)
            v32 = v64.astype(np.float32)
            if (not np.isfinite(v64).all() or not np.array_equal(v64, v32.astype(np.float64))
                    or np.any(v32.view(np.uint32) & 0xFFFF)):
                raise ValueError("set_values must already be finite exact BF16")
            original = checkpoint.read_indices(name, indices)
            if not np.isfinite(original).all():
                raise ValueError("source coordinates must be finite")
            difference = v64 - original
            if np.any(difference == 0):
                raise ValueError("declared coordinates must be non-noop")
            count += len(indices)
            if count > 1000000:
                raise ValueError("total edit count exceeds one million")
            if matrix and sum(len(e.get("coordinates", [])) for e in edits) > 262144:
                raise ValueError("matrix edit count exceeds 262,144")
            square += float(difference @ difference)
            edit["_flat_indices"] = indices
        spec["_verified_metrics"] = {"changed_elements": count, "changed_tensors": len(edits),
                                     "delta_l2": float(np.sqrt(square))}
        specs.append(spec)
    checkpoint.assert_stable()
    return plan, specs, hashes


def replay_sequence(
    candidates: list[Any], apply: Callable[[Any | None], None],
    evaluate: Callable[..., Any], on_result: Callable[[str, Any], None],
    before_restore: Callable[[], None] = lambda: None,
) -> tuple[Any | None, list[Any], Any | None, BaseException | None]:
    """Always attempt a restored-base sweep after any evaluation failure."""
    baseline, results, repeat, failure = None, [], None, None
    try:
        apply(None)
        baseline = evaluate(store_base=True)
        on_result("base", baseline)
        for candidate in candidates:
            apply(candidate)
            result = evaluate(store_base=False)
            results.append(result)
            on_result(candidate["id"], result)
    except BaseException as error:
        # Avoid retaining CUDA tensor frames when recovering from an OOM.
        failure = error.with_traceback(None)
        failure.__context__ = failure.__cause__ = None
    finally:
        try:
            before_restore()
            apply(None)
            repeat = evaluate(store_base=baseline is None, recovery=True)
            on_result("restored_base", repeat)
        except BaseException as error:
            if failure is None:
                failure = error.with_traceback(None)
                failure.__context__ = failure.__cause__ = None
    return baseline, results, repeat, failure


PUBLIC_ANSWER_FRAME_PROTOCOL = "gemma4-public-answer-v2-forced-empty-thought"
EMPTY_THOUGHT_BOUNDARY = "<|channel>thought\n<channel|>"


def prepare_frames(tokenizer: Any, cases: dict[str, Any], *, max_frame_tokens: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Condition public tokens on an explicitly empty thought under thinking ON.

    The ON generation template leaves the model at its assistant start, where
    it can begin a thought. Append the same empty-thought closure that the
    official OFF generation template emits. This is an explicit diagnostic
    intervention, not an unchanged free-thinking ON response.
    """
    prepared: dict[str, Any] = {}
    frames: dict[str, Any] = {}
    for group, folds in cases.items():
        prepared[group], frames[group] = [], []
        for fold in folds:
            if not fold or len({case["id"] for case in fold}) != len(fold):
                raise ValueError("nonempty folds with unique case IDs required")
            rows, frame_rows = [], []
            for case in fold:
                prefixes, answers = [], []
                user = {"role": "user", "content": case["prompt"].strip()}
                generation = tokenizer.apply_chat_template(
                    [user], tokenize=False, enable_thinking=True, add_generation_prompt=True)
                off_generation = tokenizer.apply_chat_template(
                    [user], tokenize=False, enable_thinking=False, add_generation_prompt=True)
                model_start = "<|turn>model\n"
                if (not generation.endswith(model_start) or "<|think|>" not in generation
                        or not off_generation.endswith(model_start + EMPTY_THOUGHT_BOUNDARY)):
                    raise ValueError("official empty-thought public boundary contract mismatch")
                for key in ("correct", "foil"):
                    answer = case[key].strip()
                    rendered = tokenizer.apply_chat_template(
                        [user, {"role": "assistant", "content": answer}],
                        tokenize=False, enable_thinking=True)
                    marker = "KAIRYU_PUBLIC_ANSWER_BOUNDARY_6e4af21d"
                    skeleton = tokenizer.apply_chat_template(
                        [user, {"role": "assistant", "content": marker}],
                        tokenize=False, enable_thinking=True)
                    if skeleton.count(marker) != 1:
                        raise ValueError("ambiguous public answer boundary")
                    prefix, suffix = skeleton.split(marker)
                    if rendered != prefix + answer + suffix or prefix != generation:
                        raise ValueError("official thinking frame mismatch")
                    prefix += EMPTY_THOUGHT_BOUNDARY
                    pids = tokenizer.encode(prefix, add_special_tokens=False)
                    ids = tokenizer.encode(prefix + answer, add_special_tokens=False)
                    if ids[:len(pids)] != pids or len(ids) <= len(pids):
                        raise ValueError("token boundary mismatch")
                    prefixes.append(pids)
                    answers.append(ids[len(pids):])
                if prefixes[0] != prefixes[1]:
                    raise ValueError("answer prefix mismatch")
                rows.append((case["id"], prefixes[0], *answers))
                frame_rows.append({"id": case["id"],
                    "prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
                    "prefix_tokens": len(prefixes[0]), "correct_tokens": len(answers[0]),
                    "foil_tokens": len(answers[1])})
            validate_frame_rows(rows, max_frame_tokens)
            prepared[group].append(rows)
            frames[group].append(frame_rows)
    return prepared, {"schema_version": 2, "frame_protocol": PUBLIC_ANSWER_FRAME_PROTOCOL,
        "template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
        "enable_thinking": True, "forced_empty_thought": True,
        "public_boundary": EMPTY_THOUGHT_BOUNDARY,
        "scope": "official ON generation prefix plus explicit OFF-template empty-thought boundary; only public answer tokens scored, closing-turn tokens excluded; no thought read, generated, or scored; not free-thinking ON generation",
        "frames": frames}


class PublicAnswerEvaluator:
    """Shared matched public-answer margin, correct NLL and endpoint-token KL.

    Every answer token contributes to the margin and NLL. Diagnostic full-vocab
    KL uses only each correct answer's first/last token positions. Base logprobs
    remain in host RAM, and no free generation or acceptance decision occurs.
    """
    def __init__(self, net: Any, head: Any, tokenizer: Any, prepared: dict[str, Any],
                 device: str, deadline: Callable[[], None] = lambda: None,
                 answer_batching: str = "paired", execution_order: str = "original") -> None:
        if answer_batching not in {"paired", "serial"}:
            raise ValueError("answer batching must be paired or serial")
        self.answer_batching = answer_batching
        if execution_order != "original" and answer_batching != "serial":
            raise ValueError("longest-first requires serial case/answers")
        self.execution_order = execution_order
        import torch
        self.torch, self.net, self.head = torch, net, head
        self.tokenizer, self.prepared, self.device = tokenizer, prepared, device
        self.deadline = deadline
        self.base_aux: dict[tuple[str, int, int], Any] = {}

    def __call__(self, store_base: bool = False, recovery: bool = False) -> dict[str, Any]:
        torch, net, head = self.torch, self.net, self.head
        tokenizer, prepared, device = self.tokenizer, self.prepared, self.device
        deadline, base_aux = self.deadline, self.base_aux
        cap = float(net.config.final_logit_softcapping)
        values: dict[str, Any] = {}
        auxiliary: dict[str, Any] = {}
        sweep_started = time.monotonic()
        with torch.inference_mode():
            values = {group: [None] * len(folds) for group, folds in prepared.items()}
            auxiliary = {group: [None] * len(folds) for group, folds in prepared.items()}
            for group, fold_index, rows in ordered_folds(prepared, self.execution_order):
                fold_values, fold_aux = {}, {}
                packed = batches(rows) if self.execution_order == "original" else [[row] for row in rows]
                if self.answer_batching == "serial":
                    packed = [[row] for batch in packed for row in batch]
                for batch_index, batch in enumerate(packed):
                    if not recovery:
                        deadline()
                    seq, targets, lengths = [], [], []
                    for _, prefix, good, bad in batch:
                        for reply in (good, bad):
                            seq.append(prefix + reply)
                            targets.append(reply)
                            lengths.append(len(prefix))
                    width = max(map(len, seq))
                    ids = torch.tensor([s + [tokenizer.pad_token_id] * (width - len(s)) for s in seq], device=device)
                    mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seq], device=device)
                    if self.answer_batching == "serial":
                        answer_logp = []
                        for i, prefix_length in enumerate(lengths):
                            hidden = net(input_ids=ids[i:i + 1], attention_mask=mask[i:i + 1],
                                         use_cache=False, return_dict=True).last_hidden_state
                            selected = hidden[0, prefix_length - 1:len(seq[i]) - 1]
                            logits = head(selected).float()
                            if cap:
                                logits = cap * torch.tanh(logits / cap)
                            answer_logp.append(torch.log_softmax(logits, -1))
                        full_logp = torch.cat(answer_logp)
                        del answer_logp
                    else:
                        hidden = net(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
                        selected = torch.cat([hidden[i, p - 1:len(seq[i]) - 1] for i, p in enumerate(lengths)])
                        logits = head(selected).float()
                        if cap:
                            logits = cap * torch.tanh(logits / cap)
                        full_logp = torch.log_softmax(logits, -1)
                    answer = torch.tensor([t for reply in targets for t in reply], device=device)
                    logp = full_logp.gather(1, answer[:, None]).flatten()
                    means, positions, position_cases, at = [], [], [], 0
                    for reply_index, reply in enumerate(targets):
                        means.append(logp[at:at + len(reply)].mean())
                        if reply_index % 2 == 0:
                            for position in sorted({at, at + len(reply) - 1}):
                                positions.append(position)
                                position_cases.append(reply_index // 2)
                        at += len(reply)
                    means = torch.stack(means)
                    margin = (means[::2] - means[1::2]).tolist()
                    selected_logp = full_logp[positions]
                    key = (group, fold_index, batch_index)
                    if store_base:
                        base_aux[key] = selected_logp.detach().cpu().clone()
                        kl = torch.zeros(len(positions), device=device)
                    else:
                        reference = base_aux[key].to(device)
                        kl = (reference.exp() * (reference - selected_logp)).sum(-1)
                        if float(kl.min()) < -1e-4:
                            raise ValueError("KL numerical consistency failed")
                        kl = kl.clamp_min(0)
                    kl_values, nll = kl.tolist(), (-means[::2]).tolist()
                    for i, row in enumerate(batch):
                        positions_kl = [v for j, v in enumerate(kl_values) if position_cases[j] == i]
                        fold_aux[row[0]] = {"kl": sum(positions_kl) / len(positions_kl), "correct_nll": nll[i]}
                    fold_values.update({row[0]: v for row, v in zip(batch, margin)})
                    del hidden, selected, logits, full_logp, selected_logp, logp, means, kl
                    if not store_base:
                        del reference
                values[group][fold_index] = [fold_values[row[0]] for row in prepared[group][fold_index]]
                auxiliary[group][fold_index] = {key: float(np.mean([fold_aux[row[0]][key] for row in prepared[group][fold_index]]))
                                              for key in ("kl", "correct_nll")}
        flat = [float(np.mean(v)) for folds in values.values() for v in folds]
        aux = {key: [row[key] for folds in auxiliary.values() for row in folds]
               for key in ("kl", "correct_nll")}
        if not np.isfinite([*flat, *aux["kl"], *aux["correct_nll"]]).all():
            raise ValueError("nonfinite finite-replay metrics")
        return {"values": values, "means": flat, **aux, "seconds": time.monotonic() - sweep_started}


class ParameterEdits:
    """Capture only the verified union of edited BF16 coordinates and restore it.

    The enclosing replay owns the model exclusively. This is an edit restoration
    boundary, not a snapshot of every parameter in an arbitrary mutating model.
    """
    def __init__(self, checkpoint: Checkpoint, specs: list[dict[str, Any]],
                 params: dict[str, Any], device: str) -> None:
        import torch
        self.torch, self.params, self.device = torch, params, device
        self.specs = {spec["id"]: json.loads(json.dumps(spec, allow_nan=False)) for spec in specs}
        self.originals: dict[str, Any] = {}
        self.indices: dict[str, Any] = {}
        self.positions: dict[str, dict[int, int]] = {}
        names = sorted({e["tensor"] for spec in specs for e in spec["edits"]})
        for name in names:
            param = params[name]
            indices = sorted({i for spec in specs for edit in spec["edits"]
                              if edit["tensor"] == name for i in edit["_flat_indices"]})
            if tuple(param.shape) != checkpoint.tensors[name].shape or param.dtype != torch.bfloat16:
                raise ValueError("loaded parameter shape/dtype differs from original checkpoint")
            device_indices = torch.tensor(indices, dtype=torch.long, device=device)
            selected = param.detach().view(-1).index_select(0, device_indices)
            value = selected.float().cpu().numpy()
            source = checkpoint.read_indices(name, indices).astype(np.float32)
            if not np.array_equal(value.view(np.uint32), source.view(np.uint32)):
                raise ValueError("loaded selected coordinates differ from original checkpoint")
            self.indices[name] = device_indices
            self.positions[name] = {index: offset for offset, index in enumerate(indices)}
            self.originals[name] = selected.cpu().clone()

    def apply(self, spec: Any | None) -> dict[str, Any]:
        torch = self.torch
        if spec is not None and spec != self.specs.get(spec.get("id")):
            raise ValueError("RAM application requires an unchanged preverified fixed spec")
        expected = {name: original.clone() for name, original in self.originals.items()}
        with torch.no_grad():
            for name, original in self.originals.items():
                self.params[name].view(-1).index_copy_(0, self.indices[name], original.to(self.device))
            if spec is not None:
                for edit in spec["edits"]:
                    name = edit["tensor"]
                    p = self.params[name].view(-1)
                    indices = torch.tensor(edit["_flat_indices"], dtype=torch.long, device=self.device)
                    target_values = torch.tensor(edit["set_values"], dtype=torch.bfloat16, device="cpu")
                    p.index_copy_(0, indices, target_values.to(self.device))
                    positions = [self.positions[name][index] for index in edit["_flat_indices"]]
                    expected[name][positions] = target_values
            proof = []
            for name, original in self.originals.items():
                actual = self.params[name].detach().view(-1).index_select(0, self.indices[name]).cpu()
                raw = actual.view(torch.int16).numpy().tobytes()
                target = expected[name].view(torch.int16).numpy().tobytes()
                if raw != target:
                    raise ValueError("selected BF16 application/restoration byte mismatch: " + name)
                proof.append({"tensor": name, "selected_count": original.numel(),
                              "indices_sha256": hashlib.sha256(self.indices[name].cpu().numpy().astype("<i8").tobytes()).hexdigest(),
                              "original_bf16_sha256": hashlib.sha256(original.view(torch.int16).numpy().tobytes()).hexdigest(),
                              "actual_bf16_sha256": hashlib.sha256(raw).hexdigest()})
        return {"method": "selected_bf16_cpu_readback", "storage_bytes_exact": True,
                "scope": "union of selected edit coordinates; not a full-model hash", "tensors": proof}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    if args.device != "cuda:0" or not args.gpu_exclusive_approved:
        raise ValueError("explicit owner approval and exactly one exposed CUDA GPU required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one explicitly exposed CUDA GPU required")
    if torch.cuda.mem_get_info()[0] < 70 * 2**30:
        raise ValueError("at least 70 GiB free GPU memory required")
    policy = observation_policy(args, intrinsic_serial=args.answer_batching == "serial")
    checkpoint = Checkpoint(args.checkpoint)
    plan, specs, hashes = load_plan(args.plan, checkpoint)
    if "observation_policy" in plan:
        validate_observation_policy(plan["observation_policy"])
    if plan.get("observation_policy") != policy:
        raise ValueError("finite plan and requested observation policy differ")
    output = args.output.resolve()
    inputs = [checkpoint.path.resolve(), *[Path(p).resolve() for p in hashes]]
    if any(output == p or output in p.parents or p in output.parents for p in inputs):
        raise ValueError("finite output must be separate from all inputs")
    if args.output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise ValueError("finite output must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    def note(phase: str, **fields: Any) -> None:
        record = {"phase": phase, "elapsed_seconds": time.monotonic() - started, **fields}
        _save(output / "status.json", record)
        print(json.dumps(record), flush=True)
    def deadline() -> None:
        if time.monotonic() - started > args.maximum_seconds - 60:
            raise TimeoutError("finite replay time budget; reserve restored-base sweep")
    cases = json.loads(Path(plan["cases"]["path"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.path, local_files_only=True)
    prepared, frame = prepare_frames(tokenizer, cases, max_frame_tokens=policy["max_frame_tokens"] if policy else None)
    if frame != json.loads(Path(plan["frame_evidence"]["path"]).read_text()):
        raise ValueError("source and replay frame mismatch")
    labels = [f"{group}-{fold}" for group, folds in prepared.items() for fold in range(len(folds))]
    provenance = {
        "plan_sha256": hashes[str(args.plan.resolve())], "source_hashes": hashes,
        "runner_sha256": _sha(Path(__file__)), "checkpoint_identity": checkpoint.identity(hash_files=False),
        "expected_base_tree_sha256": plan["base_tree_sha256"], "full_tree_rehashed_here": False,
        "dtype": "BF16 weights; FP32 softcap/log-softmax",
        "attention": policy["attention_dispatch"]["text_config"] if policy else "eager",
        "answer_batching": args.answer_batching, "threads": args.threads, "seed": args.seed,
        "scope": "Matched teacher-forced public-answer diagnostics only; no free generation or benchmark score.",
    }
    if policy is not None:
        provenance["observation_policy"] = policy
    _save(output / "provenance.json", provenance)
    _save(output / "frame-evidence.json", frame)
    note("loading_original_bf16", candidates=len(specs), folds=len(labels))
    allocator = configure_observation_cuda(torch, args.device, policy)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        checkpoint.path, dtype=torch.bfloat16, local_files_only=True,
        attn_implementation=policy["attention_dispatch"] if policy else "eager", device_map={"": args.device}).eval()
    model.requires_grad_(False)
    if policy is not None:
        provenance["observation_execution"] = resolved_observation_execution(model, policy, allocator)
        _save(output / "provenance.json", provenance)
    net, head = model.model.language_model, model.get_output_embeddings()
    params = {"model.language_model." + n: p for n, p in net.named_parameters()}
    for i, layer in enumerate(net.layers):
        params[f"model.language_model.layers.{i}.layer_scalar"] = layer.layer_scalar
    edits_state = ParameterEdits(checkpoint, specs, params, args.device)
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("all gradients must be disabled")
    byte_proofs: dict[str, Any] = {}
    base_applications = 0
    def apply(spec: Any | None) -> None:
        nonlocal base_applications
        proof = edits_state.apply(spec)
        torch.cuda.synchronize()
        if spec is None:
            name = "base" if base_applications == 0 else "restored_base"
            base_applications += 1
        else:
            name = spec["id"]
        byte_proofs[name] = proof
        _save(output / "selected-byte-proofs.json", byte_proofs)
    evaluate = PublicAnswerEvaluator(net, head, tokenizer, prepared, args.device, deadline,
                                    answer_batching=args.answer_batching,
                                    execution_order=policy["execution_order"] if policy else "original")
    records: dict[str, Any] = {}
    def on_result(identifier: str, result: Any) -> None:
        records[identifier] = result
        _save(output / f"{identifier}.json", result)
        note("sweep_complete", candidate=identifier, seconds=result["seconds"])
    def before_restore() -> None:
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(signum, signal.SIG_IGN)
        torch.cuda.empty_cache()
    baseline, results, repeat, failure = replay_sequence(specs, apply, evaluate, on_result, before_restore)
    restoration: dict[str, Any] = {"attempted": True, "forward_completed": repeat is not None, "matched": False}
    restoration.update(storage_bytes_exact=byte_proofs.get("restored_base", {}).get("storage_bytes_exact", False),
                       byte_proof_method="selected_bf16_cpu_readback")
    if baseline is not None and repeat is not None:
        margin_error = max(abs(a - b) for group, folds in baseline["values"].items()
                           for av, bv in zip(folds, repeat["values"][group]) for a, b in zip(av, bv))
        nll_error = float(np.max(np.abs(np.array(repeat["correct_nll"]) - baseline["correct_nll"])))
        kl_error = max(repeat["kl"])
        restoration.update(max_margin_error=margin_error, max_correct_nll_error=nll_error,
                           max_kl=kl_error, matched=margin_error <= 1e-4 and nll_error <= 1e-4 and kl_error <= 1e-6)
    diagnostics = []
    if baseline is not None:
        for spec, result in zip(specs, results):
            diagnostics.append({"id": spec["id"], **spec["_verified_metrics"],
                "margin_delta": (np.array(result["means"]) - baseline["means"]).tolist(),
                "correct_nll_delta": (np.array(result["correct_nll"]) - baseline["correct_nll"]).tolist(),
                "kl_base_to_candidate": result["kl"], "seconds": result["seconds"]})
    source_stable = True
    try:
        checkpoint.assert_stable()
        if any(_sha(Path(p)) != digest for p, digest in hashes.items()):
            raise ValueError("finite replay inputs changed")
    except BaseException as error:
        source_stable = False
        if failure is None:
            failure = error
    report = {"status": "completed" if failure is None and restoration["matched"] else "incomplete",
        "labels": labels, "candidates": diagnostics, "base_restoration": restoration,
        "answer_batching": args.answer_batching, "selected_byte_proofs": byte_proofs,
        "source_stable": source_stable, "error_type": None if failure is None else type(failure).__name__,
        "elapsed_seconds": time.monotonic() - started, "gpu_peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
        "new_gradients": 0, "generated_tokens": 0, "wsp_eligibility_decided": False,
        "limits": ["Synthetic public-answer calibration, not free-generation or WSP performance.",
                   "Per-purpose gains and losses are reported without an all-fold nonregression gate.",
                   "Owner must restore its serving container after this isolated process exits."]}
    if policy is not None:
        report.update(observation_policy=policy, observation_execution=provenance["observation_execution"])
    _save(output / "base-restoration.json", restoration)
    _save(output / "summary.json", report)
    note(report["status"], candidates=len(diagnostics), restored=restoration["matched"])
    if failure is not None:
        raise failure
    if not restoration["matched"]:
        raise RuntimeError("restored-base replay did not match")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda:0"], default="cuda:0")
    parser.add_argument("--maximum-seconds", type=int, default=600)
    parser.add_argument("--answer-batching", choices=("paired", "serial"), default="paired",
                        help="serial keeps one case's good/foil padded width and forwards each answer separately")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--gpu-exclusive-approved", action="store_true")
    add_observation_arguments(parser)
    args = parser.parse_args()
    if not 1 <= args.threads <= 64:
        parser.error("threads must be 1..64")
    if not 120 <= args.maximum_seconds <= 1200:
        parser.error("maximum-seconds must be 120..1200")
    def interrupted(signum: int, frame: Any) -> None:
        raise InterruptedError(f"finite replay interrupted by signal {signum}")
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    run(args)


if __name__ == "__main__":
    main()
