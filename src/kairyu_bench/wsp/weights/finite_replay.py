"""Finite-only Gemma 4 native/sparse-matrix edit replay: original, <=3 sets, original.

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
            if matrix and sum(len(e.get("coordinates", [])) for e in edits) > 65536:
                raise ValueError("matrix edit count exceeds 65,536")
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


def prepare_frames(tokenizer: Any, cases: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
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
                for key in ("correct", "foil"):
                    answer = case[key].strip()
                    user = {"role": "user", "content": case["prompt"].strip()}
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
                    if rendered != prefix + answer + suffix or "<|think|>" not in prefix or "<|turn>model" not in prefix:
                        raise ValueError("official thinking frame mismatch")
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
            prepared[group].append(rows)
            frames[group].append(frame_rows)
    return prepared, {"template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
        "enable_thinking": True,
        "scope": "user and assistant public content via official template, answer tokens only, closing-turn tokens excluded; no thought generated or scored",
        "frames": frames}


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
            if not np.array_equal(value, checkpoint.read_indices(name, indices)):
                raise ValueError("loaded selected coordinates differ from original checkpoint")
            self.indices[name] = device_indices
            self.originals[name] = selected.cpu().clone()

    def apply(self, spec: Any | None) -> None:
        torch = self.torch
        if spec is not None and spec != self.specs.get(spec.get("id")):
            raise ValueError("RAM application requires an unchanged preverified fixed spec")
        with torch.no_grad():
            for name, original in self.originals.items():
                self.params[name].view(-1).index_copy_(0, self.indices[name], original.to(self.device))
            if spec is not None:
                for edit in spec["edits"]:
                    p = self.params[edit["tensor"]].view(-1)
                    indices = torch.tensor(edit["_flat_indices"], dtype=torch.long, device=self.device)
                    values = torch.tensor(edit["set_values"], dtype=torch.bfloat16, device=self.device)
                    p.index_copy_(0, indices, values)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    if args.device != "cuda:0" or not args.gpu_exclusive_approved:
        raise ValueError("explicit owner approval and exactly one exposed CUDA GPU required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one explicitly exposed CUDA GPU required")
    if torch.cuda.mem_get_info()[0] < 70 * 2**30:
        raise ValueError("at least 70 GiB free GPU memory required")
    checkpoint = Checkpoint(args.checkpoint)
    plan, specs, hashes = load_plan(args.plan, checkpoint)
    output = args.output.resolve()
    inputs = [checkpoint.path.resolve(), *[Path(p).resolve() for p in hashes]]
    if any(output == p or output in p.parents or p in output.parents for p in inputs):
        raise ValueError("finite output must be separate from all inputs")
    if args.output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise ValueError("finite output must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261105)
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
    prepared, frame = prepare_frames(tokenizer, cases)
    if frame != json.loads(Path(plan["frame_evidence"]["path"]).read_text()):
        raise ValueError("source and replay frame mismatch")
    labels = [f"{group}-{fold}" for group, folds in prepared.items() for fold in range(len(folds))]
    _save(output / "provenance.json", {
        "plan_sha256": hashes[str(args.plan.resolve())], "source_hashes": hashes,
        "runner_sha256": _sha(Path(__file__)), "checkpoint_identity": checkpoint.identity(hash_files=False),
        "expected_base_tree_sha256": plan["base_tree_sha256"], "full_tree_rehashed_here": False,
        "dtype": "BF16 weights; FP32 softcap/log-softmax", "attention": "eager",
        "scope": "Matched teacher-forced public-answer diagnostics only; no free generation or benchmark score.",
    })
    _save(output / "frame-evidence.json", frame)
    note("loading_original_bf16", candidates=len(specs), folds=len(labels))
    model = Gemma4ForConditionalGeneration.from_pretrained(
        checkpoint.path, dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="eager", device_map={"": args.device}).eval()
    model.requires_grad_(False)
    net, head = model.model.language_model, model.get_output_embeddings()
    cap = float(net.config.final_logit_softcapping)
    params = {"model.language_model." + n: p for n, p in net.named_parameters()}
    for i, layer in enumerate(net.layers):
        params[f"model.language_model.layers.{i}.layer_scalar"] = layer.layer_scalar
    edits_state = ParameterEdits(checkpoint, specs, params, args.device)
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("all gradients must be disabled")
    base_aux: dict[tuple[str, int, int], Any] = {}
    def apply(spec: Any | None) -> None:
        edits_state.apply(spec)
        torch.cuda.synchronize()
    def evaluate(store_base: bool = False, recovery: bool = False) -> dict[str, Any]:
        values: dict[str, Any] = {}
        auxiliary: dict[str, Any] = {}
        sweep_started = time.monotonic()
        with torch.inference_mode():
            for group, folds in prepared.items():
                values[group], auxiliary[group] = [], []
                for fold_index, rows in enumerate(folds):
                    fold_values, fold_aux = {}, {}
                    for batch_index, batch in enumerate(batches(rows)):
                        if not recovery:
                            deadline()
                        seq, targets, lengths = [], [], []
                        for _, prefix, good, bad in batch:
                            for reply in (good, bad):
                                seq.append(prefix + reply)
                                targets.append(reply)
                                lengths.append(len(prefix))
                        width = max(map(len, seq))
                        ids = torch.tensor([s + [tokenizer.pad_token_id] * (width - len(s)) for s in seq], device=args.device)
                        mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seq], device=args.device)
                        hidden = net(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
                        selected = torch.cat([hidden[i, p - 1:len(seq[i]) - 1] for i, p in enumerate(lengths)])
                        logits = head(selected).float()
                        if cap:
                            logits = cap * torch.tanh(logits / cap)
                        answer = torch.tensor([t for reply in targets for t in reply], device=args.device)
                        full_logp = torch.log_softmax(logits, -1)
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
                            kl = torch.zeros(len(positions), device=args.device)
                        else:
                            reference = base_aux[key].to(args.device)
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
                    values[group].append([fold_values[row[0]] for row in rows])
                    auxiliary[group].append({key: float(np.mean([fold_aux[row[0]][key] for row in rows]))
                                             for key in ("kl", "correct_nll")})
        flat = [float(np.mean(v)) for folds in values.values() for v in folds]
        aux = {key: [row[key] for folds in auxiliary.values() for row in folds]
               for key in ("kl", "correct_nll")}
        if not np.isfinite([*flat, *aux["kl"], *aux["correct_nll"]]).all():
            raise ValueError("nonfinite finite-replay metrics")
        return {"values": values, "means": flat, **aux, "seconds": time.monotonic() - sweep_started}
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
        "source_stable": source_stable, "error_type": None if failure is None else type(failure).__name__,
        "elapsed_seconds": time.monotonic() - started, "gpu_peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
        "new_gradients": 0, "generated_tokens": 0, "wsp_eligibility_decided": False,
        "limits": ["Synthetic public-answer calibration, not free-generation or WSP performance.",
                   "Per-purpose gains and losses are reported without an all-fold nonregression gate.",
                   "Owner must restore its serving container after this isolated process exits."]}
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
    parser.add_argument("--gpu-exclusive-approved", action="store_true")
    args = parser.parse_args()
    if not 120 <= args.maximum_seconds <= 1200:
        parser.error("maximum-seconds must be 120..1200")
    def interrupted(signum: int, frame: Any) -> None:
        raise InterruptedError(f"finite replay interrupted by signal {signum}")
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    run(args)


if __name__ == "__main__":
    main()
