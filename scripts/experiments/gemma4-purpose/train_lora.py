#!/usr/bin/env python3
"""Bounded Gemma4 LoRA SFT; explicit train data, no serving or benchmark actions."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

import recipe_common as common
import train_lora_body_kl as body_kl
from native import _SavedActivationStorage
from kairyu_bench.wsp.weights.checkpoint import Checkpoint

PROJECTION = re.compile(
    r"model\.language_model\.layers\.\d+\.(?:mlp\.(?:gate|up|down)_proj|self_attn\.(?:q|k|v|o)_proj)\.weight"
)
HYPERPARAMETERS = {
    "rank": 8,
    "alpha": 8,
    "learning_rate": 0.0001,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "accumulation": 4,
    "seed": 0,
    "max_frame_tokens": 1024,
    "relative_delta_cap": 0.005,
    "pilot_steps": 4,
    "total_steps": 64,
}
FRAME_PROTOCOL = "gemma4-sft-standard-on-answer-eos-v1"


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def check_pin(pin):
    path = Path(pin["path"]).resolve()
    if common.sha(path) != pin["sha256"]:
        raise ValueError("input SHA mismatch: " + str(path))
    return path


def read_training_rows(path, identifiers, split_summary):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row.get("task_id") for row in rows]
    if (
        not ids
        or ids != identifiers
        or len(set(ids)) != len(ids)
        or any(type(i) is not int for i in ids)
        or not set(ids).issubset(split_summary["training_ids"])
        or set(ids) & set(split_summary["holdout_ids"])
    ):
        raise ValueError("train rows must exactly match explicit IDs, never holdout")
    if any(
        not isinstance(row.get(k), str) or not row[k].strip()
        for row in rows
        for k in ("text", "code")
    ):
        raise ValueError("training prompt/reference must be nonempty")
    return rows


def prepare_training_frames(tokenizer, rows):
    """Exact official ON dialog; no invented reasoning, padding or truncation."""
    frames, evidence = [], []
    for row in rows:
        user = {"role": "user", "content": row["text"]}
        prefix = tokenizer.apply_chat_template(
            [user], tokenize=False, enable_thinking=True, add_generation_prompt=True
        )
        full = tokenizer.apply_chat_template(
            [user, {"role": "assistant", "content": row["code"]}],
            tokenize=False,
            enable_thinking=True,
            add_generation_prompt=False,
        )
        pids = tokenizer.encode(prefix, add_special_tokens=False)
        ids = tokenizer.encode(full, add_special_tokens=False)
        if not pids or ids[: len(pids)] != pids or len(ids) <= len(pids):
            raise ValueError("official prefix/answer token boundary mismatch")
        if "<|channel>thought\n<channel|>" in prefix or not prefix.endswith(
            "<|turn>model\n"
        ):
            raise ValueError("SFT must not force an empty thought")
        # Gemma4 ends this dialog with <turn|> + newline, not global <eos> id 1.
        terminator = "<turn|>\n"
        terminator_ids = tokenizer.encode(terminator, add_special_tokens=False)
        if (
            not terminator_ids
            or not full.endswith(terminator)
            or ids[-len(terminator_ids) :] != terminator_ids
        ):
            raise ValueError("official assistant turn terminator mismatch")
        frames.append({"id": row["task_id"], "ids": ids, "prefix_tokens": len(pids)})
        evidence.append(
            {
                "id": row["task_id"],
                "prefix_tokens": len(pids),
                "answer_eos_tokens": len(ids) - len(pids),
                "complete_tokens": len(ids),
                "prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
                "full_sha256": hashlib.sha256(full.encode()).hexdigest(),
                "token_ids_sha256": hashlib.sha256(
                    json.dumps(ids).encode()
                ).hexdigest(),
                "eligible": len(ids) <= 1024,
            }
        )
    return frames, {
        "schema_version": 1,
        "frame_protocol": FRAME_PROTOCOL,
        "template_sha256": hashlib.sha256(
            str(tokenizer.chat_template).encode()
        ).hexdigest(),
        "enable_thinking": True,
        "forced_empty_thought": False,
        "padding": False,
        "loss": "mean answer and official turn-terminator tokens; prompt masked; each example equally weighted",
        "official_terminator": "<turn|>\n",
        "manual_global_eos_added": False,
        "maximum_tokens": 1024,
        "truncation": False,
        "rows": evidence,
        "ineligible_ids": [row["id"] for row in evidence if not row["eligible"]],
    }


def make_adapter(torch, base, rank=8, alpha=8):
    """No PEFT dependency. Parent and optimizer parameter sets remain disjoint."""
    if type(base) is not torch.nn.Linear:
        raise ValueError("LoRA requires an ordinary untransformed Linear")

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base.requires_grad_(False)
            self.A = torch.nn.Parameter(
                torch.empty(
                    rank,
                    base.in_features,
                    device=base.weight.device,
                    dtype=torch.float32,
                )
            )
            self.B = torch.nn.Parameter(
                torch.zeros(
                    base.out_features,
                    rank,
                    device=base.weight.device,
                    dtype=torch.float32,
                )
            )
            torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.scale = float(alpha / rank)
            self.activation = None
            self.calls = 0

        @property
        def weight(self):
            return self.base.weight

        @property
        def bias(self):
            return self.base.bias

        def forward(self, x):
            y = self.base(x)
            delta = torch.nn.functional.linear(
                torch.nn.functional.linear(x.float(), self.A), self.B
            )
            result = y + (delta * self.scale).to(y.dtype)
            self.calls += 1
            if self.activation is not None:
                for key, value in (("input", x), ("output", result)):
                    value = value.detach().float()
                    entry = self.activation.setdefault(key, [0, None, None, None])
                    entry[0] += value.numel()
                    values = (
                        value.sum(dtype=torch.float64),
                        value.square().sum(dtype=torch.float64),
                        value.abs().max(),
                    )
                    for index, statistic in enumerate(values, 1):
                        old = entry[index]
                        entry[index] = (
                            statistic
                            if old is None
                            else (
                                torch.maximum(old, statistic)
                                if index == 3
                                else old + statistic
                            )
                        )
            return result

    return Adapter()


def adapter_delta_norm(torch, adapter):
    # ||BA||_F² = tr((B^T B)(A A^T)); only r×r intermediates, FP64 reductions.
    a, b = adapter.A.double(), adapter.B.double()
    square = ((a @ a.T) * (b.T @ b).T).sum()
    return square.clamp_min(0).sqrt() * abs(adapter.scale)


def project_adapters(torch, adapters, base_norms, radius):
    values = (
        torch.stack([adapter_delta_norm(torch, a) for a in adapters.values()])
        .detach()
        .cpu()
        .tolist()
    )
    records = {}
    with torch.no_grad():
        for (name, adapter), value in zip(adapters.items(), values, strict=True):
            denominator = base_norms[name]
            if denominator <= 0 or not math.isfinite(value):
                raise ValueError("invalid complete matrix norm/delta")
            ratio = value / denominator
            factor = min(1.0, radius / ratio) if ratio else 1.0
            if factor < 1.0:
                # Inward rounding avoids a real-valued cap being exceeded by F32 multiplication.
                factor *= 1.0 - 4.0 * torch.finfo(torch.float32).eps
                adapter.B.mul_(factor)
            records[name] = {
                "base_frobenius": denominator,
                "delta_l2_before_projection": value,
                "relative_before": ratio,
                "B_multiplier": factor,
            }
    after = (
        torch.stack([adapter_delta_norm(torch, a) for a in adapters.values()])
        .detach()
        .cpu()
        .tolist()
    )
    for (name, row), value in zip(records.items(), after, strict=True):
        row.update(
            delta_l2_after_projection=value, relative_after=value / base_norms[name]
        )
        if not math.isfinite(value) or value / base_norms[name] > radius * (1 + 1e-7):
            raise ValueError("realized F32 factors exceed the fixed continuous cap")
    return records


def frozen_state(parameters):
    return [
        (p.data_ptr(), p._version, p.requires_grad, tuple(p.shape), p.dtype)
        for p in parameters
    ]


def answer_loss(torch, net, head, frame, device):
    ids = torch.tensor([frame["ids"]], device=device)
    hidden = net(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
    offset = frame["prefix_tokens"]
    selected = hidden[0, offset - 1 : -1]
    logits = head(selected).float()
    cap = float(net.config.final_logit_softcapping)
    if cap:
        logits = cap * torch.tanh(logits / cap)
    targets = ids[0, offset:]
    return torch.nn.functional.cross_entropy(logits, targets), targets.numel()


def teacher_identity(plan, base_identity):
    pin = plan.get("teacher_checkpoint", plan["checkpoint"])
    checkpoint = Checkpoint(pin["path"])
    receipt = json.loads(check_pin(pin["attestation"]).read_text())
    identity = receipt.get("checkpoint", receipt)
    current = checkpoint.identity(hash_files=False)
    if (
        identity.get("hash_files") is not True
        or identity.get("identity_level") != "content_sha256"
        or identity.get("sha256") != pin["tree_sha256"]
        or len(current["files"]) != len(identity["files"])
    ):
        raise ValueError("complete explicit teacher attestation required")
    for prior, actual in zip(identity["files"], current["files"], strict=True):
        if not re.fullmatch("[0-9a-f]{64}", prior.get("sha256", "")) or any(
            prior[key] != actual[key] for key in ("path", "size", "stat")
        ):
            raise ValueError("teacher full hash/stat continuity lost")
    base_files = {row["path"]: row["sha256"] for row in base_identity["files"]}
    teacher_files = {row["path"]: row["sha256"] for row in identity["files"]}
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "processor_config.json",
    ):
        if name not in base_files or base_files[name] != teacher_files.get(name):
            raise ValueError("teacher/base tokenizer or configuration differs: " + name)
    return checkpoint, identity, current


def model_counters(output, forwards, attempts):
    teacher = body_kl.counters(output)
    return {
        **teacher,
        "student_model_forwards": forwards,
        "student_forward_attempts": attempts,
        "model_forwards": forwards + teacher["teacher_model_forwards"],
        "forward_attempts": attempts + teacher["teacher_forward_attempts"],
    }


def preflight(args):
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    use_kl = body_kl.selected(plan)
    if (
        plan.get("schema_version") != 1
        or plan.get("kind") != "gemma4_lora_sft_v1"
        or plan.get("hyperparameters") != HYPERPARAMETERS
        or plan["data"].get("split") != "train"
    ):
        raise ValueError("explicit fixed LoRA SFT plan required")
    if not (
        0 < plan["maximum_seconds"] <= 3000
        and 0 <= plan["continuation_wait_seconds"] <= 600
    ):
        raise ValueError("bounded training/continuation time required")
    source_paths = [
        Path(__file__),
        Path(body_kl.__file__),
        Path(common.__file__),
        Path(common.finite_replay.__file__),
    ]
    from native import __file__ as native_source

    source_paths.append(Path(native_source))
    pinned = [
        plan_path,
        *source_paths,
        check_pin(plan["data"]),
        check_pin(plan["data"]["split_summary"]),
        check_pin(plan["frame_evidence"]),
        check_pin(plan["input_quality"]),
        check_pin(plan["checkpoint"]["attestation"]),
    ]
    if use_kl and "teacher_checkpoint" in plan:
        pinned.append(check_pin(plan["teacher_checkpoint"]["attestation"]))
    states = {str(p.resolve()): common.input_state(p) for p in pinned}
    hashes = {str(p.resolve()): common.sha(p) for p in pinned}
    checkpoint = Checkpoint(plan["checkpoint"]["path"])
    receipt = json.loads(Path(plan["checkpoint"]["attestation"]["path"]).read_text())
    identity = receipt.get("checkpoint", receipt)
    current = checkpoint.identity(hash_files=False)
    if (
        identity.get("hash_files") is not True
        or identity.get("identity_level") != "content_sha256"
        or identity.get("sha256") != plan["checkpoint"]["tree_sha256"]
        or len(current["files"]) != len(identity["files"])
    ):
        raise ValueError("complete parent attestation required")
    for old, new in zip(identity["files"], current["files"], strict=True):
        if not re.fullmatch("[0-9a-f]{64}", old.get("sha256", "")) or any(
            old[k] != new[k] for k in ("path", "size", "stat")
        ):
            raise ValueError("parent inventory/stat no longer matches attestation")
    rows = read_training_rows(
        check_pin(plan["data"]),
        plan["data"]["ids"],
        json.loads(check_pin(plan["data"]["split_summary"]).read_text()),
    )
    quality = json.loads(check_pin(plan["input_quality"]).read_text())
    eligible = quality.get("splits", {}).get("train", {})
    if (
        quality.get("status") != "completed"
        or quality.get("container_returncode") != 0
        or eligible.get("unknown") != 0
        or eligible.get("unconfirmed_ids")
        or eligible.get("eligible_ids") != plan["data"]["ids"]
    ):
        raise ValueError(
            "all declared training references need completed CPU quality evidence"
        )
    proof = json.loads(check_pin(plan["frame_evidence"]).read_text())
    if (
        proof.get("frame_protocol")
        != (body_kl.FRAME_PROTOCOL if use_kl else FRAME_PROTOCOL)
        or (
            use_kl
            and (
                proof.get("loss_protocol") != body_kl.PROTOCOL
                or proof.get("unsupported_ids")
            )
        )
        or proof.get("ineligible_ids")
        or [r["id"] for r in proof["rows"]] != plan["data"]["ids"]
    ):
        raise ValueError(
            "complete eligible tokenizer evidence for every training row required"
        )
    mapping = {
        name: info
        for name, info in checkpoint.tensors.items()
        if PROJECTION.fullmatch(name)
    }
    if len(mapping) != 410 or any(
        info.dtype != "BF16" or len(info.shape) != 2 for info in mapping.values()
    ):
        raise ValueError("exact 410 BF16 text projections required")
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("new output only")
    output = args.output.resolve()
    for path in [
        checkpoint.path,
        *pinned,
        *(
            [Path(plan["teacher_checkpoint"]["path"])]
            if use_kl and "teacher_checkpoint" in plan
            else []
        ),
    ]:
        path = path.resolve()
        if path == output or path in output.parents or output in path.parents:
            raise ValueError("output must be disjoint from inputs")
    provenance = {"input_sha256": hashes, "input_stat": states, "checkpoint": current}
    if use_kl:
        teacher, teacher_full, teacher_current = teacher_identity(plan, identity)
        provenance.update(
            teacher_checkpoint=teacher_current, teacher_full_identity=teacher_full
        )
        common.assert_inputs(teacher, {**provenance, "checkpoint": teacher_current})
    common.assert_inputs(checkpoint, provenance)
    return plan, checkpoint, rows, proof, mapping, provenance, identity


def export_adapter(
    torch,
    directory,
    adapters,
    plan,
    provenance,
    identity,
    step,
    step_sha,
    base_unchanged,
):
    from safetensors.torch import save_file

    directory.mkdir()
    factors = {}
    for name, adapter in adapters.items():
        prefix = "base_model.model." + name.removesuffix(".weight")
        factors[prefix + ".lora_A.weight"] = adapter.A.detach().cpu().contiguous()
        factors[prefix + ".lora_B.weight"] = adapter.B.detach().cpu().contiguous()
    payload = directory / "adapter_model.safetensors"
    save_file(factors, str(payload), metadata={"format": "pt"})
    config = {
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "use_rslora": False,
        "use_dora": False,
        "fan_in_fan_out": False,
        "bias": "none",
        "rank_pattern": {},
        "alpha_pattern": {},
        "modules_to_save": None,
    }
    config_path = directory / "adapter_config.json"
    write_new(config_path, config)
    dense_rows = [
        {key: item[key] for key in ("path", "size", "sha256")}
        for item in identity["files"]
    ]
    dense_rows.sort(key=lambda item: item["path"])
    dense_tree = hashlib.sha256(
        json.dumps(dense_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt = {
        "schema_version": 1,
        "kind": "local_lora_training_v1",
        "status": "exported",
        "parent_path": str(Path(plan["checkpoint"]["path"]).resolve()),
        "parent_tree_sha256": plan["checkpoint"]["tree_sha256"],
        "parent_dense_tree_sha256": dense_tree,
        "adapter_sha256": common.sha(payload),
        "config_sha256": common.sha(config_path),
        "data_sha256": plan["data"]["sha256"],
        "source_sha256": provenance["input_sha256"],
        "steps": step,
        "step_log_sha256": step_sha,
        "base_unchanged": base_unchanged,
        "rank": 8,
        "alpha": 8,
        "dtype": "F32",
        "modules": 410,
        "fresh_parent_full_hash": False,
        "parent_proof": "prior full-content hash plus exact stat continuity",
        "claim": "training export only; saved BF16 merge and capability unmeasured",
        **(
            {
                "loss_protocol": body_kl.PROTOCOL,
                "teacher_checkpoint": plan.get(
                    "teacher_checkpoint", plan["checkpoint"]
                ),
            }
            if body_kl.selected(plan)
            else {}
        ),
    }
    receipt_path = directory / "training-receipt.json"
    write_new(receipt_path, receipt)
    with payload.open("rb") as stream:
        import struct

        length = struct.unpack("<Q", stream.read(8))[0]
        header_sha = hashlib.sha256(stream.read(length)).hexdigest()
    spec = {
        "schema_version": 2,
        "kind": "dense_lora_recipe_v1",
        "scope": "native_text410",
        "donor_path": str(payload.resolve()),
        "donor_config_path": str(config_path.resolve()),
        "donor_sha256": common.sha(payload),
        "donor_config_sha256": common.sha(config_path),
        "local_training": {
            "version": 1,
            "path": str(receipt_path.resolve()),
            "sha256": common.sha(receipt_path),
        },
        "factor_format": {
            "version": 2,
            "rank": 8,
            "dtype": "F32",
            "header_sha256": header_sha,
        },
        "coefficient": 1.0,
        "base_tree_sha256": dense_tree,
    }
    write_new(directory / "merge-spec.json", spec)
    return {
        "path": str(receipt_path),
        "sha256": common.sha(receipt_path),
        "adapter_sha256": receipt["adapter_sha256"],
        "merge_spec_sha256": common.sha(directory / "merge-spec.json"),
    }


def wait_continuation(output, pilot_sha, wait_seconds, deadline):
    """Operator approval binds this exact pilot; timeout is a normal pilot-only stop."""
    end = min(time.monotonic() + wait_seconds, deadline)
    path = output / "continue.json"
    while time.monotonic() < end:
        if path.exists():
            if path.is_symlink():
                raise ValueError("continuation cannot be a symlink")
            decision = json.loads(path.read_text())
            if (
                set(decision) != {"pilot_receipt_sha256", "continue_to_step"}
                or decision["pilot_receipt_sha256"] != pilot_sha
            ):
                raise ValueError("continuation must bind this exact pilot")
            if decision["continue_to_step"] not in (4, 64):
                raise ValueError("only stop at 4 or continue to fixed 64")
            return decision["continue_to_step"] == 64
        time.sleep(min(1.0, max(0.0, end - time.monotonic())))
    return False


def train(args, plan, checkpoint, rows, proof, mapping, provenance, identity):
    from transformers import AutoTokenizer

    start = time.monotonic()
    deadline = start + plan["maximum_seconds"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.path, local_files_only=True)
    use_kl = body_kl.selected(plan)
    frames, actual_proof = (
        body_kl.prepare_frames(tokenizer, rows, prepare_training_frames)
        if use_kl
        else prepare_training_frames(tokenizer, rows)
    )
    if actual_proof != proof or actual_proof["ineligible_ids"]:
        raise ValueError("actual tokenizer frames differ; no truncation/fallback")
    args.checkpoint = checkpoint.path
    args.threads, args.seed = 4, 0
    args.mode, args._intrinsic_serial = "chat-public-answer", True
    args.max_frame_tokens, args.text_attention_backend = 1024, "sdpa"
    args.execution_order, args.cuda_memory_cap_gib = (
        "original",
        88 if args.device != "cpu" else None,
    )
    teacher_cache = None
    if use_kl:
        teacher, _, teacher_current = teacher_identity(plan, identity)
        teacher_provenance = {**provenance, "checkpoint": teacher_current}
        teacher_metadata = {
            "teacher_checkpoint": plan.get("teacher_checkpoint", plan["checkpoint"]),
            "editing_base_checkpoint": plan["checkpoint"],
            "frame_evidence_sha256": plan["frame_evidence"]["sha256"],
            "training_ids": plan["data"]["ids"],
            "holdout_ids": [],
            "loss_protocol": body_kl.PROTOCOL,
        }
        if teacher.path.resolve() != checkpoint.path.resolve():
            args.checkpoint = teacher.path
            torch, _, teacher_model = common.load_model(args)
            torch.backends.cuda.matmul.allow_tf32 = False
            teacher_model.eval().requires_grad_(False)
            teacher_cache = body_kl.cache_teacher(
                torch,
                teacher_model,
                frames,
                args.output / "teacher-cache",
                args.device,
                deadline,
                teacher_metadata,
                lambda: common.assert_inputs(teacher, teacher_provenance),
            )
            del teacher_model
            import gc

            gc.collect()
            allocated = None
            if args.device != "cpu":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated()
            write_new(
                args.output / "teacher-release.json",
                {
                    "separate_teacher": True,
                    "allocated_bytes_after_release": allocated,
                    "base_load_started": False,
                },
            )
            if allocated not in (None, 0):
                raise ValueError(
                    "teacher CUDA tensors remain; refusing to load a second model"
                )
            args.checkpoint = checkpoint.path
    torch, _, model = common.load_model(args)
    torch.backends.cuda.matmul.allow_tf32 = False
    model.eval().requires_grad_(False)
    if any(
        getattr(module, "gradient_checkpointing", False) for module in model.modules()
    ):
        raise ValueError(
            "this bounded recipe uses eval/autograd, not gradient checkpointing"
        )
    base_parameters = list(model.parameters())
    frozen = frozen_state(base_parameters)
    if use_kl and teacher_cache is None:
        teacher_cache = body_kl.cache_teacher(
            torch,
            model,
            frames,
            args.output / "teacher-cache",
            args.device,
            deadline,
            teacher_metadata,
            lambda: common.assert_inputs(teacher, teacher_provenance),
        )
    adapters, base_norms = {}, {}
    for name, info in sorted(mapping.items()):
        module_name = name.removesuffix(".weight")
        base = model.get_submodule(module_name)
        if (
            tuple(base.weight.shape) != info.shape
            or base.weight.dtype != torch.bfloat16
        ):
            raise ValueError("loaded projection does not match parent header")
        # Full matrix norm, bounded temporary conversion, no sampled denominator.
        sums = [
            part.float().square().sum(dtype=torch.float64)
            for part in base.weight.detach().split(128)
        ]
        base_norms[name] = float(torch.stack(sums).sum().sqrt())
        adapter = make_adapter(torch, base)
        parent, key = module_name.rsplit(".", 1)
        setattr(model.get_submodule(parent), key, adapter)
        adapters[name] = adapter
    parameters = [p for a in adapters.values() for p in (a.A, a.B)]
    optimizer = torch.optim.AdamW(
        parameters, lr=0.0001, weight_decay=0.0, foreach=False
    )
    order = list(range(len(frames)))
    random.Random(0).shuffle(order)
    write_new(args.output / "frame-evidence.json", actual_proof)
    write_new(args.output / "provenance.json", provenance)
    write_new(
        args.output / "training-order.json",
        {
            "ids": [frames[i]["id"] for i in order],
            "seed": 0,
            "cycle": "same fixed order after full pass",
        },
    )
    storage = _SavedActivationStorage(torch, args.device != "cpu")
    net, head = model.model.language_model, model.get_output_embeddings()
    step, cursor, forwards, exports = 0, 0, 0, []
    forward_attempts, current_example = 0, None
    status, error = "failed", None
    try:
        with (
            (args.output / "steps.jsonl").open("x", buffering=1) as log,
            (args.output / "microbatches.jsonl").open("x", buffering=1) as examples,
        ):
            for step in range(1, 65):
                if time.monotonic() >= deadline:
                    raise TimeoutError("fixed training deadline")
                common.assert_inputs(checkpoint, provenance)
                t = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                observe = step in (1, 4, 64)
                for a in adapters.values():
                    a.activation = {} if observe else None
                    a.calls = 0
                losses, ids, tokens = [], [], 0
                objective_terms = []
                for _ in range(4):
                    frame = frames[order[cursor % len(order)]]
                    cursor += 1
                    current_example = {
                        "step": step,
                        "position": cursor,
                        "id": frame["id"],
                    }
                    forward_attempts += 1
                    with storage.context() if args.device != "cpu" else nullcontext():
                        if use_kl:
                            loss, count, terms = body_kl.body_kl_loss(
                                torch,
                                net,
                                head,
                                frame,
                                teacher_cache[frame["id"]],
                                args.device,
                            )
                            objective_terms.append(terms)
                        else:
                            loss, count = answer_loss(
                                torch, net, head, frame, args.device
                            )
                        if not torch.isfinite(loss):
                            raise ValueError("nonfinite actual CE")
                        (loss / 4).backward()
                    losses.append(terms["body_CE"] if use_kl else float(loss.detach()))
                    examples.write(
                        json.dumps(
                            {
                                **current_example,
                                "CE": losses[-1],
                                "target_tokens": count,
                                "backward_completed": True,
                                **({"loss_terms": terms} if use_kl else {}),
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    examples.flush()
                    tokens += count
                    ids.append(frame["id"])
                    forwards += 1
                    del loss
                norms = []
                for a in adapters.values():
                    if a.A.grad is None or a.B.grad is None or a.calls != 4:
                        raise ValueError(
                            "every projection must execute and receive gradients"
                        )
                    norms.extend([a.A.grad.norm(), a.B.grad.norm()])
                norm_values = torch.stack(norms).detach().cpu().tolist()
                grad = float(
                    torch.nn.utils.clip_grad_norm_(
                        parameters, 1.0, error_if_nonfinite=True
                    )
                )
                optimizer.step()
                projection = project_adapters(torch, adapters, base_norms, 0.005)
                if frozen_state(base_parameters) != frozen:
                    raise ValueError("frozen parent parameter version/storage changed")
                modules = {}
                for i, (name, a) in enumerate(adapters.items()):
                    modules[name] = {
                        "calls": a.calls,
                        "A_grad_l2": norm_values[2 * i],
                        "B_grad_l2": norm_values[2 * i + 1],
                        "A_zero_grad": norm_values[2 * i] == 0.0,
                        "B_zero_grad": norm_values[2 * i + 1] == 0.0,
                        **projection[name],
                    }
                    if observe:
                        modules[name]["activations"] = {
                            key: {
                                "elements": value[0],
                                "sum": float(value[1]),
                                "sum_squares": float(value[2]),
                                "max_abs": float(value[3]),
                            }
                            for key, value in a.activation.items()
                        }
                    a.activation = None
                record = {
                    "step": step,
                    "case_ids": ids,
                    "epoch_position": cursor,
                    "example_CE": losses,
                    "mean_CE": sum(losses) / 4,
                    "answer_EOS_tokens": tokens,
                    "grad_l2_before_clip": grad,
                    "learning_rate": 0.0001,
                    "seconds": time.monotonic() - t,
                    "modules": modules,
                    **(
                        {
                            "loss_protocol": body_kl.PROTOCOL,
                            "example_loss_terms": objective_terms,
                            "mean_total_loss": sum(
                                row["total_loss"] for row in objective_terms
                            )
                            / 4,
                            "body_target_tokens": tokens,
                        }
                        if use_kl
                        else {}
                    ),
                    "gpu_peak_allocated": torch.cuda.max_memory_allocated()
                    if args.device != "cpu"
                    else None,
                    "gpu_peak_reserved": torch.cuda.max_memory_reserved()
                    if args.device != "cpu"
                    else None,
                }
                if use_kl:
                    record.pop("answer_EOS_tokens")  # New loss masks the turn terminator.
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                print(
                    json.dumps(
                        {
                            k: record[k]
                            for k in (
                                "step",
                                "mean_CE",
                                "grad_l2_before_clip",
                                "seconds",
                                "gpu_peak_allocated",
                            )
                        }
                    ),
                    flush=True,
                )
                if step in (4, 64):
                    common.assert_inputs(checkpoint, provenance)
                    exported = export_adapter(
                        torch,
                        args.output / ("pilot" if step == 4 else "final"),
                        adapters,
                        plan,
                        provenance,
                        identity,
                        step,
                        common.sha(args.output / "steps.jsonl"),
                        True,
                    )
                    exports.append(exported)
                    if step == 4:
                        ready = {
                            "status": "pilot_ready",
                            "completed_steps": 4,
                            "modules": 410,
                            "model_forwards": forwards,
                            **(
                                model_counters(args.output, forwards, forward_attempts)
                                if use_kl
                                else {}
                            ),
                            "base_unchanged": True,
                            "export": exported,
                            "step_log_sha256": common.sha(args.output / "steps.jsonl"),
                            "plan_sha256": common.sha(args.plan),
                            "elapsed_seconds": time.monotonic() - start,
                        }
                        write_new(args.output / "pilot-ready.json", ready)
                        pause_start = time.monotonic()
                        proceed = wait_continuation(
                            args.output,
                            common.sha(args.output / "pilot-ready.json"),
                            plan["continuation_wait_seconds"],
                            deadline + plan["continuation_wait_seconds"],
                        )
                        deadline += time.monotonic() - pause_start
                        if not proceed:
                            status = "pilot_only"
                            break
            else:
                status = "completed"
    except BaseException as exc:
        error = {"class": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        unchanged = frozen_state(base_parameters) == frozen
        try:
            common.assert_inputs(checkpoint, provenance)
            inputs_unchanged = True
        except Exception as exc:
            inputs_unchanged = False
            status = "failed"
            error = error or {"class": type(exc).__name__, "message": str(exc)}
        write_new(
            args.output / "summary.json",
            {
                "schema_version": 1,
                "status": status,
                "completed_steps": len(
                    (args.output / "steps.jsonl").read_text().splitlines()
                ),
                "attempted_step": step,
                "modules": 410,
                "model_forwards": forwards,
                "forward_attempts": forward_attempts,
                "last_example": current_example,
                "base_unchanged": unchanged and inputs_unchanged,
                "target_tensors": 410,
                "trainer_plan_sha256": common.sha(args.plan),
                "data_sha256": plan["data"]["sha256"],
                "training_ids": plan["data"]["ids"],
                "holdout_gradient_ids": [],
                "maximum_seconds_excludes_approval_wait": True,
                "base_unchanged_proof": "frozen parameters/version/storage and disk stat; no new full model hash",
                "exports": exports,
                "error": error,
                "elapsed_seconds": time.monotonic() - start,
                "source_sha256": provenance["input_sha256"],
                "plan_sha256": common.sha(args.plan),
                "activation_storage": storage.report(),
                **(
                    {
                        "loss_protocol": body_kl.PROTOCOL,
                        **model_counters(args.output, forwards, forward_attempts),
                    }
                    if use_kl
                    else {}
                ),
                "capability_claim": False,
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--gpu-exclusive-approved", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    plan, checkpoint, rows, proof, mapping, provenance, identity = preflight(args)
    if args.device != "cpu" and not args.gpu_exclusive_approved:
        raise ValueError("explicit exclusive GPU-owner handoff required")
    if not args.run:
        print(
            json.dumps(
                {
                    "mode": "preview",
                    "model_forwards": 0,
                    "modules": len(mapping),
                    "cases": len(rows),
                    "plan_sha256": common.sha(args.plan),
                    "hyperparameters": plan["hyperparameters"],
                    "source_sha256": provenance["input_sha256"],
                    "maximum_seconds": plan["maximum_seconds"],
                    "continuation_wait_seconds": plan["continuation_wait_seconds"],
                    **(
                        {
                            "loss_protocol": body_kl.PROTOCOL,
                            "planned_teacher_forwards": len(rows),
                            "planned_student_forwards": 256,
                            "teacher_checkpoint": plan.get(
                                "teacher_checkpoint", plan["checkpoint"]
                            ),
                        }
                        if body_kl.selected(plan)
                        else {}
                    ),
                }
            )
        )
        return
    args.output.mkdir()
    try:
        train(args, plan, checkpoint, rows, proof, mapping, provenance, identity)
    except BaseException as exc:
        if not (args.output / "summary.json").exists():
            write_new(
                args.output / "summary.json",
                {
                    "status": "failed",
                    "completed_steps": 0,
                    "error_class": type(exc).__name__,
                    **(
                        model_counters(args.output, 0, 0)
                        if body_kl.selected(plan)
                        else {}
                    ),
                    "error": str(exc),
                    "trainer_plan_sha256": common.sha(args.plan),
                    "source_sha256": provenance["input_sha256"],
                    "capability_claim": False,
                },
            )
        raise


if __name__ == "__main__":
    main()
