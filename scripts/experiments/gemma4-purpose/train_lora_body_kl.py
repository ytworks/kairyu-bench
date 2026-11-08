"""Opt-in body-only supervision and an explicit full-frame teacher's first-token KL."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import time

PROTOCOL = {
    "kind": "body_first_token_kl_v1",
    "lambda": 1.0,
    "teacher_frame": "identical_complete_frame",
    "teacher_dtype": "F32",
}
FRAME_PROTOCOL = "gemma4-sft-standard-on-body-first-token-kl-v1"


def selected(plan):
    value = plan.get("loss_protocol")
    if value is not None and value != PROTOCOL:
        raise ValueError(
            "unknown loss protocol; explicit fixed body/first-token KL required"
        )
    if value is None and "teacher_checkpoint" in plan:
        raise ValueError("separate teacher requires the explicit KL loss protocol")
    return value is not None


def body_offset(code):
    tree = ast.parse(code)
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        None,
    )
    if function is None:
        raise ValueError("no top-level Python function body")
    statements = function.body
    if (
        statements
        and isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, ast.Constant)
        and isinstance(statements[0].value.value, str)
    ):
        statements = statements[1:]
    if not statements:
        raise ValueError("function has no body after its docstring")
    node = statements[0]
    lines = code.splitlines(keepends=True)
    byte_offset = (
        sum(len(line.encode()) for line in lines[: node.lineno - 1]) + node.col_offset
    )
    return len(code.encode()[:byte_offset].decode())


def prepare_frames(tokenizer, rows, default_prepare):
    frames, proof = default_prepare(tokenizer, rows)
    unsupported = []
    for row, frame, evidence in zip(rows, frames, proof["rows"], strict=True):
        try:
            prefix = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["text"]}],
                tokenize=False,
                enable_thinking=True,
                add_generation_prompt=True,
            )
            full = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": row["text"]},
                    {"role": "assistant", "content": row["code"]},
                ],
                tokenize=False,
                enable_thinking=True,
                add_generation_prompt=False,
            )
            # The pinned Gemma template's strip_thinking macro trims its
            # final result. Restrict mapping to whitespace-only normalization;
            # channel-bearing references need a separate explicit contract.
            raw_code = row["code"]
            if "<channel|>" in raw_code or "<|channel>" in raw_code:
                raise ValueError("channel-bearing reference is unsupported")
            code = raw_code.strip()
            leading_trim = len(raw_code) - len(raw_code.lstrip())
            trailing_trim = len(raw_code) - len(raw_code.rstrip())
            if full != prefix + code + "<turn|>\n":
                raise ValueError("reference changed beyond official whitespace trim")
            start = len(prefix) + body_offset(code)
            end = len(prefix) + len(code)
            encoded = tokenizer(
                full, add_special_tokens=False, return_offsets_mapping=True
            )
            if encoded["input_ids"] != frame["ids"]:
                raise ValueError(
                    "offset-tokenization differs from exact full-frame IDs"
                )
            terminator = tokenizer.encode("<turn|>\n", add_special_tokens=False)
            positions = [
                i
                for i, (left, right) in enumerate(encoded["offset_mapping"])
                if left >= start
                and right > left
                and right <= end
                and frame["prefix_tokens"] <= i < len(frame["ids"]) - len(terminator)
            ]
            if not positions:
                raise ValueError("empty body target after conservative boundary mask")
            frame["body_target_positions"] = positions
            evidence.update(
                official_reference_normalization="strip_thinking final whitespace trim only; channel markers rejected",
                reference_leading_trim_characters=leading_trim,
                reference_trailing_trim_characters=trailing_trim,
                raw_body_character_offset=leading_trim + body_offset(code),
                body_character_start=start,
                body_character_end=end,
                body_target_positions=positions,
                body_target_tokens=len(positions),
                boundary_rule="mask any token crossing the body/terminator boundary",
            )
        except (ValueError, SyntaxError, KeyError, TypeError, UnicodeError) as error:
            unsupported.append(row["task_id"])
            frame["body_target_positions"] = []
            evidence.update(
                eligible=False,
                unsupported_type=type(error).__name__,
                unsupported_reason=str(error),
            )
    proof.update(
        frame_protocol=FRAME_PROTOCOL,
        loss_protocol=PROTOCOL,
        loss="mean body-only CE + lambda1 parent first-token full-vocabulary KL",
        teacher_frame="same complete IDs/length as student; never prefix-only",
        unsupported_ids=unsupported,
        ineligible_ids=[row["id"] for row in proof["rows"] if not row["eligible"]],
    )
    return frames, proof


def capped_logits(torch, head, hidden, cap):
    logits = head(hidden).float()
    return cap * torch.tanh(logits / cap) if cap else logits


def frame_sha(frame):
    return hashlib.sha256(json.dumps(frame["ids"]).encode()).hexdigest()


def cache_teacher(
    torch, model, frames, output, device, deadline, metadata, assert_inputs
):
    """One frozen teacher forward per complete frame, before training adapters exist."""
    from safetensors.torch import save_file

    output.mkdir()
    net, head = model.model.language_model, model.get_output_embeddings()
    cache, records = {}, []
    attempted = 0
    started = time.monotonic()
    status, error = "failed", None
    try:
        with torch.no_grad():
            for frame in frames:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "training deadline during teacher full-frame cache"
                    )
                assert_inputs()
                attempted += 1
                (output / "progress.json").write_text(
                    json.dumps(
                        {
                            "attempted": attempted,
                            "completed": len(records),
                            "current_id": frame["id"],
                        }
                    )
                    + "\n"
                )
                ids = torch.tensor([frame["ids"]], device=device)
                hidden = net(
                    input_ids=ids, use_cache=False, return_dict=True
                ).last_hidden_state
                logits = (
                    capped_logits(
                        torch,
                        head,
                        hidden[0, frame["prefix_tokens"] - 1 : frame["prefix_tokens"]],
                        float(net.config.final_logit_softcapping),
                    )[0]
                    .detach()
                    .cpu()
                    .contiguous()
                )
                if (
                    logits.dtype != torch.float32
                    or logits.ndim != 1
                    or not torch.isfinite(logits).all()
                ):
                    raise ValueError(
                        "teacher full-vocabulary F32 logits must be finite"
                    )
                path = output / (str(frame["id"]) + ".safetensors")
                save_file({"first_token_logits": logits}, str(path))
                cache[frame["id"]] = logits
                records.append(
                    {
                        "id": frame["id"],
                        "complete_tokens": len(frame["ids"]),
                        "prefix_tokens": frame["prefix_tokens"],
                        "token_ids_sha256": frame_sha(frame),
                        "vocabulary_size": logits.numel(),
                        "dtype": "F32",
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
                del hidden, ids
            assert_inputs()
            status = "completed"
    except BaseException as exc:
        error = {"class": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "attempted_forwards": attempted,
                    "completed_forwards": len(records),
                    "seconds": time.monotonic() - started,
                    "rows": records,
                    "metadata": metadata,
                    "frame": "identical complete frame used by student; no padding or truncation",
                    "holdout_forwards": 0,
                    "error": error,
                },
                indent=2,
            )
            + "\n"
        )
    return cache


def body_kl_loss(torch, net, head, frame, teacher_logits, device):
    positions = frame["body_target_positions"]
    if not positions or any(
        i < frame["prefix_tokens"] or i >= len(frame["ids"]) for i in positions
    ):
        raise ValueError("invalid explicit body target positions")
    ids = torch.tensor([frame["ids"]], device=device)
    hidden = net(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
    cap = float(net.config.final_logit_softcapping)
    indices = torch.tensor(positions, device=device)
    body_logits = capped_logits(torch, head, hidden[0, indices - 1], cap)
    ce = torch.nn.functional.cross_entropy(body_logits, ids[0, indices])
    first_logits = capped_logits(
        torch,
        head,
        hidden[0, frame["prefix_tokens"] - 1 : frame["prefix_tokens"]],
        cap,
    )[0]
    parent = teacher_logits.detach().to(device=device, dtype=torch.float32)
    if parent.shape != first_logits.shape or not torch.isfinite(parent).all():
        raise ValueError("teacher/student vocabulary or finite-value mismatch")
    parent_logp = torch.log_softmax(parent, dim=-1)
    student_logp = torch.log_softmax(first_logits, dim=-1)
    kl = (parent_logp.exp() * (parent_logp - student_logp)).sum()
    total = ce + PROTOCOL["lambda"] * kl
    return (
        total,
        len(positions),
        {
            "body_CE": float(ce.detach()),
            "first_token_KL": float(kl.detach()),
            "total_loss": float(total.detach()),
            "lambda": PROTOCOL["lambda"],
        },
    )


def counters(output):
    path = Path(output) / "teacher-cache/summary.json"
    if not path.exists():
        return {"teacher_model_forwards": 0, "teacher_forward_attempts": 0}
    data = json.loads(path.read_text())
    return {
        "teacher_model_forwards": data["completed_forwards"],
        "teacher_forward_attempts": data["attempted_forwards"],
        "teacher_cache_summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
