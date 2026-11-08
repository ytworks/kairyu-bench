#!/usr/bin/env python3
"""Observe full MLP UP-output unit gains on a frozen Gemma model; no edits."""

from __future__ import annotations

import json
from pathlib import Path
import re
import time

import numpy as np

import objective_variants as variants
import recipe_common as common

PATTERN = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.up_proj\.weight")


def row_norms(torch, weight):
    """Float64 row costs with bounded temporary FP32 chunks."""
    if weight.ndim != 2:
        raise ValueError("UP weight must be a matrix")
    count = max(1, (16 * 1024**2) // (weight.shape[1] * 4))
    return torch.cat(
        [
            weight.detach()[start : start + count]
            .float()
            .square()
            .sum(1, dtype=torch.float64)
            .sqrt()
            for start in range(0, weight.shape[0], count)
        ]
    )


def gain_factor(torch, saved_output, upstream):
    """Contract output dot VJP, without a full FP32 activation copy."""
    if saved_output.shape != upstream.shape or saved_output.ndim < 2:
        raise ValueError("UP output/VJP shape differs")
    dimensions = tuple(range(upstream.ndim - 1))
    return torch.cat(
        [
            (
                saved_output[..., start : start + 256].float()
                * upstream[..., start : start + 256].float()
            )
            .sum(dimensions, dtype=torch.float64)
            .float()
            for start in range(0, upstream.shape[-1], 256)
        ]
    )


def answer_derivatives(
    torch,
    net,
    head,
    tokenizer,
    row,
    device,
    modules,
    fold_examples,
    objective_kind,
    verify_names,
):
    """Serial complete answers; frozen weights, embedding leaf, output hooks."""
    if objective_kind not in {"correct-logprob", "margin"} or fold_examples < 1:
        raise ValueError("invalid objective or fold size")
    selections = (
        (("correct", 1.0),)
        if objective_kind == "correct-logprob"
        else (("correct", 1.0), ("foil", -1.0))
    )
    width = len(row[1]) + max(len(row[2]), len(row[3]))
    accumulated, checks, measured, forwards = {}, {}, 0.0, 0
    for answer, sign in selections:
        embedding, probes, captured, calls, hooks = [], {}, {}, {}, []

        def embed_hook(module, inputs, output):
            value = output.detach().requires_grad_(True)
            embedding.append(value)
            return value

        def factory(name):
            def hook(module, inputs, output):
                calls[name] = calls.get(name, 0) + 1
                if calls[name] != 1 or not output.requires_grad:
                    raise ValueError(
                        "one differentiable UP forward required; recomputation unsupported"
                    )
                # The gate*UP multiplication already needs this storage for VJP.
                # detach shares it; do not clone it or retain dense down inputs.
                saved = output.detach()
                indices = None
                if name in verify_names and forwards == 0:
                    indices = (
                        torch.linspace(
                            0,
                            output.shape[-1] - 1,
                            min(4, output.shape[-1]),
                            device=device,
                        )
                        .long()
                        .unique()
                    )
                    probe = torch.zeros(
                        len(indices),
                        dtype=torch.float32,
                        device=device,
                        requires_grad=True,
                    )
                    probes[name] = (probe, indices)
                    update = torch.zeros_like(output)
                    update.index_add_(
                        -1,
                        indices,
                        (saved.index_select(-1, indices).float() * probe).to(
                            output.dtype
                        ),
                    )
                    output = output + update

                def backward(upstream):
                    captured[name] = gain_factor(torch, saved, upstream)

                output.register_hook(backward)
                return output

            return hook

        try:
            hooks.append(net.embed_tokens.register_forward_hook(embed_hook))
            for name, module in modules.items():
                hooks.append(module.register_forward_hook(factory(name)))
            values = variants.single_answer_logprob_objective(
                torch,
                net,
                head,
                tokenizer,
                row,
                answer,
                "chat-public-answer",
                device,
                width,
            )
            if len(embedding) != 1:
                raise ValueError("one embedding forward per answer required")
            derivatives = torch.autograd.grad(
                values.sum() * sign / fold_examples,
                [embedding[0], *[item[0] for item in probes.values()]],
            )
            if set(captured) != set(modules):
                raise ValueError("missing UP gain derivative")
            for name, explicit in zip(probes, derivatives[1:], strict=True):
                indices = probes[name][1]
                factor = captured[name][indices].float()
                if not torch.allclose(factor, explicit, rtol=2e-5, atol=1e-7):
                    raise ValueError("UP gain factor differs from zero-additive probe")
                checks[name] = {
                    "units": len(indices),
                    "indices": indices.cpu().tolist(),
                    "max_abs_difference": float((factor - explicit).abs().max()),
                }
            for name, value in captured.items():
                accumulated[name] = accumulated.get(name, 0) + value.double()
            measured += sign * float(values.detach()[0])
            forwards += 1
            del values, derivatives
        finally:
            for hook in hooks:
                hook.remove()
    return measured, accumulated, checks, forwards


def collect(args, checkpoint, cases, provenance):
    torch, tokenizer, model = common.load_model(args, cases=cases, legacy_limit=None)
    net, head = model.model.language_model, model.get_output_embeddings()
    if any(p.requires_grad for p in model.parameters()) or any(
        getattr(m, "gradient_checkpointing", False) for m in net.modules()
    ):
        raise ValueError(
            "UP unit collection requires frozen parameters without recomputation"
        )
    versions = {name: value._version for name, value in model.named_parameters()}
    prepared, frame = common.prepare(
        tokenizer,
        cases,
        "chat-public-answer",
        max_frame_tokens=common.frame_limit(args),
    )
    common.save(args.output / "frame-evidence.json", frame)
    modules, entries, norms, cursor = {}, [], {}, 0
    for name, tensor in sorted(checkpoint.tensors.items()):
        match = PATTERN.fullmatch(name)
        if not match:
            continue
        module = net.get_submodule(
            name.removeprefix("model.language_model.").removesuffix(".weight")
        )
        if (
            type(module) is not torch.nn.Linear
            or module.bias is not None
            or module.weight.dtype != torch.bfloat16
            or tensor.dtype != "BF16"
            or list(module.weight.shape) != list(tensor.shape)
        ):
            raise ValueError("loaded UP projection/header differs")
        modules[name] = module
        norms[name] = row_norms(torch, module.weight).cpu().numpy()
        if not np.isfinite(norms[name]).all():
            raise ValueError("nonfinite UP row cost")
        entries.append(
            {
                "tensor": name,
                "shape": list(tensor.shape),
                "count": tensor.shape[0],
                "layer": int(match[1]),
                "start": cursor,
            }
        )
        cursor += tensor.shape[0]
    if not modules:
        raise ValueError("no UP projections observed")
    coverage = {
        "units_method": "up_output_gain",
        "entries": entries,
        "observed_units": cursor,
        "observed_up_matrices": len(entries),
        "native_parameters": "frozen_not_observed",
        "full_native_gradient_arrays": False,
        "cost": "L2 of BF16 source UP weight row, accumulated FP64",
        "gain_coordinate": "multiply UP output unit by (1+g); derivative at g=0",
    }
    common.save(args.output / "coverage.json", coverage)
    np.savez_compressed(args.output / "source-up-row-norms.npz", **norms)
    records, checks = [], {}
    started = time.monotonic()
    for group, fold, rows in common.finite_replay.ordered_folds(
        prepared, common.execution_order(args)
    ):
        totals = {
            name: torch.zeros(
                module.out_features, dtype=torch.float64, device=args.device
            )
            for name, module in modules.items()
        }
        measured, forwards = {}, 0
        for index, row in enumerate(rows):
            verify = (
                {next(iter(modules)), next(reversed(modules))}
                if not records and index == 0
                else set()
            )
            value, gradients, verified, count = answer_derivatives(
                torch,
                net,
                head,
                tokenizer,
                row,
                args.device,
                modules,
                len(rows),
                args.objective,
                verify,
            )
            measured[row[0]] = value
            for name in modules:
                totals[name] += gradients[name]
            checks.update(verified)
            forwards += count
            del gradients
        arrays = {name: value.float().cpu().numpy() for name, value in totals.items()}
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("nonfinite UP gain derivative")
        reference = variants.objective_references(
            cases[group][fold], measured, args.reference_tolerance, args.objective
        )
        file = args.output / f"gradient-{group}-{fold}.npz"
        np.savez_compressed(file, **arrays)
        value_key = "margin" if args.objective == "margin" else "correct_logprob"
        ordered = prepared[group][fold]
        record = {
            "purpose": group,
            "fold": fold,
            "examples": len(rows),
            "file": file.name,
            ("margins" if args.objective == "margin" else "correct_logprobs"): [
                {"id": row[0], value_key: measured[row[0]]} for row in ordered
            ],
            "mean_" + value_key: float(np.mean([measured[row[0]] for row in ordered])),
            "microbatches": len(rows),
            "answer_batching": "serial",
            "model_forward_count": forwards,
            **reference,
        }
        records.append(record)
        common.save(
            args.output / "progress.json",
            {"completed_folds": len(records), "last": record},
        )
        del totals, arrays
    canonical = {
        (group, fold): i
        for i, (group, fold, _) in enumerate(
            common.finite_replay.ordered_folds(prepared)
        )
    }
    records.sort(key=lambda r: canonical[(r["purpose"], r["fold"])])
    common.assert_inputs(checkpoint, provenance)
    if any(
        value._version != versions[name]
        or value.grad is not None
        or value.requires_grad
        for name, value in model.named_parameters()
    ):
        raise ValueError("model parameter version/gradient changed")
    summary = {
        "status": "completed",
        "mode": "chat-public-answer",
        "units_method": "up_output_gain",
        "objective_kind": args.objective,
        "answer_batching": "serial",
        "folds": records,
        "coverage": coverage,
        "zero_additive_checks": checks,
        "seconds": time.monotonic() - started,
        "source_values_unchanged": True,
        "parameter_versions_unchanged": True,
        "candidate_selection": False,
        "full_native_gradient_arrays": False,
        "runtime": {
            "torch": torch.__version__,
            "transformers": common.transformers_version(),
            "device": args.device,
            "weight_dtype": "BF16",
            **common.observation_runtime(args, provenance),
            "gradient_checkpointing": False,
            "max_cases_per_microbatch": 1,
            "gradient_accumulation": "FP32 output-VJP factors, FP64 answer/fold accumulation",
            "gpu_peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30
            if args.device != "cpu"
            else None,
            "gpu_peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30
            if args.device != "cpu"
            else None,
        },
        "limits": [
            "Authored local gain derivatives, not free-generation capability.",
            "No native gradients, optimizer, checkpoint edits or adoption.",
            "Source bytes use prior full checkpoint attestation and current stat; no new full payload hash.",
            "UP output storage is shared with the model graph; total activation memory remains empirical.",
        ],
    }
    common.save(args.output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = common.parser(__doc__, chat=True)
    parser.set_defaults(mode="chat-public-answer", max_cases_per_microbatch=1)
    parser.add_argument("--answer-batching", choices=("serial",), default="serial")
    parser.add_argument(
        "--objective", choices=("correct-logprob", "margin"), default="margin"
    )
    args = parser.parse_args(argv)
    if args.mode != "chat-public-answer" or args.max_cases_per_microbatch != 1:
        parser.error("UP gain observation requires one complete public-answer case")
    checkpoint, cases, provenance = common.preflight(args, Path(__file__))
    variants.pin_objective(provenance, args.mode, args.objective)
    variants.attach_correct_references(args.cases, cases, provenance, args.objective)
    provenance.update(
        units_method="up_output_gain",
        answer_batching="serial",
        native_parameters="frozen_not_observed",
        virtual_unit_gains=True,
        objective_forward_layout="one complete case/answer then immediate input VJP",
    )
    if not args.run:
        print(json.dumps(provenance, indent=2))
        return 0
    args.output.mkdir(parents=True)
    common.save(args.output / "provenance.json", provenance)
    try:
        result = collect(args, checkpoint, cases, provenance)
    except BaseException as error:
        common.save(
            args.output / "failure.json",
            {"status": "incomplete", "error_type": type(error).__name__},
        )
        raise
    print(
        f"Collected {len(result['folds'])} UP-output gain folds; no candidate selected."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
