#!/usr/bin/env python3
"""Collect Gemma 4 native gain and optional virtual-block purpose derivatives."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

import recipe_common as common
import objective_variants as variants


def column_group_norms(weight, groups: int, torch):
    """L2 costs of unit virtual gains; row chunks avoid a full FP32 matrix copy."""
    if weight.ndim != 2 or groups < 1 or weight.shape[1] % groups:
        raise ValueError("invalid virtual projection grouping")
    squares = torch.zeros(weight.shape[1], dtype=torch.float64, device=weight.device)
    rows = max(1, (16 * 1024**2) // (weight.shape[1] * 4))
    for start in range(0, weight.shape[0], rows):
        part = weight.detach()[start:start + rows].float()
        squares += part.square().sum(0, dtype=torch.float64)
    return squares.reshape(groups, weight.shape[1] // groups).sum(1).sqrt().float()


def collect(args, checkpoint, cases, provenance):
    objective_kind = getattr(args, "objective", "margin")
    if objective_kind not in {"margin", "correct-logprob"} or (objective_kind == "correct-logprob" and args.mode != "chat-public-answer"):
        raise ValueError("unsupported objective/frame combination")
    answer_batching = getattr(args, "answer_batching", "paired")
    if answer_batching not in {"paired", "serial"} or (answer_batching == "serial" and (args.mode != "chat-public-answer" or args.virtual)):
        raise ValueError("serial answer batching requires chat-public-answer native-only collection")
    objective = common.objective if objective_kind == "margin" else variants.correct_logprob_objective
    torch, tokenizer, model = common.load_model(args, cases=cases, legacy_limit=None)
    net, head = model.model.language_model, model.get_output_embeddings()
    prepared, frame = common.prepare(tokenizer, cases, args.mode, max_frame_tokens=common.frame_limit(args, None))
    common.save(args.output / "frame-evidence.json", frame)
    names = {"model.language_model." + name: value for name, value in net.named_parameters()
             if value.ndim <= 1 and value.is_floating_point()}
    for index, layer in enumerate(net.layers):
        names[f"model.language_model.layers.{index}.layer_scalar"] = layer.layer_scalar
    kinds = {name: "native" for name in names}
    originals, hooks, costs = {}, [], {}
    start = time.monotonic()
    try:
        for name, value in names.items():
            original = value.detach().flatten().float().cpu().numpy().copy()
            if value.dtype != torch.bfloat16 or not np.array_equal(original, checkpoint.read(name)):
                raise ValueError("loaded native BF16 value differs from official checkpoint")
            originals[name] = original
        if args.virtual:
            for index, layer in enumerate(net.layers):
                projections = (
                    ("o_head", layer.self_attn.o_proj, net.config.num_attention_heads),
                    ("ffn_down_column", layer.mlp.down_proj, layer.mlp.down_proj.in_features),
                )
                for family, projection, count in projections:
                    key = f"virtual.layers.{index}.{family}"
                    if type(projection) is not torch.nn.Linear:
                        raise ValueError("virtual gain cost requires plain untransformed torch.nn.Linear")
                    if projection.weight.shape[1] % count:
                        raise ValueError("projection cannot be divided into explicit groups")
                    width = projection.weight.shape[1] // count
                    gate = torch.ones(count, dtype=torch.float32, device=args.device, requires_grad=True)
                    def pre(module, inputs, gate=gate, width=width):
                        scale = gate.to(inputs[0].dtype)
                        if width != 1:
                            scale = scale.repeat_interleave(width)
                        return (inputs[0] * scale, *inputs[1:])
                    hooks.append(projection.register_forward_pre_hook(pre))
                    names[key], kinds[key] = gate, "virtual"
                    costs[key] = column_group_norms(projection.weight, count, torch).cpu().numpy()
        for value in names.values():
            value.requires_grad_(True)
        ordered = sorted(names)
        coverage = {"text_layers": len(net.layers), "native_tensors": sum(v == "native" for v in kinds.values()),
                    "virtual_tensors": sum(v == "virtual" for v in kinds.values()),
                    "native_coordinates": sum(v.numel() for k, v in names.items() if kinds[k] == "native"),
                    "virtual_coordinates": sum(v.numel() for k, v in names.items() if kinds[k] == "virtual"),
                    "tensors": [{"name": k, "kind": kinds[k], "shape": list(names[k].shape),
                                 "dtype": str(names[k].dtype)} for k in ordered],
                    "limits": ["Virtual derivatives are block/column gain derivatives, not independent matrix-cell gradients.",
                               "First-order purpose sensitivity is not a causal ability assignment or finite response."]}
        common.save(args.output / "coverage.json", coverage)
        np.savez_compressed(args.output / "source-native-values.npz", **originals)
        if costs:
            np.savez_compressed(args.output / "virtual-unit-gain-l2-costs.npz", **costs)
        records = []
        for group, folds in prepared.items():
            for fold, rows in enumerate(folds):
                accum = {key: torch.zeros_like(names[key], dtype=torch.float32) for key in ordered}
                measured, unused = {}, set()
                packed = common.objective_batches(rows, args.mode, args.max_cases_per_microbatch)
                forward_count = 0
                def accumulate(values, sign=1.):
                    gradients = torch.autograd.grad(values.sum() * sign / len(rows),
                                                    [names[k] for k in ordered], allow_unused=True)
                    for key, gradient in zip(ordered, gradients, strict=True):
                        if gradient is None:
                            unused.add(key)
                        else:
                            # Every answer VJP is converted before accumulation;
                            # no BF16 .grad buffer accumulates across answers.
                            accum[key] += gradient.detach().float()
                    return values.detach().cpu().tolist()
                for batch in packed:
                    if answer_batching == "paired":
                        margins = objective(torch, net, head, tokenizer, batch, args.mode, args.device)
                        observed = accumulate(margins)
                        measured.update({row[0]: float(value) for row, value in zip(batch, observed, strict=True)})
                        forward_count += 1
                        del margins
                    else:
                        width = max(len(row[1]) + max(len(row[2]), len(row[3])) for row in batch)
                        selections = (("correct", 1.), ("foil", -1.)) if objective_kind == "margin" else (("correct", 1.),)
                        for row in batch:
                            measured[row[0]] = 0.
                            for answer_kind, sign in selections:
                                values = variants.single_answer_logprob_objective(
                                    torch, net, head, tokenizer, row, answer_kind, args.mode, args.device, width)
                                measured[row[0]] += sign * float(accumulate(values, sign)[0])
                                forward_count += 1
                                del values  # Graph is freed before the next answer forward.
                arrays = {key: value.detach().cpu().numpy() for key, value in accum.items()}
                if not all(np.isfinite(value).all() for value in arrays.values()):
                    raise ValueError("nonfinite native derivative")
                reference = variants.objective_references(cases[group][fold], measured, args.reference_tolerance, objective_kind)
                path = args.output / f"gradient-{group}-{fold}.npz"
                np.savez_compressed(path, **arrays)
                value_key = "margin" if objective_kind == "margin" else "correct_logprob"
                record = {"purpose": group, "fold": fold, "examples": len(rows),
                          ("margins" if objective_kind == "margin" else "correct_logprobs"):
                              [{"id": row[0], value_key: measured[row[0]]} for row in rows],
                          "mean_" + value_key: float(np.mean(list(measured.values()))), "microbatches": len(packed),
                          "answer_batching": answer_batching, "model_forward_count": forward_count,
                          "unused_tensors": sorted(unused), "file": path.name, **reference}
                records.append(record)
                common.save(args.output / "progress.json", {"completed_folds": len(records), "last": record})
        common.assert_inputs(checkpoint, provenance)
        for name, original in originals.items():
            if not np.array_equal(names[name].detach().flatten().float().cpu().numpy(), original):
                raise ValueError("collector changed original model weights")
        summary = {"status": "completed", "mode": args.mode, "objective_kind": objective_kind, "answer_batching": answer_batching, "coverage": coverage,
                   "folds": records, "seconds": time.monotonic() - start,
                   "runtime": {"torch": torch.__version__, "transformers": common.transformers_version(), "device": args.device, "weight_dtype": "BF16",
                               **common.observation_runtime(args, provenance), "gradients": "native BF16 autograd, accumulated FP32",
                                "max_cases_per_microbatch": args.max_cases_per_microbatch,
                                "gpu_peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30 if args.device != "cpu" else None,
                                "gpu_peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30 if args.device != "cpu" else None},
                   "objective": ("mean correct-answer token log probability (maximize)" if objective_kind == "correct-logprob" else
                                 "mean correct-minus-foil public-answer mean log probability" if args.mode == "chat-public-answer"
                                 else "mean first distinguishing next-token softcapped logit margin"),
                   "source_values_unchanged": True, "candidate_selection": False,
                   "limits": ["Authored teacher-forced diagnostics only; no generated reasoning or benchmark outcomes.",
                              "Raw-next-token and chat-public-answer coordinates must not be mixed as matching observations.",
                              "Correct-logprob arrays are not margin derivatives; downstream solver must explicitly match objective_kind.",
                              "Serial answers preserve the mathematical loss and token widths; BF16 batching/gradient reduction need not be bit identical to paired collection.",
                              "No checkpoint edits, solver, finite candidate test, or WSP acceptance decision."]}
        common.save(args.output / "summary.json", summary)
        return summary
    finally:
        for hook in hooks:
            hook.remove()
        for value in names.values():
            value.requires_grad_(False)
            value.grad = None


def main(argv=None):
    parser = common.parser(__doc__, chat=True)
    parser.add_argument("--objective", choices=("margin", "correct-logprob"), default="margin",
                        help="explicit loss; correct-logprob requires chat-public-answer and keeps good+foil batch layout")
    parser.add_argument("--answer-batching", choices=("paired", "serial"), default="paired",
                        help="serial uses one answer graph at a time; chat native-only, unchanged full tokens and pack padding")
    parser.add_argument("--virtual", action="store_true", help="also differentiate explicit O-head and MLP-down-column unit gains")
    args = parser.parse_args(argv)
    if args.answer_batching == "serial" and (args.mode != "chat-public-answer" or args.virtual):
        parser.error("serial answer batching requires chat-public-answer native-only collection")
    checkpoint, cases, provenance = common.preflight(args, Path(__file__))
    variants.pin_objective(provenance, args.mode, args.objective)
    variants.attach_correct_references(args.cases, cases, provenance, args.objective)
    provenance["virtual_unit_gains"] = args.virtual
    provenance["answer_batching"] = args.answer_batching
    if args.answer_batching == "serial":
        provenance["objective_forward_layout"] = "one case/answer per forward and immediate VJP; original paired-pack token padding width; good-minus-foil or good-only; FP32 answer-gradient accumulation"
    if not args.run:
        import json
        print(json.dumps(provenance, indent=2))
        return 0
    args.output.mkdir(parents=True)
    common.save(args.output / "provenance.json", provenance)
    try:
        result = collect(args, checkpoint, cases, provenance)
    except BaseException as error:
        common.save(args.output / "failure.json", {"status": "incomplete", "error_type": type(error).__name__})
        raise
    print(f"Collected {len(result['folds'])} explicit purpose folds; no candidate selected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
