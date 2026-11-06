#!/usr/bin/env python3
"""Collect 512 stratified true weight-coordinate derivatives per Gemma 4 projection."""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import re
import time

import numpy as np

import recipe_common as common

PATTERN = re.compile(r"^model\.language_model\.layers\.(\d+)\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj\.weight$")


def attention_geometry(config: dict, layer: int, kind: str, shape: tuple[int, int]) -> tuple[int, int]:
    """Resolve current HF layer overrides or legacy checkpoint fields, then check actual shape."""
    full = config["layer_types"][layer] == "full_attention"
    per_layer = config.get("per_layer_config")
    override: dict[str, int] = {}
    if per_layer is not None:
        if not isinstance(per_layer, dict):
            raise ValueError("per_layer_config must map layer indices to overrides")
        matches = [value for key, value in per_layer.items()
                   if str(key).isdigit() and int(key) == layer]
        if len(matches) > 1 or (matches and not isinstance(matches[0], dict)):
            raise ValueError("ambiguous per-layer attention configuration")
        override = matches[0] if matches else {}
    def positive(key, legacy_key=None):
        value = override.get(key, config.get(key))
        if per_layer is None and full and legacy_key is not None:
            value = config.get(legacy_key, value)
        if type(value) is not int or value < 1:
            raise ValueError(f"missing or invalid attention geometry: {key}")
        return value
    query_heads = positive("num_attention_heads")
    kv_heads = positive("num_key_value_heads", "num_global_key_value_heads")
    width = positive("head_dim", "global_head_dim")
    hidden = positive("hidden_size")
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    heads = query_heads if kind in ("q", "o") else kv_heads
    expected = (hidden, heads * width) if kind == "o" else (heads * width, hidden)
    if shape != expected:
        raise ValueError(f"{kind}-projection shape {shape} does not match configured heads {expected}")
    return heads, width


def sample(name: str, shape: tuple[int, int], kind: str, layer: int, config: dict, seed: int):
    """Fixed 512-cell stratification; distinct source heads never collapse into one group."""
    nr, nc = shape
    local = int.from_bytes(hashlib.sha256((str(seed) + "/" + name).encode()).digest()[:8], "little")
    rng, cells = np.random.default_rng(local), []
    def bounds(index, bands, size):
        return index * size // bands, (index + 1) * size // bands
    def cell(r0, r1, c0, c1):
        if r1 <= r0 or c1 <= c0:
            raise ValueError("matrix is too small for the explicit 512-cell strata")
        cells.append((int(rng.integers(r0, r1)), int(rng.integers(c0, c1))))
    if kind in ("q", "k", "v"):
        heads, width = attention_geometry(config, layer, kind, shape)
        row_bands = 8 if heads == 4 else 4
        column_bands = 512 // (heads * row_bands)
        if nr % heads or heads * row_bands * column_bands != 512:
            raise ValueError("unsupported head geometry for 512-cell strata")
        for head in range(heads):
            for band in range(row_bands):
                r0, r1 = bounds(band, row_bands, width)
                for column in range(column_bands):
                    c0, c1 = bounds(column, column_bands, nc)
                    cell(head * width + r0, head * width + r1, c0, c1)
        strata = {"heads": heads, "head_row_bands": row_bands, "input_column_bands": column_bands}
    elif kind == "o":
        heads, width = attention_geometry(config, layer, kind, shape)
        row_bands, head_column_bands = 4, 512 // (heads * 4)
        if nc % heads or heads * row_bands * head_column_bands != 512:
            raise ValueError("unsupported O-projection geometry for 512-cell strata")
        for head in range(heads):
            for band in range(row_bands):
                r0, r1 = bounds(band, row_bands, nr)
                for column in range(head_column_bands):
                    c0, c1 = bounds(column, head_column_bands, width)
                    cell(r0, r1, head * width + c0, head * width + c1)
        strata = {"input_heads": heads, "residual_row_bands": row_bands, "head_column_bands": head_column_bands}
    elif kind in ("gate", "up", "down"):
        row_bands, column_bands = (32, 16) if kind == "down" else (16, 32)
        for row in range(row_bands):
            r0, r1 = bounds(row, row_bands, nr)
            for column in range(column_bands):
                c0, c1 = bounds(column, column_bands, nc)
                cell(r0, r1, c0, c1)
        strata = {"output_row_bands": row_bands, "input_column_bands": column_bands}
    else:
        raise ValueError("unsupported matrix family")
    if len(cells) != 512 or len(set(cells)) != 512:
        raise ValueError("stratification must produce exactly 512 distinct coordinates")
    return np.asarray(cells, dtype=np.int64), strata


def factor_derivative(upstream, selected_inputs, rows):
    """FP32 products, FP64 token/batch accumulation; inputs were gathered at sampled columns."""
    import torch

    products = upstream.detach().index_select(-1, rows).float() * selected_inputs
    return products.reshape(-1, len(rows)).sum(0, dtype=torch.float64)


def checked_batches(rows, mode, max_cases_per_microbatch=None):
    """Use the shared packing without silently truncating an oversized chat case."""
    if mode == "chat-public-answer" and any(len(row[1]) + max(len(row[2]), len(row[3])) > 512 for row in rows):
        raise ValueError("chat matrix pilot requires prefix plus each answer <=512 tokens; no truncation")
    return common.objective_batches(rows, mode, max_cases_per_microbatch)


def batch_derivatives(torch, net, head, tokenizer, rows, mode, device, plans, modules, fold_examples, verify_names):
    """One eager forward/backward, with the shared objective and explicit fold weight."""
    captured, embedding, probes, calls, hooks = {}, [], {}, {}, []
    def embed_hook(module, inputs, output):
        value = output.detach().requires_grad_(True)
        embedding.append(value)
        return value
    def factory(name):
        sampled_rows, columns = plans[name]
        def hook(module, inputs, output):
            calls[name] = calls.get(name, 0) + 1
            if calls[name] != 1:
                raise ValueError("projection recomputation is unsupported; disable gradient checkpointing")
            if not output.requires_grad:
                raise ValueError("projection output must participate in the calibration gradient")
            selected = inputs[0].detach().index_select(-1, columns).float()
            if name in verify_names:
                probe = torch.zeros(4, dtype=torch.float32, device=device, requires_grad=True)
                probes[name] = probe
                update = torch.zeros_like(output)
                update.index_add_(-1, sampled_rows[:4], (selected[..., :4] * probe).to(output.dtype))
                output = output + update
            def backward(upstream):
                value = factor_derivative(upstream, selected, sampled_rows)
                captured[name] = captured.get(name, 0) + value
            output.register_hook(backward)
            return output
        return hook
    try:
        hooks.append(net.embed_tokens.register_forward_hook(embed_hook))
        for name, module in modules.items():
            hooks.append(module.register_forward_hook(factory(name)))
        margins = common.objective(torch, net, head, tokenizer, rows, mode, device)
        if len(embedding) != 1:
            raise ValueError("expected exactly one embedding forward per microbatch")
        derivatives = torch.autograd.grad(margins.sum() / fold_examples, [embedding[0], *probes.values()])
        if set(captured) != set(plans):
            raise ValueError("a sampled projection did not participate")
        checks = {}
        for name, explicit in zip(probes, derivatives[1:], strict=True):
            factor = captured[name][:4].float()
            checks[name] = {"cells": 4, "max_abs_difference": float((factor - explicit).abs().max())}
            if not torch.allclose(factor, explicit, rtol=2e-5, atol=1e-7):
                raise ValueError("sampled-factor derivative differs from explicit zero-additive probe")
        return margins.detach(), captured, checks
    finally:
        for hook in hooks:
            hook.remove()


def collect(args, checkpoint, cases, provenance):
    from safetensors import safe_open

    torch, tokenizer, model = common.load_model(args)
    net, head = model.model.language_model, model.get_output_embeddings()
    config = net.config.to_dict()
    prepared, frame = common.prepare(tokenizer, cases, args.mode)
    packed_folds = {group: [checked_batches(rows, args.mode, args.max_cases_per_microbatch) for rows in folds]
                    for group, folds in prepared.items()}
    if any(getattr(module, "gradient_checkpointing", False) for module in net.modules()):
        raise ValueError("matrix collector does not support gradient checkpoint recomputation")
    common.save(args.output / "frame-evidence.json", frame)
    plans, entries, modules = {}, [], {}
    coordinates, original_values = [], []
    for name, tensor in sorted(checkpoint.tensors.items()):
        match = PATTERN.fullmatch(name)
        if not match:
            continue
        layer, kind = int(match[1]), match[3]
        module = net.get_submodule(name.removeprefix("model.language_model.").removesuffix(".weight"))
        if type(module) is not torch.nn.Linear or module.weight.dtype != torch.bfloat16 or tuple(module.weight.shape) != tensor.shape:
            raise ValueError("loaded projection does not match checkpoint metadata")
        rc, strata = sample(name, tuple(module.weight.shape), kind, layer, config, args.seed)
        rows = torch.as_tensor(rc[:, 0].copy(), dtype=torch.long, device=args.device)
        columns = torch.as_tensor(rc[:, 1].copy(), dtype=torch.long, device=args.device)
        values = module.weight.detach()[rows, columns].float().cpu()
        with safe_open(str(tensor.shard), framework="pt", device="cpu") as shard:
            original = shard.get_tensor(name)[torch.from_numpy(rc[:, 0].copy()), torch.from_numpy(rc[:, 1].copy())].float()
        if not torch.equal(values, original):
            raise ValueError("sampled GPU/CPU source values differ from official BF16 weights")
        plans[name], modules[name] = (rows, columns), module
        entries.append({"tensor": name, "layer": layer, "family": kind, "shape": list(tensor.shape),
                        "start": len(entries) * 512, "count": 512, "strata": strata})
        coordinates.append(rc)
        original_values.append(values.numpy())
    if not entries:
        raise ValueError("no supported Gemma projection matrices found")
    names = list(plans)
    np.savez_compressed(args.output / "sample-plan.npz",
                        coordinates=np.concatenate(coordinates), source_values=np.concatenate(original_values))
    coverage = {"text_layers": len(net.layers), "sampled_matrices": len(entries),
                "families": dict(Counter(e["family"] for e in entries)),
                "sampled_coordinates": len(entries) * 512, "coordinates_per_matrix": 512, "seed": args.seed,
                "represented_matrix_elements": sum(int(np.prod(e["shape"])) for e in entries),
                "actual_values_crosschecked_against_safetensors": True, "entries": entries,
                "limits": ["Sampled true matrix cells, not every matrix element.",
                           "dL/dW is a local synthetic-purpose derivative, not a finite edit outcome or ability localization."]}
    common.save(args.output / "coverage.json", coverage)
    checks, records = {}, []
    start = time.monotonic()
    for group, folds in prepared.items():
        for fold, rows in enumerate(folds):
            accumulated = {name: torch.zeros(512, dtype=torch.float64, device=args.device) for name in names}
            measured = {}
            packed = packed_folds[group][fold]
            for batch_index, batch in enumerate(packed):
                verify_names = {names[0], names[-1]} if not records and batch_index == 0 else set()
                margins, captured, verified = batch_derivatives(
                    torch, net, head, tokenizer, batch, args.mode, args.device,
                    plans, modules, len(rows), verify_names)
                checks.update(verified)
                for name in names:
                    accumulated[name] += captured[name]
                measured.update({row[0]: float(value) for row, value in zip(batch, margins.cpu(), strict=True)})
                del margins, captured
            values = torch.cat([accumulated[name] for name in names]).float().cpu().numpy()
            if not np.isfinite(values).all() or any(p.grad is not None for p in net.parameters()):
                raise ValueError("nonfinite derivatives or unexpected full parameter gradients")
            reference = common.check_references(cases[group][fold], measured, args.reference_tolerance)
            file = args.output / f"gradient-{group}-{fold}.npy"
            np.save(file, values, allow_pickle=False)
            record = {"purpose": group, "fold": fold, "examples": len(rows),
                      "microbatches": len(packed), "mean_margin": float(np.mean(list(measured.values()))),
                      "margins": [{"id": row[0], "margin": measured[row[0]]} for row in rows], "file": file.name,
                      "sample_gradient_l2": float(np.linalg.norm(values.astype(np.float64))), **reference}
            records.append(record)
            common.save(args.output / "progress.json", {"completed_folds": len(records), "last": record})
        del accumulated, values
    common.assert_inputs(checkpoint, provenance)
    for index, name in enumerate(names):
        sampled_rows, columns = plans[name]
        if not np.array_equal(modules[name].weight.detach()[sampled_rows, columns].float().cpu().numpy(), original_values[index]):
            raise ValueError("collector changed sampled source weights")
    result = {"status": "completed", "mode": args.mode, "coverage": coverage, "folds": records,
              "zero_additive_checks": checks, "seconds": time.monotonic() - start,
              "objective": ("mean correct-minus-foil public-answer mean log probability" if args.mode == "chat-public-answer"
                            else "mean first distinguishing next-token softcapped logit margin"),
              "runtime": {"torch": torch.__version__, "transformers": common.transformers_version(), "device": args.device,
                          "weight_dtype": "BF16", "attention": "eager", "gradient_checkpointing": False,
                           "max_cases_per_microbatch": args.max_cases_per_microbatch,
                          "gradient_accumulation": "FP32 factors, FP64 sum, each batch divided by original fold example count",
                          "gpu_peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30 if args.device != "cpu" else None,
                          "gpu_peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30 if args.device != "cpu" else None},
              "candidate_selection": False, "source_values_unchanged": True,
              "full_model_parameter_gradients_saved": False,
              "limits": ["True derivatives at stratified matrix cells, not native/virtual gain derivatives.",
                         "Chat public-answer and raw next-token objectives are different units; do not mix collections.",
                         "Authored calibration only, not held-out or free-generation performance."]}
    common.save(args.output / "summary.json", result)
    return result


def main(argv=None):
    parser = common.parser(__doc__, chat=True)
    parser.set_defaults(mode="raw-next-token")
    args = parser.parse_args(argv)
    checkpoint, cases, provenance = common.preflight(args, Path(__file__))
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
    print(f"Collected {len(result['folds'])} sampled matrix folds; no candidate selected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
