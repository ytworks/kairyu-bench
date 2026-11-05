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
        full = config["layer_types"][layer] == "full_attention"
        heads = (config["num_attention_heads"] if kind == "q" else
                 config["num_global_key_value_heads"] if full else config["num_key_value_heads"])
        row_bands = 8 if heads == 4 else 4
        column_bands = 512 // (heads * row_bands)
        if nr % heads or heads * row_bands * column_bands != 512:
            raise ValueError("unsupported head geometry for 512-cell strata")
        width = nr // heads
        for head in range(heads):
            for band in range(row_bands):
                r0, r1 = bounds(band, row_bands, width)
                for column in range(column_bands):
                    c0, c1 = bounds(column, column_bands, nc)
                    cell(head * width + r0, head * width + r1, c0, c1)
        strata = {"heads": heads, "head_row_bands": row_bands, "input_column_bands": column_bands}
    elif kind == "o":
        heads = config["num_attention_heads"]
        row_bands, head_column_bands = 4, 512 // (heads * 4)
        if nc % heads or heads * row_bands * head_column_bands != 512:
            raise ValueError("unsupported O-projection geometry for 512-cell strata")
        width = nc // heads
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


def collect(args, checkpoint, cases, provenance):
    from safetensors import safe_open

    torch, tokenizer, model = common.load_model(args)
    net, head = model.model.language_model, model.get_output_embeddings()
    config = net.config.to_dict()
    prepared, frame = common.prepare(tokenizer, cases, args.mode)
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
    hooks, captured, embedding, probes = [], {}, [], {}
    verify_names = {names[0], names[-1]}
    verify_active = False
    def embed_hook(module, inputs, output):
        value = output.detach().requires_grad_(True)
        embedding.append(value)
        return value
    def factory(name):
        rows, columns = plans[name]
        def hook(module, inputs, output):
            if not output.requires_grad:
                raise ValueError("projection output must participate in the calibration gradient")
            selected = inputs[0].detach().index_select(-1, columns).float()
            if verify_active and name in verify_names:
                probe = torch.zeros(4, dtype=torch.float32, device=args.device, requires_grad=True)
                probes[name] = probe
                update = torch.zeros_like(output)
                update.index_add_(-1, rows[:4], (selected[..., :4] * probe).to(output.dtype))
                output = output + update
            def backward(upstream):
                value = factor_derivative(upstream, selected, rows)
                captured[name] = captured.get(name, 0) + value
            output.register_hook(backward)
            return output
        return hook
    checks, records = {}, []
    start = time.monotonic()
    try:
        hooks.append(net.embed_tokens.register_forward_hook(embed_hook))
        for name, module in modules.items():
            hooks.append(module.register_forward_hook(factory(name)))
        for group, folds in prepared.items():
            for fold, rows in enumerate(folds):
                captured.clear()
                embedding.clear()
                probes.clear()
                verify_active = not records
                margins = common.objective(torch, net, head, tokenizer, rows, args.mode, args.device)
                if len(embedding) != 1:
                    raise ValueError("expected exactly one embedding forward per raw fold")
                derivatives = torch.autograd.grad(margins.mean(), [embedding[0], *probes.values()])
                if set(captured) != set(names):
                    raise ValueError("a sampled projection did not participate")
                if verify_active:
                    for name, explicit in zip(probes, derivatives[1:], strict=True):
                        factor = captured[name][:4].float()
                        difference = (factor - explicit).abs()
                        checks[name] = {"cells": 4, "max_abs_difference": float(difference.max())}
                        if not torch.allclose(factor, explicit, rtol=2e-5, atol=1e-7):
                            raise ValueError("sampled-factor derivative differs from explicit zero-additive probe")
                values = torch.cat([captured[name] for name in names]).float().cpu().numpy()
                if not np.isfinite(values).all() or any(p.grad is not None for p in net.parameters()):
                    raise ValueError("nonfinite derivatives or unexpected full parameter gradients")
                measured = {row[0]: float(value) for row, value in zip(rows, margins.detach().cpu(), strict=True)}
                reference = common.check_references(cases[group][fold], measured, args.reference_tolerance)
                file = args.output / f"gradient-{group}-{fold}.npy"
                np.save(file, values, allow_pickle=False)
                record = {"purpose": group, "fold": fold, "examples": len(rows), "mean_margin": float(margins.detach().mean()),
                          "margins": [{"id": row[0], "margin": measured[row[0]]} for row in rows], "file": file.name,
                          "sample_gradient_l2": float(np.linalg.norm(values.astype(np.float64))), **reference}
                records.append(record)
                common.save(args.output / "progress.json", {"completed_folds": len(records), "last": record})
                del margins, derivatives, values
        common.assert_inputs(checkpoint, provenance)
        for index, name in enumerate(names):
            rows, columns = plans[name]
            if not np.array_equal(modules[name].weight.detach()[rows, columns].float().cpu().numpy(), original_values[index]):
                raise ValueError("collector changed sampled source weights")
        result = {"status": "completed", "mode": "raw-next-token", "coverage": coverage, "folds": records,
                  "zero_additive_checks": checks, "seconds": time.monotonic() - start,
                  "objective": "mean first distinguishing next-token softcapped logit margin",
                  "runtime": {"torch": torch.__version__, "transformers": common.transformers_version(), "device": args.device, "weight_dtype": "BF16", "attention": "eager"},
                  "candidate_selection": False, "source_values_unchanged": True,
                  "full_model_parameter_gradients_saved": False}
        common.save(args.output / "summary.json", result)
        return result
    finally:
        for hook in hooks:
            hook.remove()


def main(argv=None):
    parser = common.parser(__doc__, chat=False)
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
