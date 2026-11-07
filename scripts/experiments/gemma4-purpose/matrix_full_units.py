#!/usr/bin/env python3
"""Observe fixed complete MLP unit rows/columns with unchanged sampled attention cells."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

import matrix as sampled
import recipe_common as common
import objective_variants as variants

PATTERN = sampled.PATTERN


def verify_attention(coordinates, entries, support):
    """A and D share exact attention coordinates; no old gradients are consumed."""
    refs = support["reference_attention"]
    for ref in refs.values():
        if common.sha(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("reference attention artifact changed")
    old = json.loads(Path(refs["coverage"]["path"]).read_text())["entries"]
    with np.load(refs["sample_plan"]["path"], allow_pickle=False) as arrays:
        before = arrays["coordinates"]
        expected = {entry["tensor"]: before[entry["start"]:entry["start"]+entry["count"]]
                    for entry in old if PATTERN.fullmatch(entry["tensor"])[2] == "self_attn"}
    actual = {entry["tensor"]: rc for entry,rc in zip(entries,coordinates,strict=True)
              if PATTERN.fullmatch(entry["tensor"])[2] == "self_attn"}
    if set(actual) != set(expected) or any(not np.array_equal(actual[n],expected[n]) for n in actual):
        raise ValueError("full-unit collection attention coordinates differ from A")


def full_unit_samples(tensors, support):
    if support.get("schema_version") != 1 or support.get("kind") != "mlp_full_unit_support_v1":
        raise ValueError("explicit full-unit support required")
    groups = support["units"]
    if (not groups or len({row["layer"] for row in groups}) != len(groups)
            or sum(len(row["indices"]) for row in groups) > 32):
        raise ValueError("one through32 unique declared units required")
    result = {}
    for group in groups:
        layer, indices = group["layer"], group["indices"]
        if (type(layer) is not int or layer < 0 or not isinstance(indices, list)
                or not 1 <= len(indices) <= 2 or any(type(i) is not int or i < 0 for i in indices)
                or indices != sorted(set(indices))):
            raise ValueError("at most two sorted unique unit indices per layer required")
        prefix = f"model.language_model.layers.{layer}.mlp."
        names = [prefix + kind + "_proj.weight" for kind in ("gate", "up", "down")]
        if any(name not in tensors for name in names):
            raise ValueError("declared full MLP triplet missing")
        gate, up, down = [tuple(tensors[name].shape) for name in names]
        if len(gate) != 2 or gate != up or down != (gate[1], gate[0]) or max(indices) >= gate[0]:
            raise ValueError("full-unit triplet shapes/indices differ")
        for name in names:
            kind = PATTERN.fullmatch(name)[3]
            cells = ([(row, unit) for row in range(down[0]) for unit in indices] if kind == "down"
                     else [(unit, col) for unit in indices for col in range(gate[1])])
            result[name] = (np.asarray(cells, dtype=np.int64), dict(shared_hidden_indices=indices,
                coherence="complete gate/up rows and down columns", full_axis="columns" if kind == "down" else "rows"))
    return result


def cartesian_derivative(upstream, inputs, rows, columns):
    """FP32 products, FP64 accumulation; bounded token tiles, no dense parameter gradient."""
    import torch

    g = upstream.detach()
    if len(rows) != g.shape[-1]:
        g = g.index_select(-1, rows)
    g = g.reshape(-1, len(rows))
    x = inputs.reshape(-1, len(columns))
    result = torch.zeros((len(rows), len(columns)), dtype=torch.float64, device=g.device)
    for start in range(0, g.shape[0], 64):
        a, b = g[start:start+64].float(), x[start:start+64].float()
        if len(rows) <= len(columns):
            for i in range(len(rows)):
                result[i] += (a[:, i:i+1] * b).sum(0, dtype=torch.float64)
        else:
            for j in range(len(columns)):
                result[:, j] += (a * b[:, j:j+1]).sum(0, dtype=torch.float64)
    return result.flatten()


def batch_derivatives(torch, net, head, tokenizer, rows, mode, device, plans, modules, fold_examples, verify_names,
                      *, objective=None, loss_sign=1.):
    captured, embedding, probes, calls, hooks = {}, [], {}, {}, []
    def embed_hook(module, inputs, output):
        value = output.detach().requires_grad_(True)
        embedding.append(value)
        return value
    def factory(name):
        sampled_rows, columns = plans[name]
        full = PATTERN.fullmatch(name)[2] == "mlp"
        r = torch.unique(sampled_rows, sorted=True) if full else sampled_rows
        c = torch.unique(columns, sorted=True) if full else columns
        def hook(module, inputs, output):
            calls[name] = calls.get(name, 0) + 1
            if calls[name] != 1 or not output.requires_grad:
                raise ValueError("one differentiable projection invocation required; no recomputation")
            source = inputs[0].detach()
            # Full input axes alias the already existing BF16 tensor, rather
            # than duplicating it once for every selected output coordinate.
            selected = source if full and len(c) == source.shape[-1] else source.index_select(-1, c)
            if name in verify_names:
                count = min(4, len(sampled_rows))
                probe = torch.zeros(count, dtype=torch.float32, device=device, requires_grad=True)
                probes[name] = probe
                probe_inputs = source.index_select(-1, columns[:count]).float()
                update = torch.zeros_like(output)
                update.index_add_(-1, sampled_rows[:count], (probe_inputs * probe).to(output.dtype))
                output = output + update
            def backward(upstream):
                derivative = (cartesian_derivative(upstream, selected, r, c) if full
                              else sampled.factor_derivative(upstream, selected, sampled_rows))
                captured[name] = derivative
            output.register_hook(backward)
            return output
        return hook
    try:
        hooks.append(net.embed_tokens.register_forward_hook(embed_hook))
        for name, module in modules.items():
            hooks.append(module.register_forward_hook(factory(name)))
        values = (objective or common.objective)(torch, net, head, tokenizer, rows, mode, device)
        if len(embedding) != 1:
            raise ValueError("one input embedding per forward required")
        derivatives = torch.autograd.grad(values.sum()*loss_sign/fold_examples, [embedding[0], *probes.values()])
        if set(captured) != set(plans):
            raise ValueError("declared projection missing from backward")
        checks = {}
        for name, explicit in zip(probes, derivatives[1:], strict=True):
            factor = captured[name][:len(explicit)].float()
            checks[name] = dict(cells=len(explicit), max_abs_difference=float((factor-explicit).abs().max()))
            if not torch.allclose(factor, explicit, rtol=2e-5, atol=1e-7):
                raise ValueError("full-unit factor differs from exact zero-additive probe")
        return values.detach(), captured, checks
    finally:
        for hook in hooks:
            hook.remove()


def objective_derivatives(torch, net, head, tokenizer, rows, mode, device, plans, modules,
                          fold_examples, verify_names, *, objective_kind, answer_batching):
    if mode != "chat-public-answer" or answer_batching != "serial":
        raise ValueError("full-unit observations require serial public answers")
    width = max(len(row[1])+max(len(row[2]),len(row[3])) for row in rows)
    selections = (("correct",1.),("foil",-1.)) if objective_kind == "margin" else (("correct",1.),)
    accumulated, checks, measured, forwards = {}, {}, [], 0
    for row in rows:
        total = 0.
        for answer, sign in selections:
            def single(t,n,h,tok,batch,m,d):
                return variants.single_answer_logprob_objective(t,n,h,tok,batch[0],answer,m,d,width)
            values, gradients, verified = batch_derivatives(torch,net,head,tokenizer,[row],mode,device,
                plans,modules,fold_examples,verify_names if forwards == 0 else set(),objective=single,loss_sign=sign)
            total += sign*float(values[0])
            for name,value in gradients.items():
                accumulated[name] = accumulated.get(name,0)+value
            checks.update(verified)
            forwards += 1
        measured.append(total)
    return torch.tensor(measured,dtype=torch.float32,device=device),accumulated,checks,forwards

def collect(args, checkpoint, cases, provenance):
    from safetensors import safe_open

    objective_kind = args.objective
    answer_batching = args.answer_batching
    sampling_mode = "mlp-full-units"
    support = json.loads(args.unit_support.read_text())
    coherent = full_unit_samples(checkpoint.tensors, support)
    torch, tokenizer, model = common.load_model(args, cases=cases, legacy_limit=512)
    net, head = model.model.language_model, model.get_output_embeddings()
    config = net.config.to_dict()
    prepared, frame = common.prepare(tokenizer, cases, args.mode, max_frame_tokens=common.frame_limit(args, 512))
    packed_folds = {group: [sampled.checked_batches(rows, args.mode, args.max_cases_per_microbatch, common.frame_limit(args, 512), common.execution_order(args)) for rows in folds]
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
        if kind in {"gate", "up", "down"} and name not in coherent:
            continue
        module = net.get_submodule(name.removeprefix("model.language_model.").removesuffix(".weight"))
        if type(module) is not torch.nn.Linear or module.weight.dtype != torch.bfloat16 or tuple(module.weight.shape) != tensor.shape:
            raise ValueError("loaded projection does not match checkpoint metadata")
        rc, strata = (coherent[name] if name in coherent else
                      sampled.sample(name, tuple(module.weight.shape), kind, layer, config, args.seed))
        rows = torch.as_tensor(rc[:, 0].copy(), dtype=torch.long, device=args.device)
        columns = torch.as_tensor(rc[:, 1].copy(), dtype=torch.long, device=args.device)
        values = module.weight.detach()[rows, columns].float().cpu()
        with safe_open(str(tensor.shard), framework="pt", device="cpu") as shard:
            original = shard.get_tensor(name)[torch.from_numpy(rc[:, 0].copy()), torch.from_numpy(rc[:, 1].copy())].float()
        if not torch.equal(values, original):
            raise ValueError("sampled GPU/CPU source values differ from official BF16 weights")
        plans[name], modules[name] = (rows, columns), module
        entries.append({"tensor": name, "layer": layer, "family": kind, "shape": list(tensor.shape),
                        "start": sum(e["count"] for e in entries), "count": len(rc), "strata": strata})
        coordinates.append(rc)
        original_values.append(values.numpy())
    if not entries:
        raise ValueError("no supported Gemma projection matrices found")
    names = list(plans)
    verify_attention(coordinates, entries, support)
    np.savez_compressed(args.output / "sample-plan.npz",
                        coordinates=np.concatenate(coordinates), source_values=np.concatenate(original_values))
    coverage = {"sampling_mode": sampling_mode, "text_layers": len(net.layers), "sampled_matrices": len(entries),
                "families": dict(Counter(e["family"] for e in entries)),
                "sampled_coordinates": sum(e["count"] for e in entries), "coordinates_per_matrix": None, "seed": args.seed,
                "unit_support": support, "unit_support_sha256": args.unit_support_sha256,
                "represented_matrix_elements": sum(int(np.prod(e["shape"])) for e in entries),
                "actual_values_crosschecked_against_safetensors": True, "entries": entries,
                "limits": ["Sampled attention plus complete declared MLP unit rows/columns; not whole-model gradients.",
                           "dL/dW is a local synthetic-purpose derivative, not a finite edit outcome or ability localization."]}
    common.save(args.output / "coverage.json", coverage)
    checks, records = {}, []
    start = time.monotonic()
    for group, fold, rows in common.finite_replay.ordered_folds(prepared, common.execution_order(args)):
        accumulated = {name: torch.zeros(len(plans[name][0]), dtype=torch.float64, device=args.device) for name in names}
        measured, forward_count = {}, 0
        packed = packed_folds[group][fold]
        for batch_index, batch in enumerate(packed):
            verify_names = {names[0], names[-1]} if not records and batch_index == 0 else set()
            margins, captured, verified, forwards = objective_derivatives(
                torch, net, head, tokenizer, batch, args.mode, args.device,
                plans, modules, len(rows), verify_names,
                objective_kind=objective_kind, answer_batching=answer_batching)
            forward_count += forwards
            checks.update(verified)
            for name in names:
                accumulated[name] += captured[name]
            measured.update({row[0]: float(value) for row, value in zip(batch, margins.cpu(), strict=True)})
            del margins, captured
        values = torch.cat([accumulated[name] for name in names]).float().cpu().numpy()
        if not np.isfinite(values).all() or any(p.grad is not None for p in net.parameters()):
            raise ValueError("nonfinite derivatives or unexpected full parameter gradients")
        reference = variants.objective_references(cases[group][fold], measured, args.reference_tolerance, objective_kind)
        file = args.output / f"gradient-{group}-{fold}.npy"
        np.save(file, values, allow_pickle=False)
        value_key = "margin" if objective_kind == "margin" else "correct_logprob"
        record = {"purpose": group, "fold": fold, "examples": len(rows),
                  "microbatches": len(packed), "mean_" + value_key: float(np.mean(list(measured.values()))),
                  ("margins" if objective_kind == "margin" else "correct_logprobs"):
                      [{"id": row[0], value_key: measured[row[0]]} for row in prepared[group][fold]], "file": file.name,
                  "answer_batching": answer_batching, "model_forward_count": forward_count,
                  "sample_gradient_l2": float(np.linalg.norm(values.astype(np.float64))), **reference}
        records.append(record)
        common.save(args.output / "progress.json", {"completed_folds": len(records), "last": record})
        del accumulated, values
    canonical_order = {(group, fold): i for i, (group, fold, _) in enumerate(common.finite_replay.ordered_folds(prepared))}
    records.sort(key=lambda row: canonical_order[(row["purpose"], row["fold"])])
    common.assert_inputs(checkpoint, provenance)
    for index, name in enumerate(names):
        sampled_rows, columns = plans[name]
        if not np.array_equal(modules[name].weight.detach()[sampled_rows, columns].float().cpu().numpy(), original_values[index]):
            raise ValueError("collector changed sampled source weights")
    result = {"status": "completed", "mode": args.mode, "objective_kind": objective_kind,
              "answer_batching": answer_batching, "coverage": coverage, "folds": records,
              "zero_additive_checks": checks, "seconds": time.monotonic() - start,
              "objective": ("mean correct-answer token log probability (maximize)" if objective_kind == "correct-logprob" else
                            "mean correct-minus-foil public-answer mean log probability" if args.mode == "chat-public-answer"
                            else "mean first distinguishing next-token softcapped logit margin"),
              "runtime": {"torch": torch.__version__, "transformers": common.transformers_version(), "device": args.device,
                          "weight_dtype": "BF16", **common.observation_runtime(args, provenance), "gradient_checkpointing": False,
                           "max_cases_per_microbatch": args.max_cases_per_microbatch,
                          "gradient_accumulation": "FP32 factors, FP64 sum, each batch divided by original fold example count",
                          "gpu_peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30 if args.device != "cpu" else None,
                          "gpu_peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30 if args.device != "cpu" else None},
              "candidate_selection": False, "source_values_unchanged": True,
              "full_model_parameter_gradients_saved": False,
              "limits": ["True derivatives at declared full-unit and attention coordinates, not unit-gain derivatives.",
                         "Chat public-answer and raw next-token objectives are different units; do not mix collections.",
                         "Authored calibration only, not held-out or free-generation performance."]}
    common.save(args.output / "summary.json", result)
    return result


def main(argv=None):
    parser = common.parser(__doc__, chat=True)
    parser.set_defaults(mode='chat-public-answer', max_cases_per_microbatch=1)
    parser.add_argument('--objective', choices=('correct-logprob','margin'), default='margin')
    parser.add_argument('--answer-batching', choices=('serial',), default='serial')
    parser.add_argument('--unit-support', type=Path, required=True)
    parser.add_argument('--unit-support-sha256', required=True)
    args = parser.parse_args(argv)
    if args.mode != 'chat-public-answer' or args.max_cases_per_microbatch != 1:
        parser.error('full-unit collection requires one serial public-answer case')
    checkpoint,cases,provenance = common.preflight(args,Path(__file__))
    if common.sha(args.unit_support) != args.unit_support_sha256:
        raise ValueError('unit support SHA differs')
    support = json.loads(args.unit_support.read_text())
    full_unit_samples(checkpoint.tensors,support)
    if (support['source_checkpoint'] != provenance['checkpoint']
            or support['attestation_receipt_sha256'] != provenance['attestation_receipt_sha256']):
        raise ValueError('unit support checkpoint differs')
    paths = [args.unit_support.resolve(),Path(sampled.__file__).resolve()]
    for ref in [support['parent_state'],*support['reference_attention'].values()]:
        path = Path(ref['path']).resolve()
        if common.sha(path) != ref['sha256']:
            raise ValueError('support source/current parent pin changed')
        paths.append(path)
    for path in paths:
        out = args.output.resolve()
        if out == path or out in path.parents or path in out.parents:
            raise ValueError('output overlaps support input')
        provenance['input_sha256'][str(path)] = common.sha(path)
        provenance['input_stat'][str(path)] = common.input_state(path)
    variants.pin_objective(provenance,args.mode,args.objective)
    variants.attach_correct_references(args.cases,cases,provenance,args.objective)
    provenance.update(answer_batching='serial',sampling_mode='mlp-full-units',
        unit_support=dict(path=str(args.unit_support.resolve()),sha256=args.unit_support_sha256),
        objective_forward_layout='one case/answer per forward; immediate VJP; FP32 products/FP64 token sums')
    common.assert_inputs(checkpoint,provenance)
    if not args.run:
        print(json.dumps(provenance,indent=2))
        return 0
    args.output.mkdir(parents=True)
    common.save(args.output/'provenance.json',provenance)
    try:
        result = collect(args,checkpoint,cases,provenance)
    except BaseException as error:
        common.save(args.output/'failure.json',dict(status='incomplete',error_type=type(error).__name__))
        raise
    print(json.dumps(dict(status=result['status'],folds=len(result['folds']),candidate_selection=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
