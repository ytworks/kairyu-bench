#!/usr/bin/env python3
"""Collect full-projection continuous LoRA direction sensitivities at zero gates."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemma4-purpose"))
import recipe_common as common  # noqa: E402
import objective_variants as objectives  # noqa: E402
from objective_variants import correct_logprob_objective, objective_references  # noqa: E402

from kairyu_bench.wsp.weights.checkpoint import Checkpoint  # noqa: E402

PATTERN = re.compile(r"^model\.language_model\.layers\.(\d+)\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj\.weight$")
SCALE = 2.0


def mapping(base, donor, config, costs):
    """Match the exact 410 text matrices and pinned, previously audited unit costs."""
    for key, expected in {"r": 16, "lora_alpha": 32, "use_rslora": False, "use_dora": False,
                          "fan_in_fan_out": False, "bias": "none"}.items():
        if config.get(key) != expected:
            raise ValueError("ordinary rank-16, alpha-32 LoRA configuration required")
    if any(config.get(k) for k in ("rank_pattern", "alpha_pattern", "modules_to_save", "lora_bias")):
        raise ValueError("unsupported donor modification")
    if not isinstance(costs, list) or len(costs) != 410:
        raise ValueError("requires the existing 410-matrix unit-cost audit")
    by_name = {entry["base_tensor"]: entry for entry in costs}
    if len(by_name) != len(costs):
        raise ValueError("duplicate audited tensor")
    expected = {name for name in base.tensors if PATTERN.fullmatch(name)}
    if len(expected) != 410 or {int(PATTERN.fullmatch(name)[1]) for name in expected} != set(range(60)):
        raise ValueError("requires all 410 text projections across 60 layers")
    if set(by_name) != expected:
        raise ValueError("unit-cost audit differs from exact source projection inventory")
    entries, used = [], set()
    for name in sorted(expected):
        match = PATTERN.fullmatch(name)
        tensor = base.tensors[name]
        prefix = "base_model.model." + name.removesuffix(".weight")
        a_name, b_name = prefix + ".lora_A.weight", prefix + ".lora_B.weight"
        if a_name not in donor.tensors or b_name not in donor.tensors:
            raise ValueError("donor factor missing")
        a, b = donor.tensors[a_name], donor.tensors[b_name]
        row = by_name[name]
        if (tensor.dtype != "BF16" or len(tensor.shape) != 2 or a.dtype != "BF16" or b.dtype != "BF16"
                or a.shape != (16, tensor.shape[1]) or b.shape != (tensor.shape[0], 16)
                or row["adapter_A"] != a_name or row["adapter_B"] != b_name
                or row["shape"] != list(tensor.shape) or row["rank"] != 16 or row["lora_scale"] != SCALE
                or row["layer"] != int(match[1]) or row["mechanism"] != match[3] + "_proj"):
            raise ValueError("source, factor or unit-cost tuple mismatch")
        cost = row["unrounded_delta_l2"]
        if isinstance(cost, bool) or not isinstance(cost, (float, int)) or not math.isfinite(cost) or cost < 0:
            raise ValueError("invalid unit weight-space cost")
        used.update((a_name, b_name))
        entries.append({"tensor": name, "layer": int(match[1]), "family": match[3],
                        "shape": list(tensor.shape), "adapter_A": a_name, "adapter_B": b_name,
                        "unit_gate_weight_l2": float(cost), "coordinate": len(entries)})
    actual_text = {name for name in donor.tensors if name.startswith("base_model.model.model.language_model.")}
    if actual_text != used:
        raise ValueError("unmapped or additional text donor factors")
    return entries


def adapter_preflight(args, checkpoint, provenance):
    paths = [args.donor_spec.resolve(), args.unit_costs.resolve()]
    spec_state = common.input_state(paths[0])
    spec_content = paths[0].read_bytes()
    if common.input_state(paths[0]) != spec_state:
        raise ValueError("donor spec changed while reading")
    spec = json.loads(spec_content)
    if spec.get("schema_version") != 2 or spec.get("kind") != "dense_lora_recipe_v1" or spec.get("scope") != "native_text410":
        raise ValueError("requires a pinned whole-text donor recipe")
    paths += [Path(spec["donor_path"]).resolve(), Path(spec["donor_config_path"]).resolve()]
    output = args.output.resolve()
    for path in paths:
        if output == path or output in path.parents or path in output.parents or path.is_relative_to(checkpoint.root):
            raise ValueError("donor inputs/output must be disjoint from the source checkpoint")
        provenance["input_stat"][str(path)] = spec_state if path == paths[0] else common.input_state(path)
        provenance["input_sha256"][str(path)] = hashlib.sha256(spec_content).hexdigest() if path == paths[0] else common.sha(path)
    for path, expected in ((paths[1], args.unit_costs_sha256), (paths[2], spec["donor_sha256"]),
                           (paths[3], spec["donor_config_sha256"])):
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected)) or provenance["input_sha256"][str(path)] != expected:
            raise ValueError("pinned donor/config/unit-cost SHA mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", str(spec.get("donor_revision"))) or not re.fullmatch(r"[^/\s]+/[^/\s]+", str(spec.get("donor_repo"))):
        raise ValueError("pinned donor repository/revision required")
    donor = Checkpoint(paths[2])
    entries = mapping(checkpoint, donor, json.loads(paths[3].read_text()), json.loads(paths[1].read_text()))
    provenance["donor"] = {"repo": spec["donor_repo"], "revision": spec["donor_revision"],
                           "payload_sha256": spec["donor_sha256"], "config_sha256": spec["donor_config_sha256"],
                           "identity": donor.identity(hash_files=False), "unit_costs_sha256": args.unit_costs_sha256,
                           "unit_cost_source": "existing pinned FP64 low-rank Gram audit; not recomputed here",
                           "r": 16, "alpha": 32, "scale": SCALE, "gate_value": 0,
                           "recipe_coefficient_ignored": spec.get("coefficient")}
    common.assert_inputs(checkpoint, provenance)
    donor.assert_stable()
    return donor, entries


def zero_gate_function(torch):
    """Exact zero-gate forward, first-order VJP of y + gate * 2 B(Ax).

    The caller owns zero FP32 gates and checks they remain zero. Recomputing
    the detached direction in backward avoids retaining every expanded FP32
    projection output. This is not a derivative of discrete BF16 weight edits.
    Higher derivatives and nonzero gates are deliberately unsupported.
    """
    class ZeroGate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, output, inputs, a, b, gate):
            ctx.save_for_backward(inputs.detach(), a, b)
            return output

        @staticmethod
        def backward(ctx, upstream):
            if torch.is_grad_enabled():
                raise ValueError("higher-order donor gate gradients are unsupported")
            inputs, a, b = ctx.saved_tensors
            direction = torch.nn.functional.linear(torch.nn.functional.linear(inputs.float(), a), b) * SCALE
            derivative = (upstream.float() * direction).sum(dtype=torch.float64).to(torch.float32)
            return upstream, None, None, None, derivative
    return ZeroGate



def collect(args, checkpoint, cases, provenance, donor, entries):
    objective = getattr(args, "objective", "margin")
    if objective not in {"margin", "correct-logprob"} or objective == "correct-logprob" and args.mode != "chat-public-answer":
        raise ValueError("unsupported objective/frame combination")
    answer_batching = getattr(args, "answer_batching", "paired")
    if (answer_batching not in {"paired", "serial"} or
            answer_batching == "serial" and args.mode != "chat-public-answer"):
        raise ValueError("serial answers require chat-public-answer")
    if getattr(args, "max_cases_per_microbatch", 1) != 1:
        raise ValueError("donor collection requires one case per microbatch")
    objective_fn = common.objective if objective == "margin" else correct_logprob_objective
    torch, tokenizer, model = common.load_model(args)
    net, head = model.model.language_model, model.get_output_embeddings()
    if any(getattr(module, "gradient_checkpointing", False) for module in net.modules()):
        raise ValueError("donor collector does not support gradient checkpoint recomputation")
    prepared, frame = common.prepare(tokenizer, cases, args.mode)
    packed = {name: [common.objective_batches(rows, args.mode, 1) for rows in folds]
              for name, folds in prepared.items()}
    common.save(args.output / "frame-evidence.json", frame)
    gate = torch.zeros(len(entries), dtype=torch.float32, device=args.device, requires_grad=True)
    gate_version = gate._version
    operation = zero_gate_function(torch)
    state = {"enabled": False, "calls": {}}
    modules, factors, source_values, versions, hooks = {}, {}, {}, {}, []
    start = time.monotonic()
    try:
        for entry in entries:
            name, index = entry["tensor"], entry["coordinate"]
            module = net.get_submodule(name.removeprefix("model.language_model.").removesuffix(".weight"))
            if type(module) is not torch.nn.Linear or module.weight.dtype != torch.bfloat16 or list(module.weight.shape) != entry["shape"]:
                raise ValueError("loaded source must be the exact plain BF16 Linear projection")
            numel = module.weight.numel()
            indices = sorted({0, numel // 3, 2 * numel // 3, numel - 1})
            selected = module.weight.detach().flatten()[indices].float().cpu().numpy()
            if not np.array_equal(selected.view(np.uint32), checkpoint.read_indices(name, indices).astype(np.float32).view(np.uint32)):
                raise ValueError("loaded source probe bytes differ from checkpoint")
            a = torch.from_numpy(donor.read(entry["adapter_A"]).reshape(16, entry["shape"][1])).to(device=args.device, dtype=torch.float32).detach()
            b = torch.from_numpy(donor.read(entry["adapter_B"]).reshape(entry["shape"][0], 16)).to(device=args.device, dtype=torch.float32).detach()
            if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
                raise ValueError("nonfinite donor factors")
            modules[name], factors[name] = module, (a, b)
            source_values[name], versions[name] = (indices, selected.copy()), module.weight._version
            def hook(layer, inputs, output, name=name, index=index, a=a, b=b):
                if not state["enabled"]:
                    return output
                state["calls"][name] = state["calls"].get(name, 0) + 1
                if state["calls"][name] != 1:
                    raise ValueError("projection executed more than once in a microbatch")
                return operation.apply(output, inputs[0], a, b, gate[index])
            hooks.append(module.register_forward_hook(hook))
        coverage = {"text_layers": len(net.layers), "matrices": len(entries), "direction_coordinates": len(entries),
                    "matrix_elements": sum(math.prod(e["shape"]) for e in entries),
                    "families": dict(Counter(e["family"] for e in entries)), "entries": entries,
                    "basis_kind": "continuous_full_projection_LoRA_scalar_gates",
                    "unit_costs": "Frobenius norm of 2BA, from pinned audit; disjoint matrices give diagonal Gram",
                    "source_probe_values_match": True}
        common.save(args.output / "coverage.json", coverage)
        costs = np.array([e["unit_gate_weight_l2"] for e in entries], dtype=np.float64)
        np.save(args.output / "unit-gate-weight-l2.npy", costs, allow_pickle=False)
        records, overall_difference = [], 0.0
        all_exact = True
        for purpose, folds in prepared.items():
            for fold, rows in enumerate(folds):
                accumulated = np.zeros(len(entries), dtype=np.float64)
                measured, base_measured = {}, {}
                serial_accumulated = torch.zeros_like(gate, dtype=torch.float32) if answer_batching == "serial" else None
                forward_count = 0
                def observe(evaluate, sign=1.):
                    nonlocal overall_difference, all_exact, forward_count
                    if gate._version != gate_version or bool(torch.count_nonzero(gate.detach())):
                        raise ValueError("only unchanged zero gates are supported")
                    state["enabled"] = False
                    with torch.inference_mode():
                        baseline = evaluate().detach().clone()
                    state["enabled"], state["calls"] = True, {}
                    values = evaluate()
                    forward_count += 1
                    if set(state["calls"]) != set(modules):
                        raise ValueError("a donor projection did not execute")
                    if not bool(torch.isfinite(baseline).all()) or not bool(torch.isfinite(values).all()):
                        raise ValueError("nonfinite zero-gate objective")
                    difference = float((values.detach() - baseline).abs().max())
                    overall_difference = max(overall_difference, difference)
                    all_exact = all_exact and torch.equal(values.detach(), baseline)
                    if difference > args.reference_tolerance:
                        raise ValueError("zero-gate objective differs from unmodified model")
                    derivative, = torch.autograd.grad(values.sum() * sign / len(rows), gate)
                    if not bool(torch.isfinite(derivative).all()):
                        raise ValueError("nonfinite donor gate derivative")
                    if serial_accumulated is None:
                        accumulated[:] += derivative.detach().cpu().numpy().astype(np.float64)
                    else:
                        serial_accumulated.add_(derivative.detach().float())
                    observed = values.detach().cpu().tolist()
                    original = baseline.cpu().tolist()
                    del values, baseline, derivative  # Release each answer graph before the next forward.
                    return observed, original
                for batch in packed[purpose][fold]:
                    if len(batch) != 1:
                        raise ValueError("donor collection requires exactly one prepared case")
                    if answer_batching == "paired":
                        observed, original = observe(lambda: objective_fn(
                            torch, net, head, tokenizer, batch, args.mode, args.device))
                        measured.update({row[0]: float(value) for row, value in zip(batch, observed, strict=True)})
                        base_measured.update({row[0]: float(value) for row, value in zip(batch, original, strict=True)})
                    else:
                        row = batch[0]
                        width = len(row[1]) + max(len(row[2]), len(row[3]))
                        selections = (("correct", 1.), ("foil", -1.)) if objective == "margin" else (("correct", 1.),)
                        measured[row[0]] = base_measured[row[0]] = 0.
                        for answer_kind, sign in selections:
                            observed, original = observe(lambda: objectives.single_answer_logprob_objective(
                                torch, net, head, tokenizer, row, answer_kind, args.mode, args.device, width), sign)
                            measured[row[0]] += sign * float(observed[0])
                            base_measured[row[0]] += sign * float(original[0])
                if serial_accumulated is not None:
                    accumulated = serial_accumulated.detach().cpu().numpy().astype(np.float64)
                if not np.isfinite(accumulated).all():
                    raise ValueError("nonfinite accumulated donor derivative")
                reference = objective_references(cases[purpose][fold], measured, args.reference_tolerance, objective)
                if any(p.grad is not None or p.requires_grad for p in model.parameters()):
                    raise ValueError("unexpected model parameter gradients")
                if any(a.grad is not None or b.grad is not None or a.requires_grad or b.requires_grad for a, b in factors.values()):
                    raise ValueError("unexpected donor factor gradients")
                prefix = "gradient" if objective == "margin" else "gradient-correct-logprob"
                file = args.output / f"{prefix}-{purpose}-{fold}.npy"
                np.save(file, accumulated, allow_pickle=False)
                record = {"purpose": purpose, "fold": fold, "examples": len(rows), "microbatches": len(packed[purpose][fold]),
                          "answer_batching": answer_batching, "objective_forward_count": forward_count,
                          "zero_gate_baseline_forward_count": forward_count,
                          "file": file.name, "directional_derivative_l2": float(np.linalg.norm(accumulated)), **reference}
                key = "margin" if objective == "margin" else "correct_logprob"
                record["mean_" + key] = float(np.mean(list(measured.values())))
                record[key + "s"] = [{"id": row[0], key: measured[row[0]], "unmodified_" + key: base_measured[row[0]]} for row in rows]
                records.append(record)
                common.save(args.output / "progress.json", {"completed_folds": len(records), "last": record})
        common.assert_inputs(checkpoint, provenance)
        donor.assert_stable()
        for name, module in modules.items():
            indices, original = source_values[name]
            current = module.weight.detach().flatten()[indices].float().cpu().numpy()
            if module.weight._version != versions[name] or not np.array_equal(current.view(np.uint32), original.view(np.uint32)):
                raise ValueError("source parameter changed")
        if gate._version != gate_version or bool(torch.count_nonzero(gate.detach())):
            raise ValueError("zero gates changed")
        summary = {"status": "completed", "mode": args.mode, "objective": objective, "answer_batching": answer_batching, "coverage": coverage, "folds": records,
                   "zero_alpha_checks": {"all_exact": all_exact, "max_absolute_objective_difference": overall_difference,
                                         **({"max_absolute_margin_difference": overall_difference} if objective == "margin" else {}),
                                         "tolerance": args.reference_tolerance, "every_microbatch_compared": True},
                   "seconds": time.monotonic() - start, "source_values_unchanged": True,
                   "runtime": {"torch": torch.__version__, "transformers": common.transformers_version(),
                               "device": args.device, "source_dtype": "BF16", "factor_compute_dtype": "FP32",
                               "gates_dtype": "FP32", "attention": "eager", "max_cases_per_microbatch": 1,
                               "gradient_checkpointing": False, "objective": objective,
                               "answer_batching": answer_batching,
                               "gradient_accumulation_dtype": "FP32" if answer_batching == "serial" else "FP64",
                               "objective_forward_layout": ("one case/answer at original pair padding width; immediate selected-answer VJP"
                                                            if answer_batching == "serial" else "unchanged good+foil batch/padding; one selected-objective VJP"),
                               "arithmetic": ("zero-gate identity forward; FP32 2B(Ax) recomputed in backward, FP32 upstream product, FP64 sum then FP32 gate derivative; "
                                              + ("FP32 answer-VJP fold accumulation, saved F64" if answer_batching == "serial" else "FP64 microbatch accumulation")),
                               "gpu_peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30 if args.device != "cpu" else None,
                               "gpu_peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30 if args.device != "cpu" else None},
                   "candidate_selection": False, "new_checkpoints": 0, "wsp_calls": 0,
                   "limits": ["First-order sensitivity of an explicit continuous zero-gate computation, not the derivative of discrete BF16 weight editing.",
                              "Unmerged additive output and merged BF16 weights have different rounding; actual finite checks are required.",
                              "Authored calibration only, not a capability or benchmark score.",
                              "Serial and paired batching/gradient accumulation are distinct numerical conditions; compare original objective values before combining collections.",
                              "No claim about donor training-base identity or training/evaluation overlap.",
                              "Source disk stat continuity, model tensor version counters and explicit probe bytes checked; no fresh full source payload hash."]}
        common.save(args.output / "summary.json", summary)
        return summary
    finally:
        for hook in hooks:
            hook.remove()
        gate.grad = None


def main(argv=None):
    parser = common.parser(__doc__, chat=True)
    parser.set_defaults(max_cases_per_microbatch=1)
    parser.add_argument("--objective", choices=("margin", "correct-logprob"), default="margin",
                        help="Mean correct-answer logP to maximize, or legacy correct-minus-foil margin; separate collections.")
    parser.add_argument("--answer-batching", choices=("paired", "serial"), default="paired",
                        help="serial: fixed paired padding, one answer/VJP at a time and FP32 fold accumulation; paired default unchanged")
    parser.add_argument("--donor-spec", type=Path, required=True)
    parser.add_argument("--unit-costs", type=Path, required=True)
    parser.add_argument("--unit-costs-sha256", required=True)
    args = parser.parse_args(argv)
    if args.max_cases_per_microbatch != 1:
        parser.error("this bounded collector requires --max-cases-per-microbatch 1")
    if args.objective == "correct-logprob" and args.mode != "chat-public-answer":
        parser.error("correct-logprob requires --mode chat-public-answer")
    if args.answer_batching == "serial" and args.mode != "chat-public-answer":
        parser.error("serial answers require --mode chat-public-answer")
    checkpoint, cases, provenance = common.preflight(args, Path(__file__))
    objectives.pin_objective(provenance, args.mode, args.objective)
    provenance["objective"] = args.objective
    provenance["answer_batching"] = args.answer_batching
    provenance["gradient_accumulation_dtype"] = "FP32" if args.answer_batching == "serial" else "FP64"
    if args.answer_batching == "serial":
        provenance["objective_forward_layout"] = "one case/answer per forward and immediate VJP; original paired-pack padding; good-minus-foil or good-only"

    objectives.attach_correct_references(args.cases, cases, provenance, args.objective)
    donor, entries = adapter_preflight(args, checkpoint, provenance)
    if not args.run:
        print(json.dumps(provenance, indent=2))
        return 0
    args.output.mkdir(parents=True)
    common.save(args.output / "provenance.json", provenance)
    try:
        result = collect(args, checkpoint, cases, provenance, donor, entries)
    except BaseException as error:
        common.save(args.output / "failure.json", {"status": "incomplete", "error_type": type(error).__name__})
        raise
    print(f"Collected {len(result['folds'])} full-projection donor folds; no candidate selected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
