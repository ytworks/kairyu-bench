from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/experiments/gemma4_lora/donor_direction.py"
spec = importlib.util.spec_from_file_location("donor_direction", SCRIPT)
assert spec is not None and spec.loader is not None
direction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(direction)


class DonorDirectionMappingTest(unittest.TestCase):
    def test_adapter_spec_change_between_read_and_fingerprint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "spec.json"
            path.write_text("{}")
            original_read = Path.read_bytes
            def changed_read(file):
                value = original_read(file)
                if file == path:
                    file.write_text('{"changed":true}')
                return value
            args = SimpleNamespace(donor_spec=path, unit_costs=root / "costs.json")
            with patch.object(Path, "read_bytes", changed_read), self.assertRaisesRegex(ValueError, "spec changed"):
                direction.adapter_preflight(args, None, {})

    def test_full_text_mapping_preserves_costs_and_rejects_wrong_factor_or_inventory(self):
        base, donor, costs = {}, {}, []
        for layer in range(60):
            families = ("q", "o", "gate", "up", "down") if layer < 5 else ("q", "k", "v", "o", "gate", "up", "down")
            for family in families:
                group = "mlp" if family in ("gate", "up", "down") else "self_attn"
                name = f"model.language_model.layers.{layer}.{group}.{family}_proj.weight"
                prefix = "base_model.model." + name.removesuffix(".weight")
                a, b = prefix + ".lora_A.weight", prefix + ".lora_B.weight"
                base[name] = SimpleNamespace(dtype="BF16", shape=(2, 3))
                donor[a] = SimpleNamespace(dtype="BF16", shape=(16, 3))
                donor[b] = SimpleNamespace(dtype="BF16", shape=(2, 16))
                costs.append({"base_tensor": name, "adapter_A": a, "adapter_B": b, "shape": [2, 3],
                              "rank": 16, "lora_scale": 2., "layer": layer, "mechanism": family + "_proj",
                              "unrounded_delta_l2": .5})
        config = {"r": 16, "lora_alpha": 32, "use_rslora": False, "use_dora": False,
                  "fan_in_fan_out": False, "bias": "none"}
        def run(rows=costs):
            return direction.mapping(SimpleNamespace(tensors=base), SimpleNamespace(tensors=donor), config, rows)
        entries = run()
        self.assertEqual(len(entries), 410)
        self.assertEqual([e["coordinate"] for e in entries], list(range(410)))
        self.assertTrue(all(e["unit_gate_weight_l2"] == .5 for e in entries))
        bad = copy.deepcopy(costs)
        bad[0]["adapter_B"] = bad[1]["adapter_B"]
        with self.assertRaisesRegex(ValueError, "tuple"):
            run(bad)
        donor[costs[0]["adapter_A"]].shape = (15, 3)
        with self.assertRaisesRegex(ValueError, "tuple"):
            run()
        donor[costs[0]["adapter_A"]].shape = (16, 3)
        donor["base_model.model.model.language_model.extra"] = SimpleNamespace(dtype="BF16", shape=(1,))
        with self.assertRaisesRegex(ValueError, "unmapped"):
            run()


    def test_correct_reference_metadata_uses_pinned_case_bytes_and_separate_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cases.json"
            path.write_text(json.dumps({"science": [[{"id": "x", "prompt": "p", "correct": "a", "foil": "b",
                                                     "expected_margin": 999., "expected_correct_logprob": -2.}]]}))
            cases = direction.common.read_cases(path)
            provenance = {"input_stat": {str(path): direction.common.input_state(path)},
                          "input_sha256": {str(path): direction.common.sha(path)}}
            direction.objectives.attach_correct_references(path, cases, provenance, "correct-logprob")
            self.assertEqual(cases["science"][0][0]["expected_correct_logprob"], -2.)
            result = direction.objective_references(cases["science"][0], {"x": -2.}, 0., "correct-logprob")
            self.assertEqual(result["reference_cases"], 1)
            with self.assertRaisesRegex(ValueError, "reference correct"):
                direction.objective_references(cases["science"][0], {"x": -3.}, 0., "correct-logprob")
            path.write_text(path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "changed after preflight"):
                direction.objectives.attach_correct_references(path, cases, provenance, "correct-logprob")



@unittest.skipUnless(importlib.util.find_spec("torch"), "optional Torch CPU fixture")
class DonorDirectionTorchTest(unittest.TestCase):
    def test_rank16_zero_vjp_matches_literal_autograd_and_central_finite_difference(self):
        import torch
        torch.manual_seed(41)
        operation = direction.zero_gate_function(torch)
        weights = [torch.randn(4, 3) * .2, torch.randn(2, 4) * .2]
        factors = [(torch.randn(16, 3) * .1, torch.randn(4, 16) * .1),
                   (torch.randn(16, 4) * .1, torch.randn(2, 16) * .1)]
        inputs = torch.randn(2, 3)
        def forward(gates, custom):
            value = inputs
            for index, (weight, (a, b)) in enumerate(zip(weights, factors, strict=True)):
                output = torch.nn.functional.linear(value, weight)
                if custom:
                    output = operation.apply(output, value, a, b, gates[index])
                else:
                    output = output + gates[index] * 2 * torch.nn.functional.linear(torch.nn.functional.linear(value, a), b)
                value = torch.tanh(output)
            return value.square().sum()
        gates = torch.zeros(2, requires_grad=True)
        custom_value = forward(gates, True)
        custom, = torch.autograd.grad(custom_value, gates)
        literal, = torch.autograd.grad(forward(gates, False), gates)
        torch.testing.assert_close(custom, literal, rtol=2e-6, atol=1e-7)
        torch.testing.assert_close(custom_value, forward(gates.detach(), False), rtol=0, atol=0)
        finite = []
        for i in range(2):
            offset = torch.zeros(2)
            offset[i] = .001
            finite.append((forward(offset, False) - forward(-offset, False)) / .002)
        torch.testing.assert_close(custom, torch.stack(finite), rtol=.001, atol=1e-5)
        self.assertTrue(all(x.grad is None and not x.requires_grad for pair in factors for x in pair))
        # Actual BF16 zero-gate output, including signed zero, is not modified.
        value = torch.tensor([-0., 0., .48046875, -.038818359375], dtype=torch.bfloat16)
        out = operation.apply(value, inputs[0], factors[0][0], factors[0][1], gates[0])
        self.assertTrue(torch.equal(out.view(torch.int16), value.view(torch.int16)))

    def test_collect_checks_every_zero_margin_and_cleans_hooks_on_success_or_failure(self):
        import torch
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Linear(3, 4, bias=False), torch.nn.Linear(4, 2, bias=False)])
            def forward(self, value):
                return torch.tanh(self.layers[1](torch.tanh(self.layers[0](value))))
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.language_model = Net()
                self.bfloat16().requires_grad_(False)
            def get_output_embeddings(self):
                return torch.nn.Identity()
        torch.manual_seed(8)
        for objective_name, batching, fail in [(o, b, f) for o in ("margin", "correct-logprob") for b in ("paired", "serial") for f in (False, True)]:
            with self.subTest(objective=objective_name, batching=batching, fail=fail), tempfile.TemporaryDirectory() as temporary:
                model = Model()
                net = model.model.language_model
                originals = [p.detach().clone() for p in model.parameters()]
                entries, factors, source = [], {}, {}
                for index, module in enumerate(net.layers):
                    name = f"model.language_model.layers.{index}.weight"
                    source[name] = module.weight.detach().flatten().float().numpy().copy()
                    factors[f"a{index}"] = np.full((16, module.in_features), .03125, np.float32)
                    factors[f"b{index}"] = np.full((module.out_features, 16), -.0625, np.float32)
                    entries.append({"tensor": name, "coordinate": index, "shape": list(module.weight.shape),
                                    "adapter_A": f"a{index}", "adapter_B": f"b{index}",
                                    "unit_gate_weight_l2": 1., "layer": index, "family": "q"})
                checkpoint = SimpleNamespace(read_indices=lambda name, ids: source[name][ids])
                donor = SimpleNamespace(read=lambda name: factors[name].flatten().copy(), assert_stable=lambda: None)
                args = SimpleNamespace(output=Path(temporary), device="cpu", mode="chat-public-answer", objective=objective_name, answer_batching=batching, max_cases_per_microbatch=1, reference_tolerance=0.)
                rows = [("x", [1], [2], [3])]
                cases = {"science": [[{"id": "x", **({"expected_margin": 999.} if objective_name == "correct-logprob" else {})}]]}
                calls = []
                def objective(*unused):
                    value = net(torch.tensor([[.5, -.25, .125]], dtype=torch.bfloat16)).float().sum().reshape(1)
                    calls.append(1)
                    if fail and len(calls) == 2:
                        raise RuntimeError("fixture interruption")
                    return value
                with patch.object(direction.common, "load_model", return_value=(torch, None, model)),                         patch.object(direction.common, "prepare", return_value=({"science": [rows]}, {})),                         patch.object(direction.common, "objective", side_effect=objective), patch.object(direction, "correct_logprob_objective", side_effect=objective),                         patch.object(direction.objectives, "single_answer_logprob_objective", side_effect=lambda *a: objective() * (1. if a[5] == "correct" else .5)), patch.object(direction.common, "assert_inputs"):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
                            direction.collect(args, checkpoint, cases, {}, donor, entries)
                    else:
                        result = direction.collect(args, checkpoint, cases, {}, donor, entries)
                        self.assertTrue(result["zero_alpha_checks"]["all_exact"])
                        self.assertEqual(result["zero_alpha_checks"]["max_absolute_objective_difference"], 0)
                        self.assertEqual(result["objective"], objective_name)
                        forwards = 2 if batching == "serial" and objective_name == "margin" else 1
                        self.assertEqual(result["folds"][0]["objective_forward_count"], forwards)
                        self.assertEqual(result["folds"][0]["zero_gate_baseline_forward_count"], forwards)
                        self.assertEqual(result["runtime"]["gradient_accumulation_dtype"], "FP32" if batching == "serial" else "FP64")
                        self.assertEqual(result["answer_batching"], batching)
                        self.assertIn("mean_margin" if objective_name == "margin" else "mean_correct_logprob", result["folds"][0])
                        derivative = np.load(Path(temporary) / result["folds"][0]["file"], allow_pickle=False)
                        self.assertEqual(derivative.shape, (2,))
                        self.assertTrue(np.isfinite(derivative).all())
                        json.dumps(result, allow_nan=False)
                self.assertTrue(all(not module._forward_hooks for module in net.layers))
                self.assertTrue(all(torch.equal(p, old) and p.grad is None and not p.requires_grad
                                    for p, old in zip(model.parameters(), originals, strict=True)))

    def test_correct_logprob_matches_same_batch_oracle_and_rank16_vjp(self):
        import torch
        torch.manual_seed(219)
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(13, 3)
                self.projection = torch.nn.Linear(3, 4, bias=False)
                self.config = SimpleNamespace(final_logit_softcapping=2.0)
                self.seen_shapes = []
            def forward(self, input_ids, attention_mask, **unused):
                self.seen_shapes.append(tuple(input_ids.shape))
                return SimpleNamespace(last_hidden_state=self.projection(self.embedding(input_ids)))
        net = Net().requires_grad_(False)
        head = torch.nn.Linear(4, 13, bias=False).requires_grad_(False)
        tokenizer = SimpleNamespace(pad_token_id=0)
        batch = [("a", [1, 2], [3, 4], [5, 6, 7])]
        a, b = torch.randn(16, 3) * .1, torch.randn(4, 16) * .1
        op = direction.zero_gate_function(torch)
        def value(gate=None, custom=False):
            hooks = []
            if gate is not None:
                def hook(module, inputs, output):
                    if custom:
                        return op.apply(output, inputs[0], a, b, gate)
                    return output + gate * 2 * torch.nn.functional.linear(torch.nn.functional.linear(inputs[0], a), b)
                hooks.append(net.projection.register_forward_hook(hook))
            try:
                return direction.correct_logprob_objective(torch, net, head, tokenizer, batch, "chat-public-answer", "cpu").sum()
            finally:
                for handle in hooks:
                    handle.remove()
        base = value()
        gate = torch.zeros((), requires_grad=True)
        actual = value(gate, True)
        custom, = torch.autograd.grad(actual, gate)
        literal, = torch.autograd.grad(value(gate, False), gate)
        torch.testing.assert_close(actual, base, rtol=0, atol=0)
        torch.testing.assert_close(custom, literal, rtol=2e-6, atol=1e-8)
        finite = (value(torch.tensor(.005), False) - value(torch.tensor(-.005), False)) / .01
        torch.testing.assert_close(custom, finite, rtol=.003, atol=2e-5)
        ids = torch.tensor([[1, 2, 3, 4, 0], [1, 2, 5, 6, 7]])
        hidden = net(input_ids=ids, attention_mask=ids.ne(0)).last_hidden_state
        logits = 2 * torch.tanh(head(hidden) / 2)
        logp = torch.log_softmax(logits, -1)
        good = torch.stack([logp[0, 1, 3], logp[0, 2, 4]]).mean()
        bad = torch.stack([logp[1, 1, 5], logp[1, 2, 6], logp[1, 3, 7]]).mean()
        torch.testing.assert_close(base, good, rtol=0, atol=0)
        legacy = direction.common.objective(torch, net, head, tokenizer, batch, "chat-public-answer", "cpu").sum()
        torch.testing.assert_close(legacy, good - bad, rtol=0, atol=0)
        self.assertEqual(set(net.seen_shapes), {(2, 5)})
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in [*net.parameters(), *head.parameters(), a, b]))
        # Serial uses the same padded width and full answer tokens, but frees
        # each graph after one VJP. Compare accumulated signs to a literal loss.
        for objective_name in ("correct-logprob", "margin"):
            net.seen_shapes.clear()
            accumulated = torch.zeros((), dtype=torch.float32)
            measured = 0.
            selections = (("correct", 1.), ("foil", -1.)) if objective_name == "margin" else (("correct", 1.),)
            for answer_kind, sign in selections:
                def single(gate_value=None, custom=False):
                    handles = []
                    if gate_value is not None:
                        def hook(module, inputs, output):
                            if custom:
                                return op.apply(output, inputs[0], a, b, gate_value)
                            return output + gate_value * 2 * torch.nn.functional.linear(torch.nn.functional.linear(inputs[0], a), b)
                        handles.append(net.projection.register_forward_hook(hook))
                    try:
                        return direction.objectives.single_answer_logprob_objective(
                            torch, net, head, tokenizer, batch[0], answer_kind, "chat-public-answer", "cpu", 5).sum()
                    finally:
                        for handle in handles:
                            handle.remove()
                base_single = single()
                gate_single = torch.zeros((), requires_grad=True)
                observed = single(gate_single, True)
                torch.testing.assert_close(observed, base_single, rtol=0, atol=0)
                derivative, = torch.autograd.grad(observed * sign, gate_single)
                literal_derivative, = torch.autograd.grad(single(gate_single, False) * sign, gate_single)
                torch.testing.assert_close(derivative, literal_derivative, rtol=2e-6, atol=1e-8)
                finite_single = (single(torch.tensor(.005), False) - single(torch.tensor(-.005), False)) / .01
                torch.testing.assert_close(derivative, finite_single * sign, rtol=.003, atol=2e-5)
                accumulated.add_(derivative.detach().float())
                measured += sign * float(observed.detach())
                del observed, derivative
            self.assertEqual(set(net.seen_shapes), {(1, 5)})
            self.assertAlmostEqual(measured, float(good if objective_name == "correct-logprob" else good - bad), places=6)
            self.assertTrue(bool(torch.isfinite(accumulated)))
        with self.assertRaisesRegex(ValueError, "chat-public-answer"):
            direction.correct_logprob_objective(torch, net, head, tokenizer, batch, "raw-next-token", "cpu")
