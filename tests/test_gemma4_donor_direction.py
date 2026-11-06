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
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temporary:
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
                args = SimpleNamespace(output=Path(temporary), device="cpu", mode="chat-public-answer", reference_tolerance=0.)
                rows = [("x", [1], [2], [3])]
                cases = {"science": [[{"id": "x"}]]}
                calls = []
                def objective(*unused):
                    value = net(torch.tensor([[.5, -.25, .125]], dtype=torch.bfloat16)).float().sum().reshape(1)
                    calls.append(1)
                    if fail and len(calls) == 2:
                        raise RuntimeError("fixture interruption")
                    return value
                with patch.object(direction.common, "load_model", return_value=(torch, None, model)),                         patch.object(direction.common, "prepare", return_value=({"science": [rows]}, {})),                         patch.object(direction.common, "objective", side_effect=objective),                         patch.object(direction.common, "assert_inputs"):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
                            direction.collect(args, checkpoint, cases, {}, donor, entries)
                    else:
                        result = direction.collect(args, checkpoint, cases, {}, donor, entries)
                        self.assertTrue(result["zero_alpha_checks"]["all_exact"])
                        self.assertEqual(result["zero_alpha_checks"]["max_absolute_margin_difference"], 0)
                        derivative = np.load(Path(temporary) / result["folds"][0]["file"], allow_pickle=False)
                        self.assertEqual(derivative.shape, (2,))
                        self.assertTrue(np.isfinite(derivative).all())
                        json.dumps(result, allow_nan=False)
                self.assertTrue(all(not module._forward_hooks for module in net.layers))
                self.assertTrue(all(torch.equal(p, old) and p.grad is None and not p.requires_grad
                                    for p, old in zip(model.parameters(), originals, strict=True)))
