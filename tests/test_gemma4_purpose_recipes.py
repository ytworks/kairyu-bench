from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_wsp_weights import weights


RECIPES = Path(__file__).resolve().parents[1] / "scripts/experiments/gemma4-purpose"


def load(name):
    spec = importlib.util.spec_from_file_location("purpose_" + name, RECIPES / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    modules = {"recipe_common": COMMON} if name != "recipe_common" else {}
    if "JOINT" in globals():
        modules["joint"] = JOINT
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


COMMON = load("recipe_common")
MATRIX = load("matrix")
NATIVE = load("native")
JOINT = load("joint")


class PurposeRecipeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cases = self.root / "cases.json"
        self.cases.write_text(json.dumps({"code": [[{"id": "c1", "prompt": "a", "correct": " yes", "foil": " no"}]]}))

    def test_import_and_help_do_not_load_torch_transformers_or_create_outputs(self):
        code = """import runpy,sys
for name in ('native','matrix','joint'):
    sys.path.insert(0, sys.argv[1])
    runpy.run_path(sys.argv[1] + '/' + name + '.py', run_name='import_fixture')
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-B", "-c", code, str(RECIPES)],
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ("native", "matrix"):
            result = subprocess.run([sys.executable, "-B", str(RECIPES / (name + ".py")), "--help"],
                                    text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--run", result.stdout)

    def test_cases_preserve_continuation_whitespace_and_reject_ambiguous_inputs(self):
        cases = COMMON.read_cases(self.cases)
        self.assertEqual(cases["code"][0][0]["correct"], " yes")
        for malformed in ({"../escape": [[{"prompt": "a", "correct": "b", "foil": "c"}]]},
                          {"code": [[{"id": "x", "prompt": "a", "correct": "b", "foil": "c"}] * 2]},
                          {"code": [[{"prompt": "a", "correct": "b", "foil": "c", "expected_margin": float("nan")}]]}):
            self.cases.write_text(json.dumps(malformed))
            with self.assertRaises(ValueError):
                COMMON.read_cases(self.cases)

    def test_raw_distinct_token_boundary_and_reference_checks(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=True):
                return [1] + [ord(c) for c in text]
        cases = {"code": [[{"id": "x", "prompt": "a", "correct": "b", "foil": "c"}]]}
        prepared, proof = COMMON.raw_frames(Tokenizer(), cases)
        self.assertEqual(prepared["code"][0][0][2:], ([98], [99]))
        self.assertEqual(proof["frames"]["code"][0][0]["prefix_tokens"], 2)
        cases["code"][0][0]["foil"] = "bad"
        with self.assertRaises(ValueError):
            COMMON.raw_frames(Tokenizer(), cases)
        self.assertEqual(COMMON.check_references([{"id": "x", "expected_margin": .25}], {"x": .25}, 1e-6)["max_reference_error"], 0)
        with self.assertRaises(ValueError):
            COMMON.check_references([{"id": "x", "expected_margin": .25}], {"x": .26}, 1e-6)

    def test_preview_is_read_only_and_run_requires_matching_attestation(self):
        model = weights(self.root / "base/model.safetensors", {
            "model.language_model.norm.weight": ("BF16", (2,), bytes.fromhex("803f0040"))})
        output = self.root / "new"
        argv = ["--checkpoint", str(model.parent), "--cases", str(self.cases), "--output", str(output)]
        parser = COMMON.parser("fixture", chat=True)
        args = parser.parse_args(argv)
        checkpoint, _, provenance = COMMON.preflight(args, RECIPES / "native.py")
        self.assertFalse(output.exists())
        self.assertFalse(provenance["full_tree_rehashed_here"])
        args.run = True
        with self.assertRaisesRegex(ValueError, "attestation"):
            COMMON.preflight(args, RECIPES / "native.py")
        receipt = self.root / "receipt.json"
        receipt.write_text(json.dumps({"checkpoint": checkpoint.identity(hash_files=True)}))
        args.checkpoint_attestation = receipt
        args.attestation_sha256 = COMMON.sha(receipt)
        verified, _, provenance = COMMON.preflight(args, RECIPES / "native.py")
        self.assertFalse(output.exists())
        COMMON.assert_inputs(verified, provenance)
        os.utime(model, ns=(model.stat().st_atime_ns, model.stat().st_mtime_ns + 1000))
        with self.assertRaises(ValueError):
            COMMON.preflight(args, RECIPES / "native.py")
        args.output = model.parent / "output"
        with self.assertRaisesRegex(ValueError, "disjoint"):
            COMMON.preflight(args, RECIPES / "native.py")

    def test_stratified_matrix_coordinates_preserve_head_and_band_coverage(self):
        config = {"layer_types": ["full_attention", "sliding_attention"],
                  "num_attention_heads": 32, "num_global_key_value_heads": 4, "num_key_value_heads": 16}
        for kind, shape in (("q", (4096, 256)), ("k", (512, 256)), ("v", (512, 256)),
                            ("o", (256, 4096)), ("gate", (512, 256)), ("up", (512, 256)), ("down", (256, 512))):
            with self.subTest(kind=kind):
                coords, strata = MATRIX.sample(kind, shape, kind, 0, config, 0)
                again, _ = MATRIX.sample(kind, shape, kind, 0, config, 0)
                other, _ = MATRIX.sample(kind, shape, kind, 0, config, 1)
                self.assertEqual(len(set(map(tuple, coords))), 512)
                np.testing.assert_array_equal(coords, again)
                self.assertFalse(np.array_equal(coords, other))
                self.assertTrue(np.all(coords >= 0))
                self.assertTrue(np.all(coords < shape))
                if kind in ("q", "k", "v"):
                    counts = np.bincount(coords[:, 0] // (shape[0] // strata["heads"]))
                    self.assertEqual(set(counts), {512 // strata["heads"]})
                if kind == "o":
                    counts = np.bincount(coords[:, 1] // (shape[1] // strata["input_heads"]))
                    self.assertEqual(set(counts), {512 // strata["input_heads"]})
        with self.assertRaises(ValueError):
            MATRIX.sample("tiny", (2, 2), "down", 0, config, 0)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional CPU Torch fixture; no model download")
class PurposeTorchFixtures(unittest.TestCase):
    def test_factor_matches_real_weight_gradient_and_zero_additive_probe(self):
        torch.manual_seed(7)
        inputs = torch.randn(2, 3, 5, requires_grad=True)
        weight = torch.randn(4, 5, requires_grad=True)
        rows = torch.tensor([0, 2, 2, 3])
        columns = torch.tensor([1, 0, 3, 4])
        probe = torch.zeros(4, requires_grad=True)
        output = inputs @ weight.T
        added = torch.zeros_like(output)
        added.index_add_(-1, rows, inputs[..., columns] * probe)
        updated = output + added
        output_gradient = torch.randn_like(updated)
        updated.register_hook(lambda upstream: setattr(self, "factor", MATRIX.factor_derivative(upstream, inputs[..., columns].detach(), rows)))
        dweight, dprobe = torch.autograd.grad((updated * output_gradient).sum(), (weight, probe))
        torch.testing.assert_close(self.factor.float(), dweight[rows, columns], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(self.factor.float(), dprobe, rtol=1e-6, atol=1e-6)
        self.assertIsNone(weight.grad)

    def test_virtual_group_gain_derivative_and_norm_cost(self):
        inputs = torch.tensor([[1., 2., 3., 4.]])
        weight = torch.tensor([[1., 2., 3., 4.], [-1., 0., 2., 3.]], requires_grad=True)
        gates = torch.ones(2, requires_grad=True)
        output = (inputs * gates.repeat_interleave(2)) @ weight.T
        dweight, dgate = torch.autograd.grad(output.square().sum(), (weight, gates))
        reference = (dweight * weight).reshape(2, 2, 2).sum((0, 2))
        torch.testing.assert_close(dgate, reference)
        costs = NATIVE.column_group_norms(weight, 2, torch)
        torch.testing.assert_close(costs, weight.detach().reshape(2, 2, 2).square().sum((0, 2)).sqrt())
        self.assertIsNone(weight.grad)


if __name__ == "__main__":
    unittest.main()
