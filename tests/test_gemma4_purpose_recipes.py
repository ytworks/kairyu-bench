from __future__ import annotations

import importlib.util
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
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
    if "OBJECTIVES" in globals():
        modules["objective_variants"] = OBJECTIVES
    if "MATRIX" in globals():
        modules["matrix"] = MATRIX
    if "JOINT" in globals():
        modules["joint"] = JOINT
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


COMMON = load("recipe_common")
OBJECTIVES = load("objective_variants")
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
        for label in ({"split": "transfer"}, {"split": "test"}, {"split": "holdout"}, {"gradient_allowed": False}):
            marked = copy.deepcopy(cases)
            marked["code"][0][0].update(label)
            self.cases.write_text(json.dumps(marked))
            with self.assertRaisesRegex(ValueError, "cannot be gradient"):
                COMMON.read_cases(self.cases)
            self.cases.write_text(json.dumps({"cases": cases, **label}))
            with self.assertRaisesRegex(ValueError, "cannot be gradient"):
                COMMON.read_cases(self.cases)
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

    def test_chat_matrix_frame_packing_and_no_truncation(self):
        class Tokenizer:
            chat_template = "fixture official thinking template"
            def apply_chat_template(self, messages, *, tokenize, enable_thinking, add_generation_prompt=False):
                self.assert_flags = (tokenize, enable_thinking)
                prefix = ("<|turn>system\n<|think|>\n<turn|>\n" if enable_thinking else "")
                prefix += "<|turn>user\n" + messages[0]["content"] + "<turn|>\n<|turn>model\n"
                if add_generation_prompt:
                    return prefix + ("" if enable_thinking else "<|channel>thought\n<channel|>")
                return prefix + messages[1]["content"] + "<turn|>"
            def encode(self, text, add_special_tokens=False):
                return [ord(character) for character in text]
        tokenizer = Tokenizer()
        cases = {"science": [[
            {"id": "short-a", "prompt": "p" * 100, "correct": "same good", "foil": "same bad"},
            {"id": "short-b", "prompt": "p" * 110, "correct": "same other good", "foil": "same other bad"},
            {"id": "long", "prompt": "p" * 350, "correct": "same good", "foil": "same bad"}]]}
        prepared, frame = COMMON.prepare(tokenizer, cases, "chat-public-answer")
        rows = prepared["science"][0]
        self.assertEqual(tokenizer.assert_flags, (False, True))
        self.assertTrue(frame["enable_thinking"])
        self.assertTrue(frame["forced_empty_thought"])
        self.assertEqual(frame["schema_version"], 2)
        self.assertTrue("".join(map(chr, rows[0][1])).endswith("<|channel>thought\n<channel|>"))
        self.assertEqual(rows[0][2], [ord(c) for c in "same good"])
        self.assertEqual(rows[0][2][0], rows[0][3][0])  # Later answer tokens must distinguish them.
        packed = MATRIX.checked_batches(rows, "chat-public-answer")
        self.assertEqual([len(batch) for batch in packed], [2, 1])
        self.assertEqual([row[0] for batch in packed for row in batch], [row[0] for row in rows])
        single = MATRIX.checked_batches(rows, "chat-public-answer", 1)
        self.assertEqual([len(batch) for batch in single], [1, 1, 1])
        self.assertEqual(single, COMMON.objective_batches(rows, "chat-public-answer", 1))
        self.assertEqual([row for batch in single for row in batch], rows)
        self.assertEqual(COMMON.objective_batches(rows, "raw-next-token", 1), single)
        args = COMMON.parser("fixture", chat=True).parse_args([
            "--checkpoint", str(self.root), "--cases", str(self.cases), "--output", str(self.root / "out"),
            "--max-cases-per-microbatch", "1"])
        self.assertEqual(args.max_cases_per_microbatch, 1)
        too_long = [("oversize", [1] * 512, [2], [3])]
        with self.assertRaisesRegex(ValueError, "no truncation"):
            MATRIX.checked_batches(too_long, "chat-public-answer")
        self.assertEqual(MATRIX.checked_batches(too_long, "raw-next-token"), [too_long])

    def test_matrix_input_bytes_are_pinned_before_parse_and_no_implicit_dataset_read(self):
        model = weights(self.root / "base/model.safetensors", {
            "model.language_model.norm.weight": ("BF16", (2,), bytes.fromhex("803f0040"))})
        (self.root / "holdout.json").write_text("not a calibration input")
        args = COMMON.parser("matrix fixture", chat=True).parse_args([
            "--checkpoint", str(model.parent), "--cases", str(self.cases), "--output", str(self.root / "new"),
            "--mode", "chat-public-answer"])
        checkpoint, cases, provenance = COMMON.preflight(args, RECIPES / "matrix.py")
        self.assertEqual(list(cases), ["code"])
        self.assertEqual(provenance["input_sha256"][str(self.cases)], COMMON.sha(self.cases))
        self.assertNotIn(str(self.root / "holdout.json"), provenance["input_sha256"])
        self.assertFalse(args.output.exists())
        COMMON.assert_inputs(checkpoint, provenance)
        original = COMMON.read_cases
        def changed_during_parse(path, *, content):
            path.write_bytes(content + b" ")
            return original(path, content=content)
        with patch.object(COMMON, "read_cases", side_effect=changed_during_parse):
            with self.assertRaisesRegex(ValueError, "input stat changed"):
                COMMON.preflight(args, RECIPES / "matrix.py")

    def test_stratified_matrix_coordinates_preserve_head_and_band_coverage(self):
        # Actual Gemma 4 31B projection dimensions, legacy JSON vs HF 5.20 serialized overrides.
        legacy = {"layer_types": ["sliding_attention"] * 5 + ["full_attention"],
                  "hidden_size": 5376, "head_dim": 256, "global_head_dim": 512,
                  "num_attention_heads": 32, "num_global_key_value_heads": 4, "num_key_value_heads": 16}
        current = {key: value for key, value in legacy.items()
                   if key not in ("global_head_dim", "num_global_key_value_heads")}
        current["per_layer_config"] = {"05": {"head_dim": 512, "num_key_value_heads": 4}}
        for layer in (0, 5):
            width, kv = (256, 16) if layer == 0 else (512, 4)
            shapes = {"q": (32 * width, 5376), "k": (kv * width, 5376),
                      "o": (5376, 32 * width), "gate": (21504, 5376),
                      "up": (21504, 5376), "down": (5376, 21504)}
            if layer == 0:  # Full-attention K=V has no separate V tensor.
                shapes["v"] = (kv * width, 5376)
            for kind, shape in shapes.items():
                with self.subTest(kind=kind, layer=layer):
                    coords, strata = MATRIX.sample(kind, shape, kind, layer, current, 0)
                    again, _ = MATRIX.sample(kind, shape, kind, layer, legacy, 0)
                    other, _ = MATRIX.sample(kind, shape, kind, layer, current, 1)
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
        triplets = {f"model.language_model.layers.{layer}.mlp.{kind}_proj.weight":
                    SimpleNamespace(shape=shape)
                    for layer in (0, 5) for kind, shape in
                    (("gate", (21504, 5376)), ("up", (21504, 5376)), ("down", (5376, 21504)))}
        coherent = MATRIX.mlp_triplet_samples(triplets, 0)
        for layer in (0, 5):
            prefix = f"model.language_model.layers.{layer}.mlp."
            gate, up, down = [coherent[prefix + k + "_proj.weight"][0] for k in ("gate", "up", "down")]
            np.testing.assert_array_equal(gate, up)
            np.testing.assert_array_equal(np.unique(gate[:, 0]), np.unique(down[:, 1]))
            self.assertEqual(len(set(map(tuple, gate))), 512)
            self.assertEqual(len(set(map(tuple, down))), 512)
        damaged = dict(triplets)
        damaged["model.language_model.layers.0.mlp.down_proj.weight"] = SimpleNamespace(shape=(5376, 21503))
        with self.assertRaisesRegex(ValueError, "triplet shapes"):
            MATRIX.mlp_triplet_samples(damaged, 0)
        with self.assertRaisesRegex(ValueError, "does not match configured heads"):
            MATRIX.sample("wrong", (4096, 5376), "k", 5, current, 0)
        with self.assertRaises(ValueError):
            MATRIX.sample("tiny", (2, 2), "down", 0, current, 0)


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

    def test_chat_matrix_microbatches_match_full_loss_true_gradient_and_finite_difference(self):
        torch.manual_seed(71)
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(7, 5)
                self.first = torch.nn.Linear(5, 4, bias=False)
                self.last = torch.nn.Linear(4, 3, bias=False)
                self.config = SimpleNamespace(final_logit_softcapping=3.)
            def forward(self, input_ids, **kwargs):
                hidden = self.last(torch.tanh(self.first(self.embed_tokens(input_ids))))
                return SimpleNamespace(last_hidden_state=hidden)
        net, head = Net(), torch.nn.Linear(3, 7, bias=False)
        net.requires_grad_(False)
        head.requires_grad_(False)
        reference = copy.deepcopy(net)
        tokenizer = SimpleNamespace(pad_token_id=0)
        rows = [
            ("a", [1] * 120, [2, 3, 2], [2, 4]),
            ("b", [2] * 120, [3, 4, 2, 1], [3, 1, 2]),
            ("c", [1] * 430, [2, 3], [2, 4, 3]),
        ]
        modules = {"first": net.first, "last": net.last}
        plans = {
            "first": (torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 4])),
            "last": (torch.tensor([0, 1, 1, 2]), torch.tensor([0, 1, 2, 3])),
        }
        reference.first.weight.requires_grad_(True)
        reference.last.weight.requires_grad_(True)
        full = COMMON.objective(torch, reference, head, tokenizer, rows, "chat-public-answer", "cpu")
        gradients = torch.autograd.grad(full.mean(), [reference.first.weight, reference.last.weight])
        for objective_kind in ("margin", "correct-logprob"):
            objective = COMMON.objective if objective_kind == "margin" else OBJECTIVES.correct_logprob_objective
            oracle_values = objective(torch, reference, head, tokenizer, rows, "chat-public-answer", "cpu")
            oracle_gradients = torch.autograd.grad(oracle_values.mean(), [reference.first.weight, reference.last.weight])
            for batching in ("paired", "serial"):
                for limit in (None, 1):
                    sums = {name: torch.zeros(4, dtype=torch.float64) for name in modules}
                    measured = {}
                    packed = MATRIX.checked_batches(rows, "chat-public-answer", limit)
                    self.assertEqual(len(packed), 2 if limit is None else 3)
                    for index, batch in enumerate(packed):
                        margins, captured, checks, forwards = MATRIX.objective_derivatives(
                            torch, net, head, tokenizer, batch, "chat-public-answer", "cpu",
                            plans, modules, len(rows), set(modules) if index == 0 else set(),
                            objective_kind=objective_kind, answer_batching=batching)
                        self.assertEqual(forwards, 1 if batching == "paired" else len(batch) * (2 if objective_kind == "margin" else 1))
                        if index == 0:
                            self.assertEqual(sum(item["cells"] for item in checks.values()), 8)
                        for name in sums:
                            sums[name] += captured[name]
                        measured.update({row[0]: value for row, value in zip(batch, margins, strict=True)})
                    torch.testing.assert_close(torch.stack([measured[row[0]] for row in rows]), oracle_values.detach(), rtol=1e-6, atol=2e-7)
                    for name, gradient in zip(plans, oracle_gradients, strict=True):
                        r, c = plans[name]
                        torch.testing.assert_close(sums[name].float(), gradient[r, c], rtol=2e-5, atol=1e-7)
            # FP32 log-probability subtraction needs a resolvable finite-difference step.
            epsilon = 1e-2
            with torch.no_grad():
                old = net.first.weight[0, 1].clone()
                net.first.weight[0, 1] = old + epsilon
                plus = objective(torch, net, head, tokenizer, rows, "chat-public-answer", "cpu").mean()
                net.first.weight[0, 1] = old - epsilon
                minus = objective(torch, net, head, tokenizer, rows, "chat-public-answer", "cpu").mean()
                net.first.weight[0, 1] = old
            self.assertAlmostEqual(float(sums["first"][0]), float((plus - minus) / (2 * epsilon)), delta=3e-5)
        with unittest.mock.patch.object(OBJECTIVES, "single_answer_logprob_objective", side_effect=ValueError("forward failed")):
            with self.assertRaisesRegex(ValueError, "forward failed"):
                MATRIX.objective_derivatives(torch, net, head, tokenizer, rows, "chat-public-answer", "cpu",
                                             plans, modules, len(rows), set(modules),
                                             objective_kind="correct-logprob", answer_batching="serial")
        self.assertTrue(all(not value._forward_hooks for value in net.modules()))
        # Native collector's shared batching and original fold denominator retain the same objective.
        native_sums = [torch.zeros_like(reference.first.weight), torch.zeros_like(reference.last.weight)]
        for batch in COMMON.objective_batches(rows, "chat-public-answer", 1):
            margin = COMMON.objective(torch, reference, head, tokenizer, batch, "chat-public-answer", "cpu")
            values = torch.autograd.grad(margin.sum() / len(rows), [reference.first.weight, reference.last.weight])
            for total, value in zip(native_sums, values, strict=True):
                total += value
        for total, expected in zip(native_sums, gradients, strict=True):
            torch.testing.assert_close(total, expected, rtol=2e-5, atol=1e-7)
        # Mathematical objective/gradient oracle for serial one-answer graphs.
        width = max(len(row[1]) + max(len(row[2]), len(row[3])) for row in rows)
        for kind in ("margin", "correct-logprob"):
            fn = COMMON.objective if kind == "margin" else OBJECTIVES.correct_logprob_objective
            values = fn(torch, reference, head, tokenizer, rows, "chat-public-answer", "cpu")
            expected = torch.autograd.grad(values.mean(), [reference.first.weight, reference.last.weight])
            serial = [torch.zeros_like(value) for value in expected]
            observed = []
            for row in rows:
                total = 0.
                for answer, sign in ((("correct", 1.), ("foil", -1.)) if kind == "margin" else (("correct", 1.),)):
                    value = OBJECTIVES.single_answer_logprob_objective(torch, reference, head, tokenizer, row, answer, "chat-public-answer", "cpu", width)
                    terms = torch.autograd.grad(value.sum() * sign / len(rows), [reference.first.weight, reference.last.weight])
                    for acc, term in zip(serial, terms, strict=True):
                        acc += term.float()
                    total += sign * float(value.detach()[0])
                observed.append(total)
            np.testing.assert_allclose(observed, values.detach().numpy(), rtol=1e-6, atol=3e-7)
            for actual, oracle in zip(serial, expected, strict=True):
                torch.testing.assert_close(actual, oracle, rtol=2e-5, atol=1e-7)
        self.assertTrue(all(value.grad is None for value in net.parameters()))
        self.assertTrue(all(not value._forward_hooks for value in net.modules()))

    def test_native_correct_logprob_collects_separate_loss_and_restores_on_reference_failure(self):
        torch.manual_seed(32)
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(7, 3, dtype=torch.bfloat16)
                self.norm = torch.nn.Module()
                self.norm.register_parameter("weight", torch.nn.Parameter(torch.tensor([1., 1.5, .75], dtype=torch.bfloat16)))
                self.layers = []
                self.events = []
                self.config = SimpleNamespace(final_logit_softcapping=3.)
            def forward(self, input_ids, **kwargs):
                self.events.append(("forward", tuple(input_ids.shape)))
                hidden = self.embedding(input_ids) * self.norm.weight
                if hidden.requires_grad:
                    hidden.register_hook(lambda unused: self.events.append(("backward", None)))
                return SimpleNamespace(last_hidden_state=hidden)
        net = Net()
        head = torch.nn.Linear(3, 7, bias=False, dtype=torch.bfloat16)
        net.requires_grad_(False)
        head.requires_grad_(False)
        model = SimpleNamespace(model=SimpleNamespace(language_model=net), get_output_embeddings=lambda: head)
        tokenizer = SimpleNamespace(pad_token_id=0)
        rows = [("a", [1, 2], [2, 3, 2], [2, 4]), ("b", [2, 1], [3, 4], [3, 1, 2])]
        name = "model.language_model.norm.weight"
        original = net.norm.weight.detach().float().numpy().copy()
        checkpoint = SimpleNamespace(read=lambda key: original.copy() if key == name else self.fail(key))
        for kind, layout in [(kind, layout) for kind in ("margin", "correct-logprob") for layout in ("paired", "serial")]:
            objective = COMMON.objective if kind == "margin" else OBJECTIVES.correct_logprob_objective
            expected_gradient = torch.zeros_like(net.norm.weight, dtype=torch.float32)
            expected_values = {}
            net.norm.weight.requires_grad_(True)
            for batch in COMMON.objective_batches(rows, "chat-public-answer", 1):
                if layout == "paired":
                    values = objective(torch, net, head, tokenizer, batch, "chat-public-answer", "cpu")
                    gradient, = torch.autograd.grad(values.sum() / len(rows), [net.norm.weight])
                    expected_gradient += gradient.float()
                    expected_values.update({row[0]: float(value) for row, value in zip(batch, values.detach(), strict=True)})
                else:
                    row = batch[0]
                    expected_values[row[0]] = 0.
                    width = len(row[1]) + max(len(row[2]), len(row[3]))
                    for answer, sign in ((("correct", 1.), ("foil", -1.)) if kind == "margin" else (("correct", 1.),)):
                        values = OBJECTIVES.single_answer_logprob_objective(torch, net, head, tokenizer, row, answer, "chat-public-answer", "cpu", width)
                        gradient, = torch.autograd.grad(values.sum() * sign / len(rows), [net.norm.weight])
                        expected_gradient += gradient.float()
                        expected_values[row[0]] += sign * float(values.detach()[0])
            net.norm.weight.requires_grad_(False)
            reference_key = "expected_margin" if kind == "margin" else "expected_correct_logprob"
            cases = {"science": [[{"id": row[0], reference_key: expected_values[row[0]]} for row in rows]]}
            if kind == "correct-logprob":
                for row in cases["science"][0]:
                    row["expected_margin"] = 999.  # Must not interpret this as the selected objective.
            for fail in (False, True):
                with self.subTest(objective=kind, layout=layout, failure=fail), tempfile.TemporaryDirectory() as temporary:
                    args = SimpleNamespace(output=Path(temporary), mode="chat-public-answer", device="cpu",
                                           virtual=False, max_cases_per_microbatch=1, objective=kind, answer_batching=layout, reference_tolerance=0.)
                    supplied = copy.deepcopy(cases)
                    if fail:
                        supplied["science"][0][0][reference_key] += 1.
                    net.events.clear()
                    with patch.object(COMMON, "load_model", return_value=(torch, tokenizer, model)), \
                         patch.object(COMMON, "prepare", return_value=({"science": [rows]}, {})), \
                         patch.object(COMMON, "assert_inputs"), \
                         patch.object(COMMON, "transformers_version", return_value="fixture"):
                        if fail:
                            with self.assertRaisesRegex(ValueError, "reference"):
                                NATIVE.collect(args, checkpoint, supplied, {})
                        else:
                            report = NATIVE.collect(args, checkpoint, supplied, {})
                            self.assertEqual(report["objective_kind"], kind)
                            self.assertEqual(report["answer_batching"], layout)
                            record = report["folds"][0]
                            selected = "mean_margin" if kind == "margin" else "mean_correct_logprob"
                            self.assertAlmostEqual(record[selected], np.mean(list(expected_values.values())))
                            self.assertNotIn("mean_margin" if kind != "margin" else "mean_correct_logprob", record)
                            self.assertEqual(record["reference_cases"], 2)
                            self.assertEqual(record["microbatches"], 2)
                            with np.load(Path(temporary) / "gradient-science-0.npz", allow_pickle=False) as saved:
                                np.testing.assert_array_equal(saved[name], expected_gradient.numpy())
                    count = 4 if layout == "serial" and kind == "margin" else 2
                    self.assertEqual([event[0] for event in net.events], ["forward", "backward"] * count)
                    self.assertTrue(all(shape == (1 if layout == "serial" else 2, 5) for event, shape in net.events if event == "forward"))
                    np.testing.assert_array_equal(net.norm.weight.detach().float().numpy(), original)
                    self.assertFalse(net.norm.weight.requires_grad)
                    self.assertIsNone(net.norm.weight.grad)

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
