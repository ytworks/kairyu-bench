"""CPU numerical boundaries for the bounded LoRA optimizer recipe."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/experiments/gemma4-purpose"
sys.path.insert(0, str(SCRIPTS))
recipe = importlib.import_module("train_lora")


class FrameTests(unittest.TestCase):
    def test_official_terminator_is_not_global_eos_and_no_padding(self):
        class Tokenizer:
            chat_template = "synthetic ordinary ON template"
            eos_token_id = 1

            def apply_chat_template(self, messages, **kwargs):
                assert kwargs["enable_thinking"] is True
                prefix = "<|think|>" + messages[0]["content"] + "<|turn>model\n"
                return (
                    prefix
                    if kwargs["add_generation_prompt"]
                    else prefix + messages[1]["content"] + "<turn|>\n"
                )

            def encode(self, text, **kwargs):
                assert kwargs["add_special_tokens"] is False
                return [ord(c) + 2 for c in text]

        rows = [{"task_id": 601, "text": "Return one.", "code": "def f(): return 1"}]
        frames, proof = recipe.prepare_training_frames(Tokenizer(), rows)
        self.assertEqual(proof["ineligible_ids"], [])
        self.assertFalse(proof["manual_global_eos_added"])
        self.assertNotIn(1, frames[0]["ids"])
        self.assertEqual(len(frames[0]["ids"]), proof["rows"][0]["complete_tokens"])
        long = rows[0] | {"code": "x" * 1100}
        full, invalid = recipe.prepare_training_frames(Tokenizer(), [long])
        self.assertEqual(invalid["ineligible_ids"], [601])
        self.assertGreater(len(full[0]["ids"]), 1024)  # retained, never truncated

    def test_holdout_and_incomplete_training_inventory_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(
                json.dumps({"task_id": 601, "text": "p", "code": "c"}) + "\n"
            )
            summary = {"training_ids": [601, 602], "holdout_ids": [603]}
            self.assertEqual(len(recipe.read_training_rows(path, [601], summary)), 1)
            for ids in ([601, 602], [603]):
                with self.assertRaises(ValueError):
                    recipe.read_training_rows(path, ids, summary)

    def test_continuation_binds_exact_pilot_and_timeout_stops(self):
        import time

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertFalse(
                recipe.wait_continuation(output, "a" * 64, 0, time.monotonic() + 2)
            )
            path = output / "continue.json"
            path.write_text(
                json.dumps({"pilot_receipt_sha256": "b" * 64, "continue_to_step": 64})
            )
            with self.assertRaises(ValueError):
                recipe.wait_continuation(output, "a" * 64, 1, time.monotonic() + 2)
            path.write_text(
                json.dumps({"pilot_receipt_sha256": "a" * 64, "continue_to_step": 64})
            )
            self.assertTrue(
                recipe.wait_continuation(output, "a" * 64, 1, time.monotonic() + 2)
            )


@unittest.skipIf(torch is None, "requires existing Torch CPU environment")
class LoraOptimizerTests(unittest.TestCase):
    def test_two_steps_reach_A_and_B_preserve_base_and_f32_merge_function(self):
        torch.manual_seed(0)
        base = torch.nn.Linear(4, 5, bias=False, dtype=torch.float32)
        original = base.weight.detach().clone()
        adapter = recipe.make_adapter(torch, base, rank=2, alpha=2)
        x, target = torch.randn(9, 4), torch.randn(9, 5)
        self.assertTrue(torch.equal(base(x), adapter(x)))
        opt = torch.optim.AdamW(
            [adapter.A, adapter.B], lr=0.01, weight_decay=0, foreach=False
        )
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            (adapter(x) - target).square().mean().backward()
            self.assertGreater(float(adapter.B.grad.norm()), 0)
            if step == 0:
                self.assertEqual(float(adapter.A.grad.norm()), 0)
            else:
                self.assertGreater(float(adapter.A.grad.norm()), 0)
            opt.step()
        self.assertIsNone(base.weight.grad)
        self.assertTrue(torch.equal(base.weight, original))
        merged = original + adapter.scale * (adapter.B @ adapter.A)
        torch.testing.assert_close(
            adapter(x), torch.nn.functional.linear(x, merged), rtol=2e-6, atol=2e-6
        )

    def test_gram_cap_projects_only_B_and_reports_full_denominator(self):
        torch.manual_seed(1)
        base = torch.nn.Linear(7, 9, bias=False, dtype=torch.bfloat16)
        adapter = recipe.make_adapter(torch, base, rank=3, alpha=6)
        with torch.no_grad():
            adapter.B.normal_()
        a, w = adapter.A.detach().clone(), base.weight.detach().clone()
        direct = (adapter.B.double() @ adapter.A.double() * adapter.scale).norm()
        torch.testing.assert_close(
            recipe.adapter_delta_norm(torch, adapter), direct, atol=1e-12, rtol=1e-12
        )
        denominator = float(w.double().norm())
        result = recipe.project_adapters(
            torch, {"weight": adapter}, {"weight": denominator}, 0.005
        )["weight"]
        self.assertLess(result["B_multiplier"], 1)
        self.assertLessEqual(result["relative_after"], 0.005 * (1 + 1e-7))
        self.assertTrue(torch.equal(adapter.A, a))
        self.assertTrue(torch.equal(base.weight, w))
        # BF16 storage is tested by an independent once-rounded oracle; do not
        # require BF16 unmerged activations to equal merged matmul bit-for-bit.
        merged_f32 = w.float() + (adapter.B @ adapter.A) * adapter.scale
        actual = merged_f32.to(torch.bfloat16)
        bits = merged_f32.detach().numpy().view("uint32")
        oracle = ((bits.astype("uint64") + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(
            "uint16"
        )
        import numpy as np

        np.testing.assert_array_equal(
            actual.view(torch.uint16).detach().numpy(), oracle
        )


if __name__ == "__main__":
    unittest.main()
