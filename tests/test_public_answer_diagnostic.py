from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/experiments/gemma4-purpose"))
import public_answer_diagnostic as diagnostic  # noqa: E402


class PublicAnswerDiagnosticTests(unittest.TestCase):
    def test_explicit_unique_selection_and_preview_do_not_load_model(self):
        cases = {"code": [[{"id": "a", "prompt": "public", "correct": "yes", "foil": "no"}]]}
        self.assertEqual(diagnostic.select_cases(cases, ["a"])[0][2]["correct"], "yes")
        for identifiers in ([], ["a", "a"], ["missing"], ["a", "b", "c", "d"]):
            with self.assertRaises(ValueError):
                diagnostic.select_cases(cases, identifiers)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            with mock.patch.object(diagnostic.common, "preflight", return_value=(None, cases, {})), \
                 mock.patch.object(diagnostic.common, "load_model") as load, \
                 mock.patch("sys.stdout", new_callable=io.StringIO) as printed:
                self.assertEqual(diagnostic.main(["--checkpoint", directory, "--cases", directory,
                                                  "--output", str(output), "--case-id", "a"]), 0)
            load.assert_not_called()
            self.assertFalse(output.exists())
            self.assertIn('"model_forwards": 0', printed.getvalue())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional Torch CPU contract")
    def test_native_and_fp32_softcaps_are_measured_not_forced_bit_identical(self):
        import torch

        raw = torch.tensor([[0.7, 2.3, -4.2], [1.1, -0.8, 3.7]], dtype=torch.bfloat16)
        official = (torch.tanh(raw / 2.) * 2.).float()
        native = diagnostic.cap_logits(torch, raw, 2., upcast=False)
        fp32 = diagnostic.cap_logits(torch, raw, 2., upcast=True)
        result = diagnostic.numeric_comparison(torch, official, native, fp32,
                                               torch.tensor([1, 2]), channel_token_id=0)
        self.assertEqual(result["custom_native_softcap"]["max_abs_logit_difference_from_official"], 0)
        self.assertGreater(result["custom_fp32_softcap"]["max_abs_logit_difference_from_official"], 0)
        self.assertEqual(result["answer_tokens"], 2)
        expected = torch.log_softmax(official, -1)[[0, 1], [1, 2]].mean()
        self.assertAlmostEqual(result["official_full_model"]["correct_mean_logprob"], float(expected))
        self.assertFalse(result["logits_saved"])


if __name__ == "__main__":
    unittest.main()
