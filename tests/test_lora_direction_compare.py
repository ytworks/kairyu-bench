from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from kairyu_bench.wsp.weights.checkpoint import Checkpoint
from test_wsp_weights import weights

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/experiments/compare_lora_directions.py"
spec = importlib.util.spec_from_file_location("compare_lora_directions", SCRIPT)
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


class LoraFactorComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.left, self.right, self.mapping = {}, {}, []
        self.oracle = []
        base = {}
        for i, (a, b, c, d) in enumerate([
            ([[1, 2, -1, 0]], [[1], [2], [-1]], [[1, 0, 2, -1], [-2, 1, 0, 1]], [[2, -1], [0, 3], [-1, 1]]),
            ([[1, 0, -1]], [[8], [4]], [[1, 1, 0], [0, -1, 2]], [[-1, 0], [1, 2]]),
            ([[1, -1]], [[0], [0]], [[1, 0], [0, 1]], [[1, 2], [3, 4]]),
        ]):
            a, b, c, d = [np.asarray(x, dtype=np.float32) for x in (a, b, c, d)]
            name = f"layer.{i}.weight"
            an, bn = f"layer.{i}.lora_A.weight", f"layer.{i}.lora_B.weight"
            for n, data in ((an, a), (bn, b)):
                raw = (data.view(np.uint32) >> 16).astype("<u2").tobytes()
                self.left[n] = ("BF16", data.shape, raw)
            for n, data in ((an, c), (bn, d)):
                self.right[n] = ("F32", data.shape, data.astype("<f4").tobytes())
            shape = (b.shape[0], a.shape[1])
            base[name] = ("F32", shape, np.zeros(shape, dtype="<f4").tobytes())
            self.mapping.append({"tensor": name, "shape": list(shape),
                                 "left": {"A": an, "B": bn}, "right": {"A": an, "B": bn},
                                 "groups": {"layer": i, "family": "projection"}})
            self.oracle.append((b.astype(float) @ a.astype(float), .75*(d.astype(float) @ c.astype(float))))
        self.base = weights(self.root/"base/weights.safetensors", base)
        self.lp = weights(self.root/"left.safetensors", self.left)
        self.rp = weights(self.root/"right.safetensors", self.right)
        self.lc, self.rc = self.root/"left.json", self.root/"right.json"
        self.lc.write_text(json.dumps({"peft_type": "LORA", "r": 1, "lora_alpha": 2, "bias": "none"}))
        self.rc.write_text(json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 6, "bias": "none"}))
        self.receipt = self.root/"source-receipt.json"
        self.receipt.write_text(json.dumps({"source_tree_sha256": "b"*64}))
        self.plan = {
            "schema_version": 1,
            "source": {"path": str(self.base), "identity": Checkpoint(self.base).identity(hash_files=False),
                       "receipt_path": str(self.receipt), "receipt_sha256": observer.sha(self.receipt),
                       "declared_tree_sha256": "b"*64},
            "left": {"path": str(self.lp), "sha256": observer.sha(self.lp),
                     "config_path": str(self.lc), "config_sha256": observer.sha(self.lc), "coefficient": .5},
            "right": {"path": str(self.rp), "sha256": observer.sha(self.rp),
                      "config_path": str(self.rc), "config_sha256": observer.sha(self.rc), "coefficient": .25},
            "mapping": self.mapping}
        self.pp, self.output = self.root/"plan.json", self.root/"report.json"

    def run_observer(self, *, run=True):
        self.pp.write_text(json.dumps(self.plan))
        return observer.analyze(self.pp, observer.sha(self.pp), self.output, run=run)

    def test_mixed_rank_dtype_full_geometry_preview_and_unchanged_inputs(self):
        before = [observer.sha(p) for p in (self.base, self.lp, self.rp)]
        with mock.patch.object(observer, "factor", side_effect=AssertionError("preview read factors")):
            self.assertEqual(self.run_observer(run=False)["status"], "preview")
        self.assertFalse(self.output.exists())
        result = self.run_observer()
        for row, (left, right) in zip(result["modules"], self.oracle, strict=True):
            self.assertAlmostEqual(row["left_l2"], float(np.linalg.norm(left)), places=12)
            self.assertAlmostEqual(row["right_l2"], float(np.linalg.norm(right)), places=12)
            self.assertAlmostEqual(row["dot"], float(np.sum(left*right)), places=12)
        left, right = [np.concatenate([pair[i].ravel() for pair in self.oracle]) for i in (0, 1)]
        expected = float(left @ right / (np.linalg.norm(left)*np.linalg.norm(right)))
        self.assertAlmostEqual(result["global_geometry"]["cosine"], expected, places=12)
        self.assertEqual(result["groups"]["family"]["projection"], result["global_geometry"])
        self.assertIsNone(result["modules"][2]["cosine"])
        self.assertFalse(result["source_full_payload_hashed_here"])
        self.assertEqual(before, [observer.sha(p) for p in (self.base, self.lp, self.rp)])
        self.assertEqual(json.loads(self.output.read_text())["status"], "completed")
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.run_observer()

    def test_unsupported_or_ambiguous_mapping_and_payload_pins_reject_before_output(self):
        original = self.rc.read_bytes()
        self.rc.write_text(json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 6, "use_dora": True}))
        self.plan["right"]["config_sha256"] = observer.sha(self.rc)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.run_observer()
        self.rc.write_bytes(original)
        self.plan["right"]["config_sha256"] = observer.sha(self.rc)
        original_name = self.mapping[1]["right"]["A"]
        self.mapping[1]["right"]["A"] = self.mapping[0]["right"]["A"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.run_observer()
        self.mapping[1]["right"]["A"] = original_name
        self.plan["right"]["sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "SHA/stat"):
            self.run_observer()
        self.assertFalse(self.output.exists())
        original_right = self.plan["right"]
        self.plan["right"] = dict(self.plan["left"])
        self.plan["left"]["sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "conflicting input pins"):
            self.run_observer()
        self.plan["left"]["sha256"] = observer.sha(self.lp)
        self.plan["right"] = original_right

    def test_nonfinite_and_midrun_mutation_fail_without_publishing(self):
        original = self.rp.read_bytes()
        changed = dict(self.right)
        dtype, shape, _ = changed["layer.0.lora_A.weight"]
        changed["layer.0.lora_A.weight"] = (dtype, shape, np.full(shape, np.inf, dtype="<f4").tobytes())
        weights(self.rp, changed)
        self.plan["right"]["sha256"] = observer.sha(self.rp)
        with self.assertRaisesRegex(ValueError, "nonfinite factor"):
            self.run_observer()
        self.rp.write_bytes(original)
        self.plan["right"]["sha256"] = observer.sha(self.rp)
        actual = observer.factor
        changed_once = False
        def mutate(*args):
            nonlocal changed_once
            result = actual(*args)
            if not changed_once:
                self.lc.write_bytes(self.lc.read_bytes()+b" ")
                changed_once = True
            return result
        with mock.patch.object(observer, "factor", mutate), self.assertRaisesRegex(ValueError, "changed"):
            self.run_observer()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
