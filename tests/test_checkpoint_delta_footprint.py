from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from kairyu_bench.wsp.weights.checkpoint import Checkpoint

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/experiments/checkpoint-delta-footprint.py"
SPEC = importlib.util.spec_from_file_location("delta_footprint", SCRIPT)
assert SPEC and SPEC.loader
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def checkpoint(root, tensors):
    root.mkdir()
    payload = b""
    header = {}
    for name, words, shape, dtype in tensors:
        data = np.asarray(words, dtype="<u2").tobytes()
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [len(payload), len(payload) + len(data)]}
        payload += data
    raw = json.dumps(header).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def declared(root):
    states = TOOL.inventory(root)
    rows = [{"path": n, "size": s["size"],
             "sha256": hashlib.sha256((root / n).read_bytes()).hexdigest()}
            for n, s in states.items()]
    return {"checkpoint_path": str(root), "inventory_stat": states,
            "files": rows, "tree_sha256": TOOL.digest(rows)}


class DeltaFootprintTests(unittest.TestCase):
    def test_chunked_counts_signed_zero_and_nonfinite(self):
        name = "model.language_model.layers.2.mlp.down_proj.weight"
        left = [0x0000, 0x8000, 0x3F80, 0x4000, 0x7FC1, 0x7F80, 0xFF80, 0x0001, 0x7F7F]
        right = [0x8000, 0x0000, 0x4000, 0x4000, 0x7FC2, 0x7F80, 0x3F80, 0x0002, 0xFF7F]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for directory, words in [(root / "a", left), (root / "b", right)]:
                checkpoint(directory, [(name, words, [9], "BF16"),
                                       ("same", [0x7FC0, 0x7F80, 0x0000], [3], "BF16")])
            result = TOOL.footprint(Checkpoint(root / "a"), Checkpoint(root / "b"), 4)
            av = (np.asarray(left, dtype=np.uint32) << 16).view(np.float32).astype(np.float64)
            bv = (np.asarray(right, dtype=np.uint32) << 16).view(np.float32).astype(np.float64)
            finite = np.isfinite(av) & np.isfinite(bv)
            expected = float(np.sqrt(np.sum((bv[finite] - av[finite]) ** 2)))
            g = result["global"]
            self.assertEqual(g["stored_bit_changed_elements"], 7)
            self.assertEqual(g["numeric_changed_finite_elements"], 3)
            self.assertEqual(g["base_nonfinite_elements"], 5)
            self.assertEqual(g["candidate_nonfinite_elements"], 4)
            self.assertEqual(g["nonfinite_pairs"], 5)
            self.assertIsNone(g["delta_l2"])
            self.assertAlmostEqual(g["finite_delta_l2"] / expected, 1)
            self.assertEqual(result["unchanged_stored_bit_tensors"], 1)
            self.assertEqual(result["raw_payload_bytes_read"], 48)
            self.assertGreater(result["identical_chunk_bytes_skipping_decode"], 0)
            self.assertEqual(result["layers"]["model.language_model.layers.2"]["stored_bit_changed_elements"], 7)
            self.assertEqual(result["families"]["down_proj"]["numeric_changed_finite_elements"], 3)

    def test_finite_reference_and_layout_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            checkpoint(root / "a", [("x", [0x3F80, 0x4000, 0x4040, 0x4080], [2, 2], "BF16")])
            checkpoint(root / "b", [("x", [0x4000, 0x3F80, 0x4040, 0x4000], [2, 2], "BF16")])
            g = TOOL.footprint(Checkpoint(root / "a"), Checkpoint(root / "b"), 6)["global"]
            self.assertEqual(g["stored_bit_changed_elements"], 3)
            self.assertEqual(g["numeric_changed_finite_elements"], 3)
            self.assertAlmostEqual(g["delta_l2"], np.sqrt(6))
            checkpoint(root / "shape", [("x", [0x4000] * 4, [4], "BF16")])
            checkpoint(root / "missing", [("y", [0x4000] * 4, [2, 2], "BF16")])
            checkpoint(root / "dtype", [("x", [0x4000] * 4, [2, 2], "F16")])
            for name in ["shape", "missing", "dtype"]:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    TOOL.footprint(Checkpoint(root / "a"), Checkpoint(root / name), 4)

    def test_preview_run_identity_mutation_and_fresh_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            checkpoint(root / "a", [("x", [0x3F80, 0x0000], [2], "BF16")])
            checkpoint(root / "b", [("x", [0x4000, 0x8000], [2], "BF16")])
            evidence = root / "prior.json"
            evidence.write_text('{"test_prior_full_hash": true}')
            plan = {"schema_version": 1, "kind": "bf16-delta-footprint-plan",
                    "base": declared(root / "a"), "candidate": declared(root / "b"),
                    "expected_tensors": 1, "chunk_bytes": 2,
                    "prior_evidence": [{"path": str(evidence), "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}]}
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(Checkpoint, "read_raw", side_effect=AssertionError("preview read payload")):
                result = TOOL.execute(path, digest, root / "preview.json", False)
            self.assertEqual(result["status"], "preview")
            result = TOOL.execute(path, digest, root / "result.json", True)
            self.assertEqual(result["global"]["stored_bit_changed_elements"], 2)
            self.assertEqual(result["global"]["numeric_changed_finite_elements"], 1)
            self.assertEqual(result["global"]["delta_l2"], 1)
            self.assertFalse(result["fresh_payload_hash_performed"])
            with self.assertRaises(ValueError):
                TOOL.execute(path, digest, root / "result.json", True)
            original = TOOL.footprint

            def mutate(*args, **kwargs):
                r = original(*args, **kwargs)
                (root / "b/extra.json").write_text("{}")
                return r

            with patch.object(TOOL, "footprint", side_effect=mutate), self.assertRaises(ValueError):
                TOOL.execute(path, digest, root / "mutated.json", True)
            self.assertFalse((root / "mutated.json").exists())


if __name__ == "__main__":
    unittest.main()
