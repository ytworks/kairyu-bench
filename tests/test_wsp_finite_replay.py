from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from kairyu_bench.wsp.weights.checkpoint import Checkpoint
from kairyu_bench.wsp.weights.finite_replay import batches, load_plan, replay_sequence
from test_wsp_weights import weights


class ReplayTests(unittest.TestCase):
    def test_base_candidates_base_order_and_failure_restore(self) -> None:
        calls = []
        def apply(candidate):
            calls.append(("apply", None if candidate is None else candidate["id"]))
        def evaluate(**flags):
            calls.append(("evaluate", flags))
            return {"value": len(calls)}
        baseline, results, repeat, failure = replay_sequence(
            [{"id": "one"}, {"id": "two"}], apply, evaluate,
            lambda name, result: calls.append(("save", name)))
        self.assertIsNotNone(baseline)
        self.assertEqual(len(results), 2)
        self.assertIsNotNone(repeat)
        self.assertIsNone(failure)
        self.assertEqual([c[1] for c in calls if c[0] == "apply"], [None, "one", "two", None])
        calls.clear()
        def broken(**flags):
            if not flags.get("store_base") and not flags.get("recovery"):
                raise MemoryError("synthetic allocation error")
            return {"base": True}
        _, results, repeat, failure = replay_sequence(
            [{"id": "one"}, {"id": "two"}], apply, broken, lambda *args: None)
        self.assertEqual(results, [])
        self.assertIsNotNone(repeat)
        self.assertIsInstance(failure, MemoryError)
        self.assertIsNone(failure.__traceback__)
        self.assertEqual([c[1] for c in calls if c[0] == "apply"], [None, "one", None])

    def test_initial_base_failure_and_writer_failure_restore(self) -> None:
        calls = []
        def evaluate(**flags):
            calls.append(flags)
            if not flags.get("recovery"):
                raise RuntimeError("initial base failure")
            return {"base": "restored"}
        baseline, _, repeat, failure = replay_sequence([], lambda _: None, evaluate, lambda *args: None)
        self.assertIsNone(baseline)
        self.assertIsNotNone(repeat)
        self.assertIsInstance(failure, RuntimeError)
        self.assertEqual(calls[-1], {"store_base": True, "recovery": True})

    def test_plan_hash_coordinates_and_output_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "checkpoint"
            weights(model / "model.safetensors", {
                "model.language_model.norm.weight": ("BF16", (2,), b"\x80\x3f\x00\x40"),
            })
            index = model / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"model.language_model.norm.weight": "model.safetensors"}}))
            case, frame = root / "cases.json", root / "frames.json"
            case.write_text("{}")
            frame.write_text("{}")
            spec = root / "spec.json"
            document = {"id": "candidate", "edits": [{"tensor": "model.language_model.norm.weight",
                "indices": [0], "set_values": [1.0078125]}]}
            spec.write_text(json.dumps(document))
            def sha(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()
            attestation = root / "attestation.json"
            attestation.write_text(json.dumps(Checkpoint(model).identity()))
            plan = {"schema_version": 1, "architecture": "gemma4_native_bf16",
                "checkpoint_index_sha256": sha(index),
                "checkpoint_attestation": {"path": str(attestation), "sha256": sha(attestation)},
                "cases": {"path": str(case), "sha256": sha(case)},
                "frame_evidence": {"path": str(frame), "sha256": sha(frame)},
                "candidates": [{"id": "candidate", "spec": str(spec), "sha256": sha(spec)}]}
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))
            _, specs, _ = load_plan(plan_path, Checkpoint(model))
            self.assertEqual(specs[0]["_verified_metrics"]["delta_l2"], 1 / 128)
            for value in [1.001, 1.0, float("nan")]:
                document["edits"][0]["set_values"] = [value]
                spec.write_text(json.dumps(document))
                plan["candidates"][0]["sha256"] = sha(spec)
                plan_path.write_text(json.dumps(plan))
                with self.assertRaises(ValueError):
                    load_plan(plan_path, Checkpoint(model))
            document["id"] = "../escape"
            document["edits"][0]["set_values"] = [1.0078125]
            spec.write_text(json.dumps(document))
            plan["candidates"][0].update(id="../escape", sha256=sha(spec))
            plan_path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                load_plan(plan_path, Checkpoint(model))
            plan["candidates"][0]["sha256"] = "0" * 64
            plan_path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                load_plan(plan_path, Checkpoint(model))

    def test_batches_preserve_matched_order_and_bound(self) -> None:
        rows = [("long", [1] * 700, [2] * 10, [3] * 5),
                ("short", [1] * 10, [2] * 2, [3] * 5)]
        packed = batches(rows)
        self.assertEqual([row[0] for batch in packed for row in batch], ["short", "long"])
        self.assertEqual(len(packed), 2)


if __name__ == "__main__":
    unittest.main()

