from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

import test_gemma4_native_objectives as fixtures
from test_gemma4_purpose_recipes import load
from kairyu_bench.wsp.weights import editor
from kairyu_bench.wsp.weights.checkpoint import Checkpoint

PREPARE = load("prepare_normalized_objective_plan")


class NormalizedTargetTests(unittest.TestCase):
    def test_prepared_targets_flow_to_existing_bf16_editor_without_native_or_qkv_changes(self):
        fixture = fixtures.NativeCoupledTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        entries = fixture.mixed_fixture()
        plan = fixture.plan
        plan["eligible_blocks"] = ["matrix"]
        cases = {purpose: [[{"id": purpose + "-case", "prompt": "p", "correct": "good", "foil": "bad"}]]
                 for purpose in ("code", "science")}
        fixture.cases.write_text(json.dumps(cases))
        plan["cases"]["sha256"] = fixture.sha(fixture.cases)
        for group in plan["collections"].values():
            for objective, folder in group.items():
                path = Path(folder)
                summary = json.loads((path / "summary.json").read_text())
                seed = summary["folds"][0]
                field, value = (("correct_logprobs", "correct_logprob") if objective == "correct-logprob"
                                else ("margins", "margin"))
                records = []
                for purpose in cases:
                    row = copy.deepcopy(seed)
                    baseline = (-100. if purpose == "code" else -.1) if objective == "correct-logprob" else 0.
                    row.update(purpose=purpose, **{field: [{"id": purpose + "-case", value: baseline}]})
                    if objective == "correct-logprob":
                        row["mean_correct_logprob"] = baseline
                    records.append(row)
                summary["folds"] = records
                (path / "summary.json").write_text(json.dumps(summary))
                provenance = json.loads((path / "provenance.json").read_text())
                provenance["input_sha256"][str(fixture.cases)] = fixture.sha(fixture.cases)
                (path / "provenance.json").write_text(json.dumps(provenance))
                frame = json.loads((path / "frame-evidence.json").read_text())
                frame["frames"] = {purpose: [[{"id": purpose + "-case"}]] for purpose in cases}
                (path / "frame-evidence.json").write_text(json.dumps(frame))
        matrix_ll = Path(plan["collections"]["matrix"]["correct-logprob"])
        gradient = np.load(matrix_ll / "gradient.npy", allow_pickle=False)
        gate = next(e for e in entries if e["family"] == "gate")
        gradient[gate["start"]:gate["start"] + gate["count"]] = 1.
        np.save(matrix_ll / "gradient.npy", gradient)
        plan["candidates"][0]["constraints"] = [
            {"objective": objective, "purpose": purpose, "fold": 0, "minimum_delta": 0.}
            for objective in fixtures.COUPLED.OBJECTIVES for purpose in cases]
        plan["completed_collection_sha256"] = {}
        plan["completed_collection_stat"] = {}
        for group in plan["collections"].values():
            for folder in group.values():
                for path in Path(folder).iterdir():
                    plan["completed_collection_sha256"][str(path)] = fixture.sha(path)
                    plan["completed_collection_stat"][str(path)] = dict(zip(
                        ("device", "inode", "size", "mtime_ns", "ctime_ns"),
                        fixtures.COUPLED.stamp(path), strict=True))
        fixture.plan_path.write_text(json.dumps(plan))
        before = {p: fixture.sha(p) for p in fixture.checkpoint.files}
        out = fixture.root / "prepared"
        with patch.dict(sys.modules, {"propose_native_objectives": fixtures.COUPLED}), \
                patch.object(sys, "path", list(sys.path)), contextlib.redirect_stdout(io.StringIO()):
            PREPARE.main(["--source-plan", str(fixture.plan_path),
                          "--source-plan-sha256", fixture.sha(fixture.plan_path),
                          "--output-dir", str(out), "--candidate-id", "normalized",
                          "--core-purposes", "code", "--science-purposes", "science"])
        prepared = json.loads((out / "plan.json").read_text())
        constraints = prepared["candidates"][0]["constraints"]
        # Q's huge diagnostic derivative must not inflate the selected dual norm.
        self.assertAlmostEqual(constraints[0]["minimum_delta"], np.sqrt(gate["count"]) / 4)
        # The second purpose hits the headroom cap instead of the norm target.
        self.assertAlmostEqual(constraints[1]["minimum_delta"], .01)
        result = fixtures.COUPLED.solve(out / "plan.json", fixture.root / "proposal", write_spec=True)
        self.assertTrue(result["methods"][0]["predicted_constraints_satisfied"])
        self.assertEqual(result["methods"][0]["realized_blocks"]["native"]["changed_elements"], 0)
        candidate, manifest = fixture.root / "saved", fixture.root / "saved.json"
        editor.create(fixture.checkpoint.path, candidate, fixture.root / "proposal/normalized.json",
                      manifest, apply=True, reserve_bytes=0)
        editor.verify(manifest)
        saved = Checkpoint(candidate)
        for name in (fixture.norm, fixture.scalar, next(e["tensor"] for e in entries if e["family"] == "q")):
            self.assertEqual(saved.read_raw(name), fixture.checkpoint.read_raw(name))
        self.assertNotEqual(saved.read_raw(gate["tensor"]), fixture.checkpoint.read_raw(gate["tensor"]))
        self.assertEqual({p: fixture.sha(p) for p in fixture.checkpoint.files}, before)


if __name__ == "__main__":
    unittest.main()
