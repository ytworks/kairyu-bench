from __future__ import annotations

import contextlib
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

with patch.dict(sys.modules, {"propose_native_objectives": fixtures.COUPLED}):
    RADIUS = load("derive_matrix_radius")


class DerivedMatrixRadiusTests(unittest.TestCase):
    def test_calibration_to_saved_candidate_and_foreign_output_preservation(self):
        fixture = fixtures.NativeCoupledTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        entries = fixture.mixed_fixture()
        plan = fixture.plan
        plan["eligible_blocks"] = ["matrix"]
        plan["candidates"][0]["id"] = "derived-matrix"
        margin = Path(plan["collections"]["matrix"]["margin"]) / "gradient.npy"
        values = np.load(margin, allow_pickle=False)
        up = next(e for e in entries if e["family"] == "up")
        values[up["start"]] = 1.
        np.save(margin, values)
        reference = fixture.root / "reference.json"
        reference.write_text(json.dumps(plan))
        plan["reference_plan"] = {"path": str(reference), "sha256": fixture.sha(reference)}
        plan["analytic_calibration"] = {
            "calibration_seconds": 120, "candidate_seconds": 180,
            "fractional_buffer": 1 / 128, "floor": 1e-6,
            "dual_tolerance": 2e-10, "maximum_iterations": 20000,
        }
        plan["completed_collection_sha256"] = {}
        plan["completed_collection_stat"] = {}
        for group in plan["collections"].values():
            for folder in group.values():
                for path in Path(folder).iterdir():
                    plan["completed_collection_sha256"][str(path)] = fixture.sha(path)
                    plan["completed_collection_stat"][str(path)] = dict(zip(
                        ("device", "inode", "size", "mtime_ns", "ctime_ns"), RADIUS.adapter.stamp(path), strict=True))
        fixture.plan_path.write_text(json.dumps(plan))
        before = {p: fixture.sha(p) for p in fixture.checkpoint.files}
        calibration = fixture.root / "calibration"
        argv = ["--plan", str(fixture.plan_path), "--plan-sha256", fixture.sha(fixture.plan_path),
                "--output", str(calibration)]
        with contextlib.redirect_stdout(io.StringIO()):
            with patch.object(RADIUS.adapter.joint, "read_collection", side_effect=AssertionError("preview gradient")):
                RADIUS.main(argv)
            self.assertFalse(calibration.exists())
            RADIUS.main(argv + ["--run"])
        report = json.loads((calibration / "report.json").read_text())
        self.assertTrue(report["continuous_box_contains_solution"])
        self.assertTrue(report["rounded_box_necessary_conditions_all_met"])
        derived = calibration / "candidate-plan.json"
        candidate = fixture.root / "candidate"
        with contextlib.redirect_stdout(io.StringIO()):
            RADIUS.main(["--plan", str(derived), "--plan-sha256", fixture.sha(derived),
                         "--output", str(candidate), "--phase", "candidate", "--run"])
        result = json.loads((candidate / "proposals/report.json").read_text())["methods"][0]
        self.assertTrue(result["predicted_constraints_satisfied"])
        self.assertEqual(result["changed_elements"], 2)
        self.assertEqual(result["realized_blocks"]["native"]["changed_elements"], 0)
        spec_path = candidate / "proposals/derived-matrix.json"
        candidate_path, manifest = fixture.root / "saved", fixture.root / "saved.json"
        editor.create(fixture.checkpoint.path, candidate_path, spec_path, manifest, apply=True, reserve_bytes=0)
        editor.verify(manifest)
        saved = Checkpoint(candidate_path)
        qname = next(e["tensor"] for e in entries if e["family"] == "q")
        for name in (fixture.norm, fixture.scalar, qname):
            self.assertEqual(saved.read_raw(name), fixture.checkpoint.read_raw(name))
        for family in ("gate", "up"):
            name = next(e["tensor"] for e in entries if e["family"] == family)
            self.assertNotEqual(saved.read_raw(name), fixture.checkpoint.read_raw(name))
        self.assertEqual({p: fixture.sha(p) for p in fixture.checkpoint.files}, before)
        # Previously demonstrated owner bug: a rejected existing destination must
        # not gain a failure receipt or have another writer's files overwritten.
        foreign = fixture.root / "foreign"
        foreign.mkdir()
        sentinel = foreign / "started.json"
        sentinel.write_bytes(b'{"owner":"another"}\n')
        with self.assertRaisesRegex(ValueError, "fresh output"):
            RADIUS.main(["--plan", str(fixture.plan_path), "--plan-sha256", fixture.sha(fixture.plan_path),
                         "--output", str(foreign), "--run"])
        self.assertEqual(list(foreign.iterdir()), [sentinel])
        self.assertEqual(sentinel.read_bytes(), b'{"owner":"another"}\n')
        # Same source checkpoint and bounds do not authorize swapping a report
        # whose observed frame/source binding differs.
        altered_report = json.loads((calibration / "report.json").read_text())
        frame_path = str(Path(plan["collections"]["matrix"]["margin"]) / "frame-evidence.json")
        altered_report["source_sha256"][frame_path] = "0" * 64
        bad_report = fixture.root / "different-observation-report.json"
        bad_report.write_text(json.dumps(altered_report))
        bad_plan = json.loads(derived.read_text())
        bad_plan["analytic_report"] = {"path": str(bad_report), "sha256": fixture.sha(bad_report)}
        bad_path = fixture.root / "different-observation-plan.json"
        bad_path.write_text(json.dumps(bad_plan))
        rejected = fixture.root / "rejected-candidate"
        with self.assertRaisesRegex(ValueError, "calibration input/source binding"):
            RADIUS.main(["--plan", str(bad_path), "--plan-sha256", fixture.sha(bad_path),
                         "--output", str(rejected), "--phase", "candidate", "--run"])
        self.assertFalse((rejected / "proposals").exists())


if __name__ == "__main__":
    unittest.main()
