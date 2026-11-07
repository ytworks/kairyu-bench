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
POSTCHECK = load("summarize_dependency_set_proposals")


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

        # The analytic solve and persisted BF16 candidate must share the same
        # exact matrix mask. A much stronger outside gradient cannot shrink the
        # derived box, and Euclidean cost must not reuse relative scaling at b=2.
        scoped = fixtures.NativeCoupledTests()
        scoped.setUp()
        self.addCleanup(scoped.doCleanups)
        scoped_entries = scoped.mixed_fixture()
        for objective, directory in scoped.plan["collections"]["native"].items():
            renamed = scoped.root / ("native-" + objective)
            Path(directory).rename(renamed)
            scoped.plan["collections"]["native"][objective] = str(renamed)
        selected = next(e["tensor"] for e in scoped_entries if e["family"] == "q")
        excluded = next(e["tensor"] for e in scoped_entries if e["family"] == "o")
        tensor = scoped.checkpoint.tensors[selected]
        tensors = {name: (item.dtype, item.shape, scoped.checkpoint.read_raw(name))
                   for name, item in scoped.checkpoint.tensors.items()}
        tensors[selected] = ("BF16", tensor.shape, bytes.fromhex("0040") * int(np.prod(tensor.shape)))
        fixtures.weights(scoped.checkpoint.path / "model.safetensors", tensors)
        scoped.checkpoint = Checkpoint(scoped.checkpoint.path)
        identity = scoped.checkpoint.identity()
        attestation = scoped.root / "attestation.json"
        attestation.write_text(json.dumps({"checkpoint": identity}))
        p = scoped.plan
        p["checkpoint"].update(tree_sha256=identity["sha256"],
                               attestation={"path": str(attestation), "sha256": scoped.sha(attestation)})
        p["eligible_blocks"] = ["matrix"]
        p["matrix_tensors"] = [selected]
        p["matrix_families"] = ["q", "k", "o", "gate", "up", "down"]
        p["candidates"][0]["id"] = "euclidean-selected"
        p["candidates"][0]["cost_metric"] = {
            "type": "euclidean", "block_scales": {"native": 1., "matrix": 1.}}
        for family, directories in p["collections"].items():
            for folder in directories.values():
                folder = Path(folder)
                provenance = json.loads((folder / "provenance.json").read_text())
                provenance.update(checkpoint=scoped.checkpoint.identity(False),
                                  attestation_receipt_sha256=scoped.sha(attestation))
                (folder / "provenance.json").write_text(json.dumps(provenance))
                if family != "matrix":
                    continue
                with np.load(folder / "sample-plan.npz", allow_pickle=False) as arrays:
                    coordinates, source = arrays["coordinates"], arrays["source_values"].copy()
                gradient = np.zeros_like(source)
                for entry in scoped_entries:
                    if entry["tensor"] == selected:
                        source[entry["start"]:entry["start"] + entry["count"]] = 2.
                        gradient[entry["start"]] = 1.
                    elif entry["tensor"] == excluded or entry["family"] == "v":
                        gradient[entry["start"]] = 1e6
                np.savez(folder / "sample-plan.npz", coordinates=coordinates, source_values=source)
                np.save(folder / "gradient.npy", gradient)
        parent = scoped.root / "parent.json"
        parent.write_text('{"fixture":true}')
        p["parent_state"] = {"path": str(parent), "sha256": scoped.sha(parent)}
        policy = scoped.root / "functional-policy.json"
        policy.write_text(json.dumps({
            "protocol": "functional_targets_v1", "realized_raw_cap": 1.,
            "parent_state": p["parent_state"], "native_frozen": True,
            "methods": [{"id": p["candidates"][0]["id"], "target_mode": "functional",
                         "cost_metric": p["candidates"][0]["cost_metric"],
                         "matrix_families": p["matrix_families"],
                         "rounding": p["candidates"][0]["rounding"]}]}))
        p["prospective_method"] = {"path": str(policy), "sha256": scoped.sha(policy)}
        p["fixed_method_inputs"] = {str(policy): scoped.sha(policy)}
        reference = scoped.root / "reference.json"
        reference.write_text(json.dumps(p))
        p["reference_plan"] = {"path": str(reference), "sha256": scoped.sha(reference)}
        p["analytic_calibration"] = dict(plan["analytic_calibration"], metric="euclidean")
        p["completed_collection_sha256"], p["completed_collection_stat"] = {}, {}
        for group in p["collections"].values():
            for folder in group.values():
                for path in Path(folder).iterdir():
                    p["completed_collection_sha256"][str(path)] = scoped.sha(path)
                    p["completed_collection_stat"][str(path)] = dict(zip(
                        ("device", "inode", "size", "mtime_ns", "ctime_ns"), RADIUS.adapter.stamp(path), strict=True))
        scoped.plan_path.write_text(json.dumps(p))
        original_hashes = {path: scoped.sha(path) for path in scoped.checkpoint.files}
        out = scoped.root / "calibration"
        with contextlib.redirect_stdout(io.StringIO()):
            RADIUS.main(["--plan", str(scoped.plan_path), "--plan-sha256", scoped.sha(scoped.plan_path),
                         "--output", str(out), "--run"])
        analytic = json.loads((out / "report.json").read_text())
        self.assertEqual(analytic["eligible_matrix_tensors"], [selected])
        self.assertEqual(analytic["matrix_columns"], 1)
        self.assertEqual(analytic["qkv_edit_columns"], 1)
        self.assertAlmostEqual(analytic["continuous_delta_l2"], 1 / 128)
        self.assertAlmostEqual(analytic["continuous_weighted_delta_l2"], 1 / 128)
        self.assertAlmostEqual(analytic["derived_relative_radius"], 3 / 256)
        derived = out / "candidate-plan.json"
        result_dir = scoped.root / "candidate"
        with contextlib.redirect_stdout(io.StringIO()):
            RADIUS.main(["--plan", str(derived), "--plan-sha256", scoped.sha(derived),
                         "--output", str(result_dir), "--phase", "candidate", "--run"])
        spec_path = result_dir / "proposals/euclidean-selected.json"
        spec = json.loads(spec_path.read_text())
        self.assertEqual([edit["tensor"] for edit in spec["edits"]], [selected])
        self.assertEqual(spec["edits"][0]["set_values"], [2.015625])
        saved_path, manifest = scoped.root / "saved", scoped.root / "saved.json"
        editor.create(scoped.checkpoint.path, saved_path, spec_path, manifest, apply=True, reserve_bytes=0)
        editor.verify(manifest)
        saved = Checkpoint(saved_path)
        for name in scoped.checkpoint.tensors:
            if name != selected:
                self.assertEqual(saved.read_raw(name), scoped.checkpoint.read_raw(name))
        self.assertNotEqual(saved.read_raw(selected), scoped.checkpoint.read_raw(selected))
        self.assertEqual({path: scoped.sha(path) for path in scoped.checkpoint.files}, original_hashes)
        roster = scoped.root / "roster.json"
        roster.write_text(json.dumps({"parent": p["parent_state"], "rows": [{"plan": str(derived),
            "plan_sha256": scoped.sha(derived), "output": str(result_dir / "proposals"),
            "ids": ["euclidean-selected"]}]}))
        checked = scoped.root.parent / (scoped.root.name + "-derived-postcheck.json")
        self.addCleanup(checked.unlink, missing_ok=True)
        with patch("sys.argv", ["postcheck", "--roster", str(roster), "--observation-root",
                               str(scoped.root), "--output", str(checked)]), contextlib.redirect_stdout(io.StringIO()):
            POSTCHECK.main()
        realized = json.loads(checked.read_text())["results"][0]
        self.assertTrue(realized["eligible_for_finite"])
        self.assertEqual(realized["cap_metric"], "raw")
        self.assertEqual(realized["raw_delta_l2"], 1 / 64)
        self.assertEqual(realized["actual_changed"], 1)
        mismatch = json.loads(scoped.plan_path.read_text())
        mismatch["matrix_tensors"] = [excluded]
        bad_path = scoped.root / "mismatched-mask.json"
        bad_path.write_text(json.dumps(mismatch))
        with self.assertRaisesRegex(ValueError, "reference matrix mask"):
            RADIUS.main(["--plan", str(bad_path), "--plan-sha256", scoped.sha(bad_path),
                         "--output", str(scoped.root / "rejected-mask"), "--run"])
        self.assertFalse((scoped.root / "rejected-mask").exists())


if __name__ == "__main__":
    unittest.main()
