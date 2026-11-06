"""Paired likelihood/margin fitting preserves objectives, provenance and finite status."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/experiments/gemma4_lora"))
import solve_positive_module_directions as tool  # noqa: E402


class PositiveModuleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        purposes = sorted(tool.PRIMARY | tool.RELATED | {"general", "lookup", "japanese", "referent", "fact_retention"})
        self.labels = [(p, f) for p in purposes for f in range(2)]
        entries = [{"tensor": f"tensor{i}", "coordinate": i, "layer": i % 60,
                    "family": ["q", "k", "v", "o", "gate", "up", "down"][i % 7], "unit_gate_weight_l2": .01}
                   for i in range(410)]
        self.coverage = {"matrices": 410, "direction_coordinates": 410, "entries": entries}
        self.frames = {"frames": {p: [[{"id": f"{p}-{f}", "prefix_sha256": "a"*64}] for f in range(2)] for p in purposes}}
        runtime = {k: "same" for k in ("torch", "transformers", "source_dtype", "factor_compute_dtype", "gates_dtype", "attention", "arithmetic")}
        runtime["max_cases_per_microbatch"] = 1
        self.cost = self.root / "cost.json"
        self.cost.write_text("[]")
        donor = {"payload_sha256": "a"*64, "config_sha256": "b"*64, "repo": "public/donor",
                 "revision": "c"*40, "unit_costs_sha256": tool.legacy.sha(self.cost)}
        self.provenance = {"mode": "chat-public-answer", "checkpoint": {"path": str(self.root / "source")},
                           "attestation_receipt_sha256": "d"*64, "donor": donor, "input_sha256": {"cases": "e"*64},
                           "threads": 1, "seed": 0, "max_cases_per_microbatch": 1}
        for objective, name in (("margin", "margin"), ("correct-logprob", "correct")):
            directory = self.root / name
            directory.mkdir()
            folds = []
            for i, (purpose, fold) in enumerate(self.labels):
                np.save(directory / f"g{i}.npy", np.full(410, 0. if objective == "margin" else 1.))
                folds.append({"purpose": purpose, "fold": fold, "file": f"g{i}.npy"})
            summary = {"status": "completed", "source_values_unchanged": True, "mode": "chat-public-answer",
                       "objective": objective, "coverage": self.coverage, "runtime": runtime, "folds": folds,
                       "zero_alpha_checks": {"every_microbatch_compared": True, "max_absolute_objective_difference": 0, "tolerance": 1e-5}}
            self.save(directory / "summary.json", summary)
            self.save(directory / "coverage.json", self.coverage)
            self.save(directory / "provenance.json", dict(self.provenance, objective=objective))
            self.save(directory / "frame-evidence.json", self.frames)
            np.save(directory / "unit-gate-weight-l2.npy", np.full(410, .01))
        self.margin, self.correct = self.root / "margin", self.root / "correct"
        self.spec = self.root / "scalar.json"
        self.save(self.spec, {"schema_version": 2, "kind": "dense_lora_recipe_v1",
                             "donor_sha256": "a"*64, "donor_config_sha256": "b"*64,
                             "donor_repo": "public/donor", "donor_revision": "c"*40, "coefficient": .25})
        self.review = self.root / "review.json"
        self.review.write_text('{"label":"fixture revision 2"}')
        self.method_review = self.root / "method-review.json"
        self.method_review.write_text('{"scope":"design-only"}')
        self.plan = tool.make_plans(self.margin, self.correct, self.spec, self.cost,
                                   self.review, tool.legacy.sha(self.review), "fixture-v2",
                                   self.method_review, tool.legacy.sha(self.method_review))
        self.plan_path = self.root / "plans.json"
        self.save(self.plan_path, self.plan)

    @staticmethod
    def save(path, value):
        path.write_text(json.dumps(value))

    def test_full_joint_solve_emits_only_continuous_fp32_specs_and_separate_objectives(self):
        report = tool.solve(self.plan_path, self.root / "out")
        self.assertEqual(report["status"], "completed")
        self.assertEqual(len(report["results"]), 6)
        for row in report["results"]:
            self.assertEqual(row["status"], "continuous_proposal")
            self.assertEqual(len(row["constraints"]), 84)
            self.assertEqual([c["objective"] for c in row["constraints"]], ["correct-logprob"]*42 + ["margin"]*42)
            self.assertTrue(np.all(np.asarray(row["predicted_response"]) >= row["minimum_delta"]))
            spec = json.loads(Path(row["spec_path"]).read_text())
            coefs = np.array(list(spec["module_coefficients"].values()))
            self.assertEqual(len(coefs), 410)
            self.assertTrue(np.array_equal(coefs, coefs.astype(np.float32).astype(np.float64)))
            self.assertEqual(spec["qualification"], "continuous_prediction_only")
            self.assertTrue(spec["actual_bf16_metrics_required"])
        with self.assertRaisesRegex(ValueError, "fresh"):
            tool.solve(self.plan_path, self.root / "out")

    def test_swapped_objective_frame_and_source_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "objective"):
            tool.paired_collections(self.correct, self.margin)
        path = self.correct / "frame-evidence.json"
        changed = copy.deepcopy(self.frames)
        changed["frames"]["code"][0][0]["prefix_sha256"] = "f"*64
        self.save(path, changed)
        with self.assertRaisesRegex(ValueError, "frame"):
            tool.paired_collections(self.margin, self.correct)
        self.save(path, self.frames)
        path = self.correct / "provenance.json"
        self.save(path, dict(self.provenance, objective="correct-logprob", attestation_receipt_sha256="f"*64))
        with self.assertRaisesRegex(ValueError, "provenance"):
            tool.paired_collections(self.margin, self.correct)

    def test_modified_review_or_omitted_objective_constraints_reject_before_output(self):
        self.review.write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            tool.solve(self.plan_path, self.root / "out")
        self.assertFalse((self.root / "out").exists())
        self.review.write_text('{"label":"fixture revision 2"}')
        self.plan["plans"][0]["constraints"] = self.plan["plans"][0]["constraints"][:42]
        self.save(self.plan_path, self.plan)
        with self.assertRaisesRegex(ValueError, "84"):
            tool.solve(self.plan_path, self.root / "out")

    def test_retrospective_audit_uses_only_old_folds_and_negates_correct_nll(self):
        old = self.root / "old-margin"
        shutil.copytree(self.margin, old)
        summary = json.loads((old / "summary.json").read_text())
        summary["folds"] = [f for f in summary["folds"] if f["purpose"] not in tool.ADVANCED]
        self.save(old / "summary.json", summary)
        frames = {"frames": {p: f for p, f in self.frames["frames"].items() if p not in tool.ADVANCED}}
        self.save(old / "frame-evidence.json", frames)
        specs = self.root / "specs"
        specs.mkdir()
        path = specs / "old-candidate.json"
        self.save(path, {"module_coefficients": {e["tensor"]: -.0001 for e in self.coverage["entries"]}})
        groups = {p: {"folds": [{"fold": f, "correct_nll_delta": .02} for f in range(2)]} for p in frames["frames"]}
        actual = self.root / "actual.json"
        self.save(actual, {"candidates": [{"id": "old-candidate", "spec_sha256": tool.legacy.sha(path), "groups": groups}]})
        report = tool.audit_old(self.margin, self.correct, old, actual, specs)
        self.assertEqual(report["advanced_v1_rows_used"], 0)
        row = report["candidates"][0]
        self.assertEqual(row["folds"], 34)
        self.assertEqual(row["predicted_correct_ll_negative"], 34)
        self.assertEqual(row["actual_correct_ll_negative"], 34)
        self.assertAlmostEqual(row["actual_correct_ll_mean"], -.02)
        self.assertEqual(row["sign_agreement"], 34)

    def test_headroom_protocol_caps_targets_prunes_duplicate_and_rejects_old_frame(self):
        arguments = (self.margin, self.correct, self.spec, self.cost, self.review,
                     tool.legacy.sha(self.review), "fixture-final", self.method_review, tool.legacy.sha(self.method_review))
        oracle = self.root / "oracle.json"
        self.save(oracle, {"status": "completed", "source_stat_continuity": True,
                          "provenance": {"attestation_receipt_sha256": self.provenance["attestation_receipt_sha256"],
                                         "frame_evidence": {"frame_protocol": tool.FRAME_PROTOCOL, "forced_empty_thought": True}},
                          "rows": [{"id": str(i), "frame": "shared_public_final_boundary",
                                    "official_full_model": {"correct_mean_logprob": -1.},
                                    "custom_native_softcap": {"max_abs_logit_difference_from_official": 0.,
                                                             "mean_logprob_difference_from_official": 0.}} for i in range(3)]})
        with self.assertRaisesRegex(ValueError, "forced-empty-thought"):
            tool.make_plans(*arguments, headroom_aware=True, frame_oracle=oracle, frame_oracle_sha=tool.legacy.sha(oracle))
        for directory in (self.margin, self.correct):
            summary = json.loads((directory / "summary.json").read_text())
            summary["answer_batching"] = "serial"
            summary["runtime"]["gradient_accumulation_dtype"] = "FP32"
            for row in summary["folds"]:
                value = -.002 if row["purpose"] == "code" else -.02 if row["purpose"].startswith("science") else -.1
                row.update(mean_correct_logprob=value, examples=1,
                           correct_logprobs=[{"id": f"{row['purpose']}-{row['fold']}", "correct_logprob": value}])
            self.save(directory / "summary.json", summary)
            self.save(directory / "frame-evidence.json", dict(self.frames, schema_version=2,
                      frame_protocol=tool.FRAME_PROTOCOL, enable_thinking=True, forced_empty_thought=True,
                      public_boundary="<|channel>thought\n<channel|>"))
        plan = tool.make_plans(*arguments, headroom_aware=True, frame_oracle=oracle, frame_oracle_sha=tool.legacy.sha(oracle))
        self.assertEqual(plan["schema_version"], 3)
        by_id = {row["id"]: row for row in plan["plans"]}
        balanced = by_id["donor-positive-balanced"]["constraints"]
        self.assertEqual(next(r["minimum_delta"] for r in balanced if r["purpose"] == "code" and r["objective"] == "correct-logprob"), 0.)
        self.assertEqual(next(r["minimum_delta"] for r in balanced if r["purpose"] == "science_numerics" and r["objective"] == "correct-logprob"), .005)
        self.assertEqual(by_id["donor-positive-science"]["duplicate_of"], "donor-positive-balanced")
        self.save(self.plan_path, plan)
        report = tool.solve(self.plan_path, self.root / "headroom-out")
        self.assertEqual(report["schema_version"], 3)
        self.assertEqual(report["results"][1]["status"], "duplicate_constraints_pruned")
        self.assertTrue(all(r["status"] in {"continuous_proposal", "duplicate_constraints_pruned"} for r in report["results"]))
        bad = json.loads(oracle.read_text())
        bad["rows"][0]["custom_native_softcap"]["mean_logprob_difference_from_official"] = .1
        self.save(oracle, bad)
        with self.assertRaisesRegex(ValueError, "oracle"):
            tool.make_plans(*arguments, headroom_aware=True, frame_oracle=oracle, frame_oracle_sha=tool.legacy.sha(oracle))

    def test_exact_saturation_preservation_does_not_demand_fictitious_positive_gain(self):
        values, report = tool.solve_headroom_coefficients(np.zeros((2, 3)), np.zeros(2), np.ones(3), np.ones(3, bool))
        self.assertTrue(np.array_equal(values, np.zeros(3)))
        self.assertEqual(report["status"], "no_change_solution")
        self.assertEqual(report["minimum_delta"], [0., 0.])
        self.assertEqual(report["numerical_safety_margin"], [0., 0.])

    def test_cpu_timeout_records_unattempted_without_emitting_candidate(self):
        with mock.patch.object(tool.legacy, "solve_coefficients", side_effect=TimeoutError):
            report = tool.solve(self.plan_path, self.root / "out")
        self.assertEqual(report["status"], "time_budget_exhausted")
        self.assertEqual(len(report["unattempted"]), 6)
        self.assertEqual(report["results"], [])
        self.assertEqual(list((self.root / "out").glob("donor-positive-*.json")), [])


if __name__ == "__main__":
    unittest.main()
