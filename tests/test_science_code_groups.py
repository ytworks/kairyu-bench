from __future__ import annotations

import argparse
import copy
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

import test_gemma4_native_objectives as fixtures
from test_gemma4_purpose_recipes import load
from kairyu_bench.wsp.weights import editor
from kairyu_bench.wsp.weights.checkpoint import Checkpoint

BUILDER = load("prepare_science_code_groups")
POSTCHECK = load("summarize_dependency_set_proposals")


class ScienceCodeGroupTest(unittest.TestCase):
    def test_agreement_scope_targets_and_saved_bytes_share_the_same_columns(self):
        f = fixtures.NativeCoupledTests("runTest")
        f.setUp()
        self.addCleanup(f.doCleanups)
        entries = f.mixed_fixture()
        for objective, directory in f.plan["collections"]["native"].items():
            old_path = Path(directory)
            new_path = f.root / ("native-" + objective)
            old_path.rename(new_path)
            f.plan["collections"]["native"][objective] = str(new_path)
        output_entries = sorted(
            (e for e in entries if e["family"] == "o"), key=lambda e: e["layer"]
        )
        selected, excluded = output_entries
        purposes = ["science", "code"]
        cases = {
            p: [[dict(id=f"{p}-{i}", prompt="p", correct="c", foil="f")]
                for i in range(2)]
            for p in purposes
        }
        f.cases.write_text(json.dumps(cases))
        f.plan["cases"]["sha256"] = f.sha(f.cases)
        frame = dict(
            schema_version=2,
            frame_protocol="gemma4-public-answer-v2-forced-empty-thought",
            forced_empty_thought=True,
            frames={p: [[dict(id=f"{p}-{i}")] for i in range(2)] for p in purposes},
        )
        constraints = [
            dict(objective=o, purpose=p, fold=i, minimum_delta=0.)
            for o in BUILDER.adapter.OBJECTIVES
            for p in purposes for i in range(2)
        ]
        f.plan["candidates"][0]["constraints"] = constraints
        # Distinct science/code folds agree in layer0 but oppose each other
        # through a much stronger gradient in layer1 of the same O family.
        for family, collections in f.plan["collections"].items():
            for objective, directory in collections.items():
                path = Path(directory)
                summary = json.loads((path / "summary.json").read_text())
                row_template = summary["folds"][0]
                field, value = (
                    ("correct_logprobs", "correct_logprob")
                    if objective == "correct-logprob" else ("margins", "margin")
                )
                rows = []
                for purpose in purposes:
                    for fold in range(2):
                        row = copy.deepcopy(row_template)
                        baseline = -1. if objective == "correct-logprob" else 0.
                        row.update(purpose=purpose, fold=fold)
                        row[field] = [dict(id=f"{purpose}-{fold}", **{value: baseline})]
                        row["mean_correct_logprob" if objective == "correct-logprob"
                            else "mean_margin"] = baseline
                        if family == "matrix":
                            gradient = np.zeros(sum(e["count"] for e in entries))
                            gradient[selected["start"]] = 1.
                            gradient[excluded["start"]] = 100. if purpose == "science" else -100.
                            row["file"] = f"{purpose}-{fold}.npy"
                            np.save(path / row["file"], gradient)
                        rows.append(row)
                summary["folds"] = rows
                (path / "summary.json").write_text(json.dumps(summary))
                provenance = json.loads((path / "provenance.json").read_text())
                provenance["input_sha256"] = {str(f.cases): f.sha(f.cases)}
                (path / "provenance.json").write_text(json.dumps(provenance))
                (path / "frame-evidence.json").write_text(json.dumps(frame))
        parent = f.root / "parent.json"
        parent.write_text('{"fixture":true}')
        parent_ref = dict(path=str(parent), sha256=f.sha(parent))
        f.plan["parent_state"] = parent_ref
        f.plan["completed_collection_sha256"] = {
            str(path): f.sha(path)
            for collections in f.plan["collections"].values()
            for directory in collections.values()
            for path in Path(directory).iterdir() if path.is_file()
        }
        f.plan_path.write_text(json.dumps(f.plan))
        review = f.root / "review.json"
        review.write_text(json.dumps(dict(
            verified_requested_model=True, review_usable=True,
            assistant_models=["claude-opus-5-5"], tool_uses=0,
        )))
        method = dict(
            id="joint-group", selection="fold_agreement",
            science_purposes=["science"], code_purposes=["code"],
            additional_primary_purposes=[],
            matrix_families=["o", "gate", "up", "down"],
            relative_floor=1e-6, ll_multiplier=1/64,
            margin_multiplier=1/64, ll_headroom_fraction=.1,
            retention_lower=-.002,
            rounding=dict(relative_radius=.025, absolute_radius=1e-6, rounds=16),
        )
        policy = f.root / "policy.json"
        policy.write_text(json.dumps(dict(
            parent_state=parent_ref, native_frozen=True, realized_weighted_cap=2.,
            review=dict(path=str(review), sha256=f.sha(review)),
            methods=[method],
        )))
        output = f.root / "prepared"
        args = argparse.Namespace(
            template=f.plan_path, policy=policy, policy_sha=f.sha(policy),
            output=output,
        )
        before = {p: f.sha(p) for p in f.checkpoint.files}
        result = BUILDER.build(args)
        self.assertEqual(result["skipped"], 0)
        self.assertEqual(len(result["plans"]), 1)
        prepared = Path(result["plans"][0]["plan"])
        plan = json.loads(prepared.read_text())
        report = json.loads((output / "preparation.json").read_text())["reports"][0]
        expected = {e["tensor"] for e in entries
                    if e["layer"] == 0 and e["family"] in method["matrix_families"]}
        self.assertEqual(set(plan["matrix_tensors"]), expected)
        self.assertEqual(plan["eligible_blocks"], ["matrix"])
        self.assertEqual([g["selected"] for g in report["groups"]], [True, False])
        for row in report["targets"]:
            self.assertEqual(row["dual_norm"], 1.)
            self.assertEqual(row["minimum_delta"], 1/64)
        proposal = f.root / "proposal"
        solved = BUILDER.adapter.solve(prepared, proposal, write_spec=True)
        actual = solved["methods"][0]
        self.assertTrue(actual["predicted_constraints_satisfied"])
        self.assertEqual(actual["realized_blocks"]["native"]["changed_elements"], 0)
        spec = proposal / "joint-group.json"
        edits = json.loads(spec.read_text())["edits"]
        self.assertEqual([edit["tensor"] for edit in edits], [selected["tensor"]])
        def postcheck(plan_path, proposal_path, label):
            roster = f.root / (label + "-roster.json")
            roster.write_text(json.dumps(dict(
                parent=parent_ref, rows=[dict(plan=str(plan_path),
                    plan_sha256=f.sha(plan_path), output=str(proposal_path),
                    ids=["joint-group"])],
            )))
            checked = f.root / (label + "-postcheck.json")
            argv = ["postcheck", "--roster", str(roster), "--observation-root",
                    str(f.root), "--output", str(checked)]
            # The public CLI requires output outside the observation directory.
            outside = f.root.parent / (f.root.name + "-" + label + ".json")
            self.addCleanup(outside.unlink, missing_ok=True)
            argv[-1] = str(outside)
            with mock.patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                POSTCHECK.main()
            return json.loads(outside.read_text())["results"][0]

        accepted = postcheck(prepared, proposal, "cap-two")
        self.assertTrue(accepted["eligible_for_finite"])
        self.assertEqual(accepted["declared_weighted_delta_l2"], 1/64)
        self.assertEqual(accepted["metric_cap"], 2.)
        # A separately frozen lower-cap policy must reject the same real BF16
        # step. This protects the behavior, not just a displayed cap value.
        lower_policy = f.root / "lower-policy.json"
        document = json.loads(policy.read_text())
        document["realized_weighted_cap"] = 1/128
        lower_policy.write_text(json.dumps(document))
        lower_args = argparse.Namespace(template=f.plan_path, policy=lower_policy,
            policy_sha=f.sha(lower_policy), output=f.root / "lower-prepared")
        lower_result = BUILDER.build(lower_args)
        lower_plan = Path(lower_result["plans"][0]["plan"])
        lower_proposal = f.root / "lower-proposal"
        BUILDER.adapter.solve(lower_plan, lower_proposal, write_spec=True)
        rejected = postcheck(lower_plan, lower_proposal, "cap-small")
        self.assertFalse(rejected["eligible_for_finite"])
        self.assertEqual(rejected["declared_weighted_delta_l2"], 1/64)
        self.assertEqual(rejected["metric_cap"], 1/128)
        saved_path, manifest = f.root / "saved", f.root / "saved.json"
        editor.create(f.checkpoint.path, saved_path, spec, manifest,
                      apply=True, reserve_bytes=0)
        editor.verify(manifest)
        saved = Checkpoint(saved_path)
        for name in f.checkpoint.tensors:
            self.assertEqual(saved.read_raw(name) == f.checkpoint.read_raw(name),
                             name != selected["tensor"])
        self.assertEqual({p: f.sha(p) for p in f.checkpoint.files}, before)
        self.assertEqual(saved.read_raw(f.scalar), bytes.fromhex("0080"))
        # A second publication cannot replace the first immutable prepared plans.
        with self.assertRaisesRegex(ValueError, "fresh nonoverlapping"):
            BUILDER.build(args)


if __name__ == "__main__":
    unittest.main()
