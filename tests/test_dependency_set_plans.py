from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import unittest

import numpy as np

import test_gemma4_native_objectives as fixtures
from test_gemma4_purpose_recipes import load

BUILDER = load("prepare_dependency_set_plans")


class DependencyPlanTests(unittest.TestCase):
    def test_derived_scopes_metrics_targets_and_foreign_output(self):
        f = fixtures.NativeCoupledTests("runTest")
        f.setUp()
        self.addCleanup(f.doCleanups)
        entries = f.mixed_fixture()
        core = ["code", "implementation", "workflow", "algorithm"]
        science = [
            "science_numerics",
            "science_contracts",
            "science_advanced_state",
            "science_advanced_counting",
            "science_advanced_probability",
            "science_advanced_simulation",
        ]
        purposes = (
            core
            + science
            + [
                "instruction",
                "tools",
                "lookup",
                "general",
                "reasoning",
                "japanese",
                "referent",
                "fact_retention",
                "tool_call",
                "tool_no_call",
                "tool_arguments",
            ]
        )
        cases = {
            p: [
                [dict(id=f"{p}-{i}", prompt="fixture", correct="good", foil="bad")]
                for i in range(2)
            ]
            for p in purposes
        }
        f.cases.write_text(json.dumps(cases))
        frame = dict(
            schema_version=2,
            frame_protocol="gemma4-public-answer-v2-forced-empty-thought",
            forced_empty_thought=True,
            frames={p: [[dict(id=f"{p}-{i}")] for i in range(2)] for p in purposes},
        )
        outputs = {}
        for family, collections in f.plan["collections"].items():
            for objective, directory in collections.items():
                path = Path(directory)
                summary = json.loads((path / "summary.json").read_text())
                template = summary["folds"][0]
                field, value = (
                    ("correct_logprobs", "correct_logprob")
                    if objective == "correct-logprob"
                    else ("margins", "margin")
                )
                rows = []
                for purpose in purposes:
                    for fold in range(2):
                        baseline = (
                            (-100.0 if purpose in core else -0.01)
                            if objective == "correct-logprob"
                            else 0.0
                        )
                        row = copy.deepcopy(template)
                        row.update(purpose=purpose, fold=fold)
                        row[field] = [dict(id=f"{purpose}-{fold}", **{value: baseline})]
                        row[
                            "mean_correct_logprob"
                            if objective == "correct-logprob"
                            else "mean_margin"
                        ] = baseline
                        rows.append(row)
                summary["folds"] = rows
                (path / "summary.json").write_text(json.dumps(summary))
                provenance = json.loads((path / "provenance.json").read_text())
                provenance["input_sha256"] = {str(f.cases): f.sha(f.cases)}
                (path / "provenance.json").write_text(json.dumps(provenance))
                (path / "frame-evidence.json").write_text(json.dumps(frame))
                if family == "native" and objective == "correct-logprob":
                    np.savez(
                        path / "gradient.npz",
                        **{f.norm: np.array([2.0, 0.0]), f.scalar: np.array(0.0)},
                    )
                if family == "matrix" and objective == "margin":
                    gradient = np.load(path / "gradient.npy", allow_pickle=False)
                    gradient[
                        next(e["start"] for e in entries if e["family"] == "up")
                    ] = 2.0
                    np.save(path / "gradient.npy", gradient)
                outputs[family + "-" + objective] = dict(
                    path=str(path), files={p.name: f.sha(p) for p in path.iterdir()}
                )
        observation = f.root / "observation"
        observation.mkdir()
        parent = f.root / "parent.json"
        parent.write_text('{"fixture": true}')
        parent_ref = dict(
            path=str(parent),
            sha256=f.sha(parent),
            tree=f.plan["checkpoint"]["tree_sha256"],
        )
        observation_plan = dict(
            parent=parent_ref,
            checkpoint_attestation=f.plan["checkpoint"]["attestation"],
            cases=dict(path=str(f.cases), sha256=f.sha(f.cases)),
        )
        for name, data in (
            ("plan.json", observation_plan),
            ("completion.json", dict(status="completed", same_container_restored=True)),
            ("collector-output-hashes.json", dict(outputs=outputs)),
            ("postchecks.json", dict(fixture=True)),
            ("independent-numeric-review.json", dict(fixture=True)),
        ):
            (observation / name).write_text(json.dumps(data))
        review = f.root / "review.json"
        review.write_text(
            json.dumps(
                dict(
                    verified_requested_model=True,
                    review_usable=True,
                    labels_match=True,
                    labels="ABCDEFGH",
                    tool_uses=0,
                    assistant_models=["claude-opus-5-5"],
                )
            )
        )
        review.with_name("method-disposition.json").write_text('{"fixture": true}')
        output = f.root / "output"
        args = argparse.Namespace(
            repo=Path(__file__).resolve().parents[1],
            observation=observation,
            output=output,
            parent_sha=f.sha(parent),
            review=review,
            review_sha=f.sha(review),
        )
        before = {p: f.sha(p) for p in f.checkpoint.files}
        BUILDER.build(args)
        plans = {
            label: json.loads((output / "plans" / f"{label}.json").read_text())
            for label in "ABCDEFGH"
        }

        def target(label, objective, purpose):
            return next(
                r["minimum_delta"]
                for r in plans[label]["candidates"][0]["constraints"]
                if (r["objective"], r["purpose"], r["fold"]) == (objective, purpose, 0)
            )

        prep = json.loads((output / "preparation.json").read_text())
        self.assertAlmostEqual(prep["F_scales"]["native"], 2.0)
        self.assertAlmostEqual(prep["F_scales"]["matrix"], np.sqrt(1e12 + 1.0))
        self.assertAlmostEqual(
            target("F", "correct-logprob", "code"), 0.25 * np.sqrt(2.0)
        )
        self.assertGreater(
            target("E", "correct-logprob", "code"),
            target("D", "correct-logprob", "code"),
        )
        self.assertGreater(target("D", "correct-logprob", science[0]), 0.0)
        self.assertLess(target("G", "correct-logprob", science[0]), 0.0)
        self.assertAlmostEqual(target("H", "margin", "code"), 0.5)
        self.assertAlmostEqual(target("H", "correct-logprob", "code"), 0.0625)
        self.assertNotIn("block_scales", plans["A"]["candidates"][0]["cost_metric"])
        self.assertEqual(plans["B"]["matrix_families"], ["q", "k"])
        self.assertEqual({p: f.sha(p) for p in f.checkpoint.files}, before)
        self.assertFalse((output / "proposals").exists())
        saved = {
            p.relative_to(output): f.sha(p) for p in output.rglob("*") if p.is_file()
        }
        with self.assertRaisesRegex(ValueError, "fresh output"):
            BUILDER.build(args)
        self.assertEqual(
            saved,
            {p.relative_to(output): f.sha(p) for p in output.rglob("*") if p.is_file()},
        )
