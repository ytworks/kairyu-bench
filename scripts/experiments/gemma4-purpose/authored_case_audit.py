#!/usr/bin/env python3
"""Audit the frozen authored 120-case F01/F02 curriculum; no model or network calls."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def summarize(values):
    return dict(min=min(values), median=statistics.median(values), max=max(values), sum=sum(values))


def concentration(rows):
    gains = [r for r in rows if r["margin_delta"] > 0]
    top = max(rows, key=lambda r: r["margin_delta"])
    positive_sum = sum(r["margin_delta"] for r in gains)
    net = sum(r["margin_delta"] for r in rows)
    return dict(count=len(rows), gains=len(gains), losses=sum(r["margin_delta"] < 0 for r in rows),
                ties=sum(r["margin_delta"] == 0 for r in rows),
                net_sum=net, mean=net / len(rows), positive_sum=positive_sum,
                top_id=top["id"], top_delta=top["margin_delta"],
                top_positive_sum_share=top["margin_delta"] / positive_sum if positive_sum else None,
                remaining_net_sum=net-top["margin_delta"],
                pair_sign_changes=[r["id"] for r in rows
                                   if (r["base_margin"] > 0) != (r["candidate_margin"] > 0)])


def run_git(argv, cwd, env):
    result = subprocess.run(["git", *argv], cwd=cwd, env=env, capture_output=True,
                            text=True, timeout=20, check=True)
    return result.stdout.strip()


def demonstrations(cases, output):
    """Execute only the explicitly reviewed snippets below, in bounded CPU fixtures."""
    def function(case_id):
        namespace = {}
        exec(compile(cases[case_id]["correct"], case_id, "exec"), namespace)
        return next(value for name, value in namespace.items()
                    if name in ("solve", case_id))

    demos = {}
    closed = []

    class Handle:
        def close(self):
            closed.append(True)
            raise RuntimeError("close failed")

    def action(_handle):
        raise ValueError("action failed")

    try:
        function("call_and_close")(Handle(), action)
    except Exception as error:
        demos["call_and_close"] = dict(raised=type(error).__name__,
                                      chained_context=type(error.__context__).__name__, close_calls=len(closed))
    demos["segment_count"] = dict(total=0, capacity=0.5, returned=function("segment_count")(0, 0.5),
                                  contract_zero_result=0)
    demos["best_schedule"] = dict(jobs=[[1, 1, 5], [0, 2, 10]],
                                  returned=function("best_schedule")([(1, 1, 5), (0, 2, 10)]),
                                  mathematical_empty_interval_optimum=15)
    for case_id, arguments, expected in (
        ("science-cumulative-trapezoid", (np.array([]), np.array([])), [0]),
        ("science-advanced-reset-running-stats", (np.array([]), np.array([], dtype=bool)), [0, 2]),
        ("science-advanced-capped-reservoir-chain", (0, np.array([]), np.array([]), 1), [0, 3]),
    ):
        answer = function(case_id)(*arguments)
        demos[case_id] = dict(actual_shape=list(answer.shape), contract_shape=expected, domain="empty arrays")
    for case_id, arguments in (
        ("chain-whitened-fit-1", (np.eye(2, dtype=np.float32), np.ones(2, dtype=np.float32),
                                  np.ones(2, dtype=np.float32))),
        ("chain-quadrature-mixture-1", (np.array([0, 1], dtype=np.float32),
                                       np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32))),
        ("chain-boundary-response-1", (np.eye(2, dtype=np.int64), np.array([2, 3]),
                                       np.array([0]), np.array([1]))),
    ):
        answer = function(case_id)(*arguments)
        demos[case_id] = dict(actual_dtype=str(answer.dtype), contract_dtype="float64",
                              shape=list(answer.shape))
    for case_id, filename in (("commit_only_fix", "app.py"), ("record_verified_rename", "legacy.py")):
        directory = output / case_id
        directory.mkdir()
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null")
        run_git(["init", "-q"], directory, env)
        run_git(["config", "user.name", "Authored case audit"], directory, env)
        run_git(["config", "user.email", "audit@example.invalid"], directory, env)
        run_git(["config", "core.hooksPath", "/dev/null"], directory, env)
        (directory / filename).write_text("print('initial')\n")
        (directory / "notes.txt").write_text("original notes\n")
        (directory / "test_fixture.py").write_text("import unittest\nclass Smoke(unittest.TestCase):\n    def test_fixture(self): self.assertTrue(True)\n")
        run_git(["add", filename, "notes.txt"], directory, env)
        run_git(["commit", "-q", "-m", "fixture"], directory, env)
        (directory / "notes.txt").write_text("unrelated staged edit\n")
        run_git(["add", "notes.txt"], directory, env)
        if case_id == "commit_only_fix":
            (directory / filename).write_text("print('intended fix')\n")
        result = subprocess.run(["sh", "-c", cases[case_id]["correct"]], cwd=directory, env=env,
                                capture_output=True, text=True, timeout=30, check=False)
        (directory / "command-receipt.json").write_text(json.dumps(dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr), indent=2)+"\n")
        assert result.returncode == 0
        names = run_git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"], directory, env).splitlines()
        assert "notes.txt" in names
        demos[case_id] = dict(unrelated_edit_pre_staged=True, exit_code=result.returncode,
                              committed_paths=names, unrelated_notes_were_committed=True,
                              fixture_directory=str(directory))
    return demos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "New output required; preserve earlier audits."
    args.output.mkdir(parents=True)
    root = args.project
    case_path = root / "analysis/code90-resume-20261106/long-helper-chain-curriculum-v1/train-cases.json"
    before = sha(case_path)
    assert before == "bd6fcba70e93b1870ee882f6e92893b93490ad4099af7a9a290ba3d324d68951"
    dataset = read(case_path)
    replay = root / "analysis/fresh-purpose-20261107-f02/finite-gpu6-v2/work/replay"
    frame = read(replay / "frame-evidence.json")
    inputs = {str(case_path): before, str(replay / "frame-evidence.json"): sha(replay / "frame-evidence.json")}
    by_id, metadata = {}, []
    for purpose, folds in dataset.items():
        assert len(folds) == 2
        for fold, cases in enumerate(folds):
            for index, case in enumerate(cases):
                assert case["id"] not in by_id
                by_id[case["id"]] = case
                f = frame["frames"][purpose][fold][index]
                assert f["id"] == case["id"]
                executable = purpose in ("implementation", "algorithm") or purpose.startswith("science")
                ast_equal = (ast.dump(ast.parse(case["correct"])) == ast.dump(ast.parse(case["foil"]))
                             if executable else None)
                metadata.append(dict(id=case["id"], purpose=purpose, fold=fold, case_index=index,
                                     prompt_chars=len(case["prompt"]), correct_chars=len(case["correct"]),
                                     foil_chars=len(case["foil"]), **{k: f[k] for k in
                                     ("prefix_tokens", "correct_tokens", "foil_tokens")},
                                     stripped_answers_equal=case["correct"].strip() == case["foil"].strip(),
                                     executable_ast_equal=ast_equal))
    assert len(metadata) == 120 and len(dataset) == 19
    joined = {}
    for name, directory in (("f01", "fresh-purpose-20261107-f01/finite-gpu6-v4"),
                             ("f02", "fresh-purpose-20261107-f02/finite-gpu6-v2")):
        path = root / "analysis" / directory / "numeric-analysis.json"
        numeric = read(path)
        inputs[str(path)] = sha(path)
        lookup = {(x["purpose"], x["fold"], x["case_index"]): x for x in numeric["cases"]}
        rows = [dict(m, **{k: lookup[(m["purpose"], m["fold"], m["case_index"])][k]
                              for k in ("base_margin", "candidate_margin", "margin_delta")}) for m in metadata]
        joined[name] = dict(all_cases=rows, descriptive_all_case_concentration=concentration(rows),
                           purposes={p: concentration([x for x in rows if x["purpose"] == p]) for p in dataset},
                           fold_margin_means={p: [statistics.mean(x["margin_delta"] for x in rows
                                                                if x["purpose"] == p and x["fold"] == f)
                                                 for f in (0, 1)] for p in dataset})
    frame_paths = [
        root / "analysis/code90-resume-20261106/science-gpu-runs/provisional-av-long-purpose-v3" / mode / "frame-evidence.json"
        for mode in ("matrix-correct-logprob", "matrix-margin")
    ] + [root / "analysis/fresh-purpose-20261107-f01/finite-gpu6-v4/work/replay/frame-evidence.json"]
    frame_matches = []
    for path in frame_paths:
        assert read(path) == frame
        inputs[str(path)] = sha(path)
        frame_matches.append(dict(path=str(path), exact_equal=True))
    demos = demonstrations(by_id, args.output)
    assert sha(case_path) == before
    output = dict(
        schema_version=1, scope="Post-freeze authored-case semantics/length audit, no new candidate or model evaluation.",
        cases=120, purposes=19, folds=2, original_cases_sha256=before, original_unchanged=True,
        literal_equal_answers=sum(x["stripped_answers_equal"] for x in metadata),
        executable_equal_ast_pairs=[x["id"] for x in metadata if x["executable_ast_equal"]],
        duplicate_stripped_prompts=120-len({x["prompt"].strip() for x in by_id.values()}),
        token_lengths={k: summarize([x[k] for x in metadata])
                       for k in ("prefix_tokens", "correct_tokens", "foil_tokens")},
        max_complete_frame_tokens=max(x["prefix_tokens"]+max(x["correct_tokens"], x["foil_tokens"]) for x in metadata),
        unequal_answer_token_lengths=sum(x["correct_tokens"] != x["foil_tokens"] for x in metadata),
        purpose_fold_counts={p: [len(f) for f in fs] for p, fs in dataset.items()},
        frame_protocol={k: v for k, v in frame.items() if k != "frames"},
        exact_frame_matches=frame_matches, finite_results=joined, bounded_cpu_demonstrations=demos,
        per_case_correct_ll_available=False,
        ll_limitation="Finite replay persists correct NLL only per fold; single-case LL contribution cannot be reconstructed.",
        concentration_limitation="Case-delta sums are descriptive, not the 76-constraint objective, semantic accuracy or WSP scores.",
        inputs_sha256=inputs, script_sha256=sha(Path(__file__)), numpy_version=np.__version__)
    path = args.output / "facts.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(output=str(path), sha256=sha(path), counts=output["purpose_fold_counts"],
                          demonstrations=demos, f02=joined["f02"]["descriptive_all_case_concentration"])))
    

if __name__ == "__main__":
    main()
