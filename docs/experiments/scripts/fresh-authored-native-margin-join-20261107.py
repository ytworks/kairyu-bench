"""Read-only case join of retained native answers, fixed48 verdicts and F02 margins."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {
        "cases": args.root / "analysis/code90-resume-20261106/long-helper-chain-curriculum-v1/train-cases.json",
        "native": args.repo / "docs/experiments/artifacts/av-authored120-native-v2-result-20261107.json",
        "execution48": args.repo / "docs/experiments/records/2026-11-07-authored48-execution-v2/summary.json",
        "stage120": args.repo / "docs/experiments/records/2026-11-07-authored48-execution-v2/all120-fixed48-stage.json",
    }
    evidence = args.repo / "docs/experiments/evidence-2026-11-07-f02"
    for key, filename in [("base", "finite-v2-replay-base.json"), ("candidate", "finite-v2-replay-fresh-code-joint-f02.json"), ("restored", "finite-v2-replay-restored_base.json"), ("finite_plan", "finite-v2-finite-plan.json"), ("frames", "finite-v2-replay-frame-evidence.json")]:
        inputs[key] = evidence / filename
    pins = {key: {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size} for key, path in inputs.items()}
    d = {key: json.loads(path.read_text()) for key, path in inputs.items()}
    assert pins["cases"]["sha256"] == d["finite_plan"]["cases"]["sha256"]
    assert d["native"]["approved_plan"]["serving"]["tree_sha256"] == d["finite_plan"]["base_tree_sha256"]
    native_rows = [row for batch in d["native"]["batches"] for row in batch["responses"]["responses"]]
    answers = {x["id"]: x for x in native_rows}
    execution = {x["id"]: x for x in d["execution48"]["all48"]}
    stage = {x["id"]: x for x in d["stage120"]}
    assert len(answers) == len(native_rows) == len(stage) == 120 and len(execution) == 48
    rows = []
    for purpose, folds in d["cases"].items():
        assert len(folds) == len(d["base"]["values"][purpose]) == len(d["candidate"]["values"][purpose]) == len(d["frames"]["frames"][purpose])
        for fold, cases in enumerate(folds):
            assert len(cases) == len(d["base"]["values"][purpose][fold]) == len(d["candidate"]["values"][purpose][fold]) == len(d["frames"]["frames"][purpose][fold])
            for index, case in enumerate(cases):
                key = case["id"]
                frame = d["frames"]["frames"][purpose][fold][index]
                assert frame["id"] == key
                answer = answers[key]
                observed = execution.get(key)
                if observed:
                    assert hashlib.sha256(answer["public_answer"].encode()).hexdigest() == observed["public_answer_sha256"]
                    assert observed["passed"] is stage[key]["passed"]
                base = d["base"]["values"][purpose][fold][index]
                candidate = d["candidate"]["values"][purpose][fold][index]
                assert base == d["restored"]["values"][purpose][fold][index]
                literal = {"status": "not_evaluated_non_guard", "semantic_correctness_inferred": False}
                if key.startswith("guard_code_"):
                    value, correct, foil = answer.get("public_answer"), case.get("correct"), case.get("foil")
                    if any(not isinstance(x, str) for x in [value, correct, foil]) or answer.get("error_type"):
                        literal["status"] = "indeterminate_missing_or_error"
                    else:
                        match_correct, match_foil = value == correct, value == foil
                        literal.update({"status": "both_literal_match" if match_correct and match_foil else "correct_literal_match" if match_correct else "foil_literal_match" if match_foil else "neither_literal_match", "correct_equal": match_correct, "foil_equal": match_foil, "comparison": "raw Unicode string equality; no strip, code parsing, execution or semantic equivalence"})
                rows.append({"id": key, "purpose": purpose, "fold": fold, "case_index": index, "teacher_forced_case": case, "teacher_forced_frame": frame, "av_base_margin": base, "f02_candidate_margin": candidate, "f02_margin_delta": candidate - base, "margin_direction": "increase" if candidate > base else "decrease" if candidate < base else "tie", "av_native_public_generation": answer, "fixed48_stage": stage[key], "fixed48_execution": observed, "guard_literal": literal})
    assert len(rows) == len({x["id"] for x in rows}) == 120
    assert {x["id"] for x in rows} == set(answers) == set(stage)
    groups = {}
    for label, selected in [("fixed48_pass", [x for x in rows if x["fixed48_stage"]["passed"] is True]), ("fixed48_fail", [x for x in rows if x["fixed48_stage"]["passed"] is False]), ("outside_fixed48_unscored", [x for x in rows if x["fixed48_stage"]["passed"] is None])]:
        deltas = [x["f02_margin_delta"] for x in selected]
        groups[label] = {"cases": len(selected), "margin_increases": sum(v > 0 for v in deltas), "margin_decreases": sum(v < 0 for v in deltas), "ties": sum(v == 0 for v in deltas), "mean_delta": statistics.mean(deltas), "median_delta": statistics.median(deltas), "min_delta": min(deltas), "max_delta": max(deltas)}
    guards = [x for x in rows if x["id"].startswith("guard_code_")]
    assert len(guards) == 10
    dominant = next(x for x in guards if x["id"] == "guard_code_0_1")
    positives = sum(max(x["f02_margin_delta"], 0) for x in guards)
    net = sum(x["f02_margin_delta"] for x in guards)
    focus = {"id": dominant["id"], "native_public_answer": dominant["av_native_public_generation"]["public_answer"], "correct": dominant["teacher_forced_case"]["correct"], "foil": dominant["teacher_forced_case"]["foil"], "base_margin": dominant["av_base_margin"], "f02_margin": dominant["f02_candidate_margin"], "margin_delta": dominant["f02_margin_delta"], "guard_positive_delta_sum": positives, "guard_net_delta_sum": net, "share_of_guard_positive_delta": dominant["f02_margin_delta"] / positives, "share_of_guard_net_delta": dominant["f02_margin_delta"] / net, "denominator_note": "Both denominators are sums over all 10 guard cases, not total WSP code or solver objective. Net share may exceed 1 because other cases lose margin."}
    for key, path in inputs.items():
        assert sha(path) == pins[key]["sha256"]
    result = {"schema_version": 1, "created_epoch": time.time(), "scope": "Read-only association of retained measurements, no new inference or answer execution", "inputs": pins, "source_sha256": sha(__file__), "joined_cases": 120, "fixed48_cases": 48, "outside_fixed48_unscored": 72, "extra_helper_execution_merged": False, "groups": groups, "guard_literal_counts": {key: sum(x["guard_literal"]["status"] == key for x in guards) for key in ["correct_literal_match", "foil_literal_match", "both_literal_match", "neither_literal_match", "indeterminate_missing_or_error"]}, "dominant_case": focus, "rows": rows, "limitations": ["Teacher-forced margin = mean correct-token logprob minus mean foil-token logprob, forced empty thought frame; it is not native thinking generation accuracy.", "AV native answers and fixed48 execution are from AV, not generated F02 answers; no claim of F02 native improvement or regression follows.", "Literal equality is examined only for 10 primitive guard outputs and is not extended to code semantic correctness or general accuracy.", "The three contract-question cases remain in fixed48 with their original stratum; passing sampled tests does not settle untested contracts.", "All72 outside fixed48 remain unscored here; any later helper16 results are separate.", "These are known authored calibration cases, not unseen benchmarks or official WSP scores.", "This is descriptive case-wise association; case sets and score regimes differ, so no causal correlation or weight-localization conclusion is established."], "new_model_calls": 0, "generated_code_executions": 0, "new_candidate_selection": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ["joined_cases", "fixed48_cases", "outside_fixed48_unscored", "groups", "guard_literal_counts", "dominant_case"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
