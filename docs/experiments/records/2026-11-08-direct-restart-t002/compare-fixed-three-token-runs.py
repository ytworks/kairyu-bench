#!/usr/bin/env python3
"""T002 fixed-three-run evidence comparison. No model/API/service operations."""
import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
RESTART = ROOT / "restart-direct-20261108"
TRIAL = RESTART / "trial-002-final-branch-ablation"
O2 = RESTART / "observation-002-token-logprobs"
RUN_IDS = [f"restart-direct-arc-batch01-candidate-t002-r{i}-20261108" for i in (1, 2, 3)]
ORIGINAL_IDS = [
    "restart-direct-arc-batch01-off-control-20261108",
    "restart-direct-arc-batch01-off-control2-20261108",
    "restart-direct-arc-batch01-off-control3-20261108",
    "restart-direct-o002-token-logprobs-r1-20261108",
    "restart-direct-o002-token-logprobs-r2-20261108",
]
WRAPPER_SHA = "aab6c223eaeb940fbbd4acb156f3dbb94c03e74bc89885a8ce9cd6cddf78c8b8"
MANIFEST_SHA = "a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b"
EVIDENCE, ERRORS = {}, []


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path, lines=False):
    raw = path.read_bytes()
    EVIDENCE[str(path)] = {"sha256": sha(raw), "bytes": len(raw)}
    return [json.loads(x) for x in raw.splitlines() if x.strip()] if lines else json.loads(raw)


def check(condition, code, **where):
    if not condition:
        ERRORS.append({"code": code, **where})
    return condition


def keyed(rows, key):
    out = {key(x): x for x in rows}
    check(len(out) == len(rows), "duplicate_records")
    return out


def timestamp(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_run(run_id, ids, prompts):
    base = ROOT / "results" / run_id
    run = read(base / "run.json")
    items = keyed(read(base / "items.jsonl", True), lambda x: x["id"])
    responses = keyed(read(base / "responses.jsonl", True), lambda x: x["id"])
    check(run["status"] == "completed", "run_not_completed", run_id=run_id)
    check(run.get("tier") == 10 and run.get("split") == "dev", "tier_split", run_id=run_id)
    check(run.get("manifest_sha256") == MANIFEST_SHA, "manifest", run_id=run_id)
    check(set(items) == set(responses) == set(ids), "item_set", run_id=run_id)
    check(not run.get("unscored"), "unscored", run_id=run_id)
    values = {}
    for item_id in ids:
        item, response = items[item_id], responses[item_id]
        check(item.get("error") is None and response.get("error") is None, "item_error", run_id=run_id, item_id=item_id)
        outcome = item["outcome"]
        check(outcome == response["outcome"], "response_item_outcome", run_id=run_id, item_id=item_id)
        ll, lengths, gold = outcome["logliks"], outcome["lengths"], outcome["gold"]
        row = prompts[item_id]
        check(lengths == list(map(len, row["choices"]["text"])), "character_denominator", run_id=run_id, item_id=item_id)
        check(row["choices"]["label"][gold] == row["answerKey"], "gold_reference", run_id=run_id, item_id=item_id)
        check(len(ll) == len(lengths) == 4 and all(math.isfinite(x) for x in ll), "choice_array", run_id=run_id, item_id=item_id)
        norm = [x / n for x, n in zip(ll, lengths)]
        predicted = max(range(4), key=lambda i: norm[i])
        score = float(predicted == gold)
        check(item["score"] == score and item["detail"]["predicted"] == predicted and item["detail"]["gold"] == gold,
              "official_argmax_score", run_id=run_id, item_id=item_id)
        margin = norm[gold] - max(v for i, v in enumerate(norm) if i != gold)
        values[item_id] = {"logliks": ll, "lengths": lengths, "gold": gold, "normalized_logliks": norm,
                           "margin": margin, "predicted": predicted, "score": item["score"]}
    return {
        "run_id": run_id, "status": run["status"], "items": values,
        "elapsed_timestamp_seconds": (timestamp(run["finished_at"]) - timestamp(run["started_at"])).total_seconds(),
        "official_timing": run.get("timing"), "unscored": run.get("unscored", []),
        "failed_items": [x for x, item in items.items() if item.get("error")],
    }


def load_capture(path, run, ids, prompts, source_hashes=None):
    raw = read(path)
    run_id = run["run_id"]
    check(path.stat().st_size <= 2 * 1024**2, "capture_capacity", run_id=run_id)
    check(raw["capture_complete"] is True and raw["capture_errors"] == {} and raw["official_returncode"] == 0,
          "capture_incomplete", run_id=run_id)
    check(raw["recorded_calls"] == raw["finished_call_records"] == raw["expected_calls"] == 40,
          "capture_call_count", run_id=run_id)
    check(raw["logical_client_calls"] == {"prompt_logprobs": 40, "tokenize": 81}, "logical_call_count", run_id=run_id)
    check(raw["wrapper_sha256"] == WRAPPER_SHA, "wrapper_hash", run_id=run_id)
    check(not raw["raw_response_modified"] and not raw["scorer_modified"] and not raw["environment_or_auth_saved"],
          "capture_scope", run_id=run_id)
    if source_hashes is not None:
        check(raw["source_sha256"] == source_hashes, "source_hashes", run_id=run_id)
    records = keyed(raw["records"], lambda x: (x["item_id"], x["choice_index"]))
    check(set(records) == {(x, c) for x in ids for c in range(4)}, "capture_key_set", run_id=run_id)
    for (item_id, choice), rec in records.items():
        where = {"run_id": run_id, "item_id": item_id, "choice_index": choice}
        row = prompts[item_id]
        prefix = row["prompt_prefix"]
        expected_text = {"head": sha(prefix.encode()), "whole": sha((prefix + row["continuations"][choice]).encode())}
        check(rec["text_sha256"] == expected_text, "expected_text_hash", **where)
        check(rec["token_ids_sha256"] == sha(json.dumps(rec["token_ids"], separators=(",", ":")).encode()),
              "token_array_hash", **where)
        start, stop, bos = rec["continuation_start"], rec["continuation_stop"], rec["bos_tokens"]
        check(bos == 1 and start == bos + rec["tokenization_lengths"]["head"]
              and stop == bos + rec["tokenization_lengths"]["whole"]
              and stop == rec["prompt_tokens"] == len(rec["token_ids"])
              and len(rec["continuation_logprobs"]) == stop - start, "boundary_consistency", **where)
        finite = all(v is not None and math.isfinite(v) for v in rec["continuation_logprobs"])
        check(finite, "missing_nonfinite_logprob", **where)
        if finite:
            total = 0.0
            for value in rec["continuation_logprobs"]:
                total += value
            check(total == rec["continuation_sum"] == rec["official_sum"] == run["items"][item_id]["logliks"][choice]
                  and rec["official_sum_matches"], "official_sequential_sum", **where)
    return {"run_id": run_id, "records": records, "metadata": {k: v for k, v in raw.items() if k != "records"},
            "bytes": path.stat().st_size, "path": str(path)}


def distribution(values):
    if not values:
        return {"count": 0}
    absolute = sorted(abs(x) for x in values)
    def quantile(q):
        index = (len(absolute) - 1) * q
        low = int(index)
        high = min(low + 1, len(absolute) - 1)
        return absolute[low] + (absolute[high] - absolute[low]) * (index - low)
    boundaries = [0.001, 0.01, 0.1, 1.0]
    bins = {"exact_zero": sum(x == 0 for x in values)}
    lower = 0
    for upper in boundaries:
        bins[f"{lower}<abs<={upper}"] = sum(lower < abs(x) <= upper for x in values)
        lower = upper
    bins["abs>1.0"] = sum(abs(x) > 1 for x in values)
    return {"count": len(values), "changed": sum(x != 0 for x in values),
            "signed_min": min(values), "signed_max": max(values), "max_absolute": max(absolute),
            "mean_absolute": sum(absolute) / len(absolute),
            "rms": math.sqrt(sum(x*x for x in values) / len(values)),
            "absolute_quantiles": {str(q): quantile(q) for q in (0.5, 0.9, 0.95, 0.99)}, "bins": bins}


def compare_tokens(a, b, ids):
    """Signed effects are b minus a; only token-aligned inputs are compared."""
    fields = ("token_ids", "token_ids_sha256", "text_sha256", "continuation_start",
              "continuation_stop", "bos_tokens", "prompt_tokens", "tokenization_lengths")
    all_deltas, choices, largest = [], [], []
    mismatches = []
    for item_id in ids:
        for choice in range(4):
            x, y = a["records"][(item_id, choice)], b["records"][(item_id, choice)]
            equal = all(x[f] == y[f] for f in fields)
            if not equal:
                mismatches.append({"item_id": item_id, "choice_index": choice,
                                   "fields": [f for f in fields if x[f] != y[f]]})
                continue
            dx = [v - u for u, v in zip(x["continuation_logprobs"], y["continuation_logprobs"])]
            all_deltas.extend(dx)
            positions = [i for i, v in enumerate(dx) if v != 0]
            choices.append({"item_id": item_id, "choice_index": choice, "token_positions": len(dx),
                            "changed_relative_positions": positions, "max_abs_delta": max(map(abs, dx), default=0),
                            "choice_sum_delta": y["official_sum"] - x["official_sum"]})
            for pos, delta in enumerate(dx):
                if delta:
                    largest.append({"item_id": item_id, "choice_index": choice, "relative_position": pos,
                                    "absolute_position": x["continuation_start"] + pos, "delta": delta})
    return {"reference_run_id": a["run_id"], "comparison_run_id": b["run_id"],
            "input_boundary_equal_pairs": 40 - len(mismatches), "input_boundary_mismatches": mismatches,
            "token_delta_distribution": distribution(all_deltas), "choices": choices,
            "largest_absolute_changes": sorted(largest, key=lambda x: abs(x["delta"]), reverse=True)[:10]}


def main():
    plan = read(TRIAL / "candidate-evaluation-plan.json")
    screen = read(TRIAL / "registered-per-item-screen.json")
    check([r["run_id"] for r in plan["runs"]] == RUN_IDS, "fixed_candidate_run_ids")
    check(screen["candidate_outputs_seen"] == 0, "screen_preregistered")
    ids = plan["fixed_target_ids"]
    check(len(ids) == 10 and len(set(ids)) == 10, "fixed_target_count")
    prompts = keyed(read(RESTART / "arc-observation-tail-plan/selected-dev10-prompts.json")["selected_rows"], lambda x: x["id"])
    originals = [load_run(x, ids, prompts) for x in ORIGINAL_IDS]
    thresholds = {x["item_id"]: x for x in screen["thresholds"]}
    check(set(thresholds) == set(ids), "screen_item_set")
    for item_id in ids:
        threshold = thresholds[item_id]
        actual_margins = [x["items"][item_id]["margin"] for x in originals]
        check([x["run_id"] for x in threshold["original_controls"]] == ORIGINAL_IDS, "screen_control_ids", item_id=item_id)
        check([x["margin"] for x in threshold["original_controls"]] == actual_margins, "screen_control_margins", item_id=item_id)
        check(threshold["original_maximum_margin"] == max(actual_margins), "screen_maximum", item_id=item_id)
    for evidence in screen["source_evidence"]:
        check(sha(Path(evidence["path"]).read_bytes()) == evidence["sha256"], "registered_evidence_hash")
    control_captures = [load_capture(O2 / f"token-logprobs-r{i}.json", originals[i+2], ids, prompts) for i in (1, 2)]
    sources = control_captures[0]["metadata"]["source_sha256"]
    check(control_captures[1]["metadata"]["source_sha256"] == sources, "original_source_hashes")
    candidates = [load_run(x, ids, prompts) for x in RUN_IDS]
    captures = [load_capture(TRIAL / f"token-logprobs-r{i}.json", candidates[i-1], ids, prompts, sources) for i in (1, 2, 3)]
    check(sum(x["bytes"] for x in captures) <= 6 * 1024**2, "combined_capture_capacity")
    items = []
    for item_id in ids:
        original = [x["items"][item_id] for x in originals]
        candidate = [x["items"][item_id] for x in candidates]
        scores = [x["score"] for x in candidate]
        margins = [x["margin"] for x in candidate]
        threshold = thresholds[item_id]["original_maximum_margin"]
        items.append({"item_id": item_id, "original_five_scores": [x["score"] for x in original],
                      "candidate_three_scores": scores, "candidate_three_predicted": [x["predicted"] for x in candidate],
                      "original_five_margins": [x["margin"] for x in original], "candidate_three_margins": margins,
                      "registered_original_maximum_margin": threshold,
                      "candidate_min_margin_minus_original_max": min(margins) - threshold,
                      "all_three_official_correct": all(x == 1 for x in scores),
                      "registered_screen_passed": all(x == 1 for x in scores) and all(x > threshold for x in margins),
                      "candidate_normalized_logliks": [x["normalized_logliks"] for x in candidate],
                      "gold": candidate[0]["gold"], "character_lengths": candidate[0]["lengths"]})
    token_comparisons = [compare_tokens(a, b, ids) for a in control_captures for b in captures]
    for comparison in token_comparisons:
        check(not comparison["input_boundary_mismatches"], "candidate_original_input_boundary",
              reference_run_id=comparison["reference_run_id"], comparison_run_id=comparison["comparison_run_id"])
    output = {
        "schema_version": 1, "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "verified_fixed_three_complete" if not ERRORS else "validation_failed",
        "scope": "T002 fixed target10 single-replica observer-OFF diagnostics; not canonical full evaluation.",
        "run_order": RUN_IDS, "original_choice_sum_controls": ORIGINAL_IDS,
        "original_token_controls": ORIGINAL_IDS[-2:], "items": items,
        "screen_precedence": {"effective": str(TRIAL / "registered-per-item-screen.json"),
                             "rule": screen["rule"],
                             "stale_operational_plan_wording": "original_controls_reuse and supplementary scope still name primary three. Later registered per-item screen and scientific creation guard require ALL FIVE. No plan mutation here."},
        "official_scoring": {"correct_counts": [sum(x["items"][i]["score"] == 1 for i in ids) for x in candidates],
                            "screen_passed_item_ids": [x["item_id"] for x in items if x["registered_screen_passed"]],
                            "confirmed_canonical_improvement": 0, "canonical_full_evaluation_performed_by_this_script": False,
                            "regressions_outside_target10": "unmeasured", "adoption": "not decided by this diagnostic"},
        "capture": {"actual_bytes": [x["bytes"] for x in captures], "total_bytes": sum(x["bytes"] for x in captures),
                    "per_run_cap_bytes": 2*1024**2, "total_cap_bytes": 6*1024**2, "raw_evidence_retained": True,
                    "wrapper_sha256": WRAPPER_SHA, "source_sha256": sources,
                    "expected_sum_checks": 200, "expected_candidate_official_score_checks": 30,
                    "expected_token_comparison_choice_pairs": 240},
        "original_to_candidate_token_comparisons": token_comparisons,
        "candidate_pairwise_token_comparisons": [compare_tokens(a, b, ids) for a, b in itertools.combinations(captures, 2)],
        "run_timing": [{k: v for k, v in x.items() if k != "items"} for x in candidates],
        "candidate_engine_timestamp_elapsed_sum_seconds": sum(x["elapsed_timestamp_seconds"] for x in candidates),
        "steps": {"prompt_logprob_completion_requests": 120, "tokenize_client_calls": 243,
                  "total_client_calls": 363, "transport_retries": "not inferred"},
        "validation_errors": ERRORS,
        "limits": [
            "Only two original controls have per-token records; five-control token envelope is not claimed.",
            "These descriptive observed differences are not calibrated statistical noise bounds.",
            "Effect is saved-weight intervention in the provided target+MTP4 runtime, including possible shared-state or scheduling mediation; no target-only or kernel-cause inference.",
            "Sequential token sums and official character-normalized argmax are checked independently. Matching logged inputs does not prove undocumented server state equivalence.",
            "Run timestamp durations exclude separate process startup/restart and outer process wall time; never combine these timings as interchangeable.",
            "No new model request, weight edit, coefficient fit, service operation, or evaluation-condition change is performed.",
        ],
        "evidence": EVIDENCE,
    }
    script = Path(__file__)
    output["comparison_script"] = {"path": str(script), "sha256": sha(script.read_bytes())}
    destination = TRIAL / "candidate-token-comparison.json"
    with destination.open("x") as stream:
        json.dump(output, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": output["status"], "path": str(destination), "bytes": destination.stat().st_size,
                      "sha256": sha(destination.read_bytes()), "validation_errors": len(ERRORS),
                      "correct_counts": output["official_scoring"]["correct_counts"],
                      "screen_passed_item_ids": output["official_scoring"]["screen_passed_item_ids"]}))
    return 0 if not ERRORS else 2


if __name__ == "__main__":
    raise SystemExit(main())
