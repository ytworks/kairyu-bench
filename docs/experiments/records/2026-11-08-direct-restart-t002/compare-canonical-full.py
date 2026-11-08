#!/usr/bin/env python3
"""T002 only: compare retained canonical dev10 results; no API/service operations."""
import collections
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
RESTART = ROOT / "restart-direct-20261108"
TRIAL = RESTART / "trial-002-final-branch-ablation"
ORIGINAL = ROOT / "results/wsp-direct-original-dev10-full-20261108-r1"
CANDIDATE = ROOT / "results/wsp-direct-t002-candidate-dev10-full-20261108-r1"
MANIFEST = ROOT / "worktrees/kairyu-bench-weight-tools/src/kairyu_bench/data/wsp/manifest.json"
NLL_SOURCES = {"probe-code", "probe-wikipedia-ja", "probe-wikitext"}
EVIDENCE, VALIDATION = {}, []


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference(path):
    result = {"path": str(path), "sha256": sha_file(path), "bytes": path.stat().st_size}
    EVIDENCE[str(path)] = result
    return result


def read(path, lines=False):
    reference(path)
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()] if lines else json.load(stream)


def check(condition, code, **context):
    if not condition:
        VALIDATION.append({"code": code, **context})


def stamp(text):
    return dt.datetime.fromisoformat(text.replace("Z", "+00:00"))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def error_summary(value):
    if not value:
        return None
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    match = re.match(r"([A-Za-z][A-Za-z0-9_ ]{0,80}):", text)
    return {"type_prefix": match.group(1) if match else "see_raw_evidence",
            "sha256": hashlib.sha256(text.encode()).hexdigest(), "bytes": len(text.encode()),
            "communication_failure": any(term in text.lower() for term in
                ("connectionerror", "connecterror", "connecttimeout", "readtimeout", "httperror",
                 "httpstatuserror", "remoteprotocolerror", "apiconnectionerror", "timeout", "connection refused",
                 "connection reset", "server disconnected", "broken pipe"))}


def classify(item):
    if item is None:
        return "missing"
    error = error_summary(item.get("error"))
    if error:
        return "communication_failure" if error["communication_failure"] else "unscored"
    value = item.get("score")
    if not finite(value):
        return "unscored"
    if item["source"] in NLL_SOURCES:
        return "diagnostic_na"
    if value == 1:
        return "correct"
    if value == 0:
        return "incorrect"
    return "partial_credit" if 0 < value < 1 else "unscored"


def output_flags(item):
    if item is None:
        return {"missing": True}
    samples = (item.get("outcome") or {}).get("samples", [])
    blank_content, reasoning_only, tool_only_content, fully_empty, non_content_schema, truncated = [], [], [], [], [], []
    for index, sample in enumerate(samples):
        if isinstance(sample, dict):
            if "content" not in sample:
                # BFCL multi-turn / agent samples use another schema, not an empty chat response.
                non_content_schema.append(index)
            elif not (sample.get("content") or "").strip():
                blank_content.append(index)
                has_reasoning = bool((sample.get("reasoning") or "").strip())
                has_tools = bool(sample.get("tool_calls"))
                if has_tools:
                    tool_only_content.append(index)
                elif has_reasoning:
                    reasoning_only.append(index)
                else:
                    fully_empty.append(index)
            if sample.get("finish_reason") == "length":
                truncated.append(index)
    def has_nonfinite(value):
        if isinstance(value, float):
            return not math.isfinite(value)
        if isinstance(value, dict):
            return any(has_nonfinite(v) for v in value.values())
        if isinstance(value, list):
            return any(has_nonfinite(v) for v in value)
        return False
    return {"blank_content_sample_indices": blank_content,
            "reasoning_only_sample_indices": reasoning_only,
            "tool_calls_without_content_sample_indices": tool_only_content,
            "no_content_reasoning_or_tool_calls_sample_indices": fully_empty,
            "non_content_sample_schema_indices": non_content_schema,
            "length_truncated_sample_indices": truncated,
            "nonfinite_numeric_in_item": has_nonfinite(item),
            "error": error_summary(item.get("error"))}


def adoption_blocked(rows, unavailable, validation, run_status):
    return bool(unavailable or validation or run_status != "completed" or any(
        row["candidate_status"] in {"missing", "unscored", "communication_failure"} for row in rows))


def load_run(directory, ids):
    run = read(directory / "run.json")
    # A final partial run is reportable honestly; an active/incomplete run is not.
    if not run.get("finished_at") or run.get("status") not in {"completed", "partial"}:
        raise RuntimeError("Canonical run has not reached a final harness status; no report written.")
    rows = read(directory / "items.jsonl", True)
    if len(rows) != 778:
        raise RuntimeError("Canonical final item ledger does not contain all778 rows; no report written.")
    indexed = {row["id"]: (index + 1, row) for index, row in enumerate(rows)}
    check(len(indexed) == len(rows), "duplicate_item_ids", run=directory.name)
    check(set(indexed) == set(ids), "selected_item_set", run=directory.name)
    responses = read(directory / "responses.jsonl", True)
    check(len(responses) == 778 and len({x["id"] for x in responses}) == 778
          and {x["id"] for x in responses} == set(ids), "response_item_set", run=directory.name)
    check(run.get("tier") == 10 and run.get("split") == "dev", "tier_split", run=directory.name)
    check(run.get("items") == 778 and not run.get("excluded_item_ids"), "full_scope", run=directory.name)
    report = read(directory / "report.json")
    check(report["run"]["status"] == run["status"], "report_status", run=directory.name)
    check(not report.get("missing_areas"), "missing_official_area", run=directory.name)
    return {"run": run, "indexed": indexed, "report": report, "directory": directory}


def state_counts(rows, side):
    return dict(sorted(collections.Counter(x[side + "_status"] for x in rows).items()))


def group_summary(rows):
    capabilities = [x for x in rows if not x["is_nll_diagnostic"]]
    diagnostic = [x for x in rows if x["is_nll_diagnostic"]]
    return {"items": len(rows), "capability_items": len(capabilities), "nll_diagnostic_items": len(diagnostic),
            "original_status_counts": state_counts(rows, "original"),
            "candidate_status_counts": state_counts(rows, "candidate"),
            "gain_count": sum(x["change"] == "gain" for x in capabilities),
            "loss_count": sum(x["change"] == "loss" for x in capabilities),
            "unchanged_count": sum(x["change"] == "unchanged" for x in capabilities),
            "unavailable_comparison_count": sum(x["change"] == "unavailable" for x in rows)}


def agent_evidence(directory):
    result = []
    jobs = directory / "agents/terminal-bench/raw/terminal-bench/jobs"
    if not jobs.exists():
        return {"available": False, "tasks": []}
    for path in sorted(jobs.rglob("result.json")):
        value = read(path)
        if "task_name" not in value:
            continue
        agent = value.get("agent_result") or {}
        metadata = agent.get("metadata") or {}
        trajectory = path.parent / "agent/trajectory.json"
        steps = read(trajectory).get("steps", []) if trajectory.exists() else None
        verifier = value.get("verifier_result") or {}
        exception = value.get("exception_info")
        row = {"task_name": value["task_name"], "result_evidence": reference(path),
               "started_at": value.get("started_at"), "finished_at": value.get("finished_at"),
               "official_rewards": verifier.get("rewards"),
               "exception_present": bool(exception),
               "exception_type": exception.get("exception_type") if isinstance(exception, dict) else None,
               "recorded_episodes": metadata.get("n_episodes"),
               "recorded_total_steps": len(steps) if steps is not None else None,
               "recorded_agent_steps": sum(x.get("source") == "agent" for x in steps) if steps is not None else None,
               "input_tokens": agent.get("n_input_tokens"), "output_tokens": agent.get("n_output_tokens")}
        if row["started_at"] and row["finished_at"]:
            row["recorded_elapsed_seconds"] = (stamp(row["finished_at"]) - stamp(row["started_at"])).total_seconds()
        if trajectory.exists():
            row["trajectory_evidence"] = reference(trajectory)
        result.append(row)
    return {"available": bool(result), "tasks": result}


def existing_runtime_metrics():
    path = TRIAL / "canonical-runtime-monitor.jsonl"
    if not path.exists():
        return {"available": False, "reason": "No retained monitor artifact; no new metric request issued."}
    raw = path.read_bytes()
    evidence = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    EVIDENCE[str(path)] = evidence
    snapshots = [json.loads(line) for line in raw.splitlines() if line.strip()]
    latest = snapshots[-1] if snapshots else {}
    replicas = []
    for replica in latest.get("replicas", []):
        ordinary, mtp, origins = {}, {}, {}
        for field in ("metrics", "mtp_process_lifetime_metrics"):
            for key, value in replica.get(field, {}).items():
                if not key.endswith("_total"):
                    continue
                normalized = key.removeprefix("vllm:")
                if normalized.startswith("spec_decode_"):
                    mtp[normalized] = value
                    origins[normalized] = {"field": field, "original_key": key}
                else:
                    ordinary[key] = value
        replicas.append({"service": replica.get("service"), "health": replica.get("health"),
                         "process_lifetime_counters": ordinary,
                         "mtp_process_lifetime_counters": mtp, "mtp_counter_sources": origins,
                         "mtp_counters_present_in_latest_snapshot": bool(mtp)})
    return {"available": bool(snapshots), "saved_snapshot_count": len(snapshots), "latest_at": latest.get("at"),
            "latest_replicas": replicas, "evidence": evidence,
            "ignored_created_timestamp_fields": True, "not_item_counts": True,
            "scope": "Retained *_total counters are descriptive process-lifetime values, not full-run-only deltas or a matched original comparison. *_created fields are timestamps and their aggregated values are intentionally discarded. Do not sum counters with overlapping meanings or infer MTP cause/acceptance change. No new requests, logging or instrumentation."}

def main():
    manifest = read(MANIFEST)
    selected = [(index, x) for index, x in enumerate(manifest["items"]) if x["tier"] == 10 and x["split"] == "dev"]
    ids = [x["id"] for _, x in selected]
    assert len(ids) == len(set(ids)) == 778
    original, candidate = load_run(ORIGINAL, ids), load_run(CANDIDATE, ids)
    plan = read(TRIAL / "canonical-full-evaluation-plan.json")
    check(candidate["run"]["label"] == plan["candidate_tree_sha256"], "candidate_label")
    for run in (original, candidate):
        check(run["run"]["manifest_sha256"] == sha_file(MANIFEST), "manifest_hash", run=run["directory"].name)
    baseline_status = {x["id"]: x for x in read(RESTART / "active-baseline-all-items-status.jsonl", True)}
    comparison = read(TRIAL / "candidate-token-comparison.json")
    target_ids = [x["item_id"] for x in comparison["items"]]
    rows = []
    for order, manifest_item in selected:
        item_id = manifest_item["id"]
        old_line, old = original["indexed"].get(item_id, (None, None))
        new_line, new = candidate["indexed"].get(item_id, (None, None))
        old_status, new_status = classify(old), classify(new)
        check(old_status == baseline_status[item_id]["status"], "original_classification_parity", item_id=item_id)
        old_score, new_score = old.get("score") if old else None, new.get("score") if new else None
        is_nll = manifest_item["source"] in NLL_SOURCES
        available = (old_status not in {"missing", "unscored", "communication_failure"}
                     and new_status not in {"missing", "unscored", "communication_failure"}
                     and finite(old_score) and finite(new_score))
        delta = new_score - old_score if available else None
        change = ("diagnostic_delta" if is_nll else "gain" if delta > 0 else "loss" if delta < 0 else "unchanged") if available else "unavailable"
        row = {"manifest_order": order, "id": item_id, "source": manifest_item["source"],
               "kind": old["kind"] if old else new["kind"], "area": old.get("area") if old else new.get("area"),
               "is_nll_diagnostic": is_nll, "original_status": old_status, "candidate_status": new_status,
               "original_score": old_score if finite(old_score) else None,
               "candidate_score": new_score if finite(new_score) else None, "delta_candidate_minus_original": delta,
               "change": change, "candidate_flags": output_flags(new), "original_flags": output_flags(old),
               "official_status": ("diagnostic_nll_no_pass_definition" if is_nll else
                   ("full_score_no_binary_pass_definition" if new_score == 1 else "fractional_or_similarity_score_no_binary_pass_definition")
                   if "no_binary_pass_definition" in baseline_status[item_id]["official_status"] else "official_sample_verdicts"),
               "pass_fail": None if baseline_status[item_id]["pass_fail"] is None or new_status in {"missing", "unscored", "communication_failure"} else new_status == "correct",
               "original_items_line": old_line, "candidate_items_line": new_line}
        rows.append(row)
    losses = [x["id"] for x in rows if x["change"] == "loss"]
    gains = [x["id"] for x in rows if x["change"] == "gain"]
    unavailable = [x["id"] for x in rows if x["change"] == "unavailable"]
    flags = {name: [x["id"] for x in rows if x["candidate_flags"].get(name)] for name in
             ("blank_content_sample_indices", "reasoning_only_sample_indices", "tool_calls_without_content_sample_indices",
              "no_content_reasoning_or_tool_calls_sample_indices", "non_content_sample_schema_indices",
              "length_truncated_sample_indices", "nonfinite_numeric_in_item", "error")}
    capabilities = [x for x in rows if not x["is_nll_diagnostic"]]
    unresolved = [x["id"] for x in capabilities if x["candidate_status"] != "correct"]
    summary = {
        "schema_version": 1, "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "final_canonical_comparison" if not VALIDATION else "validation_failed",
        "scope": "One terminal dev10 canonical run distributed across four same-candidate replicas, with all778 item ledger rows; terminal status may be partial and does not imply every item was successfully scored. Not four repeats.",
        "original_run_id": ORIGINAL.name, "candidate_run_id": CANDIDATE.name,
        "original_run_status": original["run"]["status"], "candidate_run_status": candidate["run"]["status"],
        "selected_items": 778, "capability_items": 746, "nll_diagnostic_items": 32,
        "counts": group_summary(rows), "gain_ids": gains, "loss_ids": losses,
        "unchanged_ids": [x["id"] for x in rows if x["change"] == "unchanged"],
        "unavailable_comparison_ids": unavailable,
        "candidate_unresolved_capability_count": len(unresolved), "candidate_unresolved_capability_ids": unresolved,
        "per_kind": {key: group_summary([x for x in rows if x["kind"] == key]) for key in ("chat", "logprob", "agent")},
        "logprob_capability_only": group_summary([x for x in rows if x["kind"] == "logprob" and not x["is_nll_diagnostic"]]),
        "per_source": {key: group_summary([x for x in rows if x["source"] == key]) for key in sorted({x["source"] for x in rows})},
        "target10": [next(x for x in rows if x["id"] == item_id) for item_id in target_ids],
        "nll32": [{"id": x["id"], "source": x["source"], "original_nll": x["original_score"],
                   "candidate_nll": x["candidate_score"], "delta": x["delta_candidate_minus_original"],
                   "status": x["candidate_status"]} for x in rows if x["is_nll_diagnostic"]],
        "candidate_problem_flags": flags,
        "official_report": {"original_overall": original["report"]["overall"], "candidate_overall": candidate["report"]["overall"],
                            "delta_overall": candidate["report"]["overall"] - original["report"]["overall"]
                            if finite(candidate["report"]["overall"]) and finite(original["report"]["overall"]) else None,
                            "original_areas": original["report"]["areas"], "candidate_areas": candidate["report"]["areas"],
                            "original_sources": original["report"]["sources"], "candidate_sources": candidate["report"]["sources"],
                            "candidate_unscored": candidate["report"]["unscored"]},
        "timing": {"original_official": original["run"]["timing"], "candidate_official": candidate["run"]["timing"],
                   "original_timestamp_elapsed_seconds": (stamp(original["run"]["finished_at"]) - stamp(original["run"]["started_at"])).total_seconds(),
                   "candidate_timestamp_elapsed_seconds": (stamp(candidate["run"]["finished_at"]) - stamp(candidate["run"]["started_at"])).total_seconds(),
                   "interpretation": "Official load/generation/wall values have their own scopes. Do not add them or treat timestamp elapsed as outer process wall time."},
        "agent_steps": {"original": agent_evidence(ORIGINAL), "candidate": agent_evidence(CANDIDATE)},
        "mtp_metrics": existing_runtime_metrics(),
        "registered_disposition": {"reject_this_fixed_candidate_due_to_regression": bool(losses),
                                  "adoption_not_permitted_by_unscored_or_invalid": adoption_blocked(rows, unavailable, VALIDATION, candidate["run"]["status"]),
                                  "adoption_decided_by_this_script": False,
                                  "further_variants_on_final_pair": False,
                                  "identity_reconstruction_storage_disposition": "Separate final guard and root decision required.",
                                  "tier10_all_capability_full_credit": not unresolved and not unavailable and not VALIDATION,
                                  "nll_has_no_official_pass": True},
        "limits": ["32 NLL values have no invented pass or non-regression threshold.",
                   "Blank content is not necessarily an empty response: reasoning-only and tool-call-bearing samples are reported separately. No-content schemas are not classified as empty. Length truncation is independent. These flags do not alter official scores.",
                   "Continuous/partial official scores retain their source semantics; score1 denotes full credit, not a new binary cutoff.",
                   "Per-replica attribution/variance cannot be inferred from an all-four distributed single run.",
                   "Raw response/output/scoring evidence remains in its original run directory; no raw duplication or deletion.",
                   "This script issues no evaluation, model request, service action, saved-weight change or coefficient fitting."],
        "validation_errors": VALIDATION, "evidence": EVIDENCE,
    }
    # Remove any nonfinite numeric metadata only from this derivative; raw evidence remains unchanged.
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    output = TRIAL / "canonical-full-all-items-comparison.jsonl"
    summary_path = TRIAL / "canonical-full-comparison.json"
    if output.exists() or summary_path.exists():
        raise RuntimeError("Comparison evidence exists; refusing to overwrite.")
    with output.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(clean(row), ensure_ascii=False, allow_nan=False) + "\n")
    summary["all_items_comparison"] = reference(output)
    summary["comparison_source"] = reference(Path(__file__))
    with summary_path.open("x") as stream:
        json.dump(clean(summary), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": summary["status"], "candidate_run_status": candidate["run"]["status"],
                      "counts": summary["counts"], "gains": len(gains), "losses": len(losses),
                      "unavailable": len(unavailable), "official_overall": summary["official_report"]["candidate_overall"],
                      "summary": {"path": str(summary_path), "sha256": sha_file(summary_path), "bytes": summary_path.stat().st_size},
                      "all_items": {"path": str(output), "sha256": sha_file(output), "bytes": output.stat().st_size}}))
    return 0 if not VALIDATION else 2


if __name__ == "__main__":
    raise SystemExit(main())
