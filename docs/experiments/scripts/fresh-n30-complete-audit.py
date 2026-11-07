#!/usr/bin/env python3
"""Audit completed canonical AV dev30; CPU reads only, never calls a model."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import time

from kairyu_bench.wsp.catalog import SOURCE_BY_NAME
from kairyu_bench.wsp.manifest import load_manifest
from kairyu_bench.wsp.report import build_report, read_items


def read(path):
    return json.loads(path.read_text())


def safe_json(data):
    if isinstance(data, float) and not math.isfinite(data):
        return {"invalid_nonfinite_float": repr(data)}
    if isinstance(data, dict):
        return {key: safe_json(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [safe_json(value) for value in data]
    return data


def save(path, data):
    path.write_text(json.dumps(safe_json(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def option(command, name):
    return command[command.index(name) + 1]



def item_evidence(expected, rows, responses):
    """Keep every canonical ID and every unexpected/duplicate record; never impute."""
    grouped, response_groups = {}, {}
    for index, row in enumerate(rows):
        grouped.setdefault(row.get("id"), []).append((index, row))
    for index, row in enumerate(responses):
        response_groups.setdefault(row.get("id"), []).append((index, row))
    canonical, issues = [], []
    def numeric_record(index, row):
        value = {key: row.get(key) for key in
                 ("id", "source", "area", "kind", "tier", "max_tokens", "samples",
                  "score", "sample_scores", "error")}
        value["record_index"] = index
        if row.get("area") == "code":
            value["detail"] = row.get("detail")
            value["generation"] = [
                {key: sample.get(key) for key in ("finish_reason", "prompt_tokens", "completion_tokens")}
                for sample in (row.get("outcome") or {}).get("samples", [])]
        return value
    for key, item in expected.items():
        records, replies = grouped.get(key, []), response_groups.get(key, [])
        entry = {"id": key, "source": item.source, "area": SOURCE_BY_NAME[item.source].area,
                 "kind": item.kind, "tier": item.tier, "max_tokens": item.max_tokens, "samples": item.samples,
                 "score": None, "sample_scores": [], "error": None, "audit_issues": [],
                 "actual_item_records": [numeric_record(index, row) for index, row in records],
                 "response_records": [{"record_index": index, "error": row.get("error"),
                                       "outcome_present": row.get("outcome") is not None}
                                      for index, row in replies]}
        if len(records) != 1:
            entry["audit_issues"].append("missing_item_record" if not records else "duplicate_item_records")
        else:
            row = records[0][1]
            for field in ("source", "area", "kind", "tier", "max_tokens", "samples"):
                if row.get(field) != entry[field]:
                    entry["audit_issues"].append("metadata_mismatch:" + field)
            value = row.get("score")
            if type(value) not in (int, float) or not math.isfinite(value):
                entry["audit_issues"].append("unknown_or_invalid_score")
                entry["raw_score"] = value
            else:
                entry["score"] = value
            entry["sample_scores"], entry["error"] = row.get("sample_scores"), row.get("error")
            if row.get("error") is not None:
                entry["audit_issues"].append("item_error_present")
            if not isinstance(row.get("sample_scores"), list) or any(
                    type(x) not in (int, float) or not math.isfinite(x) for x in row.get("sample_scores", [])):
                entry["audit_issues"].append("invalid_sample_scores")
            if entry["area"] == "code":
                entry.update({k: entry["actual_item_records"][0].get(k) for k in ("detail", "generation")})
        if len(replies) != 1:
            entry["audit_issues"].append("missing_response_record" if not replies else "duplicate_response_records")
        if any(row.get("error") is not None or row.get("outcome") is None for _, row in replies):
            entry["audit_issues"].append("response_error_or_missing_outcome")
        entry["scoring_status"] = "scored" if not entry["audit_issues"] else "unknown_or_anomalous"
        if entry["audit_issues"]:
            issues.append({"id": key, "issues": entry["audit_issues"]})
        canonical.append(entry)
    unexpected = [numeric_record(index, row) for index, row in enumerate(rows) if row.get("id") not in expected]
    unexpected_responses = [{"record_index": index, "id": row.get("id"), "error": row.get("error"),
                             "outcome_present": row.get("outcome") is not None}
                            for index, row in enumerate(responses) if row.get("id") not in expected]
    return canonical, {"item_issues": issues, "unexpected_items": unexpected,
                       "unexpected_responses": unexpected_responses,
                       "actual_item_record_count": len(rows), "actual_response_record_count": len(responses)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Fresh output required."
    # Refuse to create a scored audit from a partial run.
    meta = read(args.run / "run.json")
    if meta["status"] != "completed":
        raise RuntimeError("Run is not completed; no partial-score audit performed.")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    module = importlib.util.spec_from_file_location(
        "full_audit", Path(__file__).with_name("fresh-wsp-large-tier-audit.py"))
    audit = importlib.util.module_from_spec(module)
    module.loader.exec_module(audit)
    sha = audit.sha
    provenance = {}
    try:
        manifest = load_manifest()
        rows = read_items(args.run)
        report = read(args.run / "report.json")
        expected = {x.id: x for x in manifest.select(30, "dev")}
        responses = [json.loads(line) for line in (args.run / "responses.jsonl").read_text().splitlines()]
        numeric, anomalies = item_evidence(expected, rows, responses)
        code = [row for row in numeric if row["area"] == "code"]
        # Persist all individual facts before any completeness/runtime assertion.
        save(args.output / "all-2338-numeric-items.json", numeric)
        save(args.output / "code-90-items.json", code)
        anomalies.update(run_status=meta["status"], run_unscored=meta.get("unscored"),
                         report_unscored=report.get("unscored"), missing_areas=report.get("missing_areas"))
        save(args.output / "data-anomalies.json", anomalies)
        rebuilt = build_report(args.run)
        report_equal = safe_json(rebuilt) == safe_json(report)
        try:
            full = audit.audit_run(args.run, manifest)
            strict_error = None
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            strict_error = {"type": type(error).__name__, "message": str(error)}
            full = {
                "run_id": args.run.name, "tier": meta.get("tier"), "split": meta.get("split"),
                "label": meta.get("label"), "status": meta.get("status"), "items": len(numeric),
                "scored": sum(x["scoring_status"] == "scored" for x in numeric),
                "unscored_or_anomalous": [x["id"] for x in numeric if x["scoring_status"] != "scored"],
                "code_source_mean": report.get("areas", {}).get("code"), "overall": report.get("overall"),
                "areas": report.get("areas"), "sources": report.get("sources"), "timing": meta.get("timing"),
                "inputs": {str(args.run / name): sha(args.run / name)
                           for name in ("run.json", "report.json", "items.jsonl")},
                "score_values_scope": "Official reported aggregates retained as claims; not validated complete scores.",
            }
        complete = not (strict_error or not report_equal or anomalies["item_issues"] or
                        anomalies["unexpected_items"] or anomalies["unexpected_responses"] or
                        meta.get("unscored") or report.get("unscored") or report.get("missing_areas"))
        anomalies.update(strict_complete_audit_error=strict_error, report_exact_rebuild_equal=report_equal,
                         verified_complete=complete, full_score_ranking_permitted=complete)
        save(args.output / "data-anomalies.json", anomalies)
        assert meta["tier"] == 30 and meta["split"] == "dev" and meta["items"] == 2338
        assert len(numeric) == 2338 and len(code) == 90
        plan = read(args.study / "launch-plan-approved.json")
        actual = read(args.study / "actual-runner.json")["runner"]
        command = plan["command"]
        assert actual["command"] == command[1:]
        image = "sha256:f91bdcf997af6d7b194bec229812fd018277158f6795de7d91e58448936e7ad8"
        assert actual["image_id"] == plan["source_identity"]["actual_runner_image_id"] == image
        tree = "5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf"
        serving = plan["expected_serving"]
        assert serving["tree_sha256"] == meta["label"] == report["run"]["label"] == tree
        assert (serving["replicas"], serving["tp_size"], serving["mtp_tokens"]) == (4, 2, 4)
        assert command[3] == "http://kairyu:8000/v1"
        assert [option(command, "--" + kind + "-concurrency") for kind in ("chat", "logprob", "agent")] == ["256", "32", "4"]
        assert [command[i + 1] for i, x in enumerate(command) if x == "--logprob-endpoint"] == [
            f"http://gemma-{i}:8000/v1" for i in range(4)]
        assert not any(flag in command for flag in ("--only", "--exclude-item", "--limit", "--incomplete-manifest"))
        assert plan["selection"]["items"] == 2338 and not plan["selection"]["excluded_ids"]
        assert plan["http_trace"]["workers_expected"] == 0
        preflight = read(args.study / "launch-preflight.json")
        assert preflight["checkpoint_stats_equal"]
        assert {x["service"]: x["cid"] for x in preflight["services"]} == serving["service_cids"]
        for receipt in serving["closures"] + [serving["four_replica_closure"]]:
            path = Path(receipt["path"])
            assert sha(path) == receipt["sha256"]
            provenance[str(path)] = sha(path)
        for receipt in plan["source_identity"]["critical_archived_files"]:
            path = Path(receipt["path"])
            assert sha(path) == receipt["sha256"]
            provenance[str(path)] = sha(path)
        source_checks = {}
        archive = Path(plan["source_identity"]["archive"])
        for name in ("wsp/report.py", "wsp/catalog.py", "wsp/manifest.py", "wsp/sources/code.py", "wsp/scorers/scicode_score.py"):
            current, archived = args.repository / "src/kairyu_bench" / name, archive / "src/kairyu_bench" / name
            assert sha(current) == sha(archived)
            source_checks[name] = sha(current)
        trace_path = args.study / "http-trace/requests-client.jsonl"
        trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
        selected = [x for x in trace if x.get("run_id") == args.run.name and x.get("request_scope") == "request"]
        trace_summary = {
            "all_records": len(trace), "request_records": len(selected),
            "operations": dict(Counter(x["operation"] for x in selected)),
            "errors": sum(x.get("error") is not None for x in selected),
            "retry_attempts": sum(x.get("attempt_index", 0) > 0 for x in selected),
            "same_run_non_request_records": sum(x.get("run_id") == args.run.name and x.get("request_scope") != "request" for x in trace),
            "other_run_records": sum(x.get("run_id") != args.run.name for x in trace),
            "scope": "HTTP client traces only; no activation capture or server trace completeness claim",
        }
        agents = []
        for path in sorted((args.run / "agents/terminal-bench").rglob("result.json")):
            result = read(path)
            if "task_name" not in result:
                continue
            trajectory_path = path.parent / "agent/trajectory.json"
            trajectory = read(trajectory_path) if trajectory_path.exists() else None
            ar = result.get("agent_result") or {}
            agents.append({
                "id": "terminal-bench:" + result["task_name"].split("/")[-1],
                "episodes": (ar.get("metadata") or {}).get("n_episodes"),
                "trajectory_steps": len(trajectory["steps"]) if trajectory else None,
                "tool_calls": sum(len(x.get("tool_calls") or []) for x in trajectory["steps"]) if trajectory else None,
                "input_tokens": ar.get("n_input_tokens"), "output_tokens": ar.get("n_output_tokens"),
                "exception_type": (result.get("exception_info") or {}).get("exception_type"),
                "seconds": (datetime.fromisoformat(result["finished_at"].replace("Z", "+00:00")) -
                            datetime.fromisoformat(result["started_at"].replace("Z", "+00:00"))).total_seconds(),
            })
            provenance[str(path)] = sha(path)
            if trajectory_path.exists():
                provenance[str(trajectory_path)] = sha(trajectory_path)
        for path in sorted((args.run / "agents/swe-bench-pro").rglob("*.traj.json")):
            data = read(path)
            messages = data["messages"]
            assistant = [x for x in messages if x.get("role") == "assistant"]
            usages = [((x.get("extra") or {}).get("response") or {}).get("usage") for x in assistant]
            agent = {
                "id": "swe-bench-pro:" + data["instance_id"],
                "api_calls": data["info"]["model_stats"].get("api_calls"),
                "assistant_messages": len(assistant),
                "tool_messages": sum(x.get("role") == "tool" for x in messages),
                "announced_tool_calls": sum(len(x.get("tool_calls") or []) for x in assistant),
                "recorded_actions": sum(len((x.get("extra") or {}).get("actions") or []) for x in assistant),
                "exit_status": data["info"]["exit_status"],
                "usage_records": sum(isinstance(x, dict) for x in usages),
                "usage_scope": "Persisted assistant responses only; missing model-stat calls are not zero-filled.",
                "usage_complete_relative_api_calls": len(assistant) == data["info"]["model_stats"].get("api_calls"),
            }
            for output, key in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
                agent[output] = sum(u[key] for u in usages) if usages and all(isinstance(u, dict) and isinstance(u.get(key), int) for u in usages) else None
            agents.append(agent)
            provenance[str(path)] = sha(path)
        agent_items = {x["id"]: x for x in numeric if x["kind"] == "agent"}
        agent_counts = Counter(x["id"] for x in agents)
        trajectory_issues = {key: {"records": agent_counts[key]}
                             for key in agent_items if agent_counts[key] != 1}
        for key in agent_items:
            if not agent_counts[key]:
                agents.append({"id": key, "trajectory_missing": True})
        raw_agents = {x["id"]: x for x in rows if x["kind"] == "agent"}
        for agent in agents:
            row = raw_agents.get(agent["id"], agent_items[agent["id"]])
            agent["score"] = row.get("score")
            agent["outcome_numeric"] = {k: v for k, v in (row.get("outcome") or {}).items()
                                       if v is None or isinstance(v, (int, float, bool))}
            agent["score_detail"] = row.get("detail")
            agent["official_samples"] = [
                {key: sample.get(key) for key in ("reward", "seconds", "setup_seconds", "exception", "trial")}
                for sample in (row.get("outcome") or {}).get("samples", [])]
        summary = {
            "status": "completed_independent_canonical_audit" if complete else "completed_independent_audit_with_anomalies",
            "verified_complete": complete, "full_score_ranking_permitted": complete,
            "data_anomalies": anomalies, "created_at": datetime.now(timezone.utc).isoformat(),
            "run": full, "source_report": report["sources"], "http_trace": trace_summary,
            "agents": agents, "agent_trajectory_issues": trajectory_issues,
            "all_2338_unique_ids_and_metadata_verified": not anomalies["item_issues"] and not anomalies["unexpected_items"],
            "report_exact_rebuild_equal": report_equal,
            "code_unknown_or_anomalous": [x["id"] for x in code if x["scoring_status"] != "scored"],
            "code_failed_or_partial": [x["id"] for x in code if x["scoring_status"] == "scored" and x["score"] < 1],
            "code_counts_by_source": {name: {"full": sum(x["score"] == 1 for x in code if x["source"] == name and x["scoring_status"] == "scored"),
                                           "partial": sum(0 < x["score"] < 1 for x in code if x["source"] == name and x["scoring_status"] == "scored"),
                                           "unknown": sum(x["scoring_status"] != "scored" for x in code if x["source"] == name),
                                           "zero": sum(x["score"] == 0 for x in code if x["source"] == name and x["scoring_status"] == "scored")}
                                      for name in sorted({x["source"] for x in code})},
            "runtime": {"image": image, "checkpoint_tree": tree, "topology": {"tp": 2, "replicas": 4, "mtp_tokens": 4},
                        "concurrency": {"chat": 256, "logprob": 32, "agent": 4},
                        "actual_runner_cid": actual["cid"], "http_client_trace_only": True,
                        "comparison_limit": "New dev30 baseline; no same-condition candidate pair; tier10 and historical original results are different conditions."},
            "timing_note": "wall_seconds and generation_seconds both start before dataset loading; do not call either pure generation time.",
            "source_sha256": source_checks, "analysis_seconds": time.monotonic() - started,
        }
        provenance.update(full["inputs"])
        for path in [args.run / "responses.jsonl", trace_path, Path(__file__),
                     Path(__file__).with_name("fresh-wsp-large-tier-audit.py")] + [
                         args.study / name for name in ("launch-plan-approved.json", "actual-runner.json",
                                                        "launch-preflight.json", "run-start-verified.json")]:
            provenance[str(path)] = sha(path)
        summary["inputs_sha256"] = provenance
        save(args.output / "all-2338-numeric-items.json", numeric)
        save(args.output / "code-90-items.json", code)
        save(args.output / "summary.json", summary)
        print(json.dumps({"code": full["code_source_mean"], "overall": full["overall"],
                          "items": full["items"], "areas": full["areas"], "code_counts": summary["code_counts_by_source"],
                          "timing": full["timing"], "trace": trace_summary}, ensure_ascii=False))
    except Exception as error:
        save(args.output / "failure.json", {"error_type": type(error).__name__, "message": str(error),
                                           "analysis_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
