#!/usr/bin/env python3
"""Audit completed AV/F02 dev10 with existing strict comparison; no model calls."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from kairyu_bench.wsp.catalog import SOURCE_BY_NAME
from kairyu_bench.wsp.manifest import load_manifest
from kairyu_bench.wsp.report import compare, comparison_markdown, read_items


def read(path):
    return json.loads(path.read_text())


def option(command, name):
    return command[command.index(name) + 1]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Fresh output required."
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    spec = importlib.util.spec_from_file_location("full_audit", Path(__file__).with_name("fresh-wsp-large-tier-audit.py"))
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    sha = audit.sha
    root, repo = args.project, args.repository
    ids = ["gemma4-fresh-av-observed-dev10-20261107-v1", "gemma4-fresh-f02-observed-dev10-20261107-v1"]
    trees = ["5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf",
             "46680f4066f57c1596a1f4c66b67d23929c96b325943d981c6d74260cdbb3f5c"]
    groups = ["fresh-purpose-20261107-av-live10-v1", "fresh-purpose-20261107-f02-live10-v1"]
    manifest = load_manifest()
    expected = {x.id: x for x in manifest.select(10, "dev")}
    runs, items, configs, commands, receipts, provenance = [], [], [], [], [], {}
    for rid, tree, group in zip(ids, trees, groups, strict=True):
        directory, observation = root / "results" / rid, root / "analysis" / group
        checked = audit.audit_run(directory, manifest)
        run, report = read(directory / "run.json"), read(directory / "report.json")
        assert run["label"] == report["run"]["label"] == tree and report["run"]["status"] == "completed"
        assert run["tier"] == 10 and run["split"] == "dev"
        assert run["items"] == 778 and len(checked["areas"]) == 14
        assert not run.get("excluded_items") and not report["unscored"]
        rows = read_items(directory)
        for row in rows:
            item = expected[row["id"]]
            assert (row["source"], row["kind"], row["tier"], row["max_tokens"], row["samples"], row["area"]) == (
                item.source, item.kind, item.tier, item.max_tokens, item.samples, SOURCE_BY_NAME[item.source].area)
            assert math.isfinite(row["score"]) and all(math.isfinite(x) for x in row["sample_scores"])
        responses = [json.loads(line) for line in (directory / "responses.jsonl").read_text().splitlines()]
        assert len(responses) == len({x["id"] for x in responses}) == 778
        assert all(x["error"] is None and x["outcome"] is not None for x in responses)
        assert {x["id"] for x in responses} == expected.keys()
        command = read(observation / "wsp-command.json")["command"]
        assert command[3] == option(command, "--logprob-endpoint") == "http://gemma-0:8000/v1"
        assert [option(command, "--" + x + "-concurrency") for x in ("chat", "logprob", "agent")] == ["32", "8", "1"]
        assert "--only" not in command and "--exclude-item" not in command and "--limit" not in command
        capture = read(observation / "capture/capture.json")
        identities = read(observation / "checkpoint-identities.json")
        assert capture["checkpoint"]["sha256"] == identities["current"]["sha256"] == tree
        assert identities["current"]["hash_files"] and identities["current"]["source_stable"]
        assert capture["capture_id"] == run["observation"]["capture_id"]
        assert identities["capture_sha256"] == sha(observation / "capture/capture.json")
        assert read(observation / "plan.json")["topology"] == dict(tp=2, replicas=1, mtp_tokens=4)
        trace = [json.loads(line) for line in (observation / "capture/requests-client.jsonl").read_text().splitlines()]
        selected = [x for x in trace if x["run_id"] == rid and x.get("request_scope") == "request"]
        trace_summary = dict(rows=len(selected), operations=dict(Counter(x["operation"] for x in selected)),
                             errors=sum(x.get("error") is not None for x in selected),
                             retry_attempts=sum(x.get("attempt_index", 0) > 0 for x in selected),
                             excluded_other_run_rows=sum(x["run_id"] != rid for x in trace),
                             same_run_non_request_rows=sum(x["run_id"] == rid and x.get("request_scope") != "request" for x in trace))
        assert trace_summary["errors"] == trace_summary["retry_attempts"] == 0
        server_rows = []
        for server_path in sorted((observation / "capture").glob("server-requests-*.jsonl")):
            server_rows.extend(json.loads(line) for line in server_path.read_text().splitlines())
            provenance[str(server_path)] = sha(server_path)
        trace_summary["all_capture_server_statuses"] = dict(Counter(str(x["status"]) for x in server_rows))
        trace_summary["all_capture_server_rows"] = len(server_rows)
        trace_summary["server_scope_note"] = "Includes synthetic overhead and nested agent calls; server rows lack run_id."
        assert server_rows and all(x["status"] == 200 for x in server_rows)
        agents = []
        for path in sorted((directory / "agents").rglob("result.json")):
            result = read(path)
            if "task_name" not in result:
                continue
            trajectory_path = path.parent / "agent/trajectory.json"
            trajectory = read(trajectory_path)
            ar = result["agent_result"]
            agents.append(dict(task=result["task_name"], reward=result["verifier_result"]["rewards"]["reward"],
                               episodes=ar["metadata"]["n_episodes"],
                               trajectory_steps=len(trajectory["steps"]),
                               tool_calls=sum(len(s.get("tool_calls") or []) for s in trajectory["steps"]),
                               input_tokens=ar["n_input_tokens"], output_tokens=ar["n_output_tokens"],
                               exception_present=result["exception_info"] is not None,
                               seconds=(datetime.fromisoformat(result["finished_at"].replace("Z", "+00:00")) -
                                        datetime.fromisoformat(result["started_at"].replace("Z", "+00:00"))).total_seconds()))
            provenance[str(path)] = sha(path)
            provenance[str(trajectory_path)] = sha(trajectory_path)
        assert len(agents) == 2 and not any(x["exception_present"] for x in agents)
        checked["request_trace"] = trace_summary
        checked["agents"] = agents
        checked["source_report"] = report["sources"]
        runs.append(checked)
        items.append({x["id"]: x for x in rows})
        configs.append(capture)
        commands.append(command)
        receipts.append(read(observation / "source-dependencies.json"))
        provenance.update(checked["inputs"])
        for name in ("responses.jsonl",):
            provenance[str(directory / name)] = sha(directory / name)
        for name in ("plan.json", "wsp-command.json", "source-dependencies.json", "checkpoint-identities.json",
                     "attach.json", "capture/capture.json", "capture/requests-client.jsonl"):
            provenance[str(observation / name)] = sha(observation / name)
    assert receipts[0]["source"] == receipts[1]["source"]
    assert receipts[0]["dependencies"] == receipts[1]["dependencies"]
    for key in ("runtime", "limits", "sampling", "decode_offsets", "observation_mode", "data_kind"):
        assert configs[0][key] == configs[1][key]
    for name in ("chat_template.jinja", "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        assert next(x["sha256"] for x in configs[0]["checkpoint"]["files"] if x["path"] == name) == next(
            x["sha256"] for x in configs[1]["checkpoint"]["files"] if x["path"] == name)
    base_image_path = root / "analysis" / groups[0] / "runner-image-source.json"
    candidate_image_path = root / "analysis" / groups[1] / "actual-runner.json"
    baseline_image, candidate_image = read(base_image_path), read(candidate_image_path)
    assert baseline_image["docker_image_id"] == candidate_image["runner"]["image"] == (
        "sha256:f91bdcf997af6d7b194bec229812fd018277158f6795de7d91e58448936e7ad8")
    assert candidate_image["runner"]["command"] == commands[1][1:]
    for path in (base_image_path, candidate_image_path):
        provenance[str(path)] = sha(path)
    cache_path = root / "analysis" / groups[1] / "cache-parity.json"
    cache = read(cache_path)
    assert cache["result"]["all_equal"]
    provenance[str(cache_path)] = sha(cache_path)
    parent = root / "analysis/fresh-purpose-20261107-f02"
    amendment = repo / "docs/experiments/2026-11-07-f02-transfer-amendment.md"
    freeze = parent / "transfer-amendment-freeze.json"
    assert sha(amendment) == "1b6df2b79abb7d4db023651df632509a49d0b35e816c40c6d342d3884691a1ae"
    assert sha(freeze) == "344993b8b1b2e3622c581d363d2a4ccdfc28b914e7d22e8c51151f43a8fcdfc3"
    provenance[str(amendment)], provenance[str(freeze)] = sha(amendment), sha(freeze)
    comparison = compare(root / "results" / ids[0], root / "results" / ids[1], partial=False)
    assert not comparison["partial"] and not comparison["missing_areas"] and not comparison["unpaired_items"]
    assert len(comparison["areas"]) == 14 and len(comparison["sources"]) == 26
    for area, value in comparison["areas"].items():
        assert abs(value["base"]-runs[0]["areas"][area]) < 1e-12
        assert abs(value["candidate"]-runs[1]["areas"][area]) < 1e-12
    assert abs(comparison["overall"]["delta"]-(runs[1]["overall"]-runs[0]["overall"])) < 1e-12
    paired = []
    for item_id in sorted(expected):
        base, candidate = (mapping[item_id] for mapping in items)
        entry = {k: base[k] for k in ("id", "source", "area", "kind", "tier", "max_tokens", "samples")}
        entry.update(base_score=base["score"], candidate_score=candidate["score"],
                     delta=candidate["score"]-base["score"], base_sample_scores=base["sample_scores"],
                     candidate_sample_scores=candidate["sample_scores"])
        if entry["area"] == "code":
            entry.update(base_detail=base["detail"], candidate_detail=candidate["detail"])
        paired.append(entry)
    code = [x for x in paired if x["area"] == "code"]
    assert len(code) == 26
    dc, overall = comparison["areas"]["code"]["delta"], comparison["overall"]["delta"]
    worst = min(x["delta"] for x in comparison["areas"].values())
    decision = ("no_advancement_code_tie_or_loss" if dc <= 0 else
                "reject_retention_limit" if overall < -0.01 or worst < -0.10 else
                "provisional_permitted" if worst >= -0.05 else
                "paired_item_tradeoff_required" if overall >= 0 else "reject_moderate_loss_without_overall_gain")
    sources = {name: dict(area=runs[0]["sources"][name]["area"],
                         items=runs[0]["sources"][name]["items"],
                         base=runs[0]["sources"][name]["score"],
                         candidate=runs[1]["sources"][name]["score"],
                         delta=runs[1]["sources"][name]["score"]-runs[0]["sources"][name]["score"])
               for name in runs[0]["sources"]}
    summary = dict(status="completed_independent_full_score_audit", runs=runs,
                   score_decision=decision, strict_code_gain=dc > 0,
                   conditional_confirmation_pair_triggered=decision in ("provisional_permitted", "paired_item_tradeoff_required"),
                   proxy_gate="FAILED (permanent)", code_delta=dc, overall_delta=overall,
                   worst_area_delta=worst, all_sources=sources,
                   matched_778_unique_ids_and_metadata=True, code_changed_items=[x["id"] for x in code if x["delta"] != 0],
                   runtime_evidence=dict(actual_runner_image=baseline_image["docker_image_id"],
                                         serving_images=[read(root / "analysis" / g / "attach.json")["image_id"] for g in groups],
                                         observer_source_files_equal=len(receipts[0]["source"]),
                                         capture_runtime_sampling_limits_equal=True,
                                         checkpoint_and_generation_config_identity_verified=True,
                                         cache_inventory_equal=True,
                                         exact_condition_prior_repeat_count=0,
                                         timing_limitation="AV overlapped a GPU6 finite probe; F02 did not. Wall difference is not an isolated speed comparison."),
                   capture_completeness="Separately audited by observation owner; merged activation files deliberately not loaded here.",
                   inputs_sha256=provenance, source_sha256={
                       "wsp/report.py": sha(repo / "src/kairyu_bench/wsp/report.py"),
                       "reused_audit_script": sha(Path(__file__).with_name("fresh-wsp-large-tier-audit.py")),
                       "this_script": sha(Path(__file__))}, analysis_seconds=time.perf_counter()-started)
    save(args.output / "comparison.json", comparison)
    (args.output / "comparison.md").write_text(comparison_markdown(comparison))
    save(args.output / "all-778-paired-items.json", paired)
    save(args.output / "code-26-paired-items.json", code)
    save(args.output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "score_decision", "code_delta", "overall_delta",
                                             "worst_area_delta", "code_changed_items", "analysis_seconds")}))
    print(json.dumps(dict(areas=comparison["areas"], traces=[r["request_trace"] for r in runs],
                          agents=[r["agents"] for r in runs]), ensure_ascii=False))


if __name__ == "__main__":
    main()
