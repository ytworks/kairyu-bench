#!/usr/bin/env python3
"""Audit one finished live capture; retain code call/rank/position identities."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

from kairyu_bench.wsp.observe.core import _REQUEST, json_lines

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--artifacts", type=Path, required=True)
parser.add_argument("--results", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert not args.output.exists(), "refuse to overwrite prior audit"

started = time.time()
capture = args.artifacts / "capture"
config = json.loads((capture / "capture.json").read_text())
run = json.loads((args.results / "run.json").read_text())
run_id = run["observation"]["run_id"]
items = {x["id"]: x for x in json_lines(args.results / "items.jsonl")}
code_ids = {key for key, item in items.items() if item["area"] == "code"}
traces = {x["trace_id"]: x for p in sorted(capture.glob("requests-*.jsonl")) for x in json_lines(p)}
# Read only the small metadata prefix from the standard merged pretty JSON.
header = []
with (args.artifacts / "merged.json").open() as stream:
    for line in stream:
        if line.strip() == '"tasks": [':
            break
        header.append(line)
    else:
        raise ValueError("merged task array not found")
merged_run = json.loads("".join(header) + '"tasks": []}]}')["runs"][0]
acceptance = {}
acceptance_counts = Counter()
for path in sorted(capture.glob("workers/*/acceptance.jsonl")):
    for record in json_lines(path):
        acceptance_counts[path.parent.name] += 1
        key = (str(record["replica"]), record["step"], record["request_id"])
        prior = acceptance.setdefault(key, record)
        assert prior == record
request_regex = _REQUEST
counts = Counter()
observed_item_ids = set()
worker_activation_counts = Counter()
trace_ranks = defaultdict(set)
trace_backend_requests = defaultdict(set)
unlinked = defaultdict(lambda: {"records": 0, "ranks": set(), "phases": set(), "positions": set()})
code = {key: {"id": key, "score": items[key]["score"], "calls": {}} for key in sorted(code_ids)}
for path in sorted(capture.glob("workers/*/observations.jsonl.gz")):
    sites_doc = json.loads((path.parent / "sites.json").read_text())
    sites = {s["site_id"]: s for s in sites_doc["sites"]}
    for batch in json_lines(path):
        assert batch["record_type"] == "activation_batch"
        assert batch["row_fields"] == ["request_id", "token_position", "phase", "speculative_index", "accepted"]
        assert batch["capture_id"] == config["capture_id"]
        rank = batch["rank"]
        for row_index, row in enumerate(batch["rows"]):
            request, position, phase, spec_index, accepted = row
            counts["records"] += len(batch["site_ids"])
            worker_activation_counts[path.parent.name] += len(batch["site_ids"])
            match = request_regex.search(request)
            trace = traces.get(match.group(1)) if match else None
            if trace is None:
                entry = unlinked[request]
                entry["records"] += len(batch["site_ids"])
                entry["ranks"].add(rank)
                entry["phases"].add(phase)
                entry["positions"].add(position)
                counts["unlinked_records"] += len(batch["site_ids"])
                continue
            if trace.get("run_id") != run_id:
                counts["excluded_other_run_records"] += len(batch["site_ids"])
                continue
            counts["official_records"] += len(batch["site_ids"])
            trace_ranks[trace["trace_id"]].add(rank)
            trace_backend_requests[trace["trace_id"]].add(request)
            key = trace["item_id"]
            observed_item_ids.add(key)
            if key not in code:
                continue
            usage = trace.get("usage") or {}
            prompt = usage.get("prompt_tokens")
            offset = position - prompt if prompt is not None else None
            call = code[key]["calls"].setdefault(request, {
                "request_id": request, "call_id": trace.get("call_id"),
                "input_sha256": trace.get("input_sha256"), "seed": trace.get("seed"),
                "seconds": trace.get("seconds"), "usage": usage, "ranks": {},
            })
            a = acceptance.get((str(batch["replica"]), batch["step"], request))
            if accepted is None and a is not None and spec_index is not None:
                accepted = spec_index < a["accepted_draft_tokens"]
            group = call["ranks"].setdefault(str(rank), {"prefill_positions": set(), "decode_offsets": set(), "by_acceptance": defaultdict(set), "roles": defaultdict(set), "profile_offset_64": []})
            if phase == "prefill":
                group["prefill_positions"].add(position)
            elif offset is not None:
                group["decode_offsets"].add(offset)
            group["by_acceptance"][str(accepted)].add((phase, position))
            for site_index, site_id in enumerate(batch["site_ids"]):
                site = sites[site_id]
                group["roles"][site["role"]].add(site["layer"])
                if phase == "decode" and offset == 64:
                    metrics = dict(zip(batch["metric_names"], batch["values"][site_index][row_index], strict=True))
                    group["profile_offset_64"].append({"layer": site["layer"], "role": site["role"], "token_position": position, "accepted": accepted, "metrics": metrics})

for task in code.values():
    for call in task["calls"].values():
        for group in call["ranks"].values():
            group["prefill_positions"] = sorted(group["prefill_positions"])
            group["decode_offsets"] = sorted(group["decode_offsets"])
            group["by_acceptance"] = {key: sorted(value) for key, value in group["by_acceptance"].items()}
            group["roles"] = {key: sorted(value) for key, value in group["roles"].items()}
for entry in unlinked.values():
    for key in ["ranks", "phases", "positions"]:
        entry[key] = sorted(entry[key])
assert counts["records"] == merged_run["capture"]["counts"]["records"]
assert counts["unlinked_records"] == merged_run["capture"]["counts"]["unlinked_records"]
worker_accounting = {}
for path in sorted(capture.glob("workers/*/status.json")):
    status = json.loads(path.read_text())
    name = path.parent.name
    activation_count = worker_activation_counts[name]
    acceptance_count = acceptance_counts[name]
    worker_accounting[name] = {"parsed_activation_records": activation_count, "parsed_acceptance_records": acceptance_count, "writer_records_written": status["records_written"], "counts_equal": activation_count + acceptance_count == status["records_written"], "closed": status["closed"], "status_mtime_epoch": path.stat().st_mtime, "writer_last_completed_epoch": status["writer_queue"]["last_completed_at_unix"]}
official_traces = {key: value for key, value in traces.items() if value.get("run_id") == run_id}
missing_trace_ranks = [{"trace_id": key, "item_id": value["item_id"], "request_scope": value.get("request_scope"), "operation": value.get("operation"), "observed_ranks": sorted(trace_ranks[key])} for key, value in official_traces.items() if trace_ranks[key] != {0, 1}]
result = {
    "schema_version": 1, "run_id": run_id, "capture_id": config["capture_id"],
    "recipe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "worker_accounting": worker_accounting, "official_client_trace_count": len(official_traces),
    "official_traces_missing_expected_rank": missing_trace_ranks,
    "observed_backend_requests_per_trace": {key: len(value) for key, value in trace_backend_requests.items()},
    "coverage_limit": "Trace-rank coverage does not prove every scheduled activation. Trial-scoped agent traces may contain several backend calls; their expected call count is not inferred.",
    "capture": merged_run["capture"], "counts": dict(counts), "unlinked": dict(unlinked),
    "code_items": list(code.values()), "observed_item_ids": sorted(observed_item_ids),
    "official_items_without_activation": sorted(set(items) - observed_item_ids),
    "elapsed_seconds": time.time() - started,
    "interpretation": "All 26 code items retained. Profiles use predeclared relative offset 64, rank-separated, first execution including rejected speculation. Missing offsets are not zero. Different prompts, generated tokens and accepted states are not causal paired evidence.",
}
args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
print(json.dumps({"code_items": len(code), "counts": dict(counts), "unlinked": dict(unlinked), "seconds": result["elapsed_seconds"]}, ensure_ascii=False))
