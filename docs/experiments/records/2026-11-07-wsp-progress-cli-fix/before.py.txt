"""Read-only snapshot for this complete AV dev30 run; never ranks partial scores."""
from __future__ import annotations

import collections
import json
import os
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path

R = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
N = R / "analysis/fresh-purpose-20261107-av-dev30-dp4-v1"
RUN = R / "results/gemma4-fresh-av-dev30-dp4-20261107-v1"
CODE = {"humaneval-plus", "livecodebench", "mbpp-plus", "scicode"}


def rows(path: Path) -> tuple[list[dict], int]:
    if not path.exists():
        return [], 0
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    incomplete = int(bool(lines and not lines[-1].endswith(b"\n")))
    return [json.loads(x) for x in (lines[:-1] if incomplete else lines)], incomplete


def main() -> None:
    now = time.time()
    responses, response_tail = rows(RUN / "responses.jsonl")
    items, item_tail = rows(RUN / "items.jsonl")
    traces, trace_tail = rows(N / "http-trace/requests-client.jsonl")
    traces = [x for x in traces if x.get("run_id") == RUN.name]
    response_ids = [x["id"] for x in responses]
    per_code = {}
    for source in sorted(CODE):
        scored = [x for x in items if x["source"] == source and x.get("score") is not None]
        per_code[source] = {"responses": [x for x in response_ids if x.startswith(source + ":")], "scored_count": len(scored) if items else None, "scored_items": [{"id": x["id"], "score": x["score"]} for x in scored]}
    terminals = []
    base = RUN / "agents/terminal-bench/raw/terminal-bench/jobs"
    for path in sorted(base.glob("*/*/result.json")):
        trial = json.loads(path.read_text())
        if not trial.get("task_name"):
            continue
        trajectory = path.parent / "agent/trajectory.json"
        steps = json.loads(trajectory.read_text()).get("steps", []) if trajectory.exists() else None
        span = None
        if trial.get("started_at") and trial.get("finished_at"):
            span = (datetime.fromisoformat(trial["finished_at"]) - datetime.fromisoformat(trial["started_at"])).total_seconds()
        terminals.append({"task": trial["task_name"], "trial": trial["trial_name"], "reward": ((trial.get("verifier_result") or {}).get("rewards") or {}).get("reward"), "exception": (trial.get("exception_info") or {}).get("exception_type"), "elapsed_seconds": span, "agent_steps": sum(x.get("source") == "agent" for x in steps) if steps is not None else None, "tool_calls": sum(len(x.get("tool_calls") or []) for x in steps) if steps is not None else None})
    terminal_active = []
    finished_trials = {x["trial"] for x in terminals}
    for path in sorted(base.glob("*/*/agent/trajectory.json")):
        trial_name = path.parent.parent.name
        if trial_name in finished_trials:
            continue
        try:
            steps = json.loads(path.read_text()).get("steps", [])
            terminal_active.append({"trial": trial_name, "task": trial_name.rsplit("__", 1)[0], "agent_steps": sum(x.get("source") == "agent" for x in steps), "tool_calls": sum(len(x.get("tool_calls") or []) for x in steps), "first_step_timestamp": steps[0].get("timestamp") if steps else None, "last_step_timestamp": steps[-1].get("timestamp") if steps else None, "elapsed_seconds": None, "state": "trajectory_present_no_final_result"})
        except json.JSONDecodeError:
            terminal_active.append({"trial": trial_name, "state": "trajectory_write_in_progress", "agent_steps": None, "tool_calls": None})
    swe_active_ids = []
    for path in (RUN / "agents/swe-bench-pro/raw/swe-bench-pro/items").glob("*/instance-ids.txt"):
        swe_active_ids.extend(path.read_text().splitlines())
    swe_results = []
    for path in (RUN / "agents/swe-bench-pro/raw/swe-bench-pro/items").glob("*/evaluation/eval_results.json"):
        ids_path = path.parent.parent / "instance-ids.txt"
        expected = ids_path.read_text().strip()
        verdict = json.loads(path.read_text())
        if set(verdict) != {expected} or not isinstance(verdict[expected], bool):
            swe_results.append({"id": expected, "schema_error": True})
            continue
        detail = {"id": expected, "reward": float(verdict[expected]), "seconds_selection_to_verdict_mtime": path.stat().st_mtime - ids_path.stat().st_mtime, "agent_steps": None, "tool_calls": None, "api_calls": None}
        trajectory = path.parent.parent / "generation" / expected / (expected + ".traj.json")
        if trajectory.exists():
            data = json.loads(trajectory.read_text())
            messages = data.get("messages", [])
            detail.update({"agent_steps": sum(x.get("role") == "assistant" for x in messages), "tool_calls": sum(len(x.get("tool_calls") or []) for x in messages), "tool_response_messages": sum(x.get("role") == "tool" for x in messages), "api_calls": data["info"].get("model_stats", {}).get("api_calls"), "exit_status": data["info"].get("exit_status"), "step_definition": "assistant messages in saved mini-swe-agent trajectory"})
        swe_results.append(detail)
    api = []
    for service in json.loads((N / "launch-preflight.json").read_text())["services"]:
        try:
            with urllib.request.urlopen(f"http://{service['ip']}:8000/metrics", timeout=10) as response:
                text = response.read().decode()
            counts = {name: [float(x) for x in re.findall(r"^vllm:num_requests_" + name + r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)", text, re.M)] for name in ["running", "waiting"]}
            api.append({"service": service["service"], **counts})
        except Exception as error:
            api.append({"service": service["service"], "monitor_error_type": type(error).__name__})
    snapshot = {"epoch": now, "elapsed_seconds": now - json.loads((N / "owner-start.json").read_text())["epoch"], "run": json.loads((RUN / "run.json").read_text()), "response_completed": len(responses), "response_unique": len(set(response_ids)), "response_errors": [x["id"] for x in responses if x.get("error")], "source_completed": dict(collections.Counter(x.split(":", 1)[0] for x in response_ids)), "code": per_code, "canonical_scored": sum(x.get("score") is not None for x in items) if items else None, "canonical_unscored": sum(x.get("score") is None for x in items) if items else None, "client_trace_records": len(traces), "http_request_trace_records": sum(x.get("request_scope") == "request" for x in traces), "client_error_count": sum(bool(x.get("error")) for x in traces), "client_retry_attempts": sum((x.get("attempt_index") or 0) > 0 for x in traces), "terminal_results": terminals, "terminal_completed": len(terminals), "terminal_active": terminal_active, "swe_started_instance_ids": swe_active_ids, "swe_response_ids": [x for x in response_ids if x.startswith("swe-bench-pro:")], "swe_results": swe_results, "swe_verifier_completed": len(swe_results), "api": api, "incomplete_jsonl_tails": {"responses": response_tail, "items": item_tail, "traces": trace_tail}, "nvme_free_bytes": os.statvfs(R).f_bavail * os.statvfs(R).f_frsize, "owner_failure_exists": (N / "owner-failure.json").exists(), "owner_complete_exists": (N / "owner-complete.json").exists(), "partial_ranking": False}
    with (N / "progress.jsonl").open("a") as output:
        output.write(json.dumps(snapshot) + "\n")
    print(json.dumps(snapshot))


if __name__ == "__main__":
    main()
