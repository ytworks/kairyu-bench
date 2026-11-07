from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import time

R = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
REPO = R / "worktrees/kairyu-bench-weight-tools"
N = R / "analysis/fresh-purpose-20261107-av-behavior12-native-gpu4-v2"
REC = REPO / "docs/experiments/records/2026-11-07-native-gpu4-owner/measurement-v2"


def read(p):
    return json.loads(p.read_text())


def pin(p):
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)


plan = read(N / "root-approved-plan.json")
completion = read(N / "completion.json")
assert completion["status"] == "completed" and completion["restored"] and completion["protected_exact"]
assert completion["collections_completed"] and completion["collector_client_finished"]
assert not Path("/proc") .joinpath(str(completion["pid"])).exists()
lock = R / "candidate-evaluation.lock"
st = lock.stat()
key = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
lock_rows = [row for row in Path("/proc/locks").read_text().splitlines() if key in row]
assert not lock_rows
owner = REPO / "docs/experiments/scripts/fresh-av-native-gpu4-20261107.py"
spec = importlib.util.spec_from_file_location("close_native_owner", owner)
o = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o)
guard = o.helper(plan)
o.validate(plan, for_run=True)
o.services(plan, guard, serving=False)
target = guard.snapshot(o.TARGET)
assert target["running"] and target["health"] == "healthy"
assert all(target[k] == v for k, v in plan["target"].items() if k not in ("started_at", "running", "health"))
assert len(guard.docker("ps", "-aq", "--no-trunc").split()) == 27
expected_ids = ["behavior-" + g + str(i) for g in "ABCDEF" for i in (1, 2)]
values, objective_rows, inventory = {}, {}, []
for objective, key in (("correct-logprob", "correct_logprob"), ("margin", "margin")):
    out = N / "work" / objective
    summary = read(out / "summary.json")
    assert summary["status"] == "completed" and summary["source_values_unchanged"]
    assert read(out / "frame-evidence.json") == read(Path(plan["frame"]["path"]))
    assert summary["coverage"]["native_tensors"] == 421 and summary["coverage"]["native_coordinates"] == 1331516
    measured = {v["id"]:v[key] for fold in summary["folds"] for v in fold[key + "s"]}
    assert sorted(measured) == expected_ids and len(summary["folds"]) == 12
    values[objective] = measured
    runtime = summary["runtime"]
    storage = runtime["saved_activation_metrics"]
    objective_rows[objective] = dict(summary=pin(out / "summary.json"), seconds_excluding_model_load=summary["seconds"],
        forward_count=sum(f["model_forward_count"] for f in summary["folds"]),
        gpu_peak_allocated_gib=runtime["gpu_peak_allocated_GiB"], gpu_peak_reserved_gib=runtime["gpu_peak_reserved_GiB"],
        saved_activation_metrics={k:v for k,v in storage.items() if k != "offloaded_shapes"},
        runtime={k:v for k,v in runtime.items() if k != "saved_activation_metrics"},
        source_values_unchanged=True, frames_exact=True, unused_tensors=sorted(set(v for f in summary["folds"] for v in f["unused_tensors"])))
    inventory += [pin(p) for p in sorted(out.iterdir()) if p.is_file()]
behavior = read(Path(plan["native_behavior_result"]["path"]))
behavior_rows = {r["id"]:r["scored_result"] for r in behavior["rows"]}
rows = [dict(id=i, correct_logprob=values["correct-logprob"][i], margin=values["margin"][i],
    native_tests_passed=behavior_rows[i]["tests_passed"], native_tests_total=behavior_rows[i]["tests_total"],
    native_case_passed=behavior_rows[i]["passed"]) for i in expected_ids]
result = dict(schema_version=1, status="completed", plan=pin(N / "root-approved-plan.json"),
    completion=pin(N / "completion.json"), elapsed_owner_seconds=completion["elapsed_seconds"],
    objectives=objective_rows, cases=rows, raw_inventory=inventory, checkpoint_tree_sha256=o.TREE,
    failure_files=[str(p) for p in (N / "work").rglob("failure.json")], retries=0,
    source_file_pins_unchanged=True, checkpoint_all9_stats_unchanged=True,
    after=dict(epoch=time.time(), target=target, target_queues=o.idle(guard), protected_exact=True,
        container_count=27, owner_pid_absent=True, global_lock_rows=lock_rows),
    new_candidates=0, solver_calls=0, checkpoint_edits=0, transfer_cases=0,
    limits=["Teacher-forced empty-thought diagnostic; not native-generation accuracy or WSP.",
        "CPU bytes are cumulative copy events, not peak RAM or unique reclaimed storage.",
        "No host RAM peak measured. Collector seconds exclude model load; owner wall includes stop/load/collection/restore."])
with (N / "measurement-closure.json").open("x") as f:
    json.dump(result, f, indent=2, allow_nan=False)
    f.write("\n")
REC.mkdir(exist_ok=False)
for name in ["measurement-closure.json", "root-launch.json", "root-approved-plan.json", "launch.json", "completion.json",
             "correct-logprob-start.json", "correct-logprob-complete.json", "margin-start.json", "margin-complete.json",
             "correct-logprob.log", "margin.log", "read-only-monitor.jsonl"]:
    shutil.copyfile(N / name, REC / name)
for objective in o.OBJECTIVES:
    archive = REC / objective
    archive.mkdir()
    for p in (N / "work" / objective).glob("*.json"):
        shutil.copyfile(p, archive / p.name)
print(json.dumps(dict(closure=pin(N / "measurement-closure.json"), elapsed_owner_seconds=completion["elapsed_seconds"],
    objectives={k:{x:v for x,v in row.items() if x not in ("runtime", "saved_activation_metrics")} for k,row in objective_rows.items()},
    cases=rows, failure_files=result["failure_files"], restored=True, lock_released=True)))
