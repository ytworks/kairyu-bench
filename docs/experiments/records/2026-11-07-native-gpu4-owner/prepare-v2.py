from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import time

R = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
REPO = R / "worktrees/kairyu-bench-weight-tools"
N1 = R / "analysis/fresh-purpose-20261107-av-behavior12-native-gpu4-v1"
N2 = R / "analysis/fresh-purpose-20261107-av-behavior12-native-gpu4-v2"
GEN = R / "analysis/fresh-purpose-20261107-av-behavior12-native-v1"
SCORE = R / "analysis/fresh-purpose-20261107-av-behavior12-score-v2"
REC = REPO / "docs/experiments/records/2026-11-07-native-gpu4-owner"
PY = R / "infra/weight-tools-ci/bin/python"


def read(p):
    return json.loads(p.read_text())


def pin(p):
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def save(p, value):
    with p.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


N2.mkdir(exist_ok=False)
responses = read(GEN / "generation/train-responses.json")
report = read(SCORE / "evaluation/report.json")
receipt = read(SCORE / "execution-receipt.json")
old = read(N1 / "plan.json")
expected = ["behavior-" + g + str(i) for g in "ABCDEF" for i in (1, 2)]
answers = {r["id"]: r for r in responses["responses"]}
assert len(answers) == len(responses["responses"]) == 12 and sorted(answers) == expected
assert report["status"] == receipt["status"] == "completed"
assert receipt["report_sha256"] == pin(SCORE / "evaluation/report.json")["sha256"]
assert report["target_ids"] == expected and len(report["results"]) == 12
assert responses["cases_sha256"] == old["cases"]["sha256"]
assert report["inputs"][0]["sha256"] == old["cases"]["sha256"]
assert report["inputs"][1]["sha256"] == pin(GEN / "generation/train-responses.json")["sha256"]
rows = []
for row in report["results"]:
    answer = answers[row["id"]]
    assert row["public_answer_sha256"] == hashlib.sha256(answer["public_answer"].encode()).hexdigest()
    rows.append(dict(id=row["id"], generation={k: answer.get(k) for k in
        ("request_id", "finish_reason", "usage", "seconds", "error_type")}, scored_result=row))
assert sorted(r["id"] for r in rows) == expected
assert sum(len(r["scored_result"]["trial_results"]) for r in rows) == 192
artifacts = [GEN / name for name in ("generation/train-responses.json", "generation/receipt.json",
    "owner/owner-closure.json", "owner/normal-lock-release.json", "owner/serving-closure.json",
    "read-only-generation-summary.json")]
artifacts += [SCORE / name for name in ("evaluation/report.json", "execution-receipt.json",
    "raw-worker-outputs.json", "before.json", "after.json")]
identity = dict(schema_version=1, status="generation_and_numeric_execution_complete",
    cases=old["cases"], checkpoint_tree_sha256=responses["generation"]["checkpoint_sha256"],
    artifacts=[pin(p) for p in artifacts], case_ids=expected, rows=rows,
    generation=dict(cases=12, errors=read(GEN / "generation/receipt.json")["errors"],
                    owner_elapsed_seconds=read(GEN / "owner/owner-closure.json")["elapsed_seconds"]),
    execution=dict(passed=report["passed"], failed=report["failed"], unknown=report["unknown"],
        evaluated_cases=sum(r["status"] == "evaluated" for r in report["results"]),
        tests_passed=sum(r["tests_passed"] for r in report["results"]),
        tests_total=sum(r["tests_total"] for r in report["results"]), seconds=receipt["seconds"]),
    limits=report["limits"], transfer_cases_measured=0,
    interpretation="Authored numeric/interface fixtures only, not WSP; unknown stays unknown; B method requirements not verified by numeric fixtures.")
save(N2 / "native-behavior-result-identity.json", identity)
shutil.copytree(N1 / "source", N2 / "source")
owner = REPO / "docs/experiments/scripts/fresh-av-native-gpu4-20261107.py"
spec = importlib.util.spec_from_file_location("native_owner", owner)
o = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o)
plan = copy.deepcopy(old)
plan.update(status="prepared_full_cpu_preview_pending_root_decision", approved_for_execution=False,
    source_root=str(N2 / "source"), output_root=str(N2), container_prefix="fresh-av-behavior12-native-gpu4-20261107-v2",
    source_pins={str(p): pin(p)["sha256"] for p in [*(N2 / "source").rglob("*.py"), owner, owner.with_name("fresh-f01-finite.py")]},
    native_behavior_result=pin(N2 / "native-behavior-result-identity.json"),
    pending=["root final purpose/plan review and explicit execution authorization"],
    prior_plan=pin(N1 / "plan.json"), supersedes_reason="late-create cleanup race fixed before any GPU execution")
guard = o.helper(plan)
ids = guard.docker("ps", "-aq", "--no-trunc").split()
names = [json.loads(guard.docker("inspect", cid))[0]["Name"].lstrip("/") for cid in ids]
snapshots = {name: guard.snapshot(name) for name in names}
assert len(snapshots) == 27
plan["target"] = snapshots.pop(o.TARGET)
plan["protected"] = snapshots
assert plan["target"] == old["target"] and snapshots == old["protected"]
save(N2 / "fresh-readonly-serving.json", dict(epoch=time.time(), target=plan["target"], protected=plan["protected"],
    target_idle=o.idle(guard), matches_prior_plan=True, service_changes=0, model_calls=0))
env = {**os.environ, "PYTHONPATH":str(N2 / "source/src"), "PYTHONDONTWRITEBYTECODE":"1"}
for objective in o.OBJECTIVES:
    args = o.native_args(plan, objective)
    start = time.monotonic()
    q = subprocess.run([str(PY), "-B", *args], cwd=N2, env=env, text=True, capture_output=True)
    f = N2 / (objective + "-cpu-preview.json")
    save(f, dict(returncode=q.returncode, model_calls=0, collector_arguments=args,
                 stdout=q.stdout, stderr=q.stderr, seconds=time.monotonic()-start))
    assert q.returncode == 0
    plan["cpu_previews"][objective] = pin(f)
save(N2 / "plan.json", plan)
start = time.monotonic()
q = subprocess.run([str(PY), "-B", str(owner), "--plan", str(N2 / "plan.json")],
                   cwd=REPO, env=env, text=True, capture_output=True)
save(N2 / "owner-full-preview.json", dict(returncode=q.returncode, stdout=q.stdout, stderr=q.stderr,
    seconds=time.monotonic()-start, plan=pin(N2 / "plan.json"), model_calls=0, service_changes=0,
    execution_authorized=False))
assert q.returncode == 0, q.stderr
save(N2 / "preparation-receipt.json", dict(plan=pin(N2 / "plan.json"), owner=pin(owner),
    operation=pin(Path(__file__)), identity=pin(N2 / "native-behavior-result-identity.json"),
    model_calls=0, service_changes=0, global_lock_acquired=False, approved_for_execution=False))
archive = REC / "plan-v2"
archive.mkdir()
for f in N2.glob("*.json"):
    shutil.copyfile(f, archive / f.name)
print(json.dumps(dict(plan=pin(N2 / "plan.json"), identity=pin(N2 / "native-behavior-result-identity.json"),
                     owner_preview=pin(N2 / "owner-full-preview.json"), execution=identity["execution"])))
