"""One fixed F04 CPU proposal with existing solver; no checkpoint, model or service mutation."""
from pathlib import Path
import copy, hashlib, json, os, shutil, subprocess, time
R=Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104");B=R/"worktrees/kairyu-bench-weight-tools"
P=R/"analysis/fresh-purpose-20261107-f04-relative-v1";OLD=R/"analysis/fresh-purpose-20261107-f03-native-v1"
PR=B/"docs/experiments/records/2026-11-07-f04-relative-v1"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value):
    p=P/name;assert not p.exists();p.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
started=time.time();oldplan=OLD/"proposal-plan.json"
assert sha(oldplan)=="56a4b866a41110ea1397305e9c3aa0b2c03f996f53df8b3953b9a237c387e240"
plan=json.loads(oldplan.read_text());original=copy.deepcopy(plan)
coverage=Path(plan["collections"]["correct-logprob"])/"coverage.json"
tensors=json.loads(coverage.read_text())["tensors"]
assert len(tensors)==421 and all(t["kind"]=="native"for t in tensors)
selected=sorted(t["name"]for t in tensors if not t["name"].endswith(".layer_scalar"))
assert len(selected)==361 and len(tensors)-len(selected)==60
assert all(len(t["shape"])==1 for t in tensors if t["name"]in selected)
run=R/"results/gemma4-fresh-f03-dev30-dp4-20261107-v1/run.json"
r=json.loads(run.read_text())
state=json.loads((OLD/"parent-state.json").read_text())
state.update(comparison_status="AV is the highest completed comparable parent by root decision; F03 WSP30 is incomplete and unranked at authorization.",hypothesis="Try relative native-vector cost while fixing all60 whole-layer scalars; raw Euclidean postnorm concentration and relative all421 scalar concentration are observed, neither establishes this metric superior.",prior_parent_state=dict(path=str(OLD/"parent-state.json"),sha256=sha(OLD/"parent-state.json")),
 source_plan=dict(path=str(oldplan),sha256=sha(oldplan)),relative_geometry=dict(path=str(R/"analysis/fresh-purpose-20261107-native-relative-energy-v1/report.json"),sha256=sha(R/"analysis/fresh-purpose-20261107-native-relative-energy-v1/report.json")),
 F03_state_at_prepare=dict(path=str(run),sha256=sha(run),status=r.get("status")),
 parent_gate=dict(required_before_materialization_or_evaluation=True,if_F03_becomes_highest="Do not materialize or evaluate this AV-based F04; record unscored_reason=parent_changed and redesign from new best parent.",automatic_rank_or_parent_update=False),
 background_load="CPU proposal runs concurrently with F03 WSP; single-thread nice15/ionice idle; WSP timing is not unloaded latency.")
state["restrictions"].append("No checkpoint materialization/serving/evaluation authorized; one CPU proposal only.")
save("parent-state.json",state)
plan["parent_state"]=dict(path=str(P/"parent-state.json"),sha256=sha(P/"parent-state.json"))
plan["native_tensors"]=selected
assert len(plan["candidates"])==1
method=plan["candidates"][0];method["id"]="fresh-native-relative-f04";method["cost_metric"]={"type":"relative","floor":0.0001}
assert method["constraints"]==original["candidates"][0]["constraints"] and method["rounding"]==original["candidates"][0]["rounding"]
assert all(plan[k]==original[k]for k in ["checkpoint","cases","collections"])
save("proposal-plan.json",plan)
script=B/"scripts/experiments/gemma4-purpose/propose_native_objectives.py"
argv=[str(R/"infra/weight-tools-ci/bin/python"),"-B",str(script),"--plan",str(P/"proposal-plan.json"),"--output",str(P/"proposal"),"--maximum-seconds","120"]
env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1",PYTHONPATH=str(B/"src"),TMPDIR=str(P/"tmp"));(P/"tmp").mkdir()
save("preparation-receipt.json",dict(source_sha256=sha(__file__),solver_sha256=sha(script),plan_sha256=sha(P/"proposal-plan.json"),observed_tensors=421,eligible_tensors=361,excluded_scalar_tensors=60,original_F03_eligible_tensors=len(original["native_tensors"]),all24constraints_unchanged=True,rounding_unchanged=True,checkpoint_cases_collections_unchanged=True,model_calls=0,solver_calls_planned=1))
for name,extra in [("preview",[]),("solve",["--run","--write-spec"])]:
    command=["nice","-n","15","ionice","-c","3",*argv,*extra]
    save(name+"-command.json",dict(argv=command,environment_overrides={k:env[k]for k in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","PYTHONDONTWRITEBYTECODE","PYTHONPATH","TMPDIR"]}))
    t=time.time()
    try:
        with (P/(name+".stdout")).open("x")as out,(P/(name+".stderr")).open("x")as err:
            result=subprocess.run(command,cwd=B,env=env,stdout=out,stderr=err,timeout=150)
        rc=result.returncode
    except BaseException as error:
        save("failure.json",dict(stage=name,error_type=type(error).__name__,seconds=time.time()-t));raise
    save(name+"-execution.json",dict(returncode=rc,seconds=time.time()-t))
    if rc:
        save("failure.json",dict(stage=name,returncode=rc,stdout=str(P/(name+".stdout")),stderr=str(P/(name+".stderr"))));raise SystemExit(rc)
report=json.loads((P/"proposal/report.json").read_text());m=report["methods"][0]
save("completion.json",dict(status="cpu_proposal_completed_not_materialized",seconds=time.time()-started,solver_seconds=report["seconds"],plan_sha256=sha(P/"proposal-plan.json"),report_sha256=sha(P/"proposal/report.json"),changed_elements=m["changed_elements"],changed_tensors=m["changed_tensors"],actual_delta_l2=m["actual_delta_l2"],weighted_delta_l2=m["weighted_delta_l2"],predicted_constraints_satisfied=m["predicted_constraints_satisfied"],model_calls=0,checkpoint_created=False,F03_final_state=json.loads(run.read_text()).get("status"),parent_gate=state["parent_gate"]))
for p in P.rglob("*"):
    if p.is_file():
        q=PR/p.relative_to(P);q.parent.mkdir(parents=True,exist_ok=True)
        if not q.exists():shutil.copyfile(p,q)
print((P/"completion.json").read_text())
