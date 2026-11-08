"""Perform the preregistered same-candidate restart before replicate three."""
import pathlib,json,os,subprocess,time,fcntl,importlib.util,shutil
t=pathlib.Path(__file__).resolve().parent;r=t.parent.parent;w=r/"worktrees/kairyu-bench-weight-tools"
lock=(r/"candidate-evaluation.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for n in (1,2):
 assert json.loads((t/f"candidate-r{n}-execution.json").read_text())["returncode"]==0
 assert json.loads((t/f"token-logprobs-r{n}.json").read_text())["capture_complete"]
p=t/"restart-before-r3.json";assert not p.exists()
spec=importlib.util.spec_from_file_location("helper",w/"docs/experiments/scripts/redeploy-replica.py");h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
deploy=json.loads((t/"deployment-apply/receipt.json").read_text());freeze=json.loads((t/"candidate-freeze.json").read_text());cp=pathlib.Path(freeze["checkpoint_path"]);assert h.inventory(cp)==freeze["candidate_inventory_stat"]
name="gemma-4-31b-8gpu-gemma-0-1";old=h.inspect(name);before=h.compare_after(deploy["before"],old,cp,freeze["control_tree_sha256"]);assert old["Id"]==deploy["after"]["id"];h.idle(old)
record={"status":"restarting","container_id":old["Id"],"before":before,"previous_started_at":old["State"]["StartedAt"],"started_epoch":time.time(),"argv":["docker","restart","--timeout","30",old["Id"]],"reason":"exact preregistered process restart before third replicate"}
p.write_text(json.dumps(record,indent=2)+"\n")
q=subprocess.run(record["argv"],capture_output=True,timeout=120);record["restart_returncode"]=q.returncode
if q.returncode:
 record["status"]="restart_command_failed";p.write_text(json.dumps(record,indent=2)+"\n");raise SystemExit(q.returncode)
deadline=time.monotonic()+900
while time.monotonic()<deadline:
 current=h.inspect(name)
 assert current["Id"]==old["Id"] and shutil.disk_usage(r).free>=214748364800
 if current["State"]["Running"] and current["State"].get("Health",{}).get("Status")=="healthy":break
 time.sleep(5)
else:raise RuntimeError("restart readiness timeout")
after=h.compare_after(deploy["before"],current,cp,freeze["control_tree_sha256"]);assert h.inventory(cp)==freeze["candidate_inventory_stat"] and old["State"]["StartedAt"]!=current["State"]["StartedAt"]
record.update(status="completed",after=after,new_started_at=current["State"]["StartedAt"],seconds=time.time()-record["started_epoch"]);p.write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps({"status":record["status"],"seconds":record["seconds"],"container_id":old["Id"]}))
