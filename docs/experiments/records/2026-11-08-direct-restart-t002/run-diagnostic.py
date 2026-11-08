"""Execute exactly one already registered diagnostic, preserving all outcomes."""
import pathlib,json,os,sys,subprocess,hashlib,time,fcntl,importlib.util,shutil
t=pathlib.Path(__file__).resolve().parent;r=t.parent.parent;w=r/"worktrees/kairyu-bench-weight-tools"
n=int(sys.argv[1]);assert n in (1,2,3)
plan=json.loads((t/"candidate-evaluation-plan.json").read_text());run=plan["runs"][n-1]
assert run["replicate"]==n
wrapper=plan["token_recording_wrapper"];assert hashlib.sha256(pathlib.Path(wrapper["path"]).read_bytes()).hexdigest()==wrapper["sha256"]
lock=(r/"candidate-evaluation.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
spec=importlib.util.spec_from_file_location("helper",w/"docs/experiments/scripts/redeploy-replica.py");helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
freeze=json.loads((t/"candidate-freeze.json").read_text());candidate=pathlib.Path(freeze["checkpoint_path"]);assert helper.inventory(candidate)==freeze["candidate_inventory_stat"]
deploy=json.loads((t/"deployment-apply/receipt.json").read_text());assert deploy["status"]=="completed"
name="gemma-4-31b-8gpu-gemma-0-1";current=helper.inspect(name);actual=helper.compare_after(deploy["before"],current,candidate,freeze["control_tree_sha256"])
assert actual["id"]==deploy["after"]["id"];helper.idle(current)
live=json.loads((t/("candidate-live-load-after-restart.json" if n==3 else "candidate-live-load-receipt.json")).read_text())
assert live["status"]=="verified_saved_mount_and_runtime_identity" and live["container"]["started_at"]==current["State"]["StartedAt"]
if n>1:
 previous=json.loads((t/f"candidate-r{n-1}-execution.json").read_text());assert previous["returncode"]==0
 prior_capture=json.loads((t/f"token-logprobs-r{n-1}.json").read_text());assert prior_capture["capture_complete"]
assert shutil.disk_usage(r).free>=214748364800
out=t/f"candidate-r{n}-execution.json";assert not out.exists() and not (r/"results"/run["run_id"]).exists() and not pathlib.Path(run["token_logprobs_output"]).exists()
tmp=pathlib.Path(run["temporary_directory"]);tmp.mkdir();(tmp/"xdg").mkdir();tmp.chmod(0o775);(tmp/"xdg").chmod(0o775);t.chmod(0o775)
record={"replicate":n,"run_id":run["run_id"],"status":"running","argv":run["argv"],"started_epoch":time.time(),"runtime_snapshot":actual,"process_started_at":current["State"]["StartedAt"],"token_capture_limit_bytes":2097152}
out.write_text(json.dumps(record,indent=2)+"\n")
with (t/f"candidate-r{n}.stdout").open("wb") as so,(t/f"candidate-r{n}.stderr").open("wb") as se:p=subprocess.run(run["argv"],stdout=so,stderr=se)
record.update(status="completed" if p.returncode==0 else "process_failed",returncode=p.returncode,seconds=time.time()-record["started_epoch"],nvme_free_bytes=shutil.disk_usage(r).free)
capture=pathlib.Path(run["token_logprobs_output"])
if capture.exists():record.update(token_capture_bytes=capture.stat().st_size,token_capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest())
out.write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps({k:record[k] for k in ["replicate","run_id","returncode","seconds","nvme_free_bytes"]}),flush=True)
raise SystemExit(p.returncode)
