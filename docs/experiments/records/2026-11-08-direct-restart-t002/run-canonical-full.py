"""Run the registered canonical full benchmark once; preserve process outcomes."""
import pathlib,json,os,sys,subprocess,hashlib,time,shutil,fcntl,importlib.util,signal
t=pathlib.Path(__file__).resolve().parent;r=t.parent.parent;w=r/"worktrees/kairyu-bench-weight-tools"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 q=p.with_suffix(".tmp");q.write_text(json.dumps(d,indent=2)+"\n");q.replace(p)
lock=(r/"candidate-evaluation.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
guard=json.loads((t/"full-evaluation-start-guard.json").read_text());assert guard["full_evaluation_permitted"]
for p,digest in guard["evidence_sha256"].items():assert sha(pathlib.Path(p))==digest
plan=json.loads((t/"canonical-full-evaluation-plan.json").read_text());assert plan["items"]==778 and not plan["excluded_item_ids"] and plan["only_sources"] is None
assert os.environ.get("HF_TOKEN") and not (r/"results"/plan["run_id"]).exists()
spec=importlib.util.spec_from_file_location("helper",w/"docs/experiments/scripts/redeploy-replica.py");h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
freeze=json.loads((t/"candidate-freeze.json").read_text());assert h.inventory(pathlib.Path(plan["candidate_path"]))==freeze["candidate_inventory_stat"]
provided=json.loads((t/"full-serving-preflight-v2.json").read_text());assert provided["status"]=="passed"
for service,record in provided["replicas"].items():
 c=h.inspect("gemma-4-31b-8gpu-"+service+"-1");assert h.snapshot(c)==record["snapshot"] and c["State"]["StartedAt"]==record["started_at"];h.idle(c)
assert shutil.disk_usage(r).free>=214748364800
env=dict(os.environ,**plan["environment_nonsecret"]);out=t/"full-evaluation-execution.json";assert not out.exists()
record={"status":"running","run_id":plan["run_id"],"argv":plan["argv"],"started_epoch":time.time(),"plan_sha256":sha(t/"canonical-full-evaluation-plan.json"),"candidate_tree":plan["candidate_tree_sha256"],"environment_nonsecret":plan["environment_nonsecret"],"hf_token_present":True,"process_failures":[]}
save(out,record);samples=[]
try:
 with (t/"full-evaluation.stdout").open("wb") as so,(t/"full-evaluation.stderr").open("wb") as se:
  p=subprocess.Popen(plan["argv"],cwd=w,env=env,stdout=so,stderr=se,start_new_session=True)
  record["pid"]=p.pid;save(out,record)
  while p.poll() is None:
   free=shutil.disk_usage(r).free;samples.append({"epoch":time.time(),"nvme_free_bytes":free})
   save(t/"full-evaluation-progress.json",{"pid":p.pid,"samples":samples})
   if free<214748364800:
    record["process_failures"].append("NVMe free floor violated");os.killpg(p.pid,signal.SIGTERM)
    p.wait(timeout=60);break
   time.sleep(5)
 record.update(returncode=p.returncode,status="process_completed" if p.returncode==0 else "process_failed")
except BaseException as error:
 record["process_failures"].append(type(error).__name__);record["status"]="orchestration_failed"
 raise
finally:
 record.update(seconds=time.time()-record["started_epoch"],finished_epoch=time.time(),nvme_free_bytes=shutil.disk_usage(r).free);save(out,record)
print(json.dumps({k:record[k] for k in ["status","run_id","returncode","seconds","nvme_free_bytes"]}),flush=True)
raise SystemExit(record["returncode"])
