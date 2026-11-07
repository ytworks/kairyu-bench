"""Execute the one registered saved-weight edit under its exact capacity guards."""
import pathlib,json,subprocess,os,time,datetime,hashlib,shutil,fcntl
r=pathlib.Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104");t=pathlib.Path(__file__).resolve().parent;s=t.parent;w=r/"worktrees/kairyu-bench-weight-tools"
def save(p,d):
 q=p.with_suffix(".tmp");q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n");q.replace(p)
lock=(r/"candidate-evaluation.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
capacity=json.loads((t/"storage-creation-reservation.json").read_text());candidate=pathlib.Path(capacity["candidate"]);source=pathlib.Path(capacity["source"])
assert not candidate.exists() and candidate.parent.is_dir()
ledger=json.loads((r/"storage-state.json").read_text());assert ledger["checkpoint_count"]==0 and ledger["adopted_checkpoint"] is None
assert capacity["peak_additional_weight_upper_bound_bytes"]<=150*1024**3 and capacity["cache_peak_reserved_bytes"]<=100*1024**3
assert shutil.disk_usage(r).free>=capacity["start_free_required_bytes"]
assert hashlib.sha256(pathlib.Path(capacity["writer_source"]).read_bytes()).hexdigest()==capacity["writer_source_sha256"]
reg=json.loads((t/"registration.json").read_text());assert hashlib.sha256((t/"edit-spec.json").read_bytes()).hexdigest()==reg["edit_spec_sha256"]
review=json.loads((t/"opus55-review/receipt.json").read_text());assert review["verified_requested_model"] and review["assistant_models"]==["claude-opus-5-5"]
assert (s/"arc-observation-tail-plan/runtime-postnorm-semantics.json").is_file()
rounding=json.loads((t/"rounded-proposal-verification.json").read_text());assert all(x["bitwise_independent_match"] for x in rounding["rows"])
for suffix in ("", "2", "3"):
 run=r/"results"/("restart-direct-arc-batch01-off-control"+suffix+"-20261108")
 rows=[json.loads(l) for l in (run/"items.jsonl").read_text().splitlines()];assert len(rows)==10 and all(x["error"] is None for x in rows)
current=[]
for i in range(4):
 d=json.loads(subprocess.check_output(["docker","inspect",f"gemma-4-31b-8gpu-gemma-{i}-1"]))[0]
 assert d["State"]["Running"] and d["State"].get("Health",{}).get("Status")=="healthy"
 assert any(m["Destination"]=="/checkpoint" and pathlib.Path(m["Source"])==source and not m["RW"] for m in d["Mounts"])
 current.append({"id":d["Id"],"name":d["Name"],"image":d["Image"],"healthy":True})
reg.update(status="creating_registered_candidate",materialization_allowed_now=True,pending_before_materialization=[],creation_started_at=datetime.datetime.now(datetime.timezone.utc).isoformat());save(t/"registration.json",reg)
ledger.update(generation_allowed=True,generation_blockers=[],candidate_checkpoint=str(candidate),checkpoint_count=1,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat());ledger["storage_release"]["creation_apply_started"]=True
ledger["candidate_creation_reservation"]["state"]="writer_running"
ledger["entries"].append({"path":str(candidate),"purpose":"registered candidate001 independent saved weights","owner_trial":reg["trial_id"],"state":"creating","allocated_bytes":0,"reserved_bytes":capacity["weight_reservation_bytes"],"shared_references":[],"retain_reason":"single currently evaluated candidate","delete_condition":"reject/failure after writer/evaluator exit and reference checks"});save(r/"storage-state.json",ledger)
args=json.loads((t/"editor-preview-execution.json").read_text())["argv"];args[args.index("--reserve-bytes")+1]=str(capacity["writer_reserve_bytes_argument"]);args.append("--apply")
env=dict(os.environ,PYTHONPATH=str(w/"src"),PYTHONDONTWRITEBYTECODE="1")
start=time.time();samples=[]
with (t/"creation.stdout").open("wb") as so,(t/"creation.stderr").open("wb") as se:
 p=subprocess.Popen(args,cwd=w,env=env,stdout=so,stderr=se)
 while p.poll() is None:
  free=shutil.disk_usage(r).free;allocated=sum(f.stat().st_blocks*512 for f in candidate.rglob("*") if f.is_file()) if candidate.exists() else 0
  samples.append({"epoch":time.time(),"nvme_free_bytes":free,"candidate_allocated_bytes":allocated})
  save(t/"creation-progress.json",{"pid":p.pid,"samples":samples,"max_reserved_bytes":capacity["weight_reservation_bytes"]})
  if free<capacity["minimum_free_bytes"] or allocated>capacity["weight_reservation_bytes"]:p.terminate();p.wait();raise RuntimeError("capacity guard")
  time.sleep(5)
job={"candidate":str(candidate),"manifest":str(t/"edit-manifest.json"),"exit_code":p.returncode,"spec_sha256":reg["edit_spec_sha256"],"started_epoch":start,"seconds":time.time()-start,"argv":args}
save(t/"creation-receipt.json",{"jobs":[job],"serving_preflight":current,"monitor_samples":samples})
reg["status"]="candidate_created_pending_stream_verification" if p.returncode==0 else "creation_failed";save(t/"registration.json",reg)
print(json.dumps({"creation_exit_code":p.returncode,"seconds":job["seconds"]}),flush=True)
if p.returncode:raise SystemExit(p.returncode)
# Fresh exact streaming reconstruction also reads the entire candidate and all retained files.
chain=[str(r/"infra/weight-tools-ci/bin/python"),"-m","kairyu_bench.wsp.weights.chain",str(source),str(candidate),"--manifest",str(t/"edit-manifest.json")]
with (t/"chain-verification.json").open("wb") as so,(t/"chain-verification.stderr").open("wb") as se:q=subprocess.run(chain,cwd=w,env=env,stdout=so,stderr=se)
save(t/"chain-execution.json",{"argv":chain,"exit_code":q.returncode,"finished_at":datetime.datetime.now(datetime.timezone.utc).isoformat()})
print(json.dumps({"chain_exit_code":q.returncode}),flush=True)
reg["status"]="created_and_stream_verified" if q.returncode==0 else "stream_verification_failed";save(t/"registration.json",reg)
ledger=json.loads((r/"storage-state.json").read_text());allocated=sum(f.stat().st_blocks*512 for f in candidate.rglob("*") if f.is_file());ledger["nvme_free_bytes"]=shutil.disk_usage(r).free;ledger["updated_at"]=datetime.datetime.now(datetime.timezone.utc).isoformat();ledger["generation_allowed"]=False;ledger["generation_blockers"]=["single candidate001 under evaluation; no next candidate until trial cleanup"]
for row in ledger["entries"]:
 if row.get("path")==str(candidate):row.update(state=reg["status"],allocated_bytes=allocated)
ledger["candidate_creation_reservation"]["state"]=reg["status"];ledger["audit_generated_weight_bytes"]=allocated;save(r/"storage-state.json",ledger)
print(json.dumps({"candidate_allocated_bytes":allocated,"nvme_free_bytes":ledger["nvme_free_bytes"],"checkpoint_count":1}))
raise SystemExit(q.returncode)
