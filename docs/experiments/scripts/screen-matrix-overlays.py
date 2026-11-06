#!/usr/bin/env python3
"""Bounded native-ON transfer screen: saved-parent empty overlay plus <=6 matrix overlays.
Default preview is inert. No official benchmark, checkpoint writes, or adoption.
Source relationship: v3 overlay owner 9158b092 + prospective DP2 lifecycle cdac49af.
"""
import argparse, datetime, fcntl, hashlib, importlib.util, json, os, re, signal, subprocess, sys, time, urllib.request
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class OwnerCheckError(ValueError):pass
def need(value,message):
 if not value:raise OwnerCheckError(message)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,value):
 path=Path(path);tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");os.chmod(tmp,0o600);tmp.replace(path)
def canonical(value):
 value=dict(value);value["mounts"]=sorted(value.get("mounts",[]),key=lambda x:json.dumps(x,sort_keys=True));return value
def command(argv,timeout=120):
 result=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
 need(result.returncode==0,"owned metadata/container command failed")
 return result.stdout.strip()
def pinned(ref):
 need(sha(ref["path"])==ref["sha256"],"pinned input changed");return read(ref["path"])
def saved_identity(ref,path,tree):
 freeze=pinned(ref);need(freeze["checkpoint_path"]==path and freeze["control_tree_sha256"]==tree,"saved checkpoint identity differs")
 from kairyu_bench.wsp.weights.checkpoint import Checkpoint
 rows=Checkpoint(path).identity(hash_files=False)["files"]
 need([(r["path"],r["size"]) for r in rows]==[(r["path"],r["size"]) for r in freeze["all_checkpoint_files"]],"saved inventory differs")
 for row in rows:
  old=freeze["candidate_inventory_stat"][row["path"]]
  need(row["size"]==old["size"] and all(old[k]==v for k,v in row["stat"].items()),"saved file stat differs")
 return freeze
def snapshot(service):return guard.snapshot(guard.inspect(plan["serving_baseline"][service]["id"]))
def inside(path,root):
 path=Path(path);return path.is_absolute() and path.resolve()==path and path.is_relative_to(root)
def overlay_metadata(arm):
 op=pinned(arm["overlay_plan"]);parent=plan["parent"];draft=plan["draft"]
 need(op["kind"]=="kairyu_sparse_bf16_v1" and op["schema_version"]==1 and op["tensor_parallel_size"]==2,"overlay format/TP differs")
 need(op["physical_source"]=={"path":parent["checkpoint"],"tree_sha256":parent["tree"],"proof":parent["freeze"]},"overlay physical parent differs")
 need(op["logical_parent"]=={"path":parent["checkpoint"],"tree_sha256":parent["tree"]},"overlay chaining is unsupported")
 need(op["draft_bypass"]=={"path":draft["checkpoint"],"tree_sha256":draft["tree"],"proof":draft["freeze"]},"pinned draft differs")
 need(op["implementation_sha256"]==plan["overlay_implementation_sha256"] and op["vllm_source_sha256"]==plan["vllm_source_sha256"],"installed overlay source differs")
 spec=pinned(op["spec"])
 need(spec.get("schema_version",1)==1 and spec.get("kind") is None,"unsupported sparse spec")
 need(spec["source_checkpoint"]==parent["checkpoint"] and spec["base_tree_sha256"]==parent["tree"],"spec not current-parent increment")
 pattern=re.compile(r"model\.language_model\.layers\.\d+\.(?:self_attn\.o_proj|mlp\.(?:gate|up|down)_proj)\.weight")
 total=0
 for edit in spec["edits"]:
  need(set(edit)=={"tensor","coordinates","set_values"} and pattern.fullmatch(edit["tensor"]),"only O/MLP matrix edits supported")
  need(len(edit["coordinates"])==len(edit["set_values"]),"coordinate/value mismatch");total+=len(edit["coordinates"])
 need(total==arm["selected_coordinates"] and ((total==0)==(arm["label"]=="control")),"selected count/control differs")
 logical=hashlib.sha256(json.dumps({"kind":op["kind"],"parent":op["logical_parent"],"spec_sha256":op["spec"]["sha256"],"implementation_sha256":op["implementation_sha256"]},sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
 need(logical==arm["logical_overlay_sha256"],"logical overlay identity differs")
 need(inside(op["receipt_directory"],P/"load-receipts"),"load receipt path outside owner")
 return op
def validate(serving):
 need(sha(args.plan)==args.plan_sha and sha(args.authorization)==args.authorization_sha,"owner binding changed")
 auth=read(args.authorization)
 need(auth["status"]=="authorized" and auth["issued_by"]=="root" and auth["plan_sha256"]==args.plan_sha and auth["runtime_sha256"]==sha(__file__),"exact root execution authorization required")
 need(plan["binding_complete"] and 1<=len(plan["candidates"])<=6,"bound candidate roster required")
 need(plan["image_id"]=="sha256:33025e08d05ed9f79ba0f4fa34a62bb005e6404c3aa70af20ade5e73c1e88602","validated overlay image required")
 need(plan["executor_image"]=="sha256:c4ab50223f03fca17bf871e4d05f77001e76525a1c574b6c6ee7024db06cfdc2","validated scientific executor required")
 need(plan["topology"]=={"tp":2,"dp":1,"pp":1,"static_mtp":4,"slots":4},"reviewed four TP2 slots required")
 need([slot["gpus"] for slot in plan["slots"]]==[[0,1],[2,3],[4,5],[6,7]],"four disjoint TP2 GPU slots required")
 need(set(plan["serving_baseline"])=={"gemma-0","gemma-1","gemma-2","gemma-3"},"exact four old services required")
 for path,digest in plan["pinned_inputs"].items():need(sha(path)==digest,"source/input pin changed")
 state_file=pinned(plan["incumbent"]);parent=plan["parent"];draft=plan["draft"]
 need(all(not P.is_relative_to(Path(x["checkpoint"])) and not Path(x["checkpoint"]).is_relative_to(P) for x in (parent,draft)),"owner output overlaps saved source")
 need(state_file["checkpoint_path"]==parent["checkpoint"] and state_file["control_tree_sha256"]==parent["tree"],"parent is not current incumbent")
 saved_identity(parent["freeze"],parent["checkpoint"],parent["tree"]);saved_identity(draft["freeze"],draft["checkpoint"],draft["tree"])
 closure=pinned(plan["finite_completion"]);need(closure["status"]=="completed","finite owner not closed")
 # Exact selection/closure bytes are root-bound; no additional statistical gate.
 pinned(plan["selection_decision"])
 data=pinned(plan["transfer_cases"])
 need(data["split"]=="transfer" and data["gradient_allowed"] is False and len(data["cases"])==8,"complete transfer8 required")
 need([x["id"] for x in data["cases"]]==plan["transfer_ids"] and all(x["gradient_allowed"] is False for x in data["cases"]),"transfer case IDs/split differ")
 need(plan["generation"]=={"concurrency":8,"max_tokens":32768,"timeout":1200,"seed":0,"enable_thinking":True,"retries":0},"fixed native-ON generation protocol differs")
 need(plan["execution"]=={"catalog":"library-transfer-v1","seed":20261109,"trials":8,"seconds":30,"memory_mib":512,"cpus":1},"fixed CPU execution protocol differs")
 arms=[plan["control"],*plan["candidates"]]
 need(plan["control"]["label"]=="control" and len({a["label"] for a in arms})==len(arms),"unique arm labels required")
 for arm in arms:
  need(re.fullmatch("[a-z][a-z0-9-]{0,63}",arm["label"]) and re.fullmatch("[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}",arm["container_name"]),"safe arm/container label required")
  overlay_metadata(arm)
 need(len({a["container_name"] for a in arms})==len(arms),"duplicate temporary names")
 need(len({read(a["overlay_plan"]["path"])["receipt_directory"] for a in arms})==len(arms),"duplicate load receipts")
 needed={parent["freeze"]["path"],draft["freeze"]["path"]}
 for arm in arms:
  op=read(arm["overlay_plan"]["path"]);needed.add(op["spec"]["path"])
 # Read-only external files bind canonical host paths; no /model alias projection.
 for path in needed:
  need(inside(path,P) or path in plan["external_readonly_files"],"external proof/spec mount missing")
 for path in plan["external_readonly_files"]:
  need(Path(path).is_file() and Path(path).resolve()==Path(path) and path in plan["pinned_inputs"],"external mount not pinned regular file")
 for service,want in plan["serving_baseline"].items():
  actual=snapshot(service)
  config=guard.inspect(want["id"]);fixed=plan["original_configuration"][service]
  need(guard.digest(config["Config"])==fixed["config_sha256"] and guard.digest(config["HostConfig"])==fixed["host_config_sha256"],"original full configuration changed")
  if serving:need(canonical(actual)==canonical(want),"original service changed")
  else:
   need(not actual["running"],"owned original unexpectedly running")
   need(canonical({k:v for k,v in actual.items() if k not in ("running","health")})==canonical({k:v for k,v in want.items() if k not in ("running","health")}),"stopped original identity changed")
 for old in plan["embedding_baseline"]:
  item=guard.inspect(old["id"])
  need(item["State"]["Running"]==old["running"] and item["State"]["StartedAt"]==old["started_at"] and guard.digest(item["Config"])==old["config_sha256"] and guard.digest(item["HostConfig"])==old["host_config_sha256"],"embedding changed")
def publish():
 if owned:save(P/"execution-status.json",state)
def endpoint(cid):
 item=guard.inspect(cid);ips=[v["IPAddress"] for v in item["NetworkSettings"]["Networks"].values() if v.get("IPAddress")]
 need(len(ips)==1,"single temporary endpoint required");return "http://"+ips[0]+":8000/v1"
def wait_ready(record,deadline):
 while time.monotonic()<deadline:
  need(guard.inspect(record["container_id"])["State"]["Running"],"temporary server exited")
  ep=endpoint(record["container_id"])
  try:
   with urllib.request.urlopen(ep.removesuffix("/v1")+"/health",timeout=5) as response:
    if response.status==200:return ep
  except Exception:pass
  time.sleep(2)
 raise TimeoutError("temporary readiness deadline")
def counters(ep,cid,path,idle=True):
 command([plan["python"],plan["tools"]["counters"]["path"],"snapshot","--endpoint",ep.removesuffix("/v1")+"/metrics","--model","gemma-4-31b","--container-id",cid,"--attestation-sha256",args.plan_sha,"--output",str(path),"--capture"],60)
 result=read(path)
 if idle:
  for metric in ("vllm:num_requests_running","vllm:num_requests_waiting"):
   values=[float(row["value"]) for row in result["series"] if row["metric"]==metric]
   need(values and all(v==0 for v in values),"server not idle")
 return {"path":str(path),"sha256":sha(path)}
def proofs(arm):
 op=overlay_metadata(arm);directory=Path(op["receipt_directory"]);ref=arm["overlay_plan"]
 rows=[read(directory/f"target-rank-{rank}.json") for rank in range(2)]
 for row in rows:
  need(row["status"]=="verified_selected_words_all_tp_ranks" and row["storage_bytes_exact"] and row["plan"]==ref and row["logical_overlay_sha256"]==arm["logical_overlay_sha256"] and row["selected_coordinates"]==arm["selected_coordinates"],"all-rank selected words differ")
 need(rows[0]==rows[1],"rank aggregate receipts differ")
 for rank in range(2):
  bypass=read(directory/f"draft_bypass-rank-{rank}.json")
  need(bypass["status"]=="pinned_draft_bypassed" and bypass["draft_tensor_edits"]==0 and bypass["plan"]==ref and bypass["draft_tree_sha256"]==plan["draft"]["tree"],"MTP bypass differs")
 return {"logical_overlay_sha256":arm["logical_overlay_sha256"],"physical_source_tree":plan["parent"]["tree"],"selected_coordinates":arm["selected_coordinates"],"all_rank_exact":True,"empty_is_not_full_gpu_hash":arm["selected_coordinates"]==0,"receipt_sha256":{p.name:sha(p) for p in sorted(directory.glob("*.json"))}}
def create(arm,slot,base):
 rec={"label":arm["label"],"name":arm["container_name"],"slot":slot["id"],"status":"creating"};temporary.append(rec);publish()
 cmd=list(base["Config"]["Cmd"]);need(cmd[0]=="/checkpoint","server template model argument")
 cmd[0]=plan["parent"]["checkpoint"]
 for flag in ("--load-format","--model-loader-extra-config","--safetensors-load-strategy","--data-parallel-size","--data-parallel-rank"):need(flag not in cmd,"unexpected loader/DP template override")
 need(cmd[cmd.index("--tensor-parallel-size")+1]=="2","template must be TP2")
 index=cmd.index("--speculative-config")+1;draft=json.loads(cmd[index])
 need(draft["num_speculative_tokens"]==4 and "num_speculative_tokens_per_batch_size" not in draft,"static MTP4 required")
 draft["model"]=plan["draft"]["checkpoint"];cmd[index]=json.dumps(draft,separators=(",",":"))
 ref=arm["overlay_plan"];cmd+=["--load-format","kairyu_sparse_bf16_v1","--model-loader-extra-config",json.dumps({"overlay_plan":ref["path"],"overlay_plan_sha256":ref["sha256"]},separators=(",",":"))]
 cache=P/"cache"/arm["label"];cache.mkdir(parents=True)
 env=guard.environments(base["Config"]["Env"])
 env["PYTHONPATH"]="/opt/kairyu-sparse-overlay"+(":"+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
 env.update(XDG_CACHE_HOME=str(cache),XDG_CONFIG_HOME=str(cache/"config"),VLLM_CONFIG_ROOT=str(cache/"config/vllm"),HF_HOME=str(cache/"hf"),VLLM_CACHE_ROOT=str(cache/"vllm"),TORCHINDUCTOR_CACHE_DIR=str(cache/"torch"),TRITON_CACHE_DIR=str(cache/"triton"),FLASHINFER_WORKSPACE_BASE=str(cache/"flashinfer-base"))
 if env.get("VLLM_PLUGINS"):env["VLLM_PLUGINS"]=",".join(dict.fromkeys([*env["VLLM_PLUGINS"].split(","),"kairyu_sparse_overlay"]))
 need(not any("\n" in k+v or "\r" in k+v for k,v in env.items()),"multiline environment unsupported")
 ef=P/(".env-"+arm["label"]);fd=os.open(ef,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,"w") as stream:stream.write("\n".join(k+"="+v for k,v in env.items())+"\n")
 argv=["docker","create","--name",arm["container_name"],"--label","kairyu.overlay.screen.owner="+plan["owner_label"],"--runtime","nvidia","--gpus",'"device='+",".join(map(str,slot["gpus"]))+'"',"--user","1003:1003","--network",base["HostConfig"]["NetworkMode"],"--ipc",base["HostConfig"]["IpcMode"],"--cpuset-cpus",slot["cpuset"],"--shm-size",str(base["HostConfig"]["ShmSize"]),"--env-file",str(ef),"--no-healthcheck"]
 if base["Config"].get("WorkingDir"):argv+=["--workdir",base["Config"]["WorkingDir"]]
 models=next(m for m in base["Mounts"] if m["Destination"]=="/models");need(not models["RW"],"models mount must be readonly")
 mounts=[(models["Source"],"/models",True),(plan["parent"]["checkpoint"],plan["parent"]["checkpoint"],True),(plan["draft"]["checkpoint"],plan["draft"]["checkpoint"],True),(str(P),str(P),True),(str(cache),str(cache),False)]
 receipt=read(arm["overlay_plan"]["path"])["receipt_directory"];mounts.append((receipt,receipt,False))
 mounts += [(p,p,True) for p in plan["external_readonly_files"]]
 for source,target,readonly in mounts:argv+=["--mount","type=bind,src="+source+",dst="+target+(",readonly" if readonly else "")]
 try:rec["container_id"]=command([*argv,plan["image_id"],*cmd],120)
 finally:ef.unlink(missing_ok=True)
 item=guard.inspect(rec["container_id"])
 need(item["Image"]==plan["image_id"] and item["Config"]["User"]=="1003:1003" and item["Config"]["Cmd"]==cmd and item["Config"]["Entrypoint"]==base["Config"]["Entrypoint"],"temporary image/command/user differs")
 devices=item["HostConfig"].get("DeviceRequests") or []
 need(len(devices)==1 and set(devices[0].get("DeviceIDs") or [])==set(map(str,slot["gpus"])),"temporary GPU scope differs")
 rec.update(status="loading",image_id=item["Image"],command_sha256=guard.digest(cmd),environment_sha256=guard.digest(env));publish();return rec
def cleanup(record):
 if record.get("removed"):return
 query=subprocess.run(["docker","inspect",record.get("container_id",record["name"])],capture_output=True,text=True,timeout=20)
 if query.returncode==0:
  item=json.loads(query.stdout)[0]
  need(item["Image"]==plan["image_id"] and (item["Config"].get("Labels") or {}).get("kairyu.overlay.screen.owner")==plan["owner_label"],"refuse foreign temporary cleanup")
  command(["docker","rm","--force",item["Id"]],120)
 record["removed"]=True
def startup_diagnostic(record):
 if record.get("status")!="loading" or not record.get("container_id"):return
 item=guard.inspect(record["container_id"])
 need(item["Image"]==plan["image_id"] and (item["Config"].get("Labels") or {}).get("kairyu.overlay.screen.owner")==plan["owner_label"],"diagnostic ownership differs")
 result=subprocess.run(["docker","logs","--tail","2000",item["Id"]],capture_output=True,timeout=20)
 raw=(result.stdout+b"\n"+result.stderr)[-1048576:];directory=P/"private-startup";directory.mkdir(exist_ok=True,mode=0o700)
 path=directory/(record["label"]+".log");fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,"wb") as stream:stream.write(raw)
 text=raw.decode(errors="replace");classes=["PermissionError","FileNotFoundError","ModuleNotFoundError","ImportError","ValueError","RuntimeError","TypeError","AssertionError","KeyError","OSError","OutOfMemoryError"]
 record["startup_diagnostic"]={"exit_code":item["State"].get("ExitCode"),"oom_killed":item["State"].get("OOMKilled"),"private_log_sha256":sha(path),"bytes":len(raw),"error_classes":[c for c in classes if re.search(r"(?<![A-Za-z])"+c+r"(?=[:\s])",text)],"permission_denied":"Permission denied" in text,"no_module_named":"No module named" in text,"cuda_out_of_memory":"CUDA out of memory" in text,"raw_log_not_published":True}
def launch(argv,log):
 with Path(log).open("xb") as stream:child=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 children.append(child);return child
def wait_children(group,deadline):
 while any(child.poll() is None for child in group):
  need(time.monotonic()<deadline,"declared child time budget exhausted")
  failed=[child for child in group if child.poll() not in (None,0)]
  need(not failed,"authored generator/executor process failed")
  time.sleep(1)
 need(all(child.returncode==0 for child in group),"authored generator/executor process failed")
def generation_command(ep,output):
 g=plan["generation"]
 return [plan["python"],"-B",plan["tools"]["generate"]["path"],"--cases",plan["transfer_cases"]["path"],"--split","transfer","--endpoint",ep,"--model","gemma-4-31b","--checkpoint-sha256",plan["parent"]["tree"],"--concurrency",str(g["concurrency"]),"--max-tokens",str(g["max_tokens"]),"--timeout",str(g["timeout"]),"--seed",str(g["seed"]),"--output",str(output),"--run"]
def execution_command(generation,output):
 e=plan["execution"]
 return [plan["python"],"-B",plan["tools"]["sandbox"]["path"],"--catalog",e["catalog"],"--cases",plan["transfer_cases"]["path"],"--responses",str(generation/"transfer-responses.json"),"--docker-socket",plan["executor_socket"],"--daemon-id",plan["executor_daemon_id"],"--image",plan["executor_image"],"--seed",str(e["seed"]),"--trials",str(e["trials"]),"--seconds",str(e["seconds"]),"--memory-mib",str(e["memory_mib"]),"--cpus",str(e["cpus"]),"--output",str(output),"--run"]
def execute():
 global owned
 need(os.getuid()==1003,"UID1003 required");validate(True)
 for image in (plan["image_id"],plan["executor_image"]):need(command(["docker","image","inspect","--format","{{.Id}}",image],30)==image,"pinned local image absent")
 need(not (P/"execution-status.json").exists() and not (P/"completion.json").exists(),"owner already used")
 for name in ("cache","load-receipts","rounds"):need(not (P/name).exists(),"fresh owner directories required")
 for arm in [plan["control"],*plan["candidates"]]:
  need(subprocess.run(["docker","inspect",arm["container_name"]],capture_output=True,timeout=20).returncode!=0,"temporary name already exists")
 with (P/"execution-status.json").open("x") as stream:json.dump(state,stream)
 owned=True;publish()
 base=guard.inspect(plan["serving_baseline"][plan["server_template_service"]]["id"])
 for service,old in plan["serving_baseline"].items():counters(endpoint(old["id"]),old["id"],P/(service+"-before-stop.json"))
 for arm in [plan["control"],*plan["candidates"]]:Path(read(arm["overlay_plan"]["path"])["receipt_directory"]).mkdir(parents=True)
 validate(True);state["status"]="stopping_exact_four";publish()
 for service,old in plan["serving_baseline"].items():
  stopped.append(service);command(["docker","stop","--time","90",old["id"]]);need(not guard.inspect(old["id"])["State"]["Running"],"original did not stop")
 deadline=time.monotonic()+plan["maximum_active_seconds"];control=None
 chunks=[plan["candidates"][i:i+3] for i in range(0,len(plan["candidates"]),3)]
 for number,candidates in enumerate(chunks,1):
  validate(False);need(time.monotonic()<deadline,"declared active window exhausted")
  state["status"]="loading_round";state["active_round"]=number;publish()
  fresh=[]
  if control is None:
   control=create(plan["control"],plan["slots"][0],base);fresh.append(control)
  records=[]
  for arm,slot in zip(candidates,plan["slots"][1:]):records.append(create(arm,slot,base))
  fresh+=records
  # A single Docker start request starts all new servers; loading overlaps.
  command(["docker","start",*[record["container_id"] for record in fresh]],120)
  startup_deadline=min(deadline,time.monotonic()+plan["startup_seconds"])
  for record in fresh:record["endpoint"]=wait_ready(record,startup_deadline)
  arms=[plan["control"],*candidates];current=[control,*records]
  for arm,record in zip(arms,current,strict=True):
   record["proof"]=proofs(arm);record["status"]="ready"
  validate(False);publish()
  round_dir=P/"rounds"/str(number);round_dir.mkdir(parents=True)
  generation_jobs=[];round_record={"round":number,"started_at":now(),"arms":[]};state["rounds"].append(round_record)
  for arm,record in zip(arms,current,strict=True):
   out=round_dir/arm["label"];out.mkdir()
   row={"label":arm["label"],"container_id":record["container_id"],"logical_overlay_sha256":arm["logical_overlay_sha256"],"physical_source_tree":plan["parent"]["tree"],"load_proof":record["proof"],"output":str(out)}
   row["before_counters"]=counters(record["endpoint"],record["container_id"],out/"serving-before.json")
   round_record["arms"].append(row);generation_jobs.append(launch(generation_command(record["endpoint"],out/"generation"),out/"generation.log"))
   record["status"]="generating"
  state["status"]="native_on_generation";publish()
  wait_children(generation_jobs,min(deadline,time.monotonic()+plan["generation_seconds"]))
  execution_jobs=[]
  for arm,record,row in zip(arms,current,round_record["arms"],strict=True):
   out=Path(row["output"]);receipt=read(out/"generation/receipt.json")
   need(receipt["status"]=="completed" and receipt["received"]==8,"transfer generation did not complete eight cases")
   row["generation"]={k:receipt.get(k) for k in ("status","received","errors","seconds","expected_cases")}
   row["generation_receipt_sha256"]=sha(out/"generation/receipt.json")
   row["after_counters"]=counters(record["endpoint"],record["container_id"],out/"serving-after.json")
   command([plan["python"],plan["tools"]["counters"]["path"],"diff",str(out/"serving-before.json"),str(out/"serving-after.json"),"--output",str(out/"serving-diff.json")],60)
   execution_jobs.append(launch(execution_command(out/"generation",out/"execution"),out/"execution.log"))
  state["status"]="cpu_execution";publish()
  wait_children(execution_jobs,min(deadline,time.monotonic()+plan["execution_seconds"]))
  for record,row in zip(current,round_record["arms"],strict=True):
   out=Path(row["output"]);report=read(out/"execution/report.json")
   need(report["status"]=="completed" and len(report["results"])==8 and [r["id"] for r in report["results"]]==plan["transfer_ids"],"transfer execution catalog differs")
   row["numeric_results"]=[{k:r.get(k) for k in ("id","status","passed","tests_passed","tests_total","finish_reason","usage","seconds","validation_failures","code_sha256")} for r in report["results"]]
   row["execution_report_sha256"]=sha(out/"execution/report.json")
   row["passed"]=sum(r.get("passed") is True for r in report["results"]);row["unknown"]=sum(r.get("passed") is None for r in report["results"])
   record["status"]="ready";row["finished_at"]=now()
  round_record["status"]="completed";round_record["finished_at"]=now();publish()
  # Keep the same empty-overlay control process across rounds; remove only candidates.
  for record in records:cleanup(record)
  validate(False)
 state["screen_completed"]=True

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--plan",type=Path,required=True);parser.add_argument("--plan-sha",required=True)
parser.add_argument("--authorization",type=Path);parser.add_argument("--authorization-sha");parser.add_argument("--apply",action="store_true")
args=parser.parse_args();need(sha(args.plan)==args.plan_sha,"plan SHA differs");plan=read(args.plan);P=args.plan.resolve().parent
if not args.apply:
 print(json.dumps({"status":"prepared_no_operations","binding_complete":plan.get("binding_complete",False),"candidate_capacity":6,"planned_candidates":len(plan.get("candidates",[])),"maximum_rounds":2,"runtime_sha256":sha(__file__),"service_changes":0,"model_calls":0}));raise SystemExit(0)
need(args.authorization and args.authorization_sha,"root authorization required");os.umask(0o077)
for tool in plan["tools"].values():need(sha(tool["path"])==tool["sha256"],"tool pin differs")
sys.path.insert(0,str(Path(plan["repo"])/"src"))
sp=importlib.util.spec_from_file_location("overlay_screen_guard",plan["tools"]["redeploy"]["path"]);guard=importlib.util.module_from_spec(sp);sp.loader.exec_module(guard)
lock=(P/"owner.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
owned=False;stopped=[];temporary=[];children=[];error=None
state={"schema_version":1,"status":"preflight","started_at":now(),"started_epoch":time.time(),"plan_sha256":args.plan_sha,"temporary":temporary,"rounds":[],"model_scope":"Author-owned transfer8 native-ON screen only","official_calls":0,"checkpoint_writes":0,"adoption_performed":False}
def interrupted(sig,frame):raise InterruptedError("owner interrupted")
for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
try:execute()
except BaseException as exc:
 error=type(exc).__name__;state.update(status="failed",error_type=error)
 if isinstance(exc,OwnerCheckError):state["owner_assertion_message"]=str(exc)
 for record in temporary:
  try:startup_diagnostic(record)
  except BaseException as diagnostic:record["startup_diagnostic_error_type"]=type(diagnostic).__name__
 try:publish()
 except BaseException:pass
finally:
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,signal.SIG_IGN)
 for child in children:
  try:
   if child.poll() is None:
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:child.wait(timeout=20)
    except subprocess.TimeoutExpired:
     try:os.killpg(child.pid,signal.SIGKILL)
     except ProcessLookupError:pass
     child.wait(timeout=10)
  except BaseException as exc:state.setdefault("child_cleanup_errors",[]).append(type(exc).__name__);error=error or type(exc).__name__
 # Failure of one temporary cleanup cannot skip any other cleanup or restoration.
 for record in temporary:
  try:cleanup(record)
  except BaseException as exc:state.setdefault("temporary_cleanup_errors",[]).append({"label":record["label"],"type":type(exc).__name__});error=error or type(exc).__name__
 state["status"]="restoring_same_four";state["restored"]={}
 try:publish()
 except BaseException:pass
 restore_deadline=time.monotonic()+plan["restoration_seconds"]
 for service in stopped:
  try:command(["docker","start",plan["serving_baseline"][service]["id"]],max(1,min(120,restore_deadline-time.monotonic())));state["restored"][service]={"start_requested":True}
  except BaseException as exc:state["restored"][service]={"error_type":type(exc).__name__};error=error or type(exc).__name__
 for service in stopped:
  try:
   while time.monotonic()<restore_deadline:
    if canonical(snapshot(service))==canonical(plan["serving_baseline"][service]):break
    time.sleep(2)
   need(canonical(snapshot(service))==canonical(plan["serving_baseline"][service]),"original restoration incomplete")
   state["restored"][service]={"same_cid_exact_healthy":True}
  except BaseException as exc:state["restored"][service]={"error_type":type(exc).__name__};error=error or type(exc).__name__
 if owned:
  try:validate(True);state["final_source_serving_checks"]=True
  except BaseException as exc:state["final_check_error_type"]=type(exc).__name__;error=error or type(exc).__name__
  state.update(status="completed" if not error and state.get("screen_completed") else "failed",finished_at=now(),finished_epoch=time.time(),functional_parity_claim=False,physical_save_and_fresh_official_score_required_before_adoption=True)
  try:publish()
  finally:
   with (P/"completion.json").open("x") as stream:json.dump(state,stream,indent=2);stream.write("\n")
if not owned:print(json.dumps({"status":"preflight_failed_no_ownership","error_type":error,"existing_receipts_unchanged":True}))
raise SystemExit(0 if state["status"]=="completed" else 1)

