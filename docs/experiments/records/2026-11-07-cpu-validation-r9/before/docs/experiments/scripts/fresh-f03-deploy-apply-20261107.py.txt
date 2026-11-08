"""Apply the root-approved F03 freeze to ordinary gemma-0,1,2,3 sequentially."""
from pathlib import Path
import ast,fcntl,importlib.util,json,os,shutil,subprocess,sys,time,urllib.request
R=Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
B=R/"worktrees/kairyu-bench-weight-tools"
D=R/"analysis/fresh-purpose-20261107-f03-native-v1/deployment-dp4-v1"
sys.path.insert(0,str(B/"src"))
H=B/"docs/experiments/scripts/redeploy-replica.py"
spec=importlib.util.spec_from_file_location("guard",H)
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
source=R/"analysis/fresh-purpose-20261107-av-dev30-deployment-plan-v1/remove-completed-observer-v2.py"
assert g.sha(source)=="99a9cc25006a988d1d7cfe11b8f579ea0bbb857c4de34c39f21bf0769b5a12ce"
tree=ast.parse(source.read_text())
functions=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name in {"safe","inventory","lock_evidence"}],type_ignores=[])
exec(compile(functions,str(source),"exec"),globals())
def save(name,value):
    p=D/name;assert not p.exists();g.save(p,value)
def fetch(endpoint,path):
    with urllib.request.urlopen(endpoint+path,timeout=30)as response:
        return json.loads(response.read(1024**2))
started=time.time()
lockfile=(R/"candidate-evaluation.lock").open("a")
fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
save("global-keeper.json",dict(pid=os.getpid(),uid=os.getuid(),started_epoch=started,release="After all-four and endpoint closure; process exit also releases"))
record=Path("/mnt/nvme/kairyu/model-volumes/gemma-4-31b-8gpu/chat-checkpoint.json")
record_bytes=record.read_bytes();record_sha=g.sha(record)
assert record_sha=="8cd66daf213fe900c9dcd418e374f10111361a7d31ecb36f677e4ef1cb4bcf81"
(D/"chat-checkpoint-before.json").write_bytes(record_bytes)
pins={'f03-freeze.json': '8d82ac6cc2091ad06f07faf256bce3b6e548283292c4960231ff3633a51648d0', 'switch-plan.json': 'a651b155f83adae232b9d5eb51a9739176ae6604d7ac7051fc013b3c2f05ec97', 'four-preview-summary.json': '5549a726cc6b3206f55377f00c943782a15bb4b4c48d3442cb9631d7a2e49165', 'all-chat-before.json': '3cca79043916a18bd6118bd64222cd230291019825edaf8eb74d0a701239ced7'}
assert all(g.sha(D/name)==v for name,v in pins.items())
plan=g.read(D/"switch-plan.json");freeze=g.read(D/"f03-freeze.json");summary=g.read(D/"four-preview-summary.json")
sourcepins=summary["source_pins"]
assert all(g.sha(p)==v for p,v in sourcepins.items())
assert all(g.sha(v["path"])==v["sha256"]for v in summary["receipts"].values())
before,raw=inventory()
assert before==g.read(D/"preview-protected-before.json")and len(before)==27
assert g.inventory(Path(freeze["checkpoint_path"]))==freeze["candidate_inventory_stat"]
lock=lock_evidence()
spec2=importlib.util.spec_from_file_location("control","/home/y-takagi/kairyu/examples/gemma-4-31b-8gpu/control.py")
control=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(control)
members={s:g.compose_scope(control,s,cid)for s,cid in plan["target_ids"].items()}
for cid in plan["target_ids"].values():g.idle(raw[cid])
tmp=D/"tmp";tmp.mkdir(exist_ok=False)
env=dict(os.environ,PYTHONPATH=str(B/"src"),PYTHONDONTWRITEBYTECODE="1",TMPDIR=str(tmp))
save("four-apply-authorization.json",dict(status="root_authorized_F03_four_sequential_applies",pinned_inputs=pins,
 source_pins=sourcepins,service_order=plan["service_order"],compose_membership=members,owner=lock,
 protected_count=23,helper_timeout_seconds=1200,outer_timeout_including_rollback_seconds=2460,
 progress_review_deadline_epoch=started+600,keeper_auto_release_deadline=None,
 keeper_termination="Explicit stdin release only after all-four closure and root-approved next-owner handoff; SSH EOF also ends keeper.",
 no_new_service_after_failure=True,source_sha256=g.sha(Path(__file__)),started_epoch=started))
expected=dict(before);current_ids=dict(plan["target_ids"]);done={}
for service in plan["service_order"]:
    checkpoint_before=g.inventory(Path(freeze["checkpoint_path"]))
    assert checkpoint_before==freeze["candidate_inventory_stat"]
    assert all(g.sha(p)==v for p,v in sourcepins.items())
    current,objects=inventory();assert current==expected
    lock_evidence()
    cid=current_ids[service];g.compose_scope(control,service,cid);g.idle(objects[cid])
    preview=g.read(summary["receipts"][service]["path"])
    assert g.snapshot(objects[cid])==preview["before"]
    argv=g.read(D/(service+"-preview-command-v1.json"))["argv"]
    argv[argv.index("--output")+1]=str(D/(service+"-apply-v1"));argv.append("--apply")
    save(service+"-apply-command-v1.json",dict(argv=argv,
         environment_overrides={"PYTHONPATH":str(B/"src"),"PYTHONDONTWRITEBYTECODE":"1","TMPDIR":str(tmp)}))
    print(json.dumps(dict(phase="applying",service=service,completed=len(done),started_epoch=time.time())),flush=True)
    t=time.time()
    try:
        with (D/(service+"-apply-v1.stdout")).open("x")as out,(D/(service+"-apply-v1.stderr")).open("x")as err:
            result=subprocess.run(argv,cwd=B,env=env,stdout=out,stderr=err,timeout=2460)
        rc=result.returncode
    except BaseException as error:
        rc=None
        save(service+"-apply-owner-error.json",dict(error_type=type(error).__name__,seconds=time.time()-t))
        after,_=inventory();save(service+"-after-owner-error.json",after)
        raise
    after,objects=inventory()
    receipt=g.read(D/(service+"-apply-v1")/"receipt.json")
    save(service+"-apply-execution-v1.json",dict(returncode=rc,seconds=time.time()-t,
        receipt_sha256=g.sha(D/(service+"-apply-v1")/"receipt.json"),status=receipt["status"]))
    save(service+"-after-apply-v1.json",after)
    if rc!=0 or receipt["status"]!="completed":
        save("four-apply-stopped.json",dict(status="stopped_after_target_failure",service=service,completed=done,
           owner=lock_evidence(),receipt=receipt,seconds=time.time()-started))
        raise RuntimeError("target failed; retained helper rollback and no next target started")
    new=next(c for c in objects.values()if c["Name"]=="/gemma-4-31b-8gpu-"+service+"-1")
    new_id=new["Id"]
    checked=g.compare_after(preview["before"],new,Path(freeze["checkpoint_path"]),freeze["control_tree_sha256"])
    assert checked==receipt["after"]and new_id!=cid
    assert set(after)==(set(expected)-{cid})|{new_id}
    assert all(after[k]==v for k,v in expected.items()if k!=cid)
    expected.pop(cid);expected[new_id]=after[new_id];current_ids[service]=new_id
    g.compose_scope(control,service,new_id)
    assert g.inventory(Path(freeze["checkpoint_path"]))==freeze["candidate_inventory_stat"]
    ips=[v["IPAddress"]for v in new["NetworkSettings"]["Networks"].values()if v["IPAddress"]]
    assert len(ips)==1
    endpoint="http://"+ips[0]+":8000"
    models=fetch(endpoint,"/v1/models")
    ids=[x["id"]for x in models["data"]];assert "gemma-4-31b"in ids
    g.idle(new)
    args=new["Config"]["Cmd"]
    specarg=json.loads(args[args.index("--speculative-config")+1])
    assert specarg["num_speculative_tokens"]==4
    assert args[args.index("--tensor-parallel-size")+1]=="2"
    assert args[args.index("--served-model-name")+1]=="gemma-4-31b"
    assert "--worker-cls"not in args and "--middleware"not in args
    done[service]=dict(cid=new_id,endpoint=endpoint+"/v1",models=ids,tree=checked["tree"],
      checkpoint_mount=checked["checkpoint_path"],checkpoint_readonly=checked["checkpoint_readonly"],
      runtime_preserved=True,mtp_tokens=4,tp_size=2,native_runtime_flags_preserved=True,
      receipt_sha256=g.sha(D/(service+"-apply-v1")/"receipt.json"),seconds=time.time()-t)
    save(service+"-closure-v1.json",done[service])
    print(json.dumps(dict(phase="healthy_verified",service=service,cid=new_id,completed=len(done),seconds=time.time()-t)),flush=True)
final,objects=inventory();assert final==expected
assert all(final[k]==before[k]for k in plan["protected_container_ids"])
members={s:g.compose_scope(control,s,cid)for s,cid in current_ids.items()}
assert g.inventory(Path(freeze["checkpoint_path"]))==freeze["candidate_inventory_stat"]
assert all(g.sha(p)==v for p,v in sourcepins.items())
save("four-apply-final-state.json",final)
save("four-apply-closure.json",dict(status="all_four_ordinary_F03_healthy_verified",replicas=done,
 tree=freeze["control_tree_sha256"],freeze_sha256=g.sha(D/"f03-freeze.json"),source_pins=sourcepins,
 protected_count=23,protected_unchanged=True,container_count=27,compose_membership=members,
 owner=lock_evidence(),seconds=time.time()-started,finished_epoch=time.time(),model_generation_requests=0,
 global_checkpoint_record_modified=False,full_payload_rehashed=False,source_sha256=g.sha(Path(__file__))))
print(json.dumps(dict(status="all_four_F03_healthy",seconds=time.time()-started,
 closure_sha256=g.sha(D/"four-apply-closure.json"),replicas=done)),flush=True)

# Minimum endpoint usability checks after all four are homogeneous; no score evaluation.
def post(url,body):
    data=json.dumps(body).encode()
    with urllib.request.urlopen(urllib.request.Request(url,data=data,headers={"Content-Type":"application/json"}),timeout=120)as response:
        return json.loads(response.read(1024**2))
gateway=objects["88943a0985ba97dfb17be6b5fb54040eba2570b70ce05126bd410162d3658ffb"]
assert safe(gateway)==before[gateway["Id"]] and gateway["State"]["Health"]["Status"]=="healthy"
gips=[v["IPAddress"]for v in gateway["NetworkSettings"]["Networks"].values()if v["IPAddress"]];assert len(gips)==1
gendpoint="http://"+gips[0]+":8000/v1"
gmodels=fetch(gendpoint,"/models");assert "gemma-4-31b"in[r["id"]for r in gmodels["data"]]
checks={}
for service,row in done.items():
    body=dict(model="gemma-4-31b",prompt="Ready.",max_tokens=1,prompt_logprobs=1)
    response=post(row["endpoint"]+"/completions",body)
    save(service+"-logprob-smoke.json",dict(request=body,response=response))
    assert response["choices"][0].get("prompt_logprobs")
    checks[service]=dict(http_success=True,model=response["model"],prompt_logprob_positions=len(response["choices"][0]["prompt_logprobs"]))
body=dict(model="gemma-4-31b",messages=[dict(role="user",content="Reply with the single word ready.")],max_tokens=256)
response=post(gendpoint+"/chat/completions",body)
choice=response["choices"][0]
save("gateway-chat-smoke.json",dict(request=body,model=response.get("model"),usage=response.get("usage"),finish_reason=choice.get("finish_reason"),public_answer=choice["message"].get("content")))
assert response.get("model")=="gemma-4-31b" and choice["message"].get("content")
final,objects=inventory();assert final==expected
for cid in current_ids.values():g.idle(objects[cid])
assert g.inventory(Path(freeze["checkpoint_path"]))==freeze["candidate_inventory_stat"]
assert g.sha(record)==record_sha
updated=json.loads(record_bytes);updated.update(files=freeze["all_checkpoint_files"],path=freeze["checkpoint_path"],tree_sha256=freeze["control_tree_sha256"])
assert set(updated)==set(json.loads(record_bytes))
tmprecord=record.with_name(".chat-checkpoint-f03.tmp")
with tmprecord.open("x")as stream:
    stream.write(json.dumps(updated,indent=2,sort_keys=True)+"\n");stream.flush();os.fsync(stream.fileno())
assert g.sha(record)==record_sha
tmprecord.replace(record)
dfd=os.open(record.parent,os.O_DIRECTORY);os.fsync(dfd);os.close(dfd)

save("chat-checkpoint-after.json",g.read(record))
save("served-record-update.json",dict(before_sha256=record_sha,after_sha256=g.sha(record),changed_fields=["files","path","tree_sha256"],schema_preserved=True))
save("endpoint-closure.json",dict(status="all_four_F03_and_gateway_verified",gateway=dict(cid=gateway["Id"],endpoint=gendpoint,model_ids=[r["id"]for r in gmodels["data"]],healthy=True,unchanged_protected_service=True),replicas=done,logprob_checks=checks,chat_check=True,generation_requests=5,official_evaluation_requests=0,all_four_idle=True,protected_count=23,protected_unchanged=True,closure_sha256=g.sha(D/"four-apply-closure.json"),served_record_sha256=g.sha(record),created_epoch=time.time()))
fcntl.flock(lockfile,fcntl.LOCK_UN);lockfile.close()
save("global-lock-released.json",dict(status="normal_release",pid=os.getpid(),released_epoch=time.time(),endpoint_closure_sha256=g.sha(D/"endpoint-closure.json"),four_closure_sha256=g.sha(D/"four-apply-closure.json")))
PR=B/"docs/experiments/records/2026-11-07-f03-deployment-dp4-v1"
for p in D.rglob("*"):
    if p.is_file() and p.suffix!=".lock":
        q=PR/p.relative_to(D);q.parent.mkdir(parents=True,exist_ok=True)
        if not q.exists():shutil.copyfile(p,q)
print(json.dumps(dict(status="F03_deployment_and_endpoints_complete_lock_released",seconds=time.time()-started,closure_sha256=g.sha(D/"four-apply-closure.json"),endpoint_sha256=g.sha(D/"endpoint-closure.json"),release_sha256=g.sha(D/"global-lock-released.json"))),flush=True)
