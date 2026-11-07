"""One-time metadata freeze and four existing CLI previews; no service mutation."""
from pathlib import Path
import ast, fcntl, importlib.util, json, os, shutil, subprocess, sys, time
R=Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
B=R/"worktrees/kairyu-bench-weight-tools"
D=R/"analysis/fresh-purpose-20261107-f03-native-v1/deployment-dp4-v1"
M=D.parent/"materialization"
OLD=R/"analysis/fresh-purpose-20261107-av-dev30-deployment-plan-v1"
H=B/"docs/experiments/scripts/redeploy-replica.py"
C=Path("/home/y-takagi/kairyu/examples/gemma-4-31b-8gpu/control.py")
PR=B/"docs/experiments/records/2026-11-07-f03-deployment-dp4-v1"
sys.path.insert(0,str(B/"src"))
spec=importlib.util.spec_from_file_location("guard",H)
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
assert g.sha(H)=="2e9da820fe30ba9b2436fb600cf14a533b1c496b2fed48ac3649fffb9d2bc9ae"
source=OLD/"remove-completed-observer-v2.py"
assert g.sha(source)=="99a9cc25006a988d1d7cfe11b8f579ea0bbb857c4de34c39f21bf0769b5a12ce"
node=ast.parse(source.read_text())
exec(compile(ast.Module(body=[x for x in node.body if isinstance(x,ast.FunctionDef) and x.name in {"safe","inventory"}],type_ignores=[]),str(source),"exec"),globals())
def save(name,v):
    assert not (D/name).exists();g.save(D/name,v)
started=time.time()
assert os.getuid()==1003
with (R/"candidate-evaluation.lock").open("a") as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    D.mkdir(exist_ok=False)
    save("preview-lock.json",dict(pid=os.getpid(),uid=os.getuid(),phase="metadata_preview",started_epoch=started))
    shutil.copyfile(__file__,D/"prepare-source.py")
    pins={str(M/"fullcandidateidentity.json"):"b9e5dbd64520196507d1d53fae64b2609d5ea45559c78aee4a60cf701d6876b5",
          str(M/"verify.json"):"616806dc095ebecb04ac10d4514ec6c73f9bbd53d3f572d1ef704452320062c4",
          str(M/"edit-manifest.json"):"44afb922c0b8df5ca20d8c9bf02f4640868177dc322a9b7b3667c160482e92e6"}
    assert all(g.sha(p)==v for p,v in pins.items())
    identity=g.read(M/"fullcandidateidentity.json");verify=g.read(M/"verify.json")
    cp=Path(identity["checkpoint"]["path"])
    assert verify["verified"] and not verify["legacy_full_hashes_missing"]
    assert identity["candidate_stat_unchanged_across_verify"]
    frozen={k:{("device" if a=="dev" else a):b for a,b in v.items()}for k,v in identity["candidate_full_stat"].items()}
    assert g.inventory(cp)==frozen
    rows=sorted([{k:r[k]for k in ("path","size","sha256")}for r in identity["checkpoint"]["files"]],key=lambda r:r["path"])
    assert len(rows)==9 and {r["path"]:r["sha256"]for r in rows}=={r["path"]:r["candidate_sha256"]for r in verify["files"]}
    tree=g.digest(rows)
    sourcepins=g.read(OLD/"initial-readonly-state.json")["source_pins"]
    assert all(g.sha(p)==v for p,v in sourcepins.items())
    sourcepins[str(__file__)]=g.sha(__file__);sourcepins[str(source)]=g.sha(source)
    save("f03-freeze.json",dict(schema_version=1,checkpoint_path=str(cp),candidate_inventory_stat=frozen,
      all_checkpoint_files=rows,control_tree_sha256=tree,checkpoint_identity_tree_sha256=identity["checkpoint"]["sha256"],
      proof=dict(kind="fresh_full_editor_verify_plus_exact_full_stat_continuity",pins=pins,full_payload_rehashed_here=False),created_epoch=time.time(),source_pins=sourcepins))
    before,raw=inventory();assert len(before)==27
    target={s:next(c["Id"]for c in raw.values()if c["Name"]=="/gemma-4-31b-8gpu-"+s+"-1")for s in ["gemma-0","gemma-1","gemma-2","gemma-3"]}
    for cid in target.values():
        c=raw[cid];assert c["State"]["Running"] and c["State"]["Health"]["Status"]=="healthy"
        assert c["Image"]=="sha256:af9e746b913088cbf6c4464a31d6904a6b7e1d3ba62f0895362ddf13595f7abe"
    protected=sorted(set(before)-set(target.values()));assert len(protected)==23
    save("preview-protected-before.json",before)
    save("source-pins.json",sourcepins)
    save("preflight-host.json",dict(nvme_free_bytes=shutil.disk_usage(R).free,
      gpu_status=g.command(["nvidia-smi","--query-gpu=index,memory.used,memory.total,utilization.gpu","--format=csv,noheader"]),
      process_inventory=g.command(["ps","-eo","pid,ppid,comm"]),model_generation_requests=0))
    argv=[str(R/"infra/weight-tools-ci/bin/python"),str(H),"--control",str(C),"--service","<gemma-N>","--candidate",str(cp),"--freeze",str(D/"f03-freeze.json"),"--baseline",str(D/"all-chat-before.json"),"--output","<fresh-output>","--timeout","1200"]
    save("switch-plan.json",dict(status="unapproved_preview_only",service_order=list(target),target_ids=target,protected_container_ids=protected,
      protected_count=23,checkpoint_identity_tree=identity["checkpoint"]["sha256"],control_tree_sha256=tree,
      freeze_path=str(D/"f03-freeze.json"),freeze_sha256=g.sha(D/"f03-freeze.json"),source_pins=sourcepins,
      planned_arguments=argv,apply_extra=["--apply"],global_lock_required=True,no_next_service_after_failure=True,
      benchmark_condition=g.read(OLD/"switch-plan.json")["benchmark_condition"]))
    receipts={}
    for service in target:
        command=list(argv);command[command.index("--service")+1]=service;command[command.index("--output")+1]=str(D/(service+"-preview-v1"))
        save(service+"-preview-command-v1.json",dict(argv=command,apply=False))
        with (D/(service+"-preview-v1.stdout")).open("x")as out,(D/(service+"-preview-v1.stderr")).open("x")as err:
            result=subprocess.run(command,cwd=B,env=dict(os.environ,PYTHONPATH=str(B/"src"),PYTHONDONTWRITEBYTECODE="1"),stdout=out,stderr=err,timeout=90)
        save(service+"-preview-execution-v1.json",dict(returncode=result.returncode))
        assert result.returncode==0
        p=D/(service+"-preview-v1")/"receipt.json";receipt=g.read(p)
        assert receipt["status"]=="preview" and not receipt["apply_requested"]
        assert receipt["before"]["id"]==target[service] and receipt["candidate_identity"]["tree_sha256"]==tree
        assert not receipt["compose_remove_orphans"]
        receipts[service]=dict(path=str(p),sha256=g.sha(p),target_id=target[service],rendered_service_sha256=receipt["rendered_service_sha256"])
    assert inventory()[0]==before and g.inventory(cp)==frozen
    assert all(g.sha(p)==v for p,v in {**pins,**sourcepins}.items())
    save("four-preview-summary.json",dict(status="all_four_previews_passed_no_service_deployment",receipts=receipts,
      baseline_path=str(D/"all-chat-before.json"),baseline_sha256=g.sha(D/"all-chat-before.json"),freeze_sha256=g.sha(D/"f03-freeze.json"),source_pins=sourcepins,
      protected_count=23,all_27_containers_unchanged=True,checkpoint_stats_unchanged=True,seconds=time.time()-started,
      full_payload_rehashed=False,model_generation_requests=0,service_mutations=0,apply_authorized=False))
    fcntl.flock(lock,fcntl.LOCK_UN)
save("preview-lock-released.json",dict(pid=os.getpid(),released_epoch=time.time(),normal_release=True))
PR.mkdir(parents=True,exist_ok=False)
for p in D.rglob("*"):
    if p.is_file() and p.suffix!=".lock":
        q=PR/p.relative_to(D);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
print(json.dumps(dict(status="all_four_previews_passed",directory=str(D),summary_sha256=g.sha(D/"four-preview-summary.json"),freeze_sha256=g.sha(D/"f03-freeze.json"),plan_sha256=g.sha(D/"switch-plan.json"),control_tree=tree,checkpoint_tree=identity["checkpoint"]["sha256"],seconds=time.time()-started)))
