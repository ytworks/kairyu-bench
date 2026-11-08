"""Bind the successful saved F04 and run four existing metadata previews only."""
from pathlib import Path
import ast, fcntl, importlib.util, json, os, shutil, subprocess, sys, time
R=Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104");B=R/"worktrees/kairyu-bench-weight-tools";P=R/"analysis/fresh-purpose-20261107-f04-relative-v1"
D=P/"deployment-dp4-v1";M=P/"faststore-plan-v1/materialization-v1";PR=B/"docs/experiments/records/2026-11-07-f04-deployment-dp4-v1"
H=B/"docs/experiments/scripts/redeploy-replica.py";sys.path.insert(0,str(B/"src"))
mod=importlib.util.spec_from_file_location("guard",H);g=importlib.util.module_from_spec(mod);mod.loader.exec_module(g)
source=R/"analysis/fresh-purpose-20261107-av-dev30-deployment-plan-v1/remove-completed-observer-v2.py";assert g.sha(source)=="99a9cc25006a988d1d7cfe11b8f579ea0bbb857c4de34c39f21bf0769b5a12ce"
tree=ast.parse(source.read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in {"safe","inventory"}],type_ignores=[]),str(source),"exec"),globals())
def save(name,v):
    assert not(D/name).exists();g.save(D/name,v)
complete=g.read(M/"complete.json");assert complete["status"]=="completed"
assert not Path("/proc/1240408").exists(),"materialization owner still alive; defer preview"
manifest=R/"checkpoints/fresh-native-relative-f04-faststore-v1/manifest.json";verify=M/"verify.stdout";store=R/"checkpoints/native-faststore-av-v1/receipt.json"
assert g.sha(manifest)==complete["manifest_sha256"]and g.sha(store)==complete["store_receipt_sha256"]
proof=g.read(manifest);verification=g.read(verify);packed=g.read(store)
assert verification["verified"]and not verification["fresh_full_model_hash"]and verification["tree_sha256"]==proof["control_tree_sha256"]
assert packed["equivalence"]["all_tensor_names_dtype_shape_raw_bytes_equal"]and packed["shared_link_count"]==17
assert len(proof["all_checkpoint_files"])==9 and proof["changed_elements"]==429 and proof["changed_tensors"]==140
plan=g.read(P/"parallel-deploy-plan-v1/plan-unbound-v2.json")
assert plan["owner_sha256"]==g.sha(B/"docs/experiments/scripts/fresh-four-parallel-redeploy-20261107.py")
assert not plan["approved_for_apply"]and all(g.sha(p)==h for p,h in plan["source_pins"].items())
started=time.time()
with(R/"candidate-evaluation.lock").open("a")as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);D.mkdir(exist_ok=False)
    shutil.copyfile(__file__,D/"preview-source.py")
    before,raw=inventory();assert len(before)==27
    decision=g.read(P/"faststore-plan-v1/parent-decision.json");assert decision["parent"]=="AV"
    targets={x["service"]:x["cid"]for x in decision["services"]};assert list(targets)==plan["service_order"]
    for cid in targets.values():g.idle(raw[cid]);assert raw[cid]["State"]["Running"]
    protected=sorted(set(before)-set(targets.values()));assert len(protected)==23
    assert g.inventory(Path(proof["checkpoint_path"]))==proof["candidate_inventory_stat"]
    save("protected-before.json",before);save("preview-start.json",dict(pid=os.getpid(),epoch=started,materialization_complete_sha256=g.sha(M/"complete.json"),candidate_manifest_sha256=g.sha(manifest),verify_sha256=g.sha(verify),global_lock_held=True))
    previews={}
    for service in plan["service_order"]:
        cmd=[plan["python"],str(H),"--control",plan["control"],"--service",service,"--candidate",proof["checkpoint_path"],"--freeze",str(manifest),"--baseline",str(D/"all-chat-before.json"),"--output",str(D/(service+"-preview")),"--timeout","1200"]
        save(service+"-preview-command.json",dict(argv=cmd,apply=False))
        with(D/(service+"-preview.stdout")).open("x")as out,(D/(service+"-preview.stderr")).open("x")as err:
            result=subprocess.run(cmd,cwd=B,env=dict(os.environ,PYTHONPATH=str(B/"src"),PYTHONDONTWRITEBYTECODE="1"),stdout=out,stderr=err,timeout=90)
        save(service+"-preview-execution.json",dict(returncode=result.returncode));assert result.returncode==0
        path=D/(service+"-preview/receipt.json");record=g.read(path)
        assert record["status"]=="preview"and not record["apply_requested"]and record["before"]["id"]==targets[service]
        assert record["candidate_identity"]["tree_sha256"]==proof["control_tree_sha256"]and not record["compose_remove_orphans"]
        previews[service]=dict(path=str(path),sha256=g.sha(path))
    assert inventory()[0]==before and g.inventory(Path(proof["checkpoint_path"]))==proof["candidate_inventory_stat"]
    assert all(g.sha(p)==h for p,h in plan["source_pins"].items())
    plan.update(status="four_previews_passed_awaiting_root_apply",candidate=proof["checkpoint_path"],freeze=dict(path=str(manifest),sha256=g.sha(manifest)),verification=dict(path=str(verify),sha256=g.sha(verify)),baseline=dict(path=str(D/"all-chat-before.json"),sha256=g.sha(D/"all-chat-before.json")),protected_before=dict(path=str(D/"protected-before.json"),sha256=g.sha(D/"protected-before.json")),target_ids=targets,protected_ids=protected,previews=previews,output=str(D/"parallel-apply-v1"),materialization=dict(path=str(M/"complete.json"),sha256=g.sha(M/"complete.json")),parent_decision=dict(path=str(P/"faststore-plan-v1/parent-decision.json"),sha256=g.sha(P/"faststore-plan-v1/parent-decision.json")))
    save("plan-bound-unapproved.json",plan)
    save("four-preview-summary.json",dict(status="all_four_previews_passed",seconds=time.time()-started,previews=previews,plan_sha256=g.sha(D/"plan-bound-unapproved.json"),tree=proof["control_tree_sha256"],candidate_manifest_sha256=g.sha(manifest),verify_sha256=g.sha(verify),all27_unchanged=True,protected_count=23,all9_stats_unchanged=True,shared_nlink=17,model_calls=0,serving_mutations=0,payload_rehashed=False))
    fcntl.flock(lock,fcntl.LOCK_UN)
save("preview-lock-released.json",dict(normal_release=True,pid=os.getpid(),epoch=time.time()))
PR.mkdir(parents=True,exist_ok=False)
for p in D.rglob("*"):
    if p.is_file()and p.suffix!=".lock":
        q=PR/p.relative_to(D);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
print((D/"four-preview-summary.json").read_text())
