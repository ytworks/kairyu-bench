"""One-time AV faststore metadata preview only. No apply, model, GPU or service action."""
from pathlib import Path
import copy, hashlib, json, os, shutil, subprocess, sys, time
R=Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104");B=R/"worktrees/kairyu-bench-weight-tools"
P=R/"analysis/fresh-purpose-20261107-f04-relative-v1/faststore-plan-v1";PR=B/"docs/experiments/records/2026-11-07-f04-faststore-preview-v1"
M=R/"analysis/fresh-purpose-20261107-f03-native-v1/materialization"
OLD=R/"analysis/fresh-purpose-20261107-av-dev30-deployment-plan-v1/av-freeze.json"
STORE=R/"checkpoints/native-faststore-av-v1";BUNDLE=R/"checkpoints/fresh-native-relative-f04-faststore-v1"
SPEC=P.parent/"proposal/fresh-native-relative-f04.json";PY=R/"infra/weight-tools-ci/bin/python"
sys.path.insert(0,str(B/"src"))
from kairyu_bench.wsp.weights import native_faststore as fast
from kairyu_bench.wsp.weights.checkpoint import Checkpoint
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value):
    p=P/name;assert not p.exists();p.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
started=time.time();assert os.getuid()==1003 and not STORE.exists() and not BUNDLE.exists()
assert sha(OLD)=="0fbf034b37613d20fde3f8a536852246a4207be8c65be0aa46a9f2a1d920f176"
assert sha(M/"verify.json")=="616806dc095ebecb04ac10d4514ec6c73f9bbd53d3f572d1ef704452320062c4"
assert sha(M/"fullcandidateidentity.json")=="b9e5dbd64520196507d1d53fae64b2609d5ea45559c78aee4a60cf701d6876b5"
assert sha(SPEC)=="52e293bdd36a4c09715c8c211e035562b1eeb26aabb6490e21d4b511f2a0987b"
old=json.loads(OLD.read_text());av=Path(old["checkpoint_path"]);current=fast.inventory(av)
assert current==old["candidate_inventory_stat"]
identity=json.loads((M/"fullcandidateidentity.json").read_text());verified=json.loads((M/"verify.json").read_text())
assert verified["verified"] and not verified["legacy_full_hashes_missing"]
assert current=={k:{("device"if a=="dev"else a):v for a,v in row.items()}for k,row in identity["source_full_stat"].items()}
assert {r["path"]:r["sha256"]for r in old["all_checkpoint_files"]}=={r["path"]:r["source_sha256"]for r in verified["files"]}
proof={k:copy.deepcopy(old[k])for k in ["schema_version","checkpoint_path","candidate_inventory_stat","all_checkpoint_files","control_tree_sha256"]}
proof["proof"]=dict(kind="prior_fresh_full9_editor_verify_plus_current_full_stat_continuity",original_freeze=dict(path=str(OLD),sha256=sha(OLD)),latest_full_hash_evidence=dict(path=str(M/"verify.json"),sha256=sha(M/"verify.json")),full_stat_evidence=dict(path=str(M/"fullcandidateidentity.json"),sha256=sha(M/"fullcandidateidentity.json")),payload_rehashed_here=False)
save("av-freeze.json",proof)
cp=Checkpoint(av);selected=sorted(n for n,t in cp.tensors.items()if n.startswith("model.language_model.")and t.floating and len(t.shape)<=1)
assert len(selected)==421 and all(cp.tensors[n].dtype=="BF16"for n in selected)
save("selected-native421.json",selected)
build=[str(PY),"-B","-m","kairyu_bench.wsp.weights.native_faststore","build",str(av),str(STORE),"--selected-tensors",str(P/"selected-native421.json"),"--source-proof",str(P/"av-freeze.json"),"--source-proof-sha256",sha(P/"av-freeze.json"),"--capacity","16","--reserve-bytes",str(5*1024**3)]
create=[str(PY),"-B","-m","kairyu_bench.wsp.weights.native_faststore","create",str(STORE/"receipt.json"),"--receipt-sha256","<actual-successful-store-receipt-sha256>","--parent",str(av),"--parent-proof",str(P/"av-freeze.json"),"--parent-proof-sha256",sha(P/"av-freeze.json"),"--spec",str(SPEC),"--output",str(BUNDLE),"--reserve-bytes",str(5*1024**3)]
verify=[str(PY),"-B","-m","kairyu_bench.wsp.weights.native_faststore","verify",str(BUNDLE/"manifest.json"),"--manifest-sha256","<actual-successful-candidate-manifest-sha256>"]
env=dict(os.environ,PYTHONPATH=str(B/"src"),PYTHONDONTWRITEBYTECODE="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",TMPDIR=str(P/"tmp"));(P/"tmp").mkdir()
save("commands.json",dict(build_preview=build,build_apply=[*build,"--apply"],create_preview=create,create_apply=[*create,"--apply"],verify=verify,unbound_future_hashes_are_not_executable=True,environment_overrides={k:env[k]for k in ["PYTHONPATH","PYTHONDONTWRITEBYTECODE","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","TMPDIR"]}))
save("parent-and-execution-gate.json",dict(approved_for_apply=False,required_after_F03_complete="Root compares complete same-cohort score; if F03 becomes highest, do not build from AV and mark F04 unscored_reason=parent_changed.",actual_scope="Build metadata preview only; no store receipt exists and create/verify cannot run yet.",future_global_lock_required=True,future_pipeline="If authorized: build --apply once; bind real receipt SHA; create preview, create --apply once; bind manifest SHA; native_faststore verify once; ordinary four-chat parallel deployment with exact guards.",source_sha256={str(Path(fast.__file__)):sha(fast.__file__),str(Path(__file__)):sha(__file__)},free_before=shutil.disk_usage(R).free))
with (P/"preview.stdout").open("x")as out,(P/"preview.stderr").open("x")as err:
    result=subprocess.run(["nice","-n","15","ionice","-c","3",*build],cwd=B,env=env,stdout=out,stderr=err,timeout=30)
save("preview-execution.json",dict(returncode=result.returncode,seconds=time.time()-started,source_sha256=sha(__file__),freeze_sha256=sha(P/"av-freeze.json"),selected_sha256=sha(P/"selected-native421.json"),store_created=STORE.exists(),candidate_created=BUNDLE.exists(),payload_rehashed=False,model_calls=0))
assert result.returncode==0 and fast.inventory(av)==current and not STORE.exists() and not BUNDLE.exists()
cp.assert_stable()
for p in P.rglob("*"):
    if p.is_file():
        q=PR/p.relative_to(P);q.parent.mkdir(parents=True,exist_ok=True)
        if not q.exists():shutil.copyfile(p,q)
print((P/"preview.stdout").read_text());print((P/"preview-execution.json").read_text())
