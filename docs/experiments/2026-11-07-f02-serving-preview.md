# F02 single-replica serving preview — 2026-11-07

Current recorded result: repaired preview-v2 and apply-v2 succeeded. F02 is healthy on gemma-0, direct chat/logprob checks passed, all other 26 containers are unchanged, and no F02 WSP has started in this deployment operation. The first failed apply, exact AV recovery, stopped observer-retired object loss and preserved host evidence remain documented below.

Status at preparation: F02 saved creation has completed; full editor verification is running. No freeze or serving change is justified until that verification succeeds. This is a diagnostic candidate awaiting canonical full WSP evaluation, not an adopted improvement.

## Scope and observed preflight

The target is the existing gemma-0 service owned by the example control script at /home/y-takagi/kairyu/examples/gemma-4-31b-8gpu/control.py. The labels of the existing container identify that control directory and its compose.yaml. The ordinary AV container is 4e545c35bc0e277344b302058eb9ffa758747f3012bc8e395e417f28d97aa56e, healthy and idle at this read-only preflight. Formal baseline binding will be repeated while owning the global evaluation lock.

The current image ID is sha256:af9e746b913088cbf6c4464a31d6904a6b7e1d3ba62f0895362ddf13595f7abe. Current command SHA is f9de227590524bb32445ccde524ca54dc387431068e55b4bd10e2491f17ebc47 and environment SHA is e7bc877f2b9eb1b944340aa38c42a093bae14a5512fdb17a8c1c5091ac8bca9b. Environment values are not written to this report. CPU set is 0-15,64-79; devices are GPU 0 and 1; IPC is host. The ordinary mounts are /checkpoint (AV, read-only), /models (the existing model/assistant directory, read-only), and the existing compile cache. No observation flags or mounts remain from the completed AV run.

The target saved candidate path is checkpoints/fresh-code-joint-f02 under the experiment root. Only /checkpoint and the checkpoint-tree label may differ in the rendered candidate service. The existing three other chat replicas, embedding, gateway, UI and all other existing containers remain protected. Chat and logprob evaluation must use the direct gemma-0 endpoint; the heterogeneous gateway is not an evaluation endpoint.

## Existing deployment helper

The existing helper is docs/experiments/scripts/redeploy-replica.py, SHA da3dc93dcd1f7c44bbd64c9a0b65403424d436bef984d7964e0f50aee9aecc02. The actual owner control script SHA is fc9a78bc232f9cc331926a0737e64f4b4fe14946574f4a02bc1aacb9a800b851. The helper source and --help were checked. A documentation-only git HEAD change is not treated as a scientific condition change; exact relevant source SHA and rendered runtime configuration are checked.

The --freeze schema requires checkpoint_path, control_tree_sha256, all_checkpoint_files (path/size/SHA) and candidate_inventory_stat (device, inode, size, mtime_ns, ctime_ns, uid, gid, mode and nlink). The planned freeze derives complete file hashes from the successful saved full verify and separately pins the current full stat. That is prior complete verification plus exact stat continuity, not another fresh full payload hash.

The helper's preview imports the real control, constructs the existing compose configuration and compares image, command, environment, user, working directory, healthcheck, CPU/GPU allocation, IPC, networks, mounts and checkpoint tree. It refuses differences beyond the checkpoint reference. The normal full-stack run.sh up path is not used. Only a later explicit --apply calls compose up --no-deps --force-recreate --detach --wait on the selected service, with rollback support.

## Global lock and observation handoff

The deployment owner must hold the real candidate-evaluation.lock flock before formal freeze/baseline/preview and any subsequent serving action. The owner PID, UID and /proc/locks evidence will be recorded. Preview alone does not authorize apply.

The observer owner confirmed its completed AV run released the global lock. The planned observation artifact is analysis/fresh-purpose-f02-live10-plan-20261107-v1/plan.json. Its frozen observer source has 89 files, checked against the AV source-dependencies record. It pins runner image sha256:f91bdcf997af6d7b194bec229812fd018277158f6795de7d91e58448936e7ad8. Canonical conditions remain tier10/dev full 778 items in 14 areas, chat32/logprob8/agent1, TP2/one replica/MTP4, 8 prefill and 16 declared decode positions, and 2 GiB capture quota.

If apply is later authorized, the deployment owner verifies the new direct endpoint, checkpoint mount/tree, health and unchanged runtime conditions. The root then authorizes observation execution, the deployment owner releases the global lock, and the observer owner acquires it nonblocking and rechecks the exact new CID and idle state. The observer attaches to the F02 CID and restores that same F02 CID at the end; it does not own a switch back to AV. The planned observation plan is not rewritten during this preview preparation.

## Read-only preflight failure

One initial inspection imported the helper with host python3, which lacks NumPy, and stopped with ModuleNotFoundError before container inspection or any mutation. The inspection was rerun with the provided infra/weight-tools-ci/bin/python, successfully inspected the current ordinary container and passed the helper idle check. This was not a model, serving or benchmark failure.

## Exact one-off freeze and preview operation

The following code uses the existing helper API; it contains no --apply path. It translates only the evidence key dev to the existing helper key device and checks every statistic without dropping fields. The global owner is an already running standard flock process; /proc/locks confirms real exclusive ownership before and after preview.

```python
"""Fixed F02 freeze/preview operation; serving apply is deliberately absent."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

R = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
B = R / "worktrees/kairyu-bench-weight-tools"
OUT = R / "analysis/fresh-purpose-20261107-f02/deployment-gemma0-v1"
PROOF = R / "analysis/fresh-purpose-20261107-f02/materialization/full-verification-freeze-input.json"
CANDIDATE = R / "checkpoints/fresh-code-joint-f02"
BASE = R / "checkpoints/code90-current-ad-helper-av-v1"
CONTROL = Path("/home/y-takagi/kairyu/examples/gemma-4-31b-8gpu/control.py")
HELPER = B / "docs/experiments/scripts/redeploy-replica.py"
PYTHON = R / "infra/weight-tools-ci/bin/python"
OWNER_PID = 145741
EXPECTED_CID = "4e545c35bc0e277344b302058eb9ffa758747f3012bc8e395e417f28d97aa56e"
sys.path.insert(0, str(B / "src"))
spec = importlib.util.spec_from_file_location("f02_existing_deploy_helper", HELPER)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

def save(name, value):
    with (OUT / name).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")

def owner_proof():
    process = Path("/proc") / str(OWNER_PID)
    assert process.stat().st_uid == os.getuid() == 1003
    os.kill(OWNER_PID, 0)
    st = (R / "candidate-evaluation.lock").stat()
    key = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
    rows = [line.split() for line in Path("/proc/locks").read_text().splitlines()]
    matches = [row for row in rows if len(row) >= 6 and row[5] == key]
    assert any(row[1] == "FLOCK" and row[3] == "WRITE" and row[4] == str(OWNER_PID) for row in matches)
    return dict(pid=OWNER_PID, uid=os.getuid(), lock_identity=key, rows=matches, observed_epoch=time.time())

def container_snapshot():
    ids = guard.command(["docker", "ps", "-aq"]).split()
    rows = json.loads(guard.command(["docker", "inspect", *ids])) if ids else []
    return {row["Id"]: dict(name=row["Name"], image_id=row["Image"],
            config_sha256=guard.digest(row["Config"]), host_config_sha256=guard.digest(row["HostConfig"]),
            mounts=guard.mounts(row), networks=sorted(row["NetworkSettings"]["Networks"]),
            running=row["State"]["Running"], health=row["State"].get("Health", {}).get("Status"))
            for row in rows}

owner = owner_proof()
assert guard.sha(PROOF) == "f897d77bf5e4bf219f6c8391cbcb738c780912d1a666ff6ac5cb4ae150fa8715"
proof = guard.read(PROOF)
assert proof["status"] == "full_editor_verification_bound_to_postverify_stat"
assert Path(proof["source"]) == BASE and Path(proof["candidate"]) == CANDIDATE
for key in ("manifest", "verify", "spec"):
    assert guard.sha(proof[key]["path"]) == proof[key]["sha256"]
manifest = guard.read(proof["manifest"]["path"])
verification = guard.read(proof["verify"]["path"])
assert verification["verified"] is True and verification["legacy_full_hashes_missing"] is False
assert manifest["source"] == str(BASE) and manifest["candidate"] == str(CANDIDATE)
assert manifest["spec_sha256"] == proof["spec"]["sha256"]
assert manifest["spec"] == guard.read(proof["spec"]["path"])
assert [{k:row[k] for k in ("path","source_sha256","candidate_sha256")} for row in proof["files"]] == verification["files"]
assert [{k:row[k] for k in ("path","size","source_sha256","candidate_sha256")} for row in manifest["files"]] == proof["files"]
for key in ("changed_elements", "changed_tensors", "delta_l2"):
    assert manifest[key] == verification[key] == proof[key]
for path, expected in proof["code_sha256"].items():
    assert guard.sha(B / path) == expected
# Evidence uses "dev"; the existing helper uses the equivalent "device".
normalize = lambda rows: {name:{("device" if key == "dev" else key):value for key,value in row.items()} for name,row in rows.items()}
source_stats = normalize(proof["source_inventory_stat"])
candidate_stats = normalize(proof["candidate_inventory_stat"])
assert guard.inventory(BASE) == source_stats and guard.inventory(CANDIDATE) == candidate_stats
assert len(candidate_stats) == len(proof["files"]) == 9
rows = sorted([dict(path=row["path"],size=row["size"],sha256=row["candidate_sha256"]) for row in proof["files"]], key=lambda row:row["path"])
tree = guard.digest(rows)
assert tree == proof["candidate_tree_sha256"]
assert guard.sha(HELPER) == "da3dc93dcd1f7c44bbd64c9a0b65403424d436bef984d7964e0f50aee9aecc02"
assert guard.sha(CONTROL) == "fc9a78bc232f9cc331926a0737e64f4b4fe14946574f4a02bc1aacb9a800b851"
target = guard.inspect("gemma-4-31b-8gpu-gemma-0-1")
assert target["Id"] == EXPECTED_CID
guard.idle(target)
before = container_snapshot()
pins = {str(path):guard.sha(path) for path in (HELPER, CONTROL, CONTROL.parent/"compose.yaml", CONTROL.parent/"example.json", CONTROL.parent/"kairyu.yaml")}
freeze = dict(schema_version=1, checkpoint_path=str(CANDIDATE), control_tree_sha256=tree,
              all_checkpoint_files=rows, candidate_inventory_stat=candidate_stats,
              source_inventory_stat=source_stats, source_tree_sha256=proof["source_tree_sha256"],
              full_verification_evidence=dict(path=str(PROOF),sha256=guard.sha(PROOF)),
              manifest=proof["manifest"], verify=proof["verify"], spec=proof["spec"],
              captured_epoch=time.time(), source_pins=pins, owner=owner,
              evidence="Successful saved full editor verification plus exact postverify stat continuity; no fresh payload rehash here.",
              full_payload_rehashed_here=False)
save("freeze.json", freeze)
save("protected-containers-before.json", before)
save("lock-owner.json", owner)
argv = [str(PYTHON),str(HELPER),"--control",str(CONTROL),"--service","gemma-0",
        "--candidate",str(CANDIDATE),"--freeze",str(OUT/"freeze.json"),
        "--baseline",str(OUT/"replica-baseline.json"),"--output",str(OUT/"preview-v1"),
        "--timeout","1200"]
save("preview-command.json",dict(argv=argv,apply=False,operation_source_sha256=guard.sha(Path(__file__))))
environment = dict(os.environ, PYTHONPATH=str(B/"src"), PYTHONDONTWRITEBYTECODE="1")
started = time.time()
with (OUT/"preview.stdout").open("x") as stdout, (OUT/"preview.stderr").open("x") as stderr:
    result = subprocess.run(argv,cwd=B,env=environment,stdout=stdout,stderr=stderr)
save("preview-execution.json",dict(returncode=result.returncode,seconds=time.time()-started))
assert result.returncode == 0
after = container_snapshot()
assert after == before
assert guard.inventory(BASE) == source_stats and guard.inventory(CANDIDATE) == candidate_stats
assert all(guard.sha(path) == value for path,value in pins.items())
owner_after = owner_proof()
report = guard.read(OUT/"preview-v1/receipt.json")
assert report["status"] == "preview" and report["apply_requested"] is False
assert report["candidate_identity"]["tree_sha256"] == tree and report["before"]["id"] == EXPECTED_CID
assert report["before"]["image_id"] == "sha256:af9e746b913088cbf6c4464a31d6904a6b7e1d3ba62f0895362ddf13595f7abe"
result = dict(status="preview_passed_no_service_change",tree_sha256=tree,
              freeze_sha256=guard.sha(OUT/"freeze.json"),preview_receipt_sha256=guard.sha(OUT/"preview-v1/receipt.json"),
              baseline_sha256=guard.sha(OUT/"replica-baseline.json"),container_count=len(before),
              all_container_snapshots_unchanged=True,all_checkpoint_stats_unchanged=True,
              owner_after=owner_after,source_pins=pins,apply_requested=False,
              operation_source_sha256=guard.sha(Path(__file__)))
save("preview-summary.json",result)
print(json.dumps(result))
```

## Completed preview

F02 saved creation completed in 215.174 seconds, followed by successful full editor verification in 174.631 seconds. Full-verification evidence SHA is f897d77bf5e4bf219f6c8391cbcb738c780912d1a666ff6ac5cb4ae150fa8715. All nine complete-file source/candidate hashes and postverify full stats were bound; every recorded weight-tool source SHA also matched. This freeze operation did not rehash model payload.

The fixed one-off source above ran successfully with the supplied CPU Python. Its SHA is 02a337c64034dffcea5b01a07bf1a6a363dec40e08bef9f091685117fd880734. The existing helper preview passed, including exact rendered image/command/environment/allocation/mount parity. All 28 existing container snapshots before and after preview were identical, and all AV/F02 file stats remained identical. There was no service mutation and no benchmark call.

Artifacts are under analysis/fresh-purpose-20261107-f02/deployment-gemma0-v1:

| Artifact | SHA256 |
|---|---|
| freeze.json | 163202cf38fe36972f90bd22058850eb412372b603c12731e3a9859abe5e4a38 |
| preview-v1/receipt.json | 0dcc878f5ee02e643a344664c96becd08d72e63b40485f84385f4fde1284f2ee |
| replica-baseline.json | b427bb9ac0b15264240c06ade7de100c75a102d61ea28bb078d3bcea75ff9082 |

Candidate tree: 46680f4066f57c1596a1f4c66b67d23929c96b325943d981c6d74260cdbb3f5c. The two edited shard hashes are e6144c1c2322cd2a05ced629c754c229ec530747470b3ba423a3c905b8d6e97c and f472653dfcfe3218f33111b44e936e097a8ba01014b9b5376f33c3196980181b. All seven nonweight file hashes equal AV.

## Lock command and apply boundary

The standard host flock utility owns the lock across the preview and pending root decision. It was started through an interactive SSH command session (session 24320), with --no-fork preserving the process PID across exec. The exact Python body is:

```python
import json, os, sys, time
print(json.dumps({"status": "global_lock_held", "pid": os.getpid(),
                  "uid": os.getuid(), "started_epoch": time.time()}), flush=True)
sys.stdin.readline()
print(json.dumps({"status": "global_lock_released_on_stdin",
                  "pid": os.getpid()}), flush=True)
```

It is executed as the command argument of:

    flock --exclusive --nonblock --no-fork \
      /mnt/nvme/kairyu/wsp/gemma4-coding-20261104/candidate-evaluation.lock \
      /mnt/nvme/kairyu/wsp/gemma4-coding-20261104/infra/weight-tools-ci/bin/python -u -c '<body above>'

Observed owner PID145741, UID1003, lock identity 09:02:181180584, FLOCK ADVISORY WRITE, was verified in /proc/locks before and after preview. Sending one line to this dedicated session releases the lock; no shared service is stopped. The state is historical evidence, and actual lock ownership must be rechecked before a future action.

The concrete proposed application uses the exact preview command with the same control/service/candidate/freeze/baseline, changes output to the fresh apply-v1 directory, and adds --apply. It remains withheld until the root's decision. The target remains the ordinary AV CID recorded above. All runtime conditions must remain pinned at application, and protected containers must be checked again. This preview does not adopt F02 or claim a canonical score.

## Approved application operation

The root explicitly approved gemma-0-only apply after reviewing the pinned preview. The exact one-off application source below preserves all protected snapshots and uses the existing helper with the same inputs plus a fresh output and --apply. It makes two tiny synthetic generation requests (one chat and one prompt-logprob) solely to establish direct API connectivity after startup; these are not WSP scores. It records public-answer identity and numeric usage only, and never reads or records reasoning content.

```python
"""Apply the explicitly approved F02 gemma-0 preview, preserving all other services."""
from pathlib import Path
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import urllib.request

R=Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
B=R/"worktrees/kairyu-bench-weight-tools"
P=R/"analysis/fresh-purpose-20261107-f02/deployment-gemma0-v1"
sys.path.insert(0,str(B/"src"))
spec=importlib.util.spec_from_file_location("f02_apply_guard",B/"docs/experiments/scripts/redeploy-replica.py")
guard=importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
from kairyu_bench.wsp.client import LogprobClient

def save(name,value):
    with (P/name).open("x")as f:json.dump(value,f,indent=2,allow_nan=False);f.write("\n")

def lock_proof():
    pid=145741
    assert (Path("/proc")/str(pid)).stat().st_uid==os.getuid()==1003
    os.kill(pid,0)
    s=(R/"candidate-evaluation.lock").stat()
    key=f"{os.major(s.st_dev):02x}:{os.minor(s.st_dev):02x}:{s.st_ino}"
    rows=[x.split()for x in Path("/proc/locks").read_text().splitlines()]
    matches=[x for x in rows if len(x)>=6 and x[5]==key]
    assert any(x[1]=="FLOCK"and x[3]=="WRITE"and x[4]==str(pid)for x in matches)
    return dict(pid=pid,identity=key,rows=matches,epoch=time.time())

def snapshots():
    ids=guard.command(["docker","ps","-aq"]).split()
    cs=json.loads(guard.command(["docker","inspect",*ids]))if ids else[]
    return {c["Id"]:dict(name=c["Name"],image_id=c["Image"],config_sha256=guard.digest(c["Config"]),
            host_config_sha256=guard.digest(c["HostConfig"]),mounts=guard.mounts(c),
            networks=sorted(c["NetworkSettings"]["Networks"]),running=c["State"]["Running"],
            health=c["State"].get("Health",{}).get("Status"))for c in cs}

pins={"freeze.json":"163202cf38fe36972f90bd22058850eb412372b603c12731e3a9859abe5e4a38",
      "replica-baseline.json":"b427bb9ac0b15264240c06ade7de100c75a102d61ea28bb078d3bcea75ff9082",
      "preview-v1/receipt.json":"0dcc878f5ee02e643a344664c96becd08d72e63b40485f84385f4fde1284f2ee"}
for path,digest in pins.items():assert guard.sha(P/path)==digest
summary=guard.read(P/"preview-summary.json")
for path,digest in summary["source_pins"].items():assert guard.sha(path)==digest
freeze=guard.read(P/"freeze.json")
candidate=Path(freeze["checkpoint_path"])
base=R/"checkpoints/code90-current-ad-helper-av-v1"
assert guard.inventory(candidate)==freeze["candidate_inventory_stat"]
assert guard.inventory(base)==freeze["source_inventory_stat"]
before=snapshots()
assert before==guard.read(P/"protected-containers-before.json")
old=guard.read(P/"preview-v1/receipt.json")["before"]
assert guard.snapshot(guard.inspect("gemma-4-31b-8gpu-gemma-0-1"))==old
guard.idle(guard.inspect(old["id"]))
owner=lock_proof()
argv=guard.read(P/"preview-command.json")["argv"]
argv[argv.index("--output")+1]=str(P/"apply-v1")
assert argv[-2:]==["--timeout","1200"]
argv.append("--apply")
save("apply-authorization.json",dict(approved_scope="F02 gemma-0 only; root approved exact freeze/baseline/preview hashes",
     pinned_inputs=pins,owner=owner,argv=argv,operation_source_sha256=guard.sha(Path(__file__)),started_epoch=time.time()))
started=time.time()
with (P/"apply.stdout").open("x")as stdout,(P/"apply.stderr").open("x")as stderr:
    execution=subprocess.run(argv,cwd=B,env=dict(os.environ,PYTHONPATH=str(B/"src"),PYTHONDONTWRITEBYTECODE="1"),stdout=stdout,stderr=stderr)
save("apply-execution.json",dict(returncode=execution.returncode,seconds=time.time()-started,owner_after=lock_proof()))
report=guard.read(P/"apply-v1/receipt.json")
after=snapshots()
protected={key:value for key,value in before.items()if key!=old["id"]}
protected_equal=all(after.get(key)==value for key,value in protected.items())
save("protected-containers-after.json",dict(containers=after,protected_equal=protected_equal))
assert protected_equal
assert execution.returncode==0 and report["status"]=="completed"
current=guard.inspect("gemma-4-31b-8gpu-gemma-0-1")
verified=guard.compare_after(old,current,candidate,freeze["control_tree_sha256"])
assert current["Id"]!=old["id"]
assert set(after)==set(protected)|{current["Id"]}
assert guard.inventory(candidate)==freeze["candidate_inventory_stat"]
assert guard.inventory(base)==freeze["source_inventory_stat"]
ips=[x["IPAddress"]for x in current["NetworkSettings"]["Networks"].values()if x["IPAddress"]]
assert len(ips)==1
endpoint="http://"+ips[0]+":8000/v1"
with urllib.request.urlopen(endpoint+"/models",timeout=30)as response:models=json.load(response)
assert "gemma-4-31b"in[x["id"]for x in models["data"]]
payload={"model":"gemma-4-31b","messages":[{"role":"user","content":"Return only the integer result of 17 + 26."}],
         "temperature":0,"top_p":1,"seed":1700,"max_tokens":8,"chat_template_kwargs":{"enable_thinking":False}}
request=urllib.request.Request(endpoint+"/chat/completions",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
with urllib.request.urlopen(request,timeout=180)as response:chat=json.load(response)
assert chat["choices"] and chat.get("model")=="gemma-4-31b"
public=chat["choices"][0]["message"].get("content")or""
assert public.strip()
client=LogprobClient((endpoint,),model="gemma-4-31b",timeout=180)
tokens=client.tokenize("The number is 43.",add_special_tokens=True)
positions=client.prompt_logprobs(tokens,top_k=1)
assert len(positions)==len(tokens)>1
assert all(entry is None or all(math.isfinite(value[0])for value in entry.values())for entry in positions)
guard.idle(current)
save("direct-endpoint-smoke.json",dict(kind="synthetic_connectivity_not_benchmark",endpoint=endpoint,
     checkpoint_tree=freeze["control_tree_sha256"],model_ids=[x["id"]for x in models["data"]],
     chat=dict(public_sha256=guard.digest(public),usage=chat.get("usage"),finish_reason=chat["choices"][0].get("finish_reason")),
     logprob=dict(token_count=len(tokens),position_count=len(positions),api_calls=client.calls),
     generation_requests=2,tokenize_requests=1,model_list_requests=1,reasoning_not_read_or_recorded=True))
assert all(guard.sha(path)==digest for path,digest in summary["source_pins"].items())
assert guard.inventory(candidate)==freeze["candidate_inventory_stat"]
assert snapshots()==after
closure=dict(status="completed_healthy_direct_smoke_passed",new_cid=current["Id"],original_cid=old["id"],
             checkpoint_path=str(candidate),tree=freeze["control_tree_sha256"],direct_endpoint=endpoint,
             model="gemma-4-31b",runtime_parity=True,protected_container_count=len(protected),
             protected_unchanged=True,apply_receipt_sha256=guard.sha(P/"apply-v1/receipt.json"),
             before_after=verified,owner=lock_proof(),seconds=time.time()-started,
             global_lock_still_held=True,benchmark_started=False)
save("apply-closure.json",closure)
print(json.dumps({key:closure[key]for key in ("status","new_cid","tree","direct_endpoint","protected_container_count","seconds","global_lock_still_held")}))
```

## Failure recovery and owned temporary-container cleanup source

The first apply and the helper rollback both failed. The root authorized exact AV restoration and, after attribution, removal only of the newly created never-started rollback container. The cleanup source below records the failed operation and requires every remaining original container to match its pre-apply snapshot, except the stopped observer-retired object already removed by Compose. It requires original AV to be healthy, attributes the new object by creation interval and pre/post inventories, saves its attributes and empty logs, checks its read-only AV mount, and performs non-force docker rm on that exact new ID only. It does not remove a model, host bind data or any original running container.

```python
"""Audit failed F02 deployment and remove only its never-started rollback container."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import subprocess
import time
R=Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
P=R/"analysis/fresh-purpose-20261107-f02/deployment-gemma0-v1"
CID="0cc2ff600528d2ad4b5b187bb97ba270b8c6c7d6139fb62e488e5ed1869bde36"
RETIRED="21aaf4aed1096bd7205fb5369528d6c84d61672b01fc5ef36bfd57bbbbb690cd"
ORIGINAL="4e545c35bc0e277344b302058eb9ffa758747f3012bc8e395e417f28d97aa56e"
def read(p):return json.loads(p.read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stat(p):
 s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,
  ctime_ns=s.st_ctime_ns,uid=s.st_uid,gid=s.st_gid,mode=s.st_mode,nlink=s.st_nlink)
def inspect_all():
 ids=subprocess.check_output(["docker","ps","-aq"],text=True).split()
 return json.loads(subprocess.check_output(["docker","inspect",*ids],text=True))if ids else[]
def snapshots(cs):
 return {c["Id"]:dict(name=c["Name"],image_id=c["Image"],config_sha256=digest(c["Config"]),
  host_config_sha256=digest(c["HostConfig"]),mounts=sorted([dict(type=m["Type"],source=m["Source"],
  target=m["Destination"],read_only=not m["RW"])for m in c["Mounts"]],key=lambda x:x["target"]),
  networks=sorted(c["NetworkSettings"]["Networks"]),running=c["State"]["Running"],
  health=c["State"].get("Health",{}).get("Status"))for c in cs}
def save(name,x):
 with (P/name).open("x")as f:json.dump(x,f,indent=2);f.write("\n")
assert os.getuid()==1003
st=(R/"candidate-evaluation.lock").stat()
key=f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
locks=[x.split()for x in Path("/proc/locks").read_text().splitlines()]
assert any(len(x)>=6 and x[1]=="FLOCK"and x[3]=="WRITE"and x[4]=="145741"and x[5]==key for x in locks)
before=read(P/"protected-containers-before.json")
current=inspect_all();now=snapshots(current)
assert set(before)-set(now)=={RETIRED}
assert set(now)-set(before)=={CID}
assert all(now[cid]==value for cid,value in before.items()if cid!=RETIRED)
assert now[ORIGINAL]["running"]and now[ORIGINAL]["health"]=="healthy"
candidate=next(c for c in current if c["Id"]==CID)
saved=read(P/"temporary-created-container.json")
assert candidate["Name"]==saved["name"] and candidate["Created"]==saved["created"]
assert digest(candidate["Config"])==saved["config_sha256"]and digest(candidate["HostConfig"])==saved["hostconfig_sha256"]
assert candidate["State"]["Status"]=="created"and not candidate["State"]["Running"]and candidate["State"]["Pid"]==0
assert candidate["State"]["StartedAt"].startswith("0001-01-01")
created=datetime.fromisoformat(candidate["Created"].replace("Z","+00:00")).timestamp()
auth=read(P/"apply-authorization.json");execution=read(P/"apply-execution.json")
assert auth["started_epoch"]<=created<=auth["started_epoch"]+execution["seconds"]+2
assert CID not in before and CID!=ORIGINAL
mount=next(m for m in candidate["Mounts"]if m["Destination"]=="/checkpoint")
assert mount["Source"]==str(R/"checkpoints/code90-current-ad-helper-av-v1")and not mount["RW"]
assert candidate["Config"]["Labels"]["kairyu.chat-checkpoint-tree"]=="5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf"
assert read(P/"temporary-created-container-logs.json")["stdout_bytes"]==0
freeze=read(P/"freeze.json")
assert sha(P/"freeze.json")=="163202cf38fe36972f90bd22058850eb412372b603c12731e3a9859abe5e4a38"
for root,rows in [(Path(freeze["checkpoint_path"]),freeze["candidate_inventory_stat"]),
                 (R/"checkpoints/code90-current-ad-helper-av-v1",freeze["source_inventory_stat"])]:
 assert all(stat(root/name)==expected for name,expected in rows.items())
proof=dict(status="cleanup_prechecks_passed",target=CID,original_av_restored_unchanged=True,
 stopped_retired_removed_by_compose=RETIRED,other_26_original_containers_unchanged=True,
 target_created_during_apply_window=True,target_never_started=True,target_mounts_saved=True,
 target_logs_empty=True,all_av_f02_weight_stats_unchanged=True,
 source_freeze_sha256=sha(P/"freeze.json"),operation_source_sha256=sha(Path(__file__)),checked_epoch=time.time())
save("cleanup-prechecks.json",proof)
subprocess.run(["docker","rm",CID],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
after=snapshots(inspect_all())
assert set(after)==set(before)-{RETIRED}
assert all(after[cid]==value for cid,value in before.items()if cid!=RETIRED)
proof.update(status="only_own_never_started_container_removed",removed=CID,
             remaining_original_containers=len(after),original_av_healthy=True,finished_epoch=time.time())
save("cleanup-completion.json",proof)
print(json.dumps(proof))
```

## Apply-v1 failure and complete recovery

The approved helper invocation failed after 2.697 seconds, and its automatic rollback also failed. Its receipt reports status=failed, error_type=RuntimeError, rollback_completed=false and rollback_error_type=RuntimeError. The outer owner stopped at its protected-container assertion; synthetic chat/logprob connectivity probes were not reached and no F02 WSP began.

The before/after snapshots showed that the stopped observer-retired container 21aaf4aed1096bd7205fb5369528d6c84d61672b01fc5ef36bfd57bbbbb690cd had disappeared, and a new temporary-named container 0cc2ff600528d2ad4b5b187bb97ba270b8c6c7d6139fb62e488e5ed1869bde36 existed. The original ordinary AV container remained but was exited, with no allocated network IP. An initial status probe expecting an IP stopped with StopIteration; the recorded exited state corrected the earlier tentative statement that AV merely remained present. Presence did not mean health.

Recovery restarted only the original exact AV CID after proving all pre-apply configuration fields unchanged and verifying the global flock owner. It returned healthy and idle in 111.173 seconds. The compared fields are the helper snapshot (ID, image, command/environment hashes, user, working directory, healthcheck, CPU/GPU, IPC, mounts, networks, checkpoint path/tree, running and health); process start time necessarily changed. The exact one-off recovery operations were:

```python
from pathlib import Path
import importlib.util, json, sys, subprocess, time, os
r = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
b = r / "worktrees/kairyu-bench-weight-tools"
p = r / "analysis/fresh-purpose-20261107-f02/deployment-gemma0-v1"
sys.path.insert(0, str(b / "src"))
s = importlib.util.spec_from_file_location("f02_emergency_original_restart", b / "docs/experiments/scripts/redeploy-replica.py")
g = importlib.util.module_from_spec(s)
s.loader.exec_module(g)
old = g.read(p / "preview-v1/receipt.json")["before"]
c = g.inspect(old["id"])
snapshot = g.snapshot(c)
assert c["Id"] == "4e545c35bc0e277344b302058eb9ffa758747f3012bc8e395e417f28d97aa56e"
assert all(snapshot[k] == v for k, v in old.items() if k not in ("running", "health"))
assert not snapshot["running"]
lock = (r / "candidate-evaluation.lock").stat()
key = f"{os.major(lock.st_dev):02x}:{os.minor(lock.st_dev):02x}:{lock.st_ino}"
rows = [x.split() for x in Path("/proc/locks").read_text().splitlines()]
assert any(len(x) >= 6 and x[1] == "FLOCK" and x[3] == "WRITE" and x[4] == "145741" and x[5] == key for x in rows)
record = dict(status="restarting_original_exact_av_cid", cid=old["id"], settings_match_original=True,
              lock_owner_pid=145741, started_epoch=time.time(), helper_rollback_failed=True,
              operation="docker start original CID only")
(p / "manual-original-restart-start.json").write_text(json.dumps(record, indent=2) + "\n")
subprocess.run(["docker", "start", old["id"]], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
deadline = time.monotonic() + 600
while time.monotonic() < deadline:
    c = g.inspect(old["id"])
    now = g.snapshot(c)
    if now["running"] and now["health"] == "healthy":
        assert now == old
        g.idle(c)
        record.update(status="original_exact_av_cid_restored_healthy", snapshot=now,
                      seconds=time.time() - record["started_epoch"])
        break
    if not now["running"]:
        raise RuntimeError("original AV restart exited")
    time.sleep(5)
else:
    raise TimeoutError("original AV health not ready within600s")
(p / "manual-original-restart-completion.json").write_text(json.dumps(record, indent=2) + "\n")
```

The temporary object was attributed to this failed operation by absence from the complete pre-apply inventory, Created time inside its execution window, never-started state (created, PID0 and zero StartedAt), the read-only AV checkpoint mount and tree, saved configuration hashes, and empty logs. Its state/configuration metadata and zero-length log receipt were saved before non-force removal of that exact ID. No original service container was removed during this cleanup.

Final impact audit: all 27 surviving original containers matched their pre-apply snapshots, including healthy original AV. The 26 other original containers were unchanged. The stopped observer-retired Docker object is permanently absent and is explicitly recorded as a side effect of the failed Compose operation. Its host bind data were not removed. The observation owner separately verified capture-config SHA, both closed-worker statuses, zero pending/drop/error/quota counters, WSP run/report SHA, private-original receipt SHA, and existence of client/server traces, both compressed observation streams, acceptance/sites data, merged output and official results. Large trace payloads were not freshly rehashed. Those checks are in the AV artifact directory's post-compose-failure-host-evidence-check.json.

The model freeze SHA remains 163202cf38fe36972f90bd22058850eb412372b603c12731e3a9859abe5e4a38. All nine AV and F02 file stats still match the saved freeze. The candidate has not been successfully deployed or evaluated.

## Cause evidence and limits

The failed helper suppresses raw Compose stdout/stderr, so the specific Docker/Compose error text is unavailable. A narrow historical docker events query returned zero records; that absence is retained in apply-docker-events.json and is not treated as proof that no events occurred.

The observed topology contained two objects with the same Compose project/service identity. The observer owner reconstructed the retired object's label provenance from the saved original inspect, the pinned observe-docker.py source, and successful attach/restore CID receipts. That source retains Config.Labels and adds only observer labels during creation; restore stops and renames the observer object without changing labels. The reconstructed values are project=gemma-4-31b-8gpu, service=gemma-0, container-number=1 and oneoff=False, matching ordinary AV. This is a source-and-receipt reconstruction, not an independently saved inspect of the retired object itself. The safe evidence is retired-observer-label-evidence.json in the AV artifact directory.

Duplicate Compose identities are therefore an observed unsafe topology consistent with the failed recreate/rollback and temporary-name conflict. The precise suppressed error remains uncertain. The root is adding a guard that refuses extra same-project/same-service CIDs before preview or apply; this audit does not bypass that guard or any source pin. Any retry requires new source-aware preview evidence, because the helper source will change. No second apply was attempted here.

Incident artifacts under deployment-gemma0-v1 include apply-authorization.json, apply-execution.json, apply-v1/receipt.json, protected-containers-after.json, apply-docker-events.json, temporary-created-container.json, temporary-created-container-logs.json, manual-original-restart-start.json, manual-original-restart-completion.json, cleanup-prechecks.json and cleanup-completion.json. The global lock remained held throughout recovery and cleanup.

## Fresh preview-v2 after guard repair

The root reviewed helper SHA 2e9da820fe30ba9b2436fb600cf14a533b1c496b2fed48ac3649fffb9d2bc9ae after focused tests and an independent review. Its new guard includes stopped containers when checking Compose project/service membership, requires the exact target CID before preview and apply, permits at most one member for rollback, and forces COMPOSE_REMOVE_ORPHANS=0. The immutable candidate freeze continues to describe the same verified weight bytes; its old helper pin remains historical provenance. A new explicit source amendment pins the approved helper while retaining and rechecking every other source pin. The old failure and all prior receipts remain.

The following one-off uses a fresh 27-container protected snapshot, a new all-replica baseline and new preview output. It does not apply serving changes.

```python
"""Fresh F02 preview after the explicitly reviewed duplicate-Compose guard repair."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import time
R=Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
B=R/"worktrees/kairyu-bench-weight-tools"
P=R/"analysis/fresh-purpose-20261107-f02/deployment-gemma0-v1"
H=B/"docs/experiments/scripts/redeploy-replica.py"
EXPECTED_HELPER="2e9da820fe30ba9b2436fb600cf14a533b1c496b2fed48ac3649fffb9d2bc9ae"
sys.path.insert(0,str(B/"src"))
spec=importlib.util.spec_from_file_location("f02_repaired_deploy_guard",H)
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
def save(name,value):
 with(P/name).open("x")as f:json.dump(value,f,indent=2,allow_nan=False);f.write("\n")
def owner():
 pid=145741
 assert(Path("/proc")/str(pid)).stat().st_uid==os.getuid()==1003
 os.kill(pid,0)
 st=(R/"candidate-evaluation.lock").stat()
 key=f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
 rows=[x.split()for x in Path("/proc/locks").read_text().splitlines()]
 found=[x for x in rows if len(x)>=6 and x[5]==key]
 assert any(x[1]=="FLOCK"and x[3]=="WRITE"and x[4]==str(pid)for x in found)
 return dict(pid=pid,identity=key,rows=found,observed_epoch=time.time())
def snapshots():
 ids=g.command(["docker","ps","-aq"]).split()
 cs=json.loads(g.command(["docker","inspect",*ids]))if ids else[]
 return {c["Id"]:dict(name=c["Name"],image_id=c["Image"],config_sha256=g.digest(c["Config"]),
   host_config_sha256=g.digest(c["HostConfig"]),mounts=g.mounts(c),
   networks=sorted(c["NetworkSettings"]["Networks"]),running=c["State"]["Running"],
   health=c["State"].get("Health",{}).get("Status"))for c in cs}
lock=owner()
assert g.sha(H)==EXPECTED_HELPER
assert g.sha(P/"freeze.json")=="163202cf38fe36972f90bd22058850eb412372b603c12731e3a9859abe5e4a38"
freeze=g.read(P/"freeze.json")
assert g.inventory(Path(freeze["checkpoint_path"]))==freeze["candidate_inventory_stat"]
assert g.inventory(R/"checkpoints/code90-current-ad-helper-av-v1")==freeze["source_inventory_stat"]
old_summary=g.read(P/"preview-summary.json")
pins=dict(old_summary["source_pins"])
assert pins[str(H)]=="da3dc93dcd1f7c44bbd64c9a0b65403424d436bef984d7964e0f50aee9aecc02"
pins[str(H)]=EXPECTED_HELPER
for path,digest in pins.items():assert g.sha(path)==digest
before=snapshots()
original=g.read(P/"protected-containers-before.json")
retired="21aaf4aed1096bd7205fb5369528d6c84d61672b01fc5ef36bfd57bbbbb690cd"
assert len(before)==27 and set(before)==set(original)-{retired}
assert all(before[cid]==value for cid,value in original.items()if cid!=retired)
target=g.inspect("gemma-4-31b-8gpu-gemma-0-1")
assert target["Id"]=="4e545c35bc0e277344b302058eb9ffa758747f3012bc8e395e417f28d97aa56e"
g.idle(target)
amendment=dict(reason="Root-approved duplicate Compose membership guard and COMPOSE_REMOVE_ORPHANS=0 repair",
 old_helper_sha256=old_summary["source_pins"][str(H)],new_helper_sha256=EXPECTED_HELPER,
 candidate_weight_freeze_reused_unchanged=True,weight_freeze_sha256=g.sha(P/"freeze.json"),
 historical_freeze_source_pins_are_provenance_not_current_execution_pins=True,current_source_pins=pins,
 old_baseline_not_reused=True,new_container_count=27,owner=lock,operation_source_sha256=g.sha(Path(__file__)))
save("source-amendment-v2.json",amendment)
save("protected-containers-before-v2.json",before)
argv=g.read(P/"preview-command.json")["argv"]
argv[argv.index("--baseline")+1]=str(P/"replica-baseline-v2.json")
argv[argv.index("--output")+1]=str(P/"preview-v2")
assert "--apply"not in argv and argv[-2:]==["--timeout","1200"]
save("preview-command-v2.json",dict(argv=argv,apply=False,source_amendment_sha256=g.sha(P/"source-amendment-v2.json")))
started=time.time()
with(P/"preview-v2.stdout").open("x")as stdout,(P/"preview-v2.stderr").open("x")as stderr:
 result=subprocess.run(argv,cwd=B,env=dict(os.environ,PYTHONPATH=str(B/"src"),PYTHONDONTWRITEBYTECODE="1"),stdout=stdout,stderr=stderr)
save("preview-execution-v2.json",dict(returncode=result.returncode,seconds=time.time()-started))
assert result.returncode==0
report=g.read(P/"preview-v2/receipt.json")
assert report["status"]=="preview"and report["apply_requested"]is False
assert report["helper_sha256"]==EXPECTED_HELPER
assert report["compose_remove_orphans"]is False
assert report["compose_selection"]["container_ids"]==[target["Id"]]
assert report["compose_selection"]["includes_stopped"]is True
assert report["candidate_identity"]["tree_sha256"]==freeze["control_tree_sha256"]
assert snapshots()==before
assert g.inventory(Path(freeze["checkpoint_path"]))==freeze["candidate_inventory_stat"]
assert g.inventory(R/"checkpoints/code90-current-ad-helper-av-v1")==freeze["source_inventory_stat"]
assert all(g.sha(path)==digest for path,digest in pins.items())
summary=dict(status="fresh_preview_v2_passed_no_service_change",helper_sha256=EXPECTED_HELPER,
 tree=freeze["control_tree_sha256"],freeze_sha256=g.sha(P/"freeze.json"),
 preview_receipt_sha256=g.sha(P/"preview-v2/receipt.json"),
 baseline_sha256=g.sha(P/"replica-baseline-v2.json"),source_amendment_sha256=g.sha(P/"source-amendment-v2.json"),
 source_pins=pins,container_count=27,all_containers_unchanged=True,all_checkpoint_stats_unchanged=True,
 compose_selection=report["compose_selection"],compose_remove_orphans=False,owner=owner(),apply_requested=False)
save("preview-summary-v2.json",summary)
print(json.dumps(summary))
```

## Preview-v2 result and approved apply-v2 source

Fresh preview-v2 passed with helper2e9da820..., exact one-CID Compose selection including stopped objects, orphan removal disabled, all 27 protected snapshots unchanged and AV/F02 stats unchanged. Preview receipt SHA79adf5380f1a03f687988375b9f870a20e27f2a6dc7b1e12d62d6df820d17607, baseline SHAa08dbafefb5a05551b78ab35f3f29708815fa984f056f8e20ff4212a550795a0, source-amendment SHAc1bfb754b6bcaa5aa227df0a31ef4cff5dafb82a3572e48553ef490eb6c1ce10. Weight freeze163202cf... is unchanged. The root explicitly approved the second application against these exact inputs. This new source retains every v1 failure artifact and uses fresh v2 outputs. It rechecks the actual Compose selection and the full approved helper SHA immediately before invoking the existing helper.

```python
"""Apply the explicitly approved F02 gemma-0 preview, preserving all other services."""
from pathlib import Path
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import urllib.request

R=Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
B=R/"worktrees/kairyu-bench-weight-tools"
P=R/"analysis/fresh-purpose-20261107-f02/deployment-gemma0-v1"
sys.path.insert(0,str(B/"src"))
spec=importlib.util.spec_from_file_location("f02_apply_guard",B/"docs/experiments/scripts/redeploy-replica.py")
guard=importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
from kairyu_bench.wsp.client import LogprobClient

def save(name,value):
    with (P/name).open("x")as f:json.dump(value,f,indent=2,allow_nan=False);f.write("\n")

def lock_proof():
    pid=145741
    assert (Path("/proc")/str(pid)).stat().st_uid==os.getuid()==1003
    os.kill(pid,0)
    s=(R/"candidate-evaluation.lock").stat()
    key=f"{os.major(s.st_dev):02x}:{os.minor(s.st_dev):02x}:{s.st_ino}"
    rows=[x.split()for x in Path("/proc/locks").read_text().splitlines()]
    matches=[x for x in rows if len(x)>=6 and x[5]==key]
    assert any(x[1]=="FLOCK"and x[3]=="WRITE"and x[4]==str(pid)for x in matches)
    return dict(pid=pid,identity=key,rows=matches,epoch=time.time())

def snapshots():
    ids=guard.command(["docker","ps","-aq"]).split()
    cs=json.loads(guard.command(["docker","inspect",*ids]))if ids else[]
    return {c["Id"]:dict(name=c["Name"],image_id=c["Image"],config_sha256=guard.digest(c["Config"]),
            host_config_sha256=guard.digest(c["HostConfig"]),mounts=guard.mounts(c),
            networks=sorted(c["NetworkSettings"]["Networks"]),running=c["State"]["Running"],
            health=c["State"].get("Health",{}).get("Status"))for c in cs}

pins={"freeze.json":"163202cf38fe36972f90bd22058850eb412372b603c12731e3a9859abe5e4a38",
      "replica-baseline-v2.json":"a08dbafefb5a05551b78ab35f3f29708815fa984f056f8e20ff4212a550795a0",
      "preview-v2/receipt.json":"79adf5380f1a03f687988375b9f870a20e27f2a6dc7b1e12d62d6df820d17607"}
pins["source-amendment-v2.json"]="c1bfb754b6bcaa5aa227df0a31ef4cff5dafb82a3572e48553ef490eb6c1ce10"
for path,digest in pins.items():assert guard.sha(P/path)==digest
summary=guard.read(P/"preview-summary-v2.json")
for path,digest in summary["source_pins"].items():assert guard.sha(path)==digest
freeze=guard.read(P/"freeze.json")
candidate=Path(freeze["checkpoint_path"])
base=R/"checkpoints/code90-current-ad-helper-av-v1"
assert guard.inventory(candidate)==freeze["candidate_inventory_stat"]
assert guard.inventory(base)==freeze["source_inventory_stat"]
before=snapshots()
assert before==guard.read(P/"protected-containers-before-v2.json")
old=guard.read(P/"preview-v2/receipt.json")["before"]
assert guard.snapshot(guard.inspect("gemma-4-31b-8gpu-gemma-0-1"))==old
guard.idle(guard.inspect(old["id"]))
assert guard.sha(B/"docs/experiments/scripts/redeploy-replica.py")=="2e9da820fe30ba9b2436fb600cf14a533b1c496b2fed48ac3649fffb9d2bc9ae"
owner=lock_proof()
argv=guard.read(P/"preview-command-v2.json")["argv"]
argv[argv.index("--output")+1]=str(P/"apply-v2")
assert argv[-2:]==["--timeout","1200"]
argv.append("--apply")
control_path=Path(argv[argv.index("--control")+1])
control_spec=importlib.util.spec_from_file_location("f02_v2_actual_control",control_path)
control=importlib.util.module_from_spec(control_spec);control_spec.loader.exec_module(control)
selection=guard.compose_scope(control,"gemma-0",old["id"])
assert selection==guard.read(P/"preview-v2/receipt.json")["compose_selection"]
assert guard.read(P/"preview-v2/receipt.json")["compose_remove_orphans"]is False
save("apply-authorization-v2.json",dict(compose_selection=selection,compose_remove_orphans=False,approved_scope="F02 gemma-0 only; root approved exact freeze/baseline/preview hashes",
     pinned_inputs=pins,owner=owner,argv=argv,operation_source_sha256=guard.sha(Path(__file__)),started_epoch=time.time()))
started=time.time()
with (P/"apply-v2.stdout").open("x")as stdout,(P/"apply-v2.stderr").open("x")as stderr:
    execution=subprocess.run(argv,cwd=B,env=dict(os.environ,PYTHONPATH=str(B/"src"),PYTHONDONTWRITEBYTECODE="1"),stdout=stdout,stderr=stderr)
save("apply-execution-v2.json",dict(returncode=execution.returncode,seconds=time.time()-started,owner_after=lock_proof()))
report=guard.read(P/"apply-v2/receipt.json")
after=snapshots()
protected={key:value for key,value in before.items()if key!=old["id"]}
protected_equal=all(after.get(key)==value for key,value in protected.items())
save("protected-containers-after-v2.json",dict(containers=after,protected_equal=protected_equal))
assert protected_equal
assert execution.returncode==0 and report["status"]=="completed"
current=guard.inspect("gemma-4-31b-8gpu-gemma-0-1")
verified=guard.compare_after(old,current,candidate,freeze["control_tree_sha256"])
assert current["Id"]!=old["id"]
assert set(after)==set(protected)|{current["Id"]}
assert guard.inventory(candidate)==freeze["candidate_inventory_stat"]
assert guard.inventory(base)==freeze["source_inventory_stat"]
ips=[x["IPAddress"]for x in current["NetworkSettings"]["Networks"].values()if x["IPAddress"]]
assert len(ips)==1
endpoint="http://"+ips[0]+":8000/v1"
with urllib.request.urlopen(endpoint+"/models",timeout=30)as response:models=json.load(response)
assert "gemma-4-31b"in[x["id"]for x in models["data"]]
payload={"model":"gemma-4-31b","messages":[{"role":"user","content":"Return only the integer result of 17 + 26."}],
         "temperature":0,"top_p":1,"seed":1700,"max_tokens":8,"chat_template_kwargs":{"enable_thinking":False}}
request=urllib.request.Request(endpoint+"/chat/completions",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
with urllib.request.urlopen(request,timeout=180)as response:chat=json.load(response)
assert chat["choices"] and chat.get("model")=="gemma-4-31b"
public=chat["choices"][0]["message"].get("content")or""
assert public.strip()
client=LogprobClient((endpoint,),model="gemma-4-31b",timeout=180)
tokens=client.tokenize("The number is 43.",add_special_tokens=True)
positions=client.prompt_logprobs(tokens,top_k=1)
assert len(positions)==len(tokens)>1
assert all(entry is None or all(math.isfinite(value[0])for value in entry.values())for entry in positions)
guard.idle(current)
save("direct-endpoint-smoke-v2.json",dict(kind="synthetic_connectivity_not_benchmark",endpoint=endpoint,
     checkpoint_tree=freeze["control_tree_sha256"],model_ids=[x["id"]for x in models["data"]],
     chat=dict(public_sha256=guard.digest(public),usage=chat.get("usage"),finish_reason=chat["choices"][0].get("finish_reason")),
     logprob=dict(token_count=len(tokens),position_count=len(positions),api_calls=client.calls),
     generation_requests=2,tokenize_requests=1,model_list_requests=1,reasoning_not_read_or_recorded=True))
assert all(guard.sha(path)==digest for path,digest in summary["source_pins"].items())
assert guard.inventory(candidate)==freeze["candidate_inventory_stat"]
assert snapshots()==after
closure=dict(status="completed_healthy_direct_smoke_passed",new_cid=current["Id"],original_cid=old["id"],
             checkpoint_path=str(candidate),tree=freeze["control_tree_sha256"],direct_endpoint=endpoint,
             model="gemma-4-31b",runtime_parity=True,protected_container_count=len(protected),
             protected_unchanged=True,apply_receipt_sha256=guard.sha(P/"apply-v2/receipt.json"),
             before_after=verified,owner=lock_proof(),seconds=time.time()-started,
             global_lock_still_held=True,benchmark_started=False)
save("apply-closure-v2.json",closure)
print(json.dumps({key:closure[key]for key in ("status","new_cid","tree","direct_endpoint","protected_container_count","seconds","global_lock_still_held")}))
```

## Apply-v2 completion

The repaired helper completed and the outer owner passed all final checks. New ordinary F02 CID e275b7bcc331d6bcdd82a1729c0ec0cc7a80e581a008bd273d0b1aa953e82c02 is healthy, with the frozen checkpoint tree 46680f4066f57c1596a1f4c66b67d23929c96b325943d981c6d74260cdbb3f5c. Runtime image, command/environment, CPU/GPU, IPC, healthcheck and mounts match the approved preview except for the intended checkpoint reference. All other 26 containers match the v2 pre-apply snapshot; AV and F02 checkpoint stats remain unchanged. Total apply plus synthetic connectivity checks took 113.98006296157837 seconds.

The direct endpoint is http://172.26.0.4:8000/v1. The Compose network aliases include gemma-0, so the planned direct chat and logprob URL resolves to this replica. A model-list request, one synthetic chat generation, one tokenization request and one synthetic prompt-logprob generation succeeded. These are connectivity checks, not official WSP observations or scores. No reasoning field was read or recorded; only public-output identity and numeric usage were retained. No F02 WSP started in this operation.

| Artifact | SHA256 |
|---|---|
| apply-v2/receipt.json | 446f71c7fbbfb9f03f1ba7583ce63d3b084d0ace87df865d98e0f00d19247d56 |
| apply-closure-v2.json | 8937e7cf83207a67f065af38c80ef4ca0e1ce6cd292d93113e36d37a1cf11a83 |
| direct-endpoint-smoke-v2.json | 105abf6bbd79831e59567d48f66382611b872481d08cfa9b3c1120e43dad04d4 |
| apply-authorization-v2.json | 850090dac8b07b37eb0f63d46d29386b370a4d4460c2bc10f0ab1018867ca726 |

apply-v2-final-evidence.json binds these receipts and the network aliases. At closure the global lock is still held by PID145741/session24320; the observer owner was given the healthy CID but instructed to wait for the root handoff and lock-release notification. The first failed apply, automatic rollback failure, restoration and cleanup records remain intact.

## Completed observation handoff

The root authorized the full-778 observation owner. The observer bound plan SHA8ff5619bd55b93b9f967ae06cb8188ab6a015f690f9c1791f557bdea0a84e925 and owner SHAed51d091c05924a09ae3a3fa27c549d50286f0d9ed9fbc3912d1b31c0ac0a3a9, then reported preview success and launch readiness. The deployment owner rechecked that plan/owner/closure identity, exact healthy F02 CID/tree/stat and idle queues. It released its dedicated flock process by sending one line to session24320; the process printed global_lock_released_on_stdin and exited0.

The old PID145741 and old flock disappeared. The new observer PID211879 acquired the same real flock under UID1003. Its launch and execution receipts confirm startup; the handoff evidence observed phase attaching_observer. That phase is preserved as observed, not upgraded to a completed or scored benchmark. No model-service operation was performed by the deployment owner after release. The observer now owns attach, canonical evaluation, writer drain and restoration to the ordinary F02 CID.

Evidence: handoff-ready.json, handoff-release.json and handoff-accepted.json in the deployment artifact directory. The accepted receipt binds launch-receipt SHA 00f218e78274c7fecd006e27d774ef974628c50baca518c5dd1cf2d38e13886e and an execution snapshot SHA 370ce0d84db55b6f5574eb9e53019486bb4a38577a753dd7b330bdc1221720cc. The release time is an observed interval between pre-release and post-release checks, not a fabricated exact timestamp.
