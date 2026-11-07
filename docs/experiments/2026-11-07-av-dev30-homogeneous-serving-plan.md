# AV homogeneous four-replica plan for canonical dev30

2026-11-07; status: read-only plan and identity preparation. No cleanup,
preview, model deployment, benchmark generation, or global served-record change
has been executed by this plan. The official F02 dev10 completed 778/778 with
code 87.777778, tied with AV; the frozen decision rule did not advance F02.
AV remains the parent. This dev30 measures broader coverage before further
candidate design; it is not the final DeepSWE evaluation.

## Scope and evidence

Use the existing single-service helper
[redeploy-replica.py](scripts/redeploy-replica.py), SHA
`2e9da820fe30ba9b2436fb600cf14a533b1c496b2fed48ac3649fffb9d2bc9ae`.
No new orchestration tool is needed. The helper imports the actual owner
`/home/y-takagi/kairyu/examples/gemma-4-31b-8gpu/control.py`, SHA
`fc9a78bc232f9cc331926a0737e64f4b4fe14946574f4a02bc1aacb9a800b851`.
It pins the current image, renders normal configuration, preserves CPU/GPU,
command, environment, healthcheck and noncheckpoint mounts, and runs only the
selected chat service with `--no-deps --force-recreate --detach --wait`.
It forces `COMPOSE_REMOVE_ORPHANS=0`. Full-example `run.sh up` and the
historical stateful experiment `R/run-wsp.sh` are not part of this plan.

```bash
R=/mnt/nvme/kairyu/wsp/gemma4-coding-20261104
B=$R/worktrees/kairyu-bench-weight-tools
D=$R/analysis/fresh-purpose-20261107-av-dev30-deployment-plan-v1
AV=$R/checkpoints/code90-current-ad-helper-av-v1
CPU_PY=$R/infra/weight-tools-ci/bin/python
OWNER=/home/y-takagi/kairyu/examples/gemma-4-31b-8gpu
```

| Saved artifact under D | SHA256 |
|---|---|
| initial-readonly-state.json | `b3bc6bcd6090bbfa7d21db990492c80c22a7a162d0ab5c9ba2dbf61cb233c1d7` |
| av-freeze.json | `0fbf034b37613d20fde3f8a536852246a4207be8c65be0aa46a9f2a1d920f176` |
| switch-plan.json | `e28b20876d90fb8cca386d49a9d1c9602d582898678d5996d15cd058fe5c47b3` |

AV's bound tree is
`5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf`.
The freeze reuses all nine file hashes from the fresh AV live10
`checkpoint-identities.json/current` and compares every saved size/device/
inode/mtime/ctime to a fresh helper inventory. The full nine-field inventory
also equals the source inventory in the prior F02 deployment freeze
`163202cf38fe36972f90bd22058850eb412372b603c12731e3a9859abe5e4a38`.
It records current mode/uid/gid/nlink, sorted file path/size/hash rows and the
canonical tree. This is saved full SHA plus exact stat continuity, not a new
payload hash. The independent low-priority AV/F02 full weight diff continues
unchanged and does not block this serving plan.

The initial snapshot was taken while the observation owner's final CPU audit
was pending. It is a plan-time snapshot, not a substitute for the post-handoff
execution baseline. Raw environment values were retained only in memory;
snapshot fields store hashes plus safe identity/state/mount/Compose metadata.

| Target | Ordinary exact CID | Current checkpoint |
|---|---|---|
| gemma-0 | `e275b7bcc331d6bcdd82a1729c0ec0cc7a80e581a008bd273d0b1aa953e82c02` | F02 |
| gemma-1 | `32517829133b9a4a852eed753ae50b274343733307d65b2c01fa10172d513892` | AD |
| gemma-2 | `132833f603a6cbd1b6beed302590cbafd6d9e09bad879720f4c35f39cb806210` | original |
| gemma-3 | `64f27c717165a4e4ea37abec691b5871f87fdfc8ddcd8dd2ecddfd832e5df550` | AR |

All four ordinary targets were running and healthy. The initial host inventory
contains 28 objects: four targets, one stopped owned observer, and 23 protected
non-target containers. Every protected exact CID and safe snapshot is preserved
in the JSON; no name wildcard defines protection.

The stopped observer exact CID is
`584418d20c2701e2339a6ce964332e905819c4c9454f0417d489452d9e988ee9`.
Its name ends with `observer-retired-7ab8d99d7d78`; it has exit code 0 and PID 0.
It retains the same Compose project/service labels as ordinary gemma-0.
Consequently gemma-0 has two matching Compose CIDs including stopped objects,
whereas gemma-1/2/3 each have exactly one. The repaired helper must refuse
gemma-0 preview/apply in this state. No guard is weakened and no broad cleanup
is authorized.

## Observer handoff and exact-object cleanup plan

The observation owner restored exact ordinary F02 e275... healthy and finished
its execution owner. Both writer acknowledgement flags remain closed=false,
with queue/drop/error zero. Do not replace false with true or require a
post-stop acknowledgement that cannot be produced. The official 778/778 WSP
result is separate from this capture limitation.

Before cleanup, the owner must finish its one CPU audit of gzip EOF, record
counts, request/call/rank/acceptance coverage, preserve all available evidence
and explicitly document missing closure acknowledgements. Require inactive
writers, completed ownership release, and the real global evaluation lock
handoff. The planning agent does not release another owner's lock.

The owner reports exact retired-object attribution from its state/attach/
restore receipts and saved logs under the F02 live10 directory:
`retired-observer-logs` (directory 0700, files 0600), stdout SHA
`68cb8b7be02800b9885f7d1f3c51d0a0dd0eeda46e3142a2680cf4a8459f2d7c`,
stderr SHA
`ecf325945bf606da2c40bc928b2a04bb7f27fa5150eebb78adab66666a38ee20`.
Do not reproduce log contents in this document. Recheck the escrow receipt,
file hashes and final audit before the root authorizes disposal.

After explicit authorization, under the exclusive global flock:

1. Re-inspect all 28 exact CIDs; confirm the four ordinary targets and 23
   protected objects are unchanged, and the only duplicate is the attributed
   retired 584418... object. Confirm stopped/PID0, no active writer process,
   ordinary e275... healthy/idle, all owner receipts and host trace/result binds
   retained, and no meaningful container-only artifact remains unpreserved.
2. Save the retired object's safe config/mount/label/state hashes and log escrow
   references. Verify the exact ID belongs to this completed F02 observation;
   a name or checkpoint-path match alone is insufficient.
3. Remove only that exact stopped CID with non-force `docker rm EXACT_CID`.
   No `-v`, force removal, prune, Compose down, remove-orphans, wildcard,
   checkpoint deletion, or host bind-data deletion is included.
4. Save removal receipt and fresh 27-object snapshot. Require every surviving
   object to retain its identity and state, and each of the four Compose
   project/service scopes to contain exactly its ordinary target CID.
   Any mismatch stops before a service recreation.

Cleanup is still unexecuted. This is a concrete bounded proposal, not permission
to perform it before the owner finishes and the root approves.

## Four preview and sequential apply plan

With a fresh post-handoff protected snapshot and the global flock held, recheck
all source pins, AV nine-file stat inventory, actual runtime image and ordinary
flags, exact Compose memberships including stopped objects, idle counters,
active benchmark ownership, and NVMe headroom. No gateway benchmark may run
during the mixed intermediate states.

Create the helper's new shared `all-chat-before.json` baseline once and preview
all four unchanged target services against the same AV freeze. Each output
directory must be fresh. The direct invocation shape is:

```bash
cd "$B"
PYTHONPATH="$B/src" "$CPU_PY" \
  "$B/docs/experiments/scripts/redeploy-replica.py" \
  --control "$OWNER/control.py" --service gemma-0 \
  --candidate "$AV" --freeze "$D/av-freeze.json" \
  --baseline "$D/all-chat-before.json" \
  --output "$D/gemma-0-preview-v1" --timeout 1200
```

Repeat this existing command for gemma-1, gemma-2 and gemma-3 with distinct
preview directories. Save the baseline/receipt/source hashes and protected
snapshot equality. Preview reads metadata and creates evidence; it does not
deploy a container. Current duplicate ownership prevents a truthful four-way
preview until the exact-object cleanup is approved and complete.

After the root approves the concrete four-preview result, apply in order
0,1,2,3, using the same parameters, a fresh per-service apply directory and
`--apply`. The helper checks only its current target against the original
shared baseline, so earlier successful target replacements do not invalidate
the untouched targets' baseline. Keep an updated expected-CID map for the
already-replaced replicas, and compare the remaining targets plus all 23
unrelated protected objects after every step.

Each completed step must be healthy, match the frozen AV tree and read-only
mount, preserve image/command/environment/CPU/GPU/IPC/healthcheck and all other
mounts, and leave all other services unchanged. If a step fails, retain its
failure and target-only rollback receipts and stop the sequence. Earlier
successful replacements remain changed; do not describe a target-local rollback
as an atomic four-service rollback.

Only after all four are homogeneous may direct models/health and narrowly
scoped synthetic connectivity checks plus existing gateway routing be
verified. Bind fresh actual addresses, exact CIDs, checkpoint tree, rendered
flags and protected-state equality. The helper does not update the historical
global checkpoint record; do not use that stale record as serving evidence.
Any separate record update requires its own explicit scope after all-four
closure and is not performed by this plan.

## Planned canonical dev30 conditions

Freeze before any evaluation generation:

- Existing repository wrapper with immutable local runner
  `sha256:f91bdcf997af6d7b194bec229812fd018277158f6795de7d91e58448936e7ad8`.
- Tier 30, dev, all 2,338 manifest items including 96 diagnostic items; no
  problem/source/area reduction or custom --only.
- Existing gateway chat pool after all four replicas serve AV; repeat the
  supported --logprob-endpoint argument for each of the four direct gemma-N
  endpoints. No new router.
- Explicit chat concurrency 256, logprob concurrency 32, agent concurrency 4;
  ordinary native thinking, TP2 × four replicas, MTP4, and fixed runtime flags.
- Client HTTP trace capture only through the standard observe configuration;
  no server CaptureWorker/Middleware deployment and no activation claim.
  Retain request/attempt/retry evidence and the distinction from the prior
  live-activation dev10 capture.
- Full condition/source/dataset/runner/cache/identity receipt and unique result
  run ID before generation. Future candidate dev30 must match these conditions.

The runtime/API command is bound by the benchmark owner after homogeneous
closure. No legacy experiment wrapper, proxy optimization, new observation
lifecycle, confirmation pair, or final DeepSWE run is introduced here.

## Preparation method

Two one-off CPU metadata operations created this plan. The first ran
`docker ps -aq --no-trunc` and one `docker inspect` for those exact IDs, stored
only safe fields and canonical SHA256 of full config/environment/host config,
enumerated memberships by exact project/service labels including stopped
objects, and hashed the existing helper/wrapper/owner source files. The second
imported the existing helper's inventory/digest functions, checked the fresh
AV capture identity against all current stats and the prior full-stat snapshot,
and wrote the existing freeze schema plus the planned service/protected IDs.
Neither operation called Compose up, stopped or deleted an object, ran a model
request, or opened a GPU context. The helper's existing API, rather than a new
deployment implementation, performs every planned runtime action.


## Executed ownership, evidence preservation, and four previews

The root authorized exclusive ownership and removal only of the completed
retired observer after the final handoff. Keeper PID 319432, UID 1003, holds the
real candidate-evaluation.lock via flock --exclusive --nonblock --no-fork
(session 47281); /proc/locks was checked before and after operations. The keeper
waits on its own stdin and remains held for the root's apply decision.

The first cleanup precheck stopped before any mutation because Docker inspect
returned equivalent Mounts arrays in a different order. All 28 CIDs and every
other saved field matched. The failure is retained in
cleanup-precheck-failure-v1.json. The v2 source requires every prior difference
to be only an equal mount set, sorts both sides by target (as the existing
helper does), and repeats the complete checks; it does not discard a mount or
weaken an ownership condition.

The v2 checks passed. Fresh Docker log hashes matched the owner's preserved
stdout/stderr exactly; safe retired metadata, filesystem-diff metadata and
host bind-path existence were saved. All four ordinary targets were healthy
and idle, all 23 unrelated objects matched, and all AV stats remained fixed.
The only removal was non-force docker rm of exact stopped PID0/exit0 observer
584418d20c2701e2339a6ce964332e905819c4c9454f0417d489452d9e988ee9.
There was no -v, force, orphan cleanup, checkpoint change or host bind deletion.
The operation took 7.648049 seconds. All 27 survivors were unchanged and all
four Compose service scopes became exactly one CID.

The observer's two closed=false acknowledgements remain false. Gzip EOF,
record/status counts and all-item/call/rank evidence are independently
preserved, with the capture still partial. No status was edited by cleanup.
The temporary container layer contained runtime driver/cache/lock/temp paths
and bind mount points, not the canonical host capture/result/checkpoint files.
Its path inventory remains in retired-filesystem-diff.json; the container
object and its disposable layer no longer exist.

Cleanup completion SHA:
2e20687df668012383253872716df35f006bb505b6e811908fc9001fa8535c7a.
The existing helper then passed all four previews in 2.934715 seconds. Each
receipt requires exact single-CID Compose ownership, orphan removal disabled,
the same AV freeze, existing image/flags/resources and only the intended
checkpoint change. All 27 containers and AV nine-file stats remained unchanged.
No deployment or benchmark generation occurred.

Shared all-chat-before.json SHA:
615955c06b056fa07a6de8fb9f66fada9146be8ce8b2ff3de79fe1313e2c5f54.
Four-preview summary SHA:
ef6982572eeefd138f62a72f3338ce60c6e7227085647b9cec1e84f80911d7e1.

| Service | Preview receipt SHA256 |
|---|---|
| gemma-0 | 2533668e91a232383b8b5eb70d20f3122f8fafadd14c27a35e5c492d87de7d22 |
| gemma-1 | 97017bf0d1cd3bdb286ccafbc7f97026579b94ce660e6a1d21c246b8015a5266 |
| gemma-2 | 9922ce79d50eafc90b904da832ccf2e301836442d53378767409694fc54d4739 |
| gemma-3 | f7b17a8b03ad9d32599cd1080599ca4dad044201ef0ff80870a679cd209183fb |


### Exact one-off source retained for reproducibility

The completed cleanup v2 source below invokes existing metadata/helper APIs.
It is a record of this exact authorized operation, not a generic cleanup tool.
Its source SHA is
99a9cc25006a988d1d7cfe11b8f579ea0bbb857c4de34c39f21bf0769b5a12ce.
Every host command ran from the NVMe worktree with PYTHONDONTWRITEBYTECODE=1
and PYTHONPATH pointing to that worktree's src.


~~~python
"""Preserve completed F02 observer evidence and remove only its authorized stopped CID."""
from pathlib import Path
import hashlib, importlib.util, json, os, subprocess, time

R=Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
B=R/"worktrees/kairyu-bench-weight-tools"
D=R/"analysis/fresh-purpose-20261107-av-dev30-deployment-plan-v1"
N2=R/"analysis/fresh-purpose-20261107-f02-live10-v1"
CID="584418d20c2701e2339a6ce964332e905819c4c9454f0417d489452d9e988ee9"
H=B/"docs/experiments/scripts/redeploy-replica.py"
spec=importlib.util.spec_from_file_location("guard",H)
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
def save(name,value):
    p=D/name
    assert not p.exists()
    g.save(p,value)
def lock_evidence():
    owner=g.read(D/"global-keeper.json")
    s=(R/"candidate-evaluation.lock").stat()
    key=f"{os.major(s.st_dev):02x}:{os.minor(s.st_dev):02x}:{s.st_ino}"
    rows=[x for x in Path("/proc/locks").read_text().splitlines() if key in x.split()]
    assert len(rows)==1 and rows[0].split()[4]==str(owner["pid"])
    assert "FLOCK  ADVISORY  WRITE" in rows[0] and owner["uid"]==os.getuid()
    return dict(owner=owner,lock_identity=key,proc_locks_line=rows[0])
def safe(c):
    cfg=c["Config"]; lab=cfg.get("Labels")or{}
    env=dict(x.split("=",1)for x in cfg.get("Env")or[]if"="in x)
    return dict(id=c["Id"],name=c["Name"],created=c["Created"],image_id=c["Image"],
      command_sha256=g.digest(dict(entrypoint=cfg.get("Entrypoint"),cmd=cfg.get("Cmd"))),
      environment_sha256=g.digest(env),config_sha256=g.digest(cfg),hostconfig_sha256=g.digest(c["HostConfig"]),
      state={k:c["State"].get(k)for k in ["Status","Running","Paused","Restarting","Dead","Pid","ExitCode","StartedAt","FinishedAt"]},
      health=c["State"].get("Health",{}).get("Status"),
      labels={k:v for k,v in lab.items()if k.startswith("com.docker.compose.")or k.startswith("kairyu.")or k.startswith("kairyu_bench.")},
      mounts=sorted([dict(type=m["Type"],source=m["Source"],target=m["Destination"],read_only=not m["RW"])for m in c["Mounts"]],key=lambda m:m["target"]),
      network={k:dict(ip=v.get("IPAddress"),aliases=v.get("Aliases"))for k,v in c["NetworkSettings"]["Networks"].items()})
def inventory():
    ids=g.command(["docker","ps","-aq","--no-trunc"]).splitlines()
    values=json.loads(g.command(["docker","inspect",*ids]))
    return {c["Id"]:safe(c)for c in values}, {c["Id"]:c for c in values}
started=time.time()
assert os.getuid()!=0
assert g.sha(D/"av-freeze.json")=="0fbf034b37613d20fde3f8a536852246a4207be8c65be0aa46a9f2a1d920f176"
assert g.sha(D/"switch-plan.json")=="e28b20876d90fb8cca386d49a9d1c9602d582898678d5996d15cd058fe5c47b3"
initial=g.read(D/"initial-readonly-state.json");plan=g.read(D/"switch-plan.json")
# Docker inspect Mounts ordering is not stable; preserve every mount and sort by target.
failure=g.read(D/"cleanup-precheck-failure-v1.json")
assert failure["added"]==failure["removed"]==[]
assert all(set(v)=={"mounts"} and sorted(v["mounts"]["before"],key=lambda m:m["target"])==sorted(v["mounts"]["after"],key=lambda m:m["target"])for v in failure["differences"].values())
for record in initial["containers"].values():record["mounts"]=sorted(record["mounts"],key=lambda m:m["target"])
assert all(g.sha(p)==v for p,v in initial["source_pins"].items())
lock=lock_evidence()
assert not Path("/proc/211879").exists()
assert g.read(N2/"execution.json")["status"]=="completed"
before,raw=inventory()
assert len(before)==28 and before==initial["containers"]
assert plan["duplicate_retired_ids"]==[CID]
assert len(plan["protected_container_ids"])==23
c=raw[CID]
assert c["State"]["Status"]=="exited" and not c["State"]["Running"] and c["State"]["Pid"]==0
assert c["State"]["ExitCode"]==0 and not c["State"]["OOMKilled"]
assert before[CID]["labels"]["com.docker.compose.project"]=="gemma-4-31b-8gpu"
assert before[CID]["labels"]["com.docker.compose.service"]=="gemma-0"
escrow=N2/"retired-observer-logs"
assert g.sha(escrow/"receipt.json")=="c24190ff4bfb16bb0c3ef0547c265d28fa4927e09c3a9af775a59a8118453f00"
receipt=g.read(escrow/"receipt.json")
assert receipt["observer_cid"]==CID and receipt["closed_worker_count"]==0
logs=subprocess.run(["docker","logs","--timestamps",CID],capture_output=True,check=True)
for name,data in [("stdout.log",logs.stdout),("stderr.log",logs.stderr)]:
    row=receipt["files"][name]
    assert row["redactions"]==0
    assert g.sha(escrow/name)==row["sha256"]==hashlib.sha256(data).hexdigest()
    assert len(data)==row["input_bytes"]==row["saved_bytes"]
diff=g.command(["docker","diff",CID])
save("retired-filesystem-diff.json",dict(command=["docker","diff",CID],lines=diff.splitlines(),
 interpretation="Runtime driver/cache/lock/temp paths and host bind mount points; capture, acceptance, status, source and compile-cache paths are host binds; no checkpoint or trial payload deletion authorized."))
for m in c["Mounts"]:
    if m["Type"]=="bind":
        assert Path(m["Source"]).exists()
for target in plan["target_ids"].values():
    g.idle(raw[target])
freeze=g.read(D/"av-freeze.json")
assert g.inventory(Path(freeze["checkpoint_path"]))==freeze["candidate_inventory_stat"]
save("retired-safe-metadata.json",before[CID])
save("cleanup-before.json",before)
command=["docker","rm",CID]
save("cleanup-prechecks.json",dict(status="passed",command=command,authorized_exact_cid=CID,
    nonforce=True,remove_volumes=False,orphan_removal=False,
    lock=lock,source_sha256=g.sha(Path(__file__)),retired_metadata_sha256=g.sha(D/"retired-safe-metadata.json"),
    logs_receipt_sha256=g.sha(escrow/"receipt.json"),log_hashes={n:v["sha256"]for n,v in receipt["files"].items()},
    protected_count=23,protected_unchanged=True,other_three_chat_unchanged=True,
    ordinary_gemma0_exact_healthy=True,writer_close_acknowledgements=0,writer_count=2,
    capture_limitations_preserved=True,full_payload_rehashed=False,started_epoch=started))
assert inventory()[0]==before
lock_evidence()
output=g.command(command).strip()
assert output==CID
after,_=inventory()
assert set(after)==set(before)-{CID}
assert all(after[k]==v for k,v in before.items()if k!=CID)
spec2=importlib.util.spec_from_file_location("control","/home/y-takagi/kairyu/examples/gemma-4-31b-8gpu/control.py")
control=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(control)
members={s:g.compose_scope(control,s,cid)for s,cid in plan["target_ids"].items()}
assert g.inventory(Path(freeze["checkpoint_path"]))==freeze["candidate_inventory_stat"]
save("cleanup-after.json",after)
save("cleanup-completion.json",dict(status="only_authorized_stopped_observer_removed",removed_cid=CID,
 command=command,remaining_containers=len(after),protected_count=23,all_survivors_unchanged=True,
 compose_membership=members,writer_closed_flags_unchanged=True,
 clarification="Original false close acknowledgements retained in original capture; no capture status edited.",
 lock=lock_evidence(),seconds=time.time()-started,finished_epoch=time.time(),
 source_sha256=g.sha(Path(__file__)),before_sha256=g.sha(D/"cleanup-before.json"),after_sha256=g.sha(D/"cleanup-after.json")))
print(json.dumps(dict(status="cleanup_completed",removed=CID,containers=len(after),protected_unchanged=23,seconds=time.time()-started,receipt_sha256=g.sha(D/"cleanup-completion.json"))))
~~~

The failed v1 source SHA is
73c01304a48a30743307edb632798dbc4cffbfcf73d923efe4d2dbc38a095545.
The exact v1-to-v2 patch below retains its reconstruction, including the
corrected descriptive closure field that v1 never reached:

~~~diff
--- remove-completed-observer.py
+++ remove-completed-observer-v2.py
@@ -31,7 +31,7 @@
       state={k:c["State"].get(k)for k in ["Status","Running","Paused","Restarting","Dead","Pid","ExitCode","StartedAt","FinishedAt"]},
       health=c["State"].get("Health",{}).get("Status"),
       labels={k:v for k,v in lab.items()if k.startswith("com.docker.compose.")or k.startswith("kairyu.")or k.startswith("kairyu_bench.")},
-      mounts=[dict(type=m["Type"],source=m["Source"],target=m["Destination"],read_only=not m["RW"])for m in c["Mounts"]],
+      mounts=sorted([dict(type=m["Type"],source=m["Source"],target=m["Destination"],read_only=not m["RW"])for m in c["Mounts"]],key=lambda m:m["target"]),
       network={k:dict(ip=v.get("IPAddress"),aliases=v.get("Aliases"))for k,v in c["NetworkSettings"]["Networks"].items()})
 def inventory():
     ids=g.command(["docker","ps","-aq","--no-trunc"]).splitlines()
@@ -42,6 +42,11 @@
 assert g.sha(D/"av-freeze.json")=="0fbf034b37613d20fde3f8a536852246a4207be8c65be0aa46a9f2a1d920f176"
 assert g.sha(D/"switch-plan.json")=="e28b20876d90fb8cca386d49a9d1c9602d582898678d5996d15cd058fe5c47b3"
 initial=g.read(D/"initial-readonly-state.json");plan=g.read(D/"switch-plan.json")
+# Docker inspect Mounts ordering is not stable; preserve every mount and sort by target.
+failure=g.read(D/"cleanup-precheck-failure-v1.json")
+assert failure["added"]==failure["removed"]==[]
+assert all(set(v)=={"mounts"} and sorted(v["mounts"]["before"],key=lambda m:m["target"])==sorted(v["mounts"]["after"],key=lambda m:m["target"])for v in failure["differences"].values())
+for record in initial["containers"].values():record["mounts"]=sorted(record["mounts"],key=lambda m:m["target"])
 assert all(g.sha(p)==v for p,v in initial["source_pins"].items())
 lock=lock_evidence()
 assert not Path("/proc/211879").exists()
@@ -99,7 +104,7 @@
 save("cleanup-after.json",after)
 save("cleanup-completion.json",dict(status="only_authorized_stopped_observer_removed",removed_cid=CID,
  command=command,remaining_containers=len(after),protected_count=23,all_survivors_unchanged=True,
- compose_membership=members,writer_closed_flags_unchanged=False,
+ compose_membership=members,writer_closed_flags_unchanged=True,
  clarification="Original false close acknowledgements retained in original capture; no capture status edited.",
  lock=lock_evidence(),seconds=time.time()-started,finished_epoch=time.time(),
  source_sha256=g.sha(Path(__file__)),before_sha256=g.sha(D/"cleanup-before.json"),after_sha256=g.sha(D/"cleanup-after.json")))
~~~

The four-preview one-off source SHA is
2c3be39f675b3ec3d2cf4c3913ba7a03aeefb5814b7fa0ae3fe9e61fff488947.
Only the pinned metadata function definitions are loaded from the prior
operation; its executable cleanup body is never rerun:

~~~python
"""Preview the existing AV checkpoint on all four ordinary services after exact retired cleanup."""
from pathlib import Path
import importlib.util,json,os,subprocess,time
R=Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
B=R/"worktrees/kairyu-bench-weight-tools"
D=R/"analysis/fresh-purpose-20261107-av-dev30-deployment-plan-v1"
H=B/"docs/experiments/scripts/redeploy-replica.py"
spec=importlib.util.spec_from_file_location("guard",H)
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
# Reuse only the already pinned one-off's metadata functions, not its executable body.
source=D/"remove-completed-observer-v2.py"
assert g.sha(source)=="99a9cc25006a988d1d7cfe11b8f579ea0bbb857c4de34c39f21bf0769b5a12ce"
import ast
tree=ast.parse(source.read_text())
functions=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name in {"safe","inventory","lock_evidence"}],type_ignores=[])
exec(compile(functions,str(source),"exec"),globals())
def save(name,value):
    p=D/name;assert not p.exists();g.save(p,value)
started=time.time()
pins={"av-freeze.json":"0fbf034b37613d20fde3f8a536852246a4207be8c65be0aa46a9f2a1d920f176",
 "switch-plan.json":"e28b20876d90fb8cca386d49a9d1c9602d582898678d5996d15cd058fe5c47b3",
 "cleanup-completion.json":"2e20687df668012383253872716df35f006bb505b6e811908fc9001fa8535c7a"}
assert all(g.sha(D/name)==v for name,v in pins.items())
plan=g.read(D/"switch-plan.json");freeze=g.read(D/"av-freeze.json")
sourcepins=g.read(D/"initial-readonly-state.json")["source_pins"]
assert all(g.sha(p)==v for p,v in sourcepins.items())
lock=lock_evidence()
before,raw=inventory()
assert before==g.read(D/"cleanup-after.json")and len(before)==27
assert g.inventory(Path(freeze["checkpoint_path"]))==freeze["candidate_inventory_stat"]
save("preview-protected-before.json",before)
spec2=importlib.util.spec_from_file_location("control","/home/y-takagi/kairyu/examples/gemma-4-31b-8gpu/control.py")
control=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(control)
selections={s:g.compose_scope(control,s,cid)for s,cid in plan["target_ids"].items()}
for cid in plan["target_ids"].values():g.idle(raw[cid])
save("four-preview-prechecks.json",dict(status="passed",source_pins=sourcepins,pinned_inputs=pins,compose_membership=selections,
 owner=lock,protected_count=23,all_27_unchanged=True,source_sha256=g.sha(Path(__file__)),started_epoch=started))
receipts={}
for service in plan["service_order"]:
    output=D/(service+"-preview-v1")
    argv=list(plan["planned_arguments"])
    argv[argv.index("--service")+1]=service
    argv[argv.index("--output")+1]=str(output)
    env=dict(os.environ,PYTHONPATH=str(B/"src"),PYTHONDONTWRITEBYTECODE="1")
    save(service+"-preview-command-v1.json",dict(argv=argv,environment_overrides={"PYTHONPATH":str(B/"src"),"PYTHONDONTWRITEBYTECODE":"1"},apply=False))
    with (D/(service+"-preview-v1.stdout")).open("x")as out,(D/(service+"-preview-v1.stderr")).open("x")as err:
        t=time.time();result=subprocess.run(argv,cwd=B,env=env,stdout=out,stderr=err,timeout=90)
    save(service+"-preview-execution-v1.json",dict(returncode=result.returncode,seconds=time.time()-t))
    assert result.returncode==0
    receipt=g.read(output/"receipt.json")
    assert receipt["status"]=="preview"and receipt["apply_requested"]is False
    assert receipt["before"]["id"]==plan["target_ids"][service]
    assert receipt["candidate_identity"]["tree_sha256"]==freeze["control_tree_sha256"]
    assert receipt["compose_selection"]==selections[service]and receipt["compose_remove_orphans"]is False
    assert receipt["helper_sha256"]==sourcepins[str(H)]
    receipts[service]=dict(path=str(output/"receipt.json"),sha256=g.sha(output/"receipt.json"),
        target_id=receipt["before"]["id"],rendered_service_sha256=receipt["rendered_service_sha256"])
assert inventory()[0]==before
assert g.inventory(Path(freeze["checkpoint_path"]))==freeze["candidate_inventory_stat"]
assert all(g.sha(p)==v for p,v in sourcepins.items())
save("four-preview-summary.json",dict(status="all_four_previews_passed_no_service_deployment",receipts=receipts,
 baseline_path=str(D/"all-chat-before.json"),baseline_sha256=g.sha(D/"all-chat-before.json"),
 freeze_sha256=g.sha(D/"av-freeze.json"),source_pins=sourcepins,protected_count=23,
 all_27_containers_unchanged=True,checkpoint_stats_unchanged=True,owner=lock_evidence(),
 seconds=time.time()-started,source_sha256=g.sha(Path(__file__)),apply_authorized=False))
print(json.dumps(dict(status="all_four_previews_passed",baseline_sha256=g.sha(D/"all-chat-before.json"),
 summary_sha256=g.sha(D/"four-preview-summary.json"),receipts=receipts,seconds=time.time()-started)))
~~~


## Apply blocked by automatic approval review; no deployment

The root reviewed the four preview receipts and authorized a sequential apply.
The execution tool's automatic approval review then rejected the proposed SSH
command before process creation. Its reason was that recreating all four live
model containers has broad service-disruption and runtime-state risks, and it
could not find direct end-user authorization for this exact multi-service
apply. The rejection explicitly forbids bypassing it through a workaround or
indirect execution. No alternative invocation of the same operation is made.

The proposed SSH command did not run: its remote apply source was not created,
no replica was replaced, no WSP request was generated, and the four ordinary
replicas remain at their prior mixed checkpoints. The already completed exact
retired cleanup and four previews remain valid historical operations. The
keeper remains held pending the parent owner's direction; CPU weight diff
continues independently. A decision must resolve explicit user approval or a
materially narrower alternative scientific plan before any dependent action.

The following source is an unexecuted, reviewable proposal only. It uses the
existing helper for each replica and retains all failures. The keeper has no
automatic timed release: a ten-minute progress review deadline was proposed,
with per-helper timeout 1,200 seconds and an outer bound of 2,460 seconds to
include a target rollback. The successful termination condition would be
all-four closure followed by a root-approved next-owner lock handoff. None of
these proposed apply outputs exists.

~~~python
"""Apply the explicitly approved AV freeze to ordinary gemma-0,1,2,3 sequentially."""
from pathlib import Path
import ast,importlib.util,json,os,subprocess,time,urllib.request
R=Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
B=R/"worktrees/kairyu-bench-weight-tools"
D=R/"analysis/fresh-purpose-20261107-av-dev30-deployment-plan-v1"
H=B/"docs/experiments/scripts/redeploy-replica.py"
spec=importlib.util.spec_from_file_location("guard",H)
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
source=D/"remove-completed-observer-v2.py"
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
pins={"av-freeze.json":"0fbf034b37613d20fde3f8a536852246a4207be8c65be0aa46a9f2a1d920f176",
 "all-chat-before.json":"615955c06b056fa07a6de8fb9f66fada9146be8ce8b2ff3de79fe1313e2c5f54",
 "four-preview-summary.json":"ef6982572eeefd138f62a72f3338ce60c6e7227085647b9cec1e84f80911d7e1"}
assert all(g.sha(D/name)==v for name,v in pins.items())
plan=g.read(D/"switch-plan.json");freeze=g.read(D/"av-freeze.json");summary=g.read(D/"four-preview-summary.json")
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
save("four-apply-authorization.json",dict(status="root_authorized_four_sequential_applies",pinned_inputs=pins,
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
save("four-apply-closure.json",dict(status="all_four_ordinary_AV_healthy_verified",replicas=done,
 tree=freeze["control_tree_sha256"],freeze_sha256=g.sha(D/"av-freeze.json"),source_pins=sourcepins,
 protected_count=23,protected_unchanged=True,container_count=27,compose_membership=members,
 owner=lock_evidence(),seconds=time.time()-started,finished_epoch=time.time(),model_generation_requests=0,
 global_checkpoint_record_modified=False,full_payload_rehashed=False,source_sha256=g.sha(Path(__file__))))
print(json.dumps(dict(status="all_four_AV_healthy",seconds=time.time()-started,
 closure_sha256=g.sha(D/"four-apply-closure.json"),replicas=done)),flush=True)
~~~


## Explicit end-user approval and successful sequential application

After the recorded automatic-review rejection, the user directly answered the
root's explicit question about recreating all four inference containers and
preserving the other 23 services:

> 4推論コンテナのAVへの切替を承認する

The exact question, response, scope and frozen identifiers are retained in
explicit-user-approval.json, SHA
fa823510d7c17e1272c3d5d946875ec4a9e0daaab389943a507361309854012f.
This new authorization was supplied to the automatic reviewer, which permitted
the same bounded operation. There was no workaround or scope expansion. The
earlier blocked state is historical and remains documented.

The same proposed source above was then saved as apply-all-four.py and executed;
its actual source SHA is
1bb6b6d4ee5b2b24ebb677295bdfe889587e534748d46c5afb8c7fa2f8f5d1b7.
Fresh source/input hashes, all 27 ordinary/protected objects, exact one-CID
Compose membership and every target's idle state were checked before starting.
The source records a ten-minute progress-review deadline and holds the keeper
until an explicitly coordinated handoff; it does not silently expire the global
lock during a service change. Each helper has a 1,200-second bound and the
outer operation permits at most 2,460 seconds including a target rollback.

All four applies and closures passed in 455.464788 seconds. Each step preserved
every other container, including already changed and not-yet-changed chat
replicas. No apply failed, no rollback was needed, and no benchmark generation
was issued. All four share AV tree
5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf,
the same read-only AV checkpoint mount, the preserved runtime image/commands/
environment/resources, TP2, MTP4 and native Gemma4 settings. All 23 unrelated
containers remain unchanged, all four Compose scopes are exact singletons,
and the host container count remains 27.

| Replica | New exact CID | Direct endpoint | Seconds |
|---|---|---|---:|
| gemma-0 | baf23e529c4e217801633d0c62b04801e3991b24f6267274104f9d3e355e62c3 | http://172.26.0.4:8000/v1 | 112.118169 |
| gemma-1 | ce1a67f4ec204a90e486092534e3f700f3abf2fbc9f1e68f87668009104ee00d | http://172.26.0.5:8000/v1 | 112.713972 |
| gemma-2 | fd4ec2eb9d7deba2ce3e790add6b50c0ee79a6390495320d8697873b32957bcf | http://172.26.0.6:8000/v1 | 117.019532 |
| gemma-3 | e8ec9f3bbfca4baa44b16590ee771a96bba1e1328a60ff0c904995f2675a31b5 | http://172.26.0.8:8000/v1 | 111.982780 |


All four direct /v1/models responses contain gemma-4-31b. The existing gateway
retains its original exact CID and remains healthy at
http://172.26.0.9:8000/v1; its read-only /models response contains gemma-4-31b
and embeddinggemma-2. These are identity/connectivity checks, not model
generation or a new capability score. No global checkpoint record or UI
configuration was changed.

four-apply-closure.json SHA:
671fc7f0b4f7853631e7dff05e5cb9d76c4d9b95b66927d32fa0393ed2ed82ee.
endpoint-closure.json SHA:
32d3850eba9b4c0d10c19d7229e526f68929620d946f2f88cf8aacbfe90c4e65.

The keeper is still held at this closure. Next ownership transfer requires the
benchmark owner's bound plan and readiness plus the root's explicit handoff
instruction, followed by release of this exact keeper and a nonblocking
acquisition/identity/idle recheck by the next owner. No WSP generation has been
started by the deployment operation.

The safe metadata, identity, rejected-precheck, approval, preview and apply
receipts are copied into records/2026-11-07-av-dev30-deployment/. Raw Docker
logs remain in the restricted host escrow, with hashes referenced rather than
their contents copied into the public PR. Full environment values were never
written into these records.
