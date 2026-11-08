"""Bind existing F06 save/review/deployment evidence; preview only, no model calls."""
import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys
import time
import uuid

R = pathlib.Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
B = R / "worktrees/kairyu-bench-weight-tools"
N = R / "analysis/fresh-purpose-20261107-f06-dev30-dp4-v1"
T = R / "analysis/fresh-purpose-20261107-f04-mbpp-lora-training-v1"
D = R / "analysis/fresh-purpose-20261107-f06-lora-deployment-plan-v1"
sys.path.insert(0, str(B / "src"))
from kairyu_bench.wsp.weights.checkpoint import Checkpoint  # noqa: E402
from kairyu_bench.wsp.weights.native_faststore import digest  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--plan-sha256", required=True)
parser.add_argument("--owner-pid", type=int, required=True)
parser.add_argument("--closure-sha256", required=True)
parser.add_argument("--endpoint-source", type=pathlib.Path, required=True)
options = parser.parse_args()
assert options.owner_pid > 1
assert len(options.plan_sha256) == len(options.closure_sha256) == 64


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pin(p, expected=None):
    h = sha(p)
    if expected is not None:
        assert h == expected, (str(p), h)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": h}


def save(p, d):
    with p.open("x") as f:
        json.dump(d, f, indent=2)
        f.write(chr(10))


def st(p):
    s = p.stat()
    return dict(
        device=s.st_dev,
        inode=s.st_ino,
        size=s.st_size,
        mtime_ns=s.st_mtime_ns,
        ctime_ns=s.st_ctime_ns,
        nlink=s.st_nlink,
        uid=s.st_uid,
        gid=s.st_gid,
        mode=s.st_mode,
    )


dpin = pin(D / "plan-root-approved-v1.json", options.plan_sha256)
deployment = json.loads(pathlib.Path(dpin["path"]).read_text())
for field in ["freeze", "candidate_identity", "materialization", "opus_receipt", "root_review_decision"]:
    item = deployment[field]
    pin(pathlib.Path(item["path"]), item["sha256"])
freeze = json.loads(pathlib.Path(deployment["freeze"]["path"]).read_text())
mp = pathlib.Path(freeze["actual_manifest"]["path"])
mpin = pin(mp, "22b0603cc2dc9a2060c76f8baf99c28eceac08b56116727173bc61a719b5c0d4")
m = json.loads(mp.read_text())
cp = pathlib.Path(m["candidate"])
vp = pathlib.Path(freeze["actual_verify"]["path"])
vpin = pin(vp, "42140a38d3f2412887952c0dfe836433f0fabb6a8aa68196978eb8e5e479788d")
v = json.loads(vp.read_text())
assert v["verified"]
assert v["changed_elements"] == m["changed_elements"] == 8193921391
assert v["delta_l2"] == m["delta_l2"] == 3.708003317513851
identitypin = pin(pathlib.Path(deployment["candidate_identity"]["path"]),
                  "50aa8056b9e48257b48c8621396030b4796c2e72d1c94fd8f64ad84a6867bdf7")
specpin = pin(pathlib.Path(m["spec_path"]),
              "44e1928332dc347bfe19048b93ddb837e79f79797d2db807e6230bb52ce78d8d")
assert m["spec_sha256"] == specpin["sha256"]
now = {x["path"]: st(cp / x["path"]) for x in freeze["all_checkpoint_files"]}
assert len(now) == 9 and now == freeze["candidate_inventory_stat"]
rows = sorted(freeze["all_checkpoint_files"], key=lambda x: x["path"])
tree = digest(rows)
assert tree == freeze["control_tree_sha256"] == m["candidate_tree_sha256"] == v["candidate_tree_sha256"]
assert tree == "0fc0ef29346c4d8a06bea8570dee96d8e0ba7038fee0d5aded8db57852474ec5"
identity = Checkpoint(cp).identity(hash_files=False)
rowmap = {x["path"]: x for x in rows}
assert set(rowmap) == {x["path"] for x in identity["files"]}
for item in identity["files"]:
    assert item["size"] == rowmap[item["path"]]["size"]
    item["sha256"] = rowmap[item["path"]]["sha256"]
identity.update(sha256=tree, hash_files=True, identity_level="content_sha256", source_stable=True)
assert now == {x: st(cp / x) for x in now}
review = json.loads(pathlib.Path(deployment["opus_receipt"]["path"]).read_text())
assert review["verified_requested_model"] and review["requested_model"] == "claude-opus-5-5"
assert review["assistant_models"] == ["claude-opus-5-5"]
decision = json.loads(pathlib.Path(deployment["root_review_decision"]["path"]).read_text())
assert decision["status"] == "proceed_to_one_full_dev30_after_verified_artifact_and_serving_identity"
assert decision["opus_receipt_sha256"] == deployment["opus_receipt"]["sha256"]
pin(D / "plan-root-approved-v1.json", options.plan_sha256)
parallel_closure = pin(
    D / "parallel-apply-v1/four-apply-closure.json", options.closure_sha256
)
parallel_release = json.loads((D / "parallel-apply-v1/lock-released.json").read_text())
assert parallel_release == dict(
    normal_release=True, pid=options.owner_pid, all_children_joined=True, success=True
)
assert not pathlib.Path(f"/proc/{options.owner_pid}").exists()
closurepin = pin(D / "four-apply-closure.json")
closure = json.loads((D / "four-apply-closure.json").read_text())
assert closure["parallel_owner_closure_sha256"] == options.closure_sha256
endpoint_source = options.endpoint_source
assert closure["source_sha256"] == sha(endpoint_source)
epin = pin(D / "endpoint-closure.json")
endpoint = json.loads((D / "endpoint-closure.json").read_text())
assert endpoint["status"] == "all_four_F06_and_gateway_verified"
assert endpoint["closure_sha256"] == closurepin["sha256"]
assert endpoint["replicas"] == closure["replicas"] and endpoint["all_four_idle"]
relpin = pin(D / "global-lock-released.json")
rel = json.loads((D / "global-lock-released.json").read_text())
assert rel["status"] == "normal_release"
assert (
    rel["endpoint_closure_sha256"] == epin["sha256"]
    and rel["four_closure_sha256"] == closurepin["sha256"]
)
assert not pathlib.Path(f"/proc/{rel['pid']}").exists()
individuals = [pin(D / f"gemma-{i}-closure-v1.json") for i in range(4)]
for service, c in closure["replicas"].items():
    assert (
        c["tree"] == tree
        and c["checkpoint_mount"] == str(cp)
        and c["checkpoint_readonly"]
        and c["runtime_preserved"]
        and c["tp_size"] == 2
        and c["mtp_tokens"] == 4
    )
    assert json.loads((D / f"{service}-closure-v1.json").read_text()) == c
assert (
    len(closure["replicas"]) == 4
    and closure["protected_unchanged"]
    and closure["protected_count"] == 23
    and closure["container_count"] == 27
)
cfg = json.loads((N / "http-trace-template.json").read_text())
cfg.update(
    capture_id=uuid.uuid4().hex,
    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    checkpoint=identity,
)
(N / "http-trace").mkdir()
save(N / "http-trace/capture.json", cfg)
evidence = {
    "created_epoch": time.time(),
    "manifest": mpin,
    "verify": vpin,
    "checkpoint": str(cp),
    "tree_sha256": tree,
    "current_full_stat": now,
    "all_checkpoint_files": rows,
    "hash_provenance": m["verification"],
    "compact_full_identity": identitypin,
    "freeze": deployment["freeze"],
    "verification_stat_evidence": freeze["stat_evidence"],
    "payload_hash_passes_in_this_binding": 0,
    "headers_and_metadata_only_read": True,
    "same_stat_before_after": True,
    "source": pin(pathlib.Path(__file__)),
    "scope": "Completed full dense recipe verify hashes plus current exact nine-key stat continuity; original internal verification snapshots were compared internally but not separately saved. No new full payload hash.",
}
save(N / "http-trace-identity-check.json", evidence)
p = json.loads((N / "launch-plan-unbound.json").read_text())
p.update(
    status="bound_ready_for_root_launch_authorization",
    approved_for_execution=False,
    closure_bound_epoch=time.time(),
)
p["command"][p["command"].index("--label") + 1] = tree
p["expected_serving"].update(
    checkpoint=str(cp),
    tree_sha256=tree,
    service_cids={k: x["cid"] for k, x in closure["replicas"].items()},
    closures=individuals,
    four_replica_closure=closurepin,
)
p["http_trace"]["config"] = pin(N / "http-trace/capture.json")
p["http_trace"]["identity_check"] = pin(N / "http-trace-identity-check.json")
p["handoff"].update(
    root_authorized=False,
    prior_owner_pid=rel["pid"],
    normal_global_release=relpin,
    endpoint_closure=epin,
    authorization="Root authorized F06 evidence binding and read-only preview only. Existing complete F04 baseline reused without rerun. Root must explicitly approve one full dev30 launch.",
)
p["prior_owner_pid"] = rel["pid"]
p["pending_deployment_bindings"] = ["root launch authorization"]
p["candidate_identity"] = p.pop("candidate_identity_pending")
p["candidate_identity"].update(
    checkpoint_path=str(cp),
    manifest_path=str(mp),
    manifest_sha256=mpin["sha256"],
    verification_receipt_path=str(vp),
    verification_receipt_sha256=vpin["sha256"],
    spec_path=specpin["path"],
    spec_sha256=specpin["sha256"],
    physical_tree_sha256=tree,
    compact_full_identity=identitypin,
    materialization=deployment["materialization"],
    freeze=deployment["freeze"],
    actual_saved_changes={"elements": m["changed_elements"], "tensors": m["changed_tensors"], "delta_l2": m["delta_l2"]},
    hash_contract="Existing completed full dense recipe verify and exact current stat continuity; no additional payload hash.",
    opus_review={"receipt_path": deployment["opus_receipt"]["path"],
                 "receipt_sha256": deployment["opus_receipt"]["sha256"],
                 "actual_model": "claude-opus-5-5", "verified": True,
                 "root_decision": deployment["root_review_decision"]},
)
p["candidate_identity"]["failed_materialization_path_reference"]["status"] = (
    "Historical failed initial create retained separately. Same destination was subsequently completed and fully verified by materialization-f06-v2; current eligibility derives solely from bound successful receipts."
)
p["deployment_plan"] = dpin
p["deployment_parallel_closure"] = parallel_closure
p["binding_source"] = pin(pathlib.Path(__file__))
p["prelaunch_requirements"][0] = (
    "F04 remains selected complete same-cohort parent (coding tie with AV; higher overall). Existing F04 baseline reused, no base rerun."
)
owner = pathlib.Path(p["owner_source"]["path"])
pin(owner, p["owner_source"]["sha256"])
save(N / "launch-plan-bound-v1.json", p)
previewcmd = [
    sys.executable,
    str(owner),
    "--plan",
    str(N / "launch-plan-bound-v1.json"),
]
t = time.time()
result = subprocess.run(previewcmd, text=True, capture_output=True)
save(
    N / "owner-preview-bound-v1.json",
    {
        "command": previewcmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "elapsed_seconds": time.time() - t,
        "plan": pin(N / "launch-plan-bound-v1.json"),
        "owner": pin(owner),
        "model_calls": 0,
    },
)
print(
    json.dumps(
        {
            "plan": pin(N / "launch-plan-bound-v1.json"),
            "preview": pin(N / "owner-preview-bound-v1.json"),
            "preview_returncode": result.returncode,
            "preview_stdout": result.stdout,
            "source": pin(pathlib.Path(__file__)),
        }
    )
)
assert result.returncode == 0
