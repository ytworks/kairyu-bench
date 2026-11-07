"""Complete root-authorized F05 endpoint checks and served record after successful parallel deployment."""

from pathlib import Path
import argparse
import ast
import fcntl
import importlib.util
import json
import os
import shutil
import subprocess  # noqa: F401 - dependency of captured inventory/lock functions
import sys
import time
import urllib.request

R = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
B = R / "worktrees/kairyu-bench-weight-tools"
D = R / "analysis/fresh-purpose-20261107-f05-native-contract-v1/deployment-dp4-v1"
sys.path.insert(0, str(B / "src"))
H = B / "docs/experiments/scripts/redeploy-replica.py"
spec = importlib.util.spec_from_file_location("guard", H)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
source = (
    R
    / "analysis/fresh-purpose-20261107-av-dev30-deployment-plan-v1/remove-completed-observer-v2.py"
)
assert (
    g.sha(source) == "99a9cc25006a988d1d7cfe11b8f579ea0bbb857c4de34c39f21bf0769b5a12ce"
)
tree = ast.parse(source.read_text())
functions = ast.Module(
    body=[
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"safe", "inventory", "lock_evidence"}
    ],
    type_ignores=[],
)
exec(compile(functions, str(source), "exec"), globals())
safe = globals()["safe"]
inventory = globals()["inventory"]


def save(name, value):
    p = D / name
    assert not p.exists()
    g.save(p, value)


def fetch(endpoint, path):
    with urllib.request.urlopen(endpoint + path, timeout=30) as response:
        return json.loads(response.read(1024**2))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--plan-sha256", required=True)
parser.add_argument("--owner-pid", type=int, required=True)
parser.add_argument("--closure-sha256", required=True)
options = parser.parse_args()
assert options.owner_pid > 1
assert len(options.plan_sha256) == len(options.closure_sha256) == 64

started = time.time()
A = D / "parallel-apply-v1"
assert (
    g.sha(D / "plan-root-approved.json")
    == options.plan_sha256
)
assert (
    g.sha(A / "four-apply-closure.json")
    == options.closure_sha256
)
assert g.read(A / "lock-released.json") == dict(
    normal_release=True, pid=options.owner_pid, all_children_joined=True, success=True
)
assert not Path(f"/proc/{options.owner_pid}").exists()
plan = g.read(D / "plan-root-approved.json")
assert plan["approved_for_apply"] is True
combined = g.read(A / "four-apply-closure.json")
assert (
    combined["status"] == "all_four_healthy_verified"
    and combined["container_count"] == 27
)
for key in ["freeze", "verification", "baseline", "protected_before"]:
    assert g.sha(plan[key]["path"]) == plan[key]["sha256"]
freeze = g.read(plan["freeze"]["path"])
verification = g.read(plan["verification"]["path"])
assert (
    verification["verified"]
    and verification["tree_sha256"] == freeze["control_tree_sha256"]
)
sourcepins = plan["source_pins"]
assert all(g.sha(p) == v for p, v in sourcepins.items())
original = g.read(plan["baseline"]["path"])["replicas"]
before = g.read(plan["protected_before"]["path"])
record = Path("/mnt/nvme/kairyu/model-volumes/gemma-4-31b-8gpu/chat-checkpoint.json")
lockfile = (R / "candidate-evaluation.lock").open("a")
fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
save(
    "endpoint-global-keeper.json",
    dict(
        pid=os.getpid(),
        started_epoch=started,
        source_sha256=g.sha(__file__),
        authorization="Root authorized endpoint chat1/directlogprob4 and files/path/tree metadata update after all4 success",
    ),
)
record_bytes = record.read_bytes()
record_sha = g.sha(record)
assert record_sha == "4e4b9a1d6670624cab7e4f53accad87970bf10635599ce2d06a2f6d2b0652552"
(D / "chat-checkpoint-before.json").write_bytes(record_bytes)
expected, objects = inventory()
assert len(expected) == 27
assert all(expected[k] == before[k] for k in plan["protected_ids"])
current_ids = {s: v["id"] for s, v in combined["replicas"].items()}
assert set(expected) == set(plan["protected_ids"]) | set(current_ids.values())
assert (
    g.inventory(Path(freeze["checkpoint_path"])) == freeze["candidate_inventory_stat"]
)
spec2 = importlib.util.spec_from_file_location("control", plan["control"])
control = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(control)
done = {}
for service, cid in current_ids.items():
    new = objects[cid]
    g.compose_scope(control, service, cid)
    g.idle(new)
    checked = g.compare_after(
        original[service],
        new,
        Path(freeze["checkpoint_path"]),
        freeze["control_tree_sha256"],
    )
    assert checked == combined["replicas"][service]
    ips = [
        v["IPAddress"]
        for v in new["NetworkSettings"]["Networks"].values()
        if v["IPAddress"]
    ]
    assert len(ips) == 1
    endpoint = "http://" + ips[0] + ":8000/v1"
    models = fetch(endpoint, "/models")
    ids = [v["id"] for v in models["data"]]
    assert "gemma-4-31b" in ids
    args = new["Config"]["Cmd"]
    assert (
        json.loads(args[args.index("--speculative-config") + 1])[
            "num_speculative_tokens"
        ]
        == 4
    )
    assert args[args.index("--tensor-parallel-size") + 1] == "2"
    assert args[args.index("--served-model-name") + 1] == "gemma-4-31b"
    assert "--worker-cls" not in args and "--middleware" not in args
    done[service] = dict(
        cid=cid,
        endpoint=endpoint,
        models=ids,
        tree=checked["tree"],
        checkpoint_mount=checked["checkpoint_path"],
        checkpoint_readonly=True,
        runtime_preserved=True,
        mtp_tokens=4,
        tp_size=2,
        native_runtime_flags_preserved=True,
        receipt_sha256=g.sha(A / (service + "-apply") / "receipt.json"),
        seconds=g.read(A / (service + "-result.json"))["seconds"],
    )
    save(service + "-closure-v1.json", done[service])
save(
    "four-apply-closure.json",
    dict(
        status="all_four_ordinary_F05_healthy_verified",
        replicas=done,
        tree=freeze["control_tree_sha256"],
        freeze_sha256=plan["freeze"]["sha256"],
        source_pins=sourcepins,
        protected_count=23,
        protected_unchanged=True,
        container_count=27,
        seconds=combined["seconds"],
        parallel_owner_closure_sha256=g.sha(A / "four-apply-closure.json"),
        model_generation_requests=0,
        full_payload_rehashed=False,
        source_sha256=g.sha(__file__),
    ),
)


# Minimum endpoint usability checks after all four are homogeneous; no score evaluation.
def post(url, body):
    data = json.dumps(body).encode()
    with urllib.request.urlopen(
        urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        ),
        timeout=120,
    ) as response:
        return json.loads(response.read(1024**2))


gateway = objects["88943a0985ba97dfb17be6b5fb54040eba2570b70ce05126bd410162d3658ffb"]
assert (
    safe(gateway) == before[gateway["Id"]]
    and gateway["State"]["Health"]["Status"] == "healthy"
)
gips = [
    v["IPAddress"]
    for v in gateway["NetworkSettings"]["Networks"].values()
    if v["IPAddress"]
]
assert len(gips) == 1
gendpoint = "http://" + gips[0] + ":8000/v1"
gmodels = fetch(gendpoint, "/models")
assert "gemma-4-31b" in [r["id"] for r in gmodels["data"]]
checks = {}
for service, row in done.items():
    body = dict(model="gemma-4-31b", prompt="Ready.", max_tokens=1, prompt_logprobs=1)
    response = post(row["endpoint"] + "/completions", body)
    save(service + "-logprob-smoke.json", dict(request=body, response=response))
    assert response["choices"][0].get("prompt_logprobs")
    checks[service] = dict(
        http_success=True,
        model=response["model"],
        prompt_logprob_positions=len(response["choices"][0]["prompt_logprobs"]),
    )
body = dict(
    model="gemma-4-31b",
    messages=[dict(role="user", content="Reply with the single word ready.")],
    max_tokens=256,
)
response = post(gendpoint + "/chat/completions", body)
choice = response["choices"][0]
save(
    "gateway-chat-smoke.json",
    dict(
        request=body,
        model=response.get("model"),
        usage=response.get("usage"),
        finish_reason=choice.get("finish_reason"),
        public_answer=choice["message"].get("content"),
    ),
)
assert response.get("model") == "gemma-4-31b" and choice["message"].get("content")
final, objects = inventory()
assert final == expected
for cid in current_ids.values():
    g.idle(objects[cid])
assert (
    g.inventory(Path(freeze["checkpoint_path"])) == freeze["candidate_inventory_stat"]
)
assert all(g.sha(p) == v for p, v in sourcepins.items())
assert g.sha(record) == record_sha
updated = json.loads(record_bytes)
updated.update(
    files=freeze["all_checkpoint_files"],
    path=freeze["checkpoint_path"],
    tree_sha256=freeze["control_tree_sha256"],
)
assert set(updated) == set(json.loads(record_bytes))
tmprecord = record.with_name(".chat-checkpoint-f05.tmp")
with tmprecord.open("x") as stream:
    stream.write(json.dumps(updated, indent=2, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())
assert g.sha(record) == record_sha
tmprecord.replace(record)
dfd = os.open(record.parent, os.O_DIRECTORY)
os.fsync(dfd)
os.close(dfd)

save("chat-checkpoint-after.json", g.read(record))
save(
    "served-record-update.json",
    dict(
        before_sha256=record_sha,
        after_sha256=g.sha(record),
        changed_fields=["files", "path", "tree_sha256"],
        schema_preserved=True,
    ),
)
save(
    "endpoint-closure.json",
    dict(
        status="all_four_F05_and_gateway_verified",
        gateway=dict(
            cid=gateway["Id"],
            endpoint=gendpoint,
            model_ids=[r["id"] for r in gmodels["data"]],
            healthy=True,
            unchanged_protected_service=True,
        ),
        replicas=done,
        logprob_checks=checks,
        chat_check=True,
        generation_requests=5,
        official_evaluation_requests=0,
        all_four_idle=True,
        protected_count=23,
        protected_unchanged=True,
        closure_sha256=g.sha(D / "four-apply-closure.json"),
        served_record_sha256=g.sha(record),
        created_epoch=time.time(),
    ),
)
fcntl.flock(lockfile, fcntl.LOCK_UN)
lockfile.close()
save(
    "global-lock-released.json",
    dict(
        status="normal_release",
        pid=os.getpid(),
        released_epoch=time.time(),
        endpoint_closure_sha256=g.sha(D / "endpoint-closure.json"),
        four_closure_sha256=g.sha(D / "four-apply-closure.json"),
    ),
)
PR = B / "docs/experiments/records/2026-11-07-f05-deployment-dp4-v1"
for p in D.rglob("*"):
    if p.is_file() and p.suffix != ".lock":
        q = PR / p.relative_to(D)
        q.parent.mkdir(parents=True, exist_ok=True)
        if not q.exists():
            shutil.copyfile(p, q)
print(
    json.dumps(
        dict(
            status="F05_deployment_and_endpoints_complete_lock_released",
            seconds=time.time() - started,
            closure_sha256=g.sha(D / "four-apply-closure.json"),
            endpoint_sha256=g.sha(D / "endpoint-closure.json"),
            release_sha256=g.sha(D / "global-lock-released.json"),
        )
    ),
    flush=True,
)
