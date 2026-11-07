"""Four existing guarded service helpers in parallel; default metadata preview only."""

from pathlib import Path
import argparse
import ast
import concurrent.futures
import fcntl
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time

R = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
B = R / "worktrees/kairyu-bench-weight-tools"
H = B / "docs/experiments/scripts/redeploy-replica.py"
sys.path.insert(0, str(B / "src"))
module = importlib.util.spec_from_file_location("guard", H)
g = importlib.util.module_from_spec(module)
module.loader.exec_module(g)
source = (
    R
    / "analysis/fresh-purpose-20261107-av-dev30-deployment-plan-v1/remove-completed-observer-v2.py"
)
assert (
    g.sha(source) == "99a9cc25006a988d1d7cfe11b8f579ea0bbb857c4de34c39f21bf0769b5a12ce"
)
tree = ast.parse(source.read_text())
exec(
    compile(
        ast.Module(
            body=[
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name in {"safe", "inventory"}
            ],
            type_ignores=[],
        ),
        str(source),
        "exec",
    ),
    globals(),
)
inventory = globals()["inventory"]
a = argparse.ArgumentParser(description=__doc__)
a.add_argument("--plan", type=Path, required=True)
a.add_argument("--apply", action="store_true")
a.add_argument("--approved-plan-sha256")
args = a.parse_args()
plan = g.read(args.plan)
services = ["gemma-0", "gemma-1", "gemma-2", "gemma-3"]
assert plan["service_order"] == services and os.getuid() == 1003
assert (
    g.sha(H) == plan["helper_sha256"] and g.sha(Path(__file__)) == plan["owner_sha256"]
)
assert all(g.sha(p) == h for p, h in plan["source_pins"].items())
if not args.apply:
    print(
        json.dumps(
            dict(
                status="unbound_plan_preview_no_service_calls",
                candidate_manifest_bound=plan["freeze"]["sha256"] is not None,
                service_count=4,
                global_lock_acquired=False,
                apply=False,
            )
        )
    )
    raise SystemExit(0)
assert plan["approved_for_apply"] is True and args.approved_plan_sha256 == g.sha(
    args.plan
)
for key in ["freeze", "verification", "baseline", "protected_before"]:
    ref = plan[key]
    assert g.sha(ref["path"]) == ref["sha256"]
freeze = g.read(plan["freeze"]["path"])
verification = g.read(plan["verification"]["path"])
assert (
    verification["verified"]
    and verification["tree_sha256"] == freeze["control_tree_sha256"]
)
assert freeze["checkpoint_path"] == plan["candidate"]
baseline = g.read(plan["baseline"]["path"])
before = g.read(plan["protected_before"]["path"])
assert len(before) == 27 and len(plan["protected_ids"]) == 23
assert set(before) == set(plan["protected_ids"]) | set(plan["target_ids"].values())
assert len(set(plan["target_ids"].values())) == 4
original = {s: baseline["replicas"][s] for s in services}
assert all(original[s]["id"] == plan["target_ids"][s] for s in services)
assert len({(v["checkpoint_path"], v["tree"]) for v in original.values()}) == 1
for s in services:
    ref = plan["previews"][s]
    assert g.sha(ref["path"]) == ref["sha256"]
    preview = g.read(ref["path"])
    assert preview["status"] == "preview" and not preview["apply_requested"]
    assert (
        preview["before"] == original[s]
        and preview["candidate_identity"]["tree_sha256"]
        == freeze["control_tree_sha256"]
    )
module = importlib.util.spec_from_file_location("control", plan["control"])
control = importlib.util.module_from_spec(module)
module.loader.exec_module(control)
assert baseline["control_sha256"] == g.sha(plan["control"])
D = Path(plan["output"])
assert not D.exists()
interrupts = []
for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
    signal.signal(sig, lambda number, frame: interrupts.append(number))


def save(name, value):
    p = D / name
    assert not p.exists()
    g.save(p, value)


def stable_protected(current):
    assert all(cid in current for cid in plan["protected_ids"])
    for cid in plan["protected_ids"]:
        old, now = dict(before[cid]), dict(current[cid])
        old.pop("health", None)
        now.pop("health", None)
        assert old == now
    selected = {
        cid
        for cid, row in current.items()
        if row["labels"].get("com.docker.compose.project") == control.PROJECT
        and row["labels"].get("com.docker.compose.service") in services
    }
    assert set(current) - selected == set(plan["protected_ids"])


def closure(expected_candidate, expected_tree):
    deadline = time.monotonic() + 60
    while True:
        current, raw = inventory()
        stable_protected(current)
        if all(current[cid] == before[cid] for cid in plan["protected_ids"]):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("protected health did not recover")
        time.sleep(1)
    assert len(current) == 27
    result = {}
    for s in services:
        c = g.inspect(control.PROJECT + "-" + s + "-1")
        g.compose_scope(control, s, c["Id"])
        result[s] = g.compare_after(
            original[s], c, Path(expected_candidate), expected_tree
        )
        g.idle(c)
    return dict(
        status="all_four_healthy_verified",
        replicas=result,
        protected_count=23,
        protected_unchanged=True,
        container_count=27,
        tree=expected_tree,
    )


with (R / "candidate-evaluation.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    D.mkdir()
    started = time.time()
    save(
        "owner-start.json",
        dict(
            pid=os.getpid(),
            uid=os.getuid(),
            epoch=started,
            plan_sha256=g.sha(args.plan),
            source_sha256=g.sha(__file__),
        ),
    )
    current, raw = inventory()
    assert current == before and not interrupts
    assert g.inventory(Path(plan["candidate"])) == freeze["candidate_inventory_stat"]
    env = control._compose_env()
    env["COMPOSE_REMOVE_ORPHANS"] = "0"
    for s in services:
        g.compose_scope(control, s, original[s]["id"])
        g.idle(raw[original[s]["id"]])
        env[s.replace("-", "_").upper() + "_CPUSET"] = original[s]["cpuset"]

    def apply_one(s):
        cmd = [
            plan["python"],
            str(H),
            "--control",
            plan["control"],
            "--service",
            s,
            "--candidate",
            plan["candidate"],
            "--freeze",
            plan["freeze"]["path"],
            "--baseline",
            plan["baseline"]["path"],
            "--output",
            str(D / (s + "-apply")),
            "--timeout",
            "1200",
            "--apply",
        ]
        save(s + "-command.json", dict(argv=cmd))
        t = time.time()
        try:
            with (
                (D / (s + ".stdout")).open("x") as out,
                (D / (s + ".stderr")).open("x") as err,
            ):
                rc = subprocess.run(
                    cmd,
                    cwd=B,
                    env=dict(
                        os.environ,
                        PYTHONPATH=str(B / "src"),
                        PYTHONDONTWRITEBYTECODE="1",
                    ),
                    stdout=out,
                    stderr=err,
                ).returncode
            receipt = D / (s + "-apply/receipt.json")
            value = dict(
                service=s,
                returncode=rc,
                receipt_sha256=g.sha(receipt) if receipt.exists() else None,
                status=g.read(receipt)["status"] if receipt.exists() else "no_receipt",
                seconds=time.time() - t,
            )
        except BaseException as error:
            value = dict(
                service=s,
                returncode=None,
                status="owner_stage_error",
                error_type=type(error).__name__,
                seconds=time.time() - t,
            )
        save(s + "-result.json", value)
        print(json.dumps(value), flush=True)
        return value

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(apply_one, services))
    save(
        "all-helper-results.json",
        dict(results=results, all_children_joined=True, interrupts=interrupts),
    )
    good = not interrupts and all(
        v["returncode"] == 0 and v["status"] == "completed" for v in results
    )
    if good:
        try:
            result = closure(plan["candidate"], freeze["control_tree_sha256"])
            assert (
                g.inventory(Path(plan["candidate"]))
                == freeze["candidate_inventory_stat"]
            )
            assert all(g.sha(p) == h for p, h in plan["source_pins"].items())
            save("four-apply-closure.json", dict(result, seconds=time.time() - started))
        except BaseException as error:
            good = False
            save("postcheck-failure.json", dict(error_type=type(error).__name__))
    if not good:
        current, _ = inventory()
        stable_protected(current)

        def rollback_one(s):
            t = time.time()
            try:
                members = g.compose_scope(control, s)["container_ids"]
                rollback = dict(
                    env,
                    VLLM_IMAGE=original[s]["image_id"],
                    CHAT_CHECKPOINT_PATH=original[s]["checkpoint_path"],
                    CHAT_CHECKPOINT_TREE=original[s]["tree"],
                )
                if members:
                    current = g.snapshot(g.inspect(members[0]))
                    assert all(
                        current[k] == original[s][k]
                        for k in [
                            "image_id",
                            "command_sha256",
                            "environment_sha256",
                            "cpuset",
                            "devices",
                            "ipc_mode",
                            "networks",
                        ]
                    )
                    assert current["checkpoint_path"] in [
                        original[s]["checkpoint_path"],
                        plan["candidate"],
                    ]
                g.check_render(
                    control,
                    rollback,
                    s,
                    raw[original[s]["id"]],
                    Path(original[s]["checkpoint_path"]),
                    original[s]["tree"],
                )
                g.compose_up(control, rollback, s, 1200)
                value = dict(
                    status="rollback_healthy",
                    after=g.compare_after(
                        original[s],
                        g.inspect(control.PROJECT + "-" + s + "-1"),
                        Path(original[s]["checkpoint_path"]),
                        original[s]["tree"],
                    ),
                )
            except BaseException as error:
                value = dict(status="rollback_failed", error_type=type(error).__name__)
            save(s + "-batch-rollback.json", dict(value, seconds=time.time() - t))
            return value

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            restored = list(pool.map(rollback_one, services))
        rollback_ok = all(v["status"] == "rollback_healthy" for v in restored)
        if rollback_ok:
            save(
                "four-rollback-closure.json",
                closure(
                    original[services[0]]["checkpoint_path"],
                    original[services[0]]["tree"],
                ),
            )
        save(
            "failure.json",
            dict(
                status="failed_rolled_back"
                if rollback_ok
                else "failed_rollback_incomplete",
                all_children_joined=True,
                original_helper_receipts_preserved=True,
                seconds=time.time() - started,
            ),
        )
    fcntl.flock(lock, fcntl.LOCK_UN)
save(
    "lock-released.json",
    dict(normal_release=True, pid=os.getpid(), all_children_joined=True, success=good),
)
print(
    json.dumps(
        dict(
            status="completed" if good else "failed",
            output=str(D),
            seconds=time.time() - started,
        )
    ),
    flush=True,
)
raise SystemExit(0 if good else 1)
