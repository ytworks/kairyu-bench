#!/usr/bin/env python3
"""Prospective ordinary-saved TP2/DP2 matched full14 owner. Preview is inert."""

import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def need(v, msg):
    if not v:
        raise ValueError(msg)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(p, v):
    p = Path(p)
    t = p.with_suffix(".tmp")
    t.write_text(json.dumps(v, indent=2, allow_nan=False) + "\n")
    os.chmod(t, 0o600)
    t.replace(p)


def canonical(v):
    v = dict(v)
    v["mounts"] = sorted(
        v.get("mounts", []), key=lambda x: json.dumps(x, sort_keys=True)
    )
    return v


def command(argv, timeout=120):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    need(p.returncode == 0, "owned command failed")
    return p.stdout.strip()


def snap(s):
    return guard.snapshot(guard.inspect(plan["serving_baseline"][s]["id"]))


def publish():
    if owned:
        save(P / "execution-status.json", state)


def saved_identity(ref, checkpoint, tree):
    need(sha(ref["path"]) == ref["sha256"], "saved freeze changed")
    f = read(ref["path"])
    need(
        f["checkpoint_path"] == checkpoint and f["control_tree_sha256"] == tree,
        "saved identity mismatch",
    )
    from kairyu_bench.wsp.weights.checkpoint import Checkpoint

    rows = Checkpoint(checkpoint).identity(hash_files=False)["files"]
    need(
        [(r["path"], r["size"]) for r in rows]
        == [(r["path"], r["size"]) for r in f["all_checkpoint_files"]],
        "saved inventory",
    )
    for r in rows:
        old = f["candidate_inventory_stat"][r["path"]]
        need(
            r["size"] == old["size"] and all(old[k] == v for k, v in r["stat"].items()),
            "saved file changed",
        )
    return f


def validate(serving):
    need(
        sha(args.plan) == args.plan_sha
        and sha(args.authorization) == args.authorization_sha,
        "owner bindings changed",
    )
    auth = read(args.authorization)
    need(
        auth["status"] == "authorized"
        and auth["issued_by"] == "root"
        and auth["plan_sha256"] == args.plan_sha
        and auth["runtime_sha256"] == sha(__file__),
        "root authorization differs",
    )
    need(
        plan["binding_complete"]
        and [a["role"] for a in plan["arms"]] == ["parent", "candidate"],
        "two arm binding",
    )
    need(
        plan["image_id"]
        == "sha256:af9e746b913088cbf6c4464a31d6904a6b7e1d3ba62f0895362ddf13595f7abe",
        "fixed base image",
    )
    need(
        plan["topology"]
        == {
            "tp": 2,
            "dp": 2,
            "pp": 1,
            "static_mtp": 4,
            "external_load_balancer": False,
        },
        "topology",
    )
    need(
        sha(plan["incumbent"]["path"]) == plan["incumbent"]["sha256"],
        "incumbent changed",
    )
    need(
        sha(plan["selection_decision"]["path"]) == plan["selection_decision"]["sha256"],
        "selected finalist changed",
    )
    for path, digest in plan["pinned_inputs"].items():
        need(sha(path) == digest, "input or code pin changed")
    need(
        len(plan["archived_source_files_sha256"]) == 76
        and len(plan["installed_source_files_sha256"]) == 82,
        "official source coverage",
    )
    for rel, h in plan["archived_source_files_sha256"].items():
        need(
            sha(Path(plan["source_path"]) / rel) == h,
            "official archived source changed",
        )
    freezes = [
        saved_identity(a["freeze"], a["checkpoint"], a["tree"]) for a in plan["arms"]
    ]
    incumbent = read(plan["incumbent"]["path"])
    need(
        incumbent["checkpoint_path"] == plan["arms"][0]["checkpoint"]
        and incumbent["control_tree_sha256"] == plan["arms"][0]["tree"],
        "parent arm not current incumbent",
    )
    need(
        freezes[1]["source_checkpoint_path"] == plan["arms"][0]["checkpoint"]
        and freezes[1]["source_tree_sha256"] == plan["arms"][0]["tree"]
        and freezes[1]["parent_state"]["sha256"] == plan["incumbent"]["sha256"],
        "candidate not exact current-parent increment",
    )
    need(
        freezes[1]["source_inventory_stat"] == freezes[0]["candidate_inventory_stat"],
        "candidate source-stat binding differs",
    )
    need(
        [a["gpus"] for a in plan["arms"]] == [[0, 1, 2, 3], [4, 5, 6, 7]],
        "two disjoint four-GPU arms required",
    )
    policies = []
    for a in plan["arms"]:
        need(
            "BIND_TREE" not in a["smoke_generation_argv"]
            and a["smoke_generation_argv"][
                a["smoke_generation_argv"].index("--checkpoint-sha256") + 1
            ]
            == a["tree"],
            "smoke tree unbound",
        )
        op = a["observer"]
        need(sha(op["path"]) == op["sha256"], "observer config changed")
        oc = read(op["path"])
        need(
            oc["schema_version"] == 1 and oc["limits"] == {"storage_bytes": 268435456},
            "observer policy",
        )
        need(
            Path(op["path"]).parent == Path(oc["output_dir"])
            and Path(oc["output_dir"]).is_dir(),
            "observer directory",
        )
        need(
            len(oc["capture_id"]) == 32
            and all(c in "0123456789abcdef" for c in oc["capture_id"]),
            "observer capture ID",
        )
        policies.append(
            {k: v for k, v in oc.items() if k not in ["capture_id", "output_dir"]}
        )
    need(
        policies[0] == policies[1]
        and plan["arms"][0]["observer"]["path"] != plan["arms"][1]["observer"]["path"],
        "matched independent observers",
    )
    saved_identity(plan["draft_freeze"], plan["draft_path"], plan["draft_tree_sha256"])
    for name in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "processor_config.json",
    ]:
        need(
            len(
                {
                    next(
                        r["sha256"]
                        for r in f["all_checkpoint_files"]
                        if r["path"] == name
                    )
                    for f in freezes
                }
            )
            == 1,
            "arm protocol metadata differs",
        )
    if serving:
        for service, want in plan["serving_baseline"].items():
            need(
                canonical(snap(service)) == canonical(want), "original service changed"
            )
    for old in plan["embedding_baseline"]:
        c = guard.inspect(old["id"])
        need(
            c["State"]["Running"] == old["running"]
            and c["State"]["StartedAt"] == old["started_at"]
            and guard.digest(c["Config"]) == old["config_sha256"]
            and guard.digest(c["HostConfig"]) == old["host_config_sha256"],
            "embedding changed",
        )


def endpoint(cid):
    c = guard.inspect(cid)
    ips = [
        x["IPAddress"]
        for x in c["NetworkSettings"]["Networks"].values()
        if x.get("IPAddress")
    ]
    need(len(ips) == 1, "single endpoint required")
    return "http://" + ips[0] + ":8000/v1"


def wait_ready(cid, deadline):
    while time.monotonic() < deadline:
        need(guard.inspect(cid)["State"]["Running"], "temporary server exited")
        ep = endpoint(cid)
        try:
            with urllib.request.urlopen(
                ep.removesuffix("/v1") + "/health", timeout=5
            ) as r:
                if r.status == 200:
                    return ep
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("startup deadline")


def counters(ep, cid, path, idle):
    argv = [
        str(PY),
        plan["counter_tool"],
        "snapshot",
        "--endpoint",
        ep.removesuffix("/v1") + "/metrics",
        "--model",
        "gemma-4-31b",
        "--container-id",
        cid,
        "--attestation-sha256",
        args.plan_sha,
        "--output",
        str(path),
        "--capture",
    ]
    command(argv, 30)
    d = read(path)
    if idle:
        for name in ["vllm:num_requests_running", "vllm:num_requests_waiting"]:
            vals = [float(r["value"]) for r in d["series"] if r["metric"] == name]
            need(vals and all(v == 0 for v in vals), "server not idle")
    return {"path": str(path), "sha256": sha(path)}


def mtp(ep):
    # Numeric-only allowlist; never persist raw /metrics response.
    with urllib.request.urlopen(ep.removesuffix("/v1") + "/metrics", timeout=10) as r:
        raw = r.read(16 * 1024 * 1024).decode()
    rows = []
    for line in raw.splitlines():
        m = re.fullmatch(
            r"(vllm:spec_decode_[a-zA-Z0-9_]+)(?:\{([^}]*)\})?\s+([-+0-9.eE]+)(?:\s+\d+)?",
            line,
        )
        if m:
            engine = re.search(r'(?:^|,)engine="([^"]*)"', m[2] or "")
            rows.append(
                {
                    "metric": m[1],
                    "engine": engine[1] if engine else None,
                    "labels_sha256": hashlib.sha256((m[2] or "").encode()).hexdigest(),
                    "value": float(m[3]),
                }
            )
    return rows


def create(a, base):
    cmd = list(base["Config"]["Cmd"])
    need(cmd[0] == "/checkpoint", "server template model")
    cmd[0] = a["checkpoint"]
    for flag in [
        "--data-parallel-size",
        "--data-parallel-rank",
        "--data-parallel-external-lb",
        "--load-format",
        "--safetensors-load-strategy",
    ]:
        need(flag not in cmd, "unexpected template override")
    need(cmd[cmd.index("--tensor-parallel-size") + 1] == "2", "template TP2")
    ix = cmd.index("--speculative-config") + 1
    draft = json.loads(cmd[ix])
    need(
        draft["num_speculative_tokens"] == 4
        and "num_speculative_tokens_per_batch_size" not in draft,
        "static MTP4 required",
    )
    draft["model"] = plan["draft_path"]
    cmd[ix] = json.dumps(draft, separators=(",", ":"))
    cmd += [
        "--data-parallel-size",
        "2",
        "--load-format",
        "safetensors",
        "--safetensors-load-strategy",
        "lazy",
    ]
    cache = P / "cache" / a["role"]
    cache.mkdir(parents=True)
    env = guard.environments(base["Config"]["Env"])
    need(
        "kairyu_sparse_overlay" not in env.get("VLLM_PLUGINS", ""),
        "ordinary saved loader only",
    )
    env.update(
        XDG_CACHE_HOME=str(cache),
        XDG_CONFIG_HOME=str(cache / "config"),
        VLLM_CONFIG_ROOT=str(cache / "config/vllm"),
        HF_HOME=str(cache / "hf"),
        VLLM_CACHE_ROOT=str(cache / "vllm"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "torch"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        FLASHINFER_WORKSPACE_BASE=str(cache / "flashinfer-base"),
    )
    need(
        not any("\n" in k + v or "\r" in k + v for k, v in env.items()),
        "environment multiline",
    )
    ef = P / (".env-" + a["role"])
    fd = os.open(ef, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(k + "=" + v for k, v in env.items()) + "\n")
    rec = {"role": a["role"], "name": a["container_name"], "status": "creating"}
    temporary.append(rec)
    publish()
    argv = [
        "docker",
        "create",
        "--name",
        a["container_name"],
        "--label",
        "kairyu.dp2.owner=" + plan["owner_label"],
        "--runtime",
        "nvidia",
        "--gpus",
        '"device=' + ",".join(map(str, a["gpus"])) + '"',
        "--user",
        "1003:1003",
        "--network",
        base["HostConfig"]["NetworkMode"],
        "--ipc",
        base["HostConfig"]["IpcMode"],
        "--cpuset-cpus",
        a["cpuset"],
        "--shm-size",
        str(base["HostConfig"]["ShmSize"]),
        "--env-file",
        str(ef),
        "--no-healthcheck",
    ]
    if base["Config"].get("WorkingDir"):
        argv += ["--workdir", base["Config"]["WorkingDir"]]
    models = next(x for x in base["Mounts"] if x["Destination"] == "/models")
    need(not models["RW"], "readonly models")
    for src, dst, ro in [
        (models["Source"], "/models", True),
        (a["checkpoint"], a["checkpoint"], True),
        (plan["draft_path"], plan["draft_path"], True),
        (str(cache), str(cache), False),
    ]:
        argv += [
            "--mount",
            "type=bind,src=" + src + ",dst=" + dst + (",readonly" if ro else ""),
        ]
    try:
        rec["container_id"] = command([*argv, plan["image_id"], *cmd])
    finally:
        ef.unlink(missing_ok=True)
    c = guard.inspect(rec["container_id"])
    need(
        c["Image"] == plan["image_id"]
        and c["Config"]["User"] == "1003:1003"
        and c["Config"]["Cmd"] == cmd
        and c["Config"]["Entrypoint"] == base["Config"]["Entrypoint"],
        "temporary config",
    )
    devices = c["HostConfig"]["DeviceRequests"]
    need(
        len(devices) == 1 and set(devices[0]["DeviceIDs"]) == set(map(str, a["gpus"])),
        "temporary GPUs",
    )
    rec.update(
        status="loading",
        command_sha256=guard.digest(cmd),
        environment_sha256=guard.digest(env),
        image_id=c["Image"],
    )
    publish()
    return rec


def startup_diagnostic(rec):
    if rec.get("status") != "loading":
        return
    c = guard.inspect(rec["container_id"])
    need(
        c["Image"] == plan["image_id"]
        and c["Config"]["Labels"].get("kairyu.dp2.owner") == plan["owner_label"],
        "diagnostic ownership",
    )
    p = subprocess.run(
        ["docker", "logs", "--tail", "2000", c["Id"]], capture_output=True, timeout=20
    )
    raw = (p.stdout + b"\n" + p.stderr)[-1048576:]
    d = P / "private-startup"
    d.mkdir(exist_ok=True, mode=0o700)
    z = d / (rec["role"] + ".log")
    fd = os.open(z, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    text = raw.decode(errors="replace")
    classes = [
        "PermissionError",
        "ValueError",
        "RuntimeError",
        "ImportError",
        "ModuleNotFoundError",
        "OutOfMemoryError",
    ]
    rec["startup_diagnostic"] = {
        "exit_code": c["State"].get("ExitCode"),
        "oom_killed": c["State"].get("OOMKilled"),
        "private_log_sha256": sha(z),
        "bytes": len(raw),
        "error_classes": [
            x for x in classes if re.search(r"(?<![A-Za-z])" + x + r"(?=[:\s])", text)
        ],
    }


def launch_child(argv, log, env=None):
    with Path(log).open("xb") as f:
        p = subprocess.Popen(
            argv, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
    children.append(p)
    return p


def wait_child(child, deadline):
    while child.poll() is None:
        need(time.monotonic() < deadline, "bounded child deadline")
        time.sleep(2)
    need(child.returncode == 0, "child failed")


def runner_capture(a, cmd, out, deadline):
    for _ in range(90):
        need(time.monotonic() < deadline, "runner capture deadline")
        for cid in command(
            ["docker", "ps", "--no-trunc", "--format", "{{.ID}}"],
            max(1, min(30, deadline - time.monotonic())),
        ).split():
            need(time.monotonic() < deadline, "runner capture deadline")
            c = guard.inspect(cid)
            if a["run_id"] not in (c["Config"].get("Cmd") or []):
                continue
            need(c["Config"]["Cmd"] == cmd[1:], "official runner args differ")
            a["_runner_id"] = cid
            image = json.loads(command(["docker", "image", "inspect", c["Image"]]))[0]
            save(
                out / "actual-runner-oci.json",
                {
                    "container_id": cid,
                    "image_id": c["Image"],
                    "rootfs": image["RootFS"],
                    "image_config_sha256": guard.digest(image["Config"]),
                    "command_sha256": guard.digest(c["Config"]["Cmd"]),
                    "source_head": plan["source_head"],
                    "credentials_persisted": False,
                },
            )
            code = "import hashlib,importlib.util,json,sys;from pathlib import Path;p=Path(importlib.util.find_spec('kairyu_bench').origin).parent;print(json.dumps({n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in json.loads(sys.argv[1])}))"
            v = json.loads(
                command(
                    [
                        "docker",
                        "exec",
                        cid,
                        "python",
                        "-B",
                        "-c",
                        code,
                        json.dumps(sorted(plan["installed_source_files_sha256"])),
                    ],
                    30,
                )
            )
            need(
                v == plan["installed_source_files_sha256"],
                "installed official source differs",
            )
            save(out / "actual-installed-source.json", v)
            return
        time.sleep(1)
    raise TimeoutError("official runner capture unavailable")


def execute():
    global owned
    need(os.getuid() == 1003, "UID1003 required")
    validate(True)
    need(
        not (P / "execution-status.json").exists()
        and not (P / "completion.json").exists(),
        "already owned",
    )
    need(
        bool(os.environ.get("HF_TOKEN")),
        "official credential must be inherited, never saved",
    )
    need(
        len(plan["serving_baseline"]) == 4
        and set(plan["serving_baseline"])
        == {"gemma-0", "gemma-1", "gemma-2", "gemma-3"},
        "exact four old services",
    )
    for a in plan["arms"]:
        need(
            not (Path(plan["results_dir"]) / a["run_id"]).exists(),
            "official output exists",
        )
        need(
            subprocess.run(
                ["docker", "inspect", a["container_name"]], capture_output=True
            ).returncode
            != 0,
            "temporary name exists",
        )
    with (P / "execution-status.json").open("x") as f:
        json.dump(state, f)
    owned = True
    publish()
    base = guard.inspect(
        plan["serving_baseline"][plan["server_template_service"]]["id"]
    )
    for svc, s in plan["serving_baseline"].items():
        counters(endpoint(s["id"]), s["id"], P / (svc + "-before-stop.json"), True)
    validate(True)
    state["status"] = "stopping_exact_four"
    publish()
    for svc, s in plan["serving_baseline"].items():
        stopped.append(svc)
        command(["docker", "stop", "--time", "90", s["id"]])
        need(not guard.inspect(s["id"])["State"]["Running"], "original did not stop")
    deadline = time.monotonic() + plan["maximum_active_seconds"]
    for a in plan["arms"]:
        create(a, base)
    command(["docker", "start", *[r["container_id"] for r in temporary]])
    for a, rec in zip(plan["arms"], temporary, strict=True):
        rec["endpoint"] = wait_ready(
            rec["container_id"],
            min(deadline, time.monotonic() + plan["startup_seconds"]),
        )
        rec["status"] = "ready"
        publish()
    validate(False)
    # One existing eight-case authored smoke per arm; correctness is recorded, never an8/8 gate.
    state["status"] = "authored_smoke"
    publish()
    smokes = []
    for a, rec in zip(plan["arms"], temporary, strict=True):
        d = P / a["role"]
        d.mkdir()
        a["_output"] = str(d)
        rec["mtp_before"] = mtp(rec["endpoint"])
        argv = [
            rec["endpoint"] if x == "BIND_ENDPOINT" else x
            for x in a["smoke_generation_argv"]
        ]
        smokes.append(launch_child(argv, d / "smoke-generation.log"))
    smoke_deadline = min(deadline, time.monotonic() + plan["smoke_seconds"])
    for child in smokes:
        wait_child(child, smoke_deadline)
    for a, rec in zip(plan["arms"], temporary, strict=True):
        d = Path(a["_output"])
        wait_child(
            launch_child(a["smoke_execution_argv"], d / "smoke-execution.log"),
            smoke_deadline,
        )
        receipt = read(Path(a["smoke_generation_output"]) / "receipt.json")
        need(
            receipt["status"] == "completed"
            and receipt["received"] == 8
            and receipt["errors"] == 0,
            "authored request smoke failed",
        )
        report = read(Path(a["smoke_execution_output"]) / "report.json")
        rec["authored_numeric"] = [
            {
                k: r.get(k)
                for k in [
                    "id",
                    "passed",
                    "status",
                    "finish_reason",
                    "tests",
                    "tests_passed",
                    "tests_total",
                    "usage",
                ]
            }
            for r in report["results"]
        ]
        after = mtp(rec["endpoint"])
        rec["mtp_after"] = after
        before = {
            (r["metric"], r["labels_sha256"]): r["value"] for r in rec["mtp_before"]
        }
        delta = [
            r["value"] - before.get((r["metric"], r["labels_sha256"]), 0)
            for r in after
            if "num_drafts" in r["metric"]
        ]
        need(delta and sum(delta) > 0, "static MTP draft activity not evidenced")
        rec["mtp_draft_delta"] = sum(delta)
    validate(False)
    state["status"] = "official_full14"
    publish()
    runs = []
    for a, rec in zip(plan["arms"], temporary, strict=True):
        d = Path(a["_output"])
        rec["before_counters"] = counters(
            rec["endpoint"], rec["container_id"], d / "serving-before.json", True
        )
        cmd = [
            str(Path(plan["source_path"]) / "kairyu-bench"),
            "wsp",
            "run",
            rec["endpoint"],
            "--results-dir",
            str(Path(plan["results_dir"])),
            "--tier",
            "10",
            "--split",
            "dev",
            "--label",
            a["tree"],
            "--run-id",
            a["run_id"],
            "--chat-concurrency",
            "32",
            "--logprob-concurrency",
            "8",
            "--agent-concurrency",
            "1",
            "--logprob-endpoint",
            rec["endpoint"],
            "--observe-config",
            a["observer"]["path"],
        ]
        a["_official_command"] = cmd
        child = launch_child(
            cmd, d / "official.log", {**os.environ, **plan["benchmark_env"]}
        )
        runs.append((a, rec, cmd, child))
        rec.update(
            official_pid=child.pid, official_started_at=now(), run_id=a["run_id"]
        )
    publish()
    for a, rec, cmd, child in runs:
        runner_capture(a, cmd, Path(a["_output"]), min(deadline, time.monotonic() + 90))
    oci = [read(Path(a["_output"]) / "actual-runner-oci.json") for a in plan["arms"]]
    need(
        all(
            {k: x[k] for k in ["image_id", "rootfs", "image_config_sha256"]}
            == {k: oci[0][k] for k in ["image_id", "rootfs", "image_config_sha256"]}
            for x in oci
        ),
        "official runner images differ",
    )
    official_deadline = min(deadline, time.monotonic() + plan["official_seconds"])
    for a, rec, cmd, child in runs:
        wait_child(child, official_deadline)
        run = Path(plan["results_dir"]) / a["run_id"]
        need(
            read(run / "run.json")["status"] == "completed"
            and (run / "report.json").is_file(),
            "official run incomplete",
        )
        rec.update(
            status="completed",
            official_finished_at=now(),
            run_sha256=sha(run / "run.json"),
            report_sha256=sha(run / "report.json"),
        )
        d = Path(a["_output"])
        rec["after_counters"] = counters(
            rec["endpoint"], rec["container_id"], d / "serving-after.json", True
        )
        command(
            [
                str(PY),
                plan["counter_tool"],
                "diff",
                str(d / "serving-before.json"),
                str(d / "serving-after.json"),
                "--output",
                str(d / "serving-diff.json"),
            ],
            30,
        )
    validate(False)
    state["pair_completed"] = True


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--plan", type=Path, required=True)
parser.add_argument("--plan-sha", required=True)
parser.add_argument("--authorization", type=Path)
parser.add_argument("--authorization-sha")
parser.add_argument("--apply", action="store_true")
args = parser.parse_args()
need(sha(args.plan) == args.plan_sha, "plan SHA")
plan = read(args.plan)
P = args.plan.resolve().parent
if not args.apply:
    print(
        json.dumps(
            {
                "status": "prepared_no_operations",
                "binding_complete": plan.get("binding_complete"),
                "arms": 2,
                "topology": plan["topology"],
                "runtime_sha256": sha(__file__),
                "service_changes": 0,
                "model_calls": 0,
            }
        )
    )
    raise SystemExit(0)
need(args.authorization and args.authorization_sha, "root authorization required")
os.umask(0o077)
PY = Path(plan["python"])
W = Path(plan["repo"])
HELPER = Path(plan["tools"]["redeploy"]["path"])
need(sha(HELPER) == plan["tools"]["redeploy"]["sha256"], "redeploy helper pin differs")
sys.path.insert(0, str(W / "src"))
sp = importlib.util.spec_from_file_location("dp2_guard", HELPER)
guard = importlib.util.module_from_spec(sp)
sp.loader.exec_module(guard)
lock = (P / "owner.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
owned = False
stopped = []
temporary = []
children = []
error = None
state = {
    "status": "preflight",
    "started_at": now(),
    "started_epoch": time.time(),
    "plan_sha256": args.plan_sha,
    "temporary": temporary,
    "source_checkpoint_writes": 0,
}


def interrupted(sig, frame):
    raise InterruptedError("owner interrupted")


for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
    signal.signal(sig, interrupted)
try:
    execute()
except BaseException as exc:
    error = type(exc).__name__
    state.update(status="failed", error_type=error)
    if isinstance(exc, ValueError):
        state["owner_assertion_message"] = str(exc)
    for rec in temporary:
        try:
            startup_diagnostic(rec)
        except BaseException as e:
            rec["diagnostic_error_type"] = type(e).__name__
    publish()
finally:
    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
        signal.signal(sig, signal.SIG_IGN)
    # Every cleanup and every original restoration is independently attempted.
    for child in children:
        try:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait(timeout=10)
        except BaseException as e:
            state.setdefault("child_cleanup_errors", []).append(type(e).__name__)
            error = error or type(e).__name__
    for a in plan["arms"]:
        if a.get("_official_command"):
            try:
                ids = (
                    [a["_runner_id"]]
                    if a.get("_runner_id")
                    else command(
                        ["docker", "ps", "--no-trunc", "--format", "{{.ID}}"], 30
                    ).split()
                )
                for cid in ids:
                    q = subprocess.run(
                        ["docker", "inspect", cid],
                        capture_output=True,
                        text=True,
                        timeout=20,
                    )
                    if q.returncode != 0:
                        continue
                    c = json.loads(q.stdout)[0]
                    if c["Config"].get("Cmd") != a["_official_command"][1:]:
                        continue
                    need(a["run_id"] in c["Config"]["Cmd"], "runner cleanup identity")
                    command(["docker", "stop", "--time", "30", c["Id"]], 60)
            except BaseException as e:
                state.setdefault("runner_cleanup_errors", []).append(type(e).__name__)
                error = error or type(e).__name__
    for rec in temporary:
        try:
            q = subprocess.run(
                ["docker", "inspect", rec.get("container_id", rec["name"])],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if q.returncode == 0:
                c = json.loads(q.stdout)[0]
                need(
                    c["Image"] == plan["image_id"]
                    and (c["Config"].get("Labels") or {}).get("kairyu.dp2.owner")
                    == plan["owner_label"],
                    "temporary cleanup identity",
                )
                command(["docker", "rm", "--force", c["Id"]], 120)
        except BaseException as e:
            state.setdefault("temporary_cleanup_errors", []).append(type(e).__name__)
            error = error or type(e).__name__
    state["status"] = "restoring_same_four"
    state["restored"] = {}
    try:
        publish()
    except BaseException as e:
        state["restore_status_error_type"] = type(e).__name__
        error = error or type(e).__name__
    restore_deadline = time.monotonic() + plan["restoration_seconds"]
    for svc in stopped:
        try:
            command(
                ["docker", "start", plan["serving_baseline"][svc]["id"]],
                max(1, min(120, restore_deadline - time.monotonic())),
            )
            state["restored"][svc] = {"start_requested": True}
        except BaseException as e:
            state["restored"][svc] = {"error_type": type(e).__name__}
            error = error or type(e).__name__
    for svc in stopped:
        try:
            while time.monotonic() < restore_deadline:
                if canonical(snap(svc)) == canonical(plan["serving_baseline"][svc]):
                    break
                time.sleep(2)
            need(
                canonical(snap(svc)) == canonical(plan["serving_baseline"][svc]),
                "original restore incomplete",
            )
            state["restored"][svc] = {"same_cid_exact_healthy": True}
        except BaseException as e:
            state["restored"][svc] = {"error_type": type(e).__name__}
            error = error or type(e).__name__
    if owned:
        try:
            validate(True)
            state["final_source_serving_checks"] = True
        except BaseException as e:
            state["final_check_error_type"] = type(e).__name__
            error = error or type(e).__name__
        state.update(
            status="completed"
            if not error and state.get("pair_completed")
            else "failed",
            finished_at=now(),
            finished_epoch=time.time(),
        )
        publish()
        with (P / "completion.json").open("x") as f:
            json.dump(state, f, indent=2)
            f.write("\n")
if not owned:
    print(
        json.dumps(
            {
                "status": "preflight_failed_no_ownership",
                "error_type": error,
                "existing_receipts_unchanged": True,
            }
        )
    )
raise SystemExit(0 if state["status"] == "completed" else 1)
