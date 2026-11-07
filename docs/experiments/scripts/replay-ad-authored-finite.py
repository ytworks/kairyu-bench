#!/usr/bin/env python3
"""Prepare/default-preview or execute root-pinned authored-only AD six-alternative finite replay."""

import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path

R = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
A = R / "analysis/code90-resume-20261106"
D = R / "runtime/code90-resume-20261106"
P = A / "current-ad-functional-six-finite-v1"
PY = R / "infra/weight-tools-ci/bin/python"
PLAN_SHA = None
AUTH_SHA = None


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def require(v, msg):
    if not v:
        raise ValueError(msg)


def save(p, d):
    t = Path(p).with_suffix(".tmp")
    t.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    os.chmod(t, 0o600)
    t.replace(p)


def canonical(s):
    d = dict(s)
    d["mounts"] = sorted(
        d.get("mounts", []), key=lambda v: json.dumps(v, sort_keys=True)
    )
    return d


spec = importlib.util.spec_from_file_location(
    "fixed_four", D / "base-alignment-fourway-m-v1/dev90-four.py"
)
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
guard, _ = F.U.helper()
state = {
    "schema_version": 1,
    "status": "preflight",
    "pid": os.getpid(),
    "started_at": stamp(),
    "started_epoch": time.time(),
    "plan_sha256": PLAN_SHA,
    "jobs": [],
    "model_calls_scope": "Plan-bound authored train cases, immutable accepted AD -> each selected alternative from AD -> restored AD finite diagnostics only",
    "benchmark_calls": 0,
    "checkpoint_writes": 0,
}
plan = None
stopped = False
active_cid = None
active_proc = None
active_job = None
owner_lock = None
owned_execution = False


def publish():
    if owned_execution:
        save(P / "execution-status.json", state)


def snapshot(svc):
    return guard.snapshot(guard.inspect("gemma-4-31b-8gpu-" + svc + "-1"))


def same_service(svc, want):
    actual = canonical(snapshot(svc))
    expected = canonical(want)
    return actual == expected


def validate(serving=True):
    require(
        sha(P / "owner-plan.json") == PLAN_SHA
        and sha(P / "root-authorization.json") == AUTH_SHA,
        "plan/authorization changed",
    )
    auth = read(P / "root-authorization.json")
    require(
        auth["status"] == "authorized"
        and auth["issued_by"] == "root"
        and auth["plan_sha256"] == PLAN_SHA
        and auth["runtime_sha256"] == sha(__file__),
        "authorization contract",
    )
    require(
        plan["stop_service"] == "gemma-0"
        and plan["physical_gpus"] == [0]
        and plan["protected_services"] == ["gemma-1", "gemma-2", "gemma-3"],
        "fixed AD finite lifecycle scope",
    )
    require(
        plan["status"] == "planned_not_executed_requires_root_authorization"
        and plan["source_freeze_required"] is False
        and len(plan["source_sha256"]) == 6,
        "source pins not bound",
    )
    require(
        plan["candidate_binding_required"] is False
        and len(plan["jobs"]) == 1
        and 1 <= len(plan["selected_candidates"]) <= 8,
        "candidate unbound",
    )
    for path, digest in plan["pinned_inputs"].items():
        require(sha(path) == digest, "finite fixed input changed")
    fixed = plan["fixed_plan"]
    require(sha(fixed["path"]) == fixed["sha256"], "finite plan changed")
    fp = read(fixed["path"])
    require(
        fp["architecture"] == "gemma4_sparse_bf16"
        and fp["candidates"] == plan["selected_candidates"],
        "exact selected sparse candidate roster required",
    )
    require(
        len({c["id"] for c in plan["selected_candidates"]})
        == len(plan["selected_candidates"]),
        "duplicate candidate ids",
    )
    require(
        sha(plan["frame"]["historical_evidence_path"])
        == plan["frame"]["historical_evidence_sha256"],
        "frame changed",
    )
    for path, digest in plan["operational_pins"].items():
        require(sha(path) == digest, "operational helper changed")
    require(
        sha(plan["parent"]["path"]) == plan["parent"]["sha256"],
        "incumbent state changed",
    )
    for path, digest in plan["source_sha256"].items():
        require(sha(path) == digest, "collector source changed")
    require(sha(plan["cases"]["path"]) == plan["cases"]["sha256"], "cases changed")
    att = plan["checkpoint_attestation"]
    require(sha(att["path"]) == att["sha256"], "attestation changed")
    data = read(att["path"])
    for key in ["source_freeze", "full_hash_evidence"]:
        pin = data[key]
        require(sha(pin["path"]) == pin["sha256"], "full hash evidence changed")
    for row in data["checkpoint"]["files"]:
        f = Path(data["checkpoint"]["path"]) / row["path"]
        s = f.stat()
        require(
            s.st_size == row["size"]
            and all(
                row["stat"][k] == v
                for k, v in {
                    "device": s.st_dev,
                    "inode": s.st_ino,
                    "mtime_ns": s.st_mtime_ns,
                    "ctime_ns": s.st_ctime_ns,
                }.items()
            ),
            "AD file stamp changed",
        )
    if serving:
        for svc, want in plan["serving_baseline"].items():
            require(same_service(svc, want), "serving snapshot changed " + svc)
    else:
        for svc in ["gemma-1", "gemma-2", "gemma-3"]:
            require(
                same_service(svc, plan["serving_baseline"][svc]),
                "preserved service changed " + svc,
            )
    for row in plan["embedding_baseline"]:
        c = guard.inspect(row["id"])
        require(
            c["Id"] == row["id"]
            and c["State"]["Running"] == row["running"]
            and c["State"]["StartedAt"] == row["started_at"]
            and guard.digest(c["Config"]) == row["config_sha256"]
            and guard.digest(c["HostConfig"]) == row["host_config_sha256"],
            "embedding changed",
        )


def stop_signal(sig, frame):
    raise InterruptedError("owner interrupted")


def run():
    global plan, stopped, active_cid, active_proc, active_job, owned_execution
    os.umask(0o077)
    require(os.getuid() == 1003, "ordinary owner required")
    plan = read(P / "owner-plan.json")
    ci = read(plan["ci"]["path"])
    require(
        sha(plan["ci"]["path"]) == plan["ci"]["sha256"]
        and ci["all_passed"]
        and ci["source_changed_during_checks"] == [],
        "final CI mismatch",
    )
    for path, digest in plan["source_sha256"].items():
        require(
            ci["source_files"].get(
                str(Path(path).relative_to(R / "worktrees/kairyu-bench-weight-tools"))
            )
            == digest,
            "CI source mismatch",
        )
    validate()
    require(
        not (P / "execution-status.json").exists()
        and not (P / "completion.json").exists(),
        "owner already started",
    )
    with (P / "execution-status.json").open("x") as initial:
        json.dump(state, initial, indent=2)
        initial.write("\n")
    owned_execution = True
    publish()
    for job in plan["jobs"]:
        log = P / (job["label"] + "-preview.log")
        with log.open("xb") as f:
            t = time.monotonic()
            r = subprocess.run(
                job["preview_argv"],
                env={**os.environ, **job["preview_env"]},
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=job["preview_timeout_seconds"],
            )
        require(r.returncode == 0, "CPU metadata preview failed " + job["label"])
        state.setdefault("previews", []).append(
            {
                "label": job["label"],
                "returncode": 0,
                "seconds": time.monotonic() - t,
                "log_sha256": sha(log),
            }
        )
        publish()
    validate()
    c = guard.inspect(plan["serving_baseline"]["gemma-0"]["id"])
    ips = [
        n["IPAddress"]
        for n in c["NetworkSettings"]["Networks"].values()
        if n.get("IPAddress")
    ]
    require(len(ips) == 1, "gemma0 network mapping")
    counter = (
        R
        / "worktrees/kairyu-bench-weight-tools/scripts/experiments/serving-counters.py"
    )
    require(
        sha(counter)
        == "82ec0621729e67ed566295030ac548ce91552a294d10042f5922e307528180ee",
        "counter helper changed",
    )
    q = subprocess.run(
        [
            str(PY),
            str(counter),
            "snapshot",
            "--endpoint",
            "http://" + ips[0] + ":8000/metrics",
            "--model",
            "gemma-4-31b",
            "--container-id",
            c["Id"],
            "--attestation-sha256",
            PLAN_SHA,
            "--output",
            str(P / "gemma0-before-stop-counters.json"),
            "--capture",
        ],
        capture_output=True,
    )
    require(q.returncode == 0, "idle metadata snapshot failed")
    metrics = read(P / "gemma0-before-stop-counters.json")
    for metric in ["vllm:num_requests_running", "vllm:num_requests_waiting"]:
        vals = [float(v["value"]) for v in metrics["series"] if v["metric"] == metric]
        require(vals and all(v == 0 for v in vals), "gemma0 not idle")
    require(
        all(
            not Path(j["output"]).exists() and not Path(j["temp_dir"]).exists()
            for j in plan["jobs"]
        ),
        "output already exists",
    )
    names = [j["run_argv"][j["run_argv"].index("--name") + 1] for j in plan["jobs"]]
    for name in names:
        q = subprocess.run(
            ["docker", "container", "inspect", name], capture_output=True
        )
        require(q.returncode != 0, "collector name already exists")
    Path(plan["jobs"][0]["output"]).mkdir()
    for j in plan["jobs"]:
        Path(j["temp_dir"]).mkdir()
    state["status"] = "stopping_exact_idle_gemma0"
    state["stop_requested_at"] = stamp()
    publish()
    stopped = True
    q = subprocess.run(["docker", "stop", "--time", "90", c["Id"]], capture_output=True)
    require(q.returncode == 0, "gemma0 stop failed")
    require(not guard.inspect(c["Id"])["State"]["Running"], "gemma0 still running")
    q = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = [x.strip().split(", ") for x in q.stdout.splitlines()]
    row = next(x for x in rows if x[0] == "0")
    state["gpu0_after_stop"] = {
        "index": 0,
        "uuid": row[1],
        "total_MiB": int(row[2]),
        "free_MiB": int(row[3]),
    }
    require(int(row[3]) >= 70 * 1024, "collector existing70GiB free requirement")
    state["status"] = "collecting"
    publish()
    collection_deadline = time.monotonic() + 900
    for job in plan["jobs"]:
        validate(False)
        require(time.monotonic() < collection_deadline, "collection window exhausted")
        active_job = job
        name = job["run_argv"][job["run_argv"].index("--name") + 1]
        rec = {
            "label": job["label"],
            "status": "running",
            "started_at": stamp(),
            "started_epoch": time.time(),
            "output": job["output"],
            "argv": job["run_argv"],
        }
        state["jobs"].append(rec)
        publish()
        with Path(job["log"]).open("xb") as log:
            active_proc = subprocess.Popen(
                job["run_argv"], stdout=log, stderr=subprocess.STDOUT
            )
            deadline = min(
                time.monotonic() + job["timeout_seconds"], collection_deadline
            )
            while active_proc.poll() is None:
                require(time.monotonic() < deadline, "bounded collector elapsed")
                if active_cid is None:
                    x = subprocess.run(
                        ["docker", "inspect", name], capture_output=True, text=True
                    )
                    if x.returncode == 0:
                        found = json.loads(x.stdout)[0]
                        require(
                            found["Image"] == plan["image"], "collector image mismatch"
                        )
                        active_cid = found["Id"]
                        rec["container_id"] = active_cid
                        publish()
                time.sleep(2)
            rc = active_proc.returncode
        active_proc = None
        active_cid = None
        active_job = None
        rec.update(
            returncode=rc,
            finished_at=stamp(),
            finished_epoch=time.time(),
            log_sha256=sha(job["log"]),
        )
        require(rc == 0, "collector nonzero exit " + job["label"])
        out = Path(job["output"])
        summary = read(out / "summary.json")
        require(
            summary["status"] == "completed"
            and len(summary["labels"]) == plan["cases"]["folds"]
            and [c["id"] for c in summary["candidates"]]
            == [c["id"] for c in plan["selected_candidates"]],
            "finite completion incomplete",
        )
        require(
            summary["source_stable"]
            and summary["base_restoration"]["matched"]
            and summary["base_restoration"]["storage_bytes_exact"],
            "finite restoration proof failed",
        )
        require(
            set(summary["selected_byte_proofs"])
            == {
                "base",
                "restored_base",
                *[c["id"] for c in plan["selected_candidates"]],
            },
            "selected byte proof coverage differs",
        )
        require(
            all(
                p["storage_bytes_exact"]
                for p in summary["selected_byte_proofs"].values()
            ),
            "selected byte proof failed",
        )
        require(
            sha(out / "frame-evidence.json")
            == plan["frame"]["historical_evidence_sha256"],
            "finite frame differs",
        )
        rec.update(
            status="completed",
            summary_sha256=sha(out / "summary.json"),
            provenance_sha256=sha(out / "provenance.json"),
            frame_sha256=sha(out / "frame-evidence.json"),
            finite_seconds=summary["elapsed_seconds"],
            base_restoration=summary["base_restoration"],
            source_stable=True,
        )
        publish()
        validate(False)
    state["finite_completed"] = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-plan", type=Path, default=P / "owner-plan.json")
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--authorization-sha")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    require(
        args.owner_plan.name == "owner-plan.json", "owner-plan.json filename required"
    )
    P = args.owner_plan.resolve().parent
    PLAN_SHA = args.plan_sha
    AUTH_SHA = args.authorization_sha
    require(sha(P / "owner-plan.json") == PLAN_SHA, "plan SHA mismatch")
    if not args.apply:
        print(
            json.dumps(
                {
                    "status": "preview_no_operations",
                    "plan_sha256": PLAN_SHA,
                    "runtime_sha256": sha(__file__),
                    "jobs": 1,
                    "candidates": len(
                        read(P / "owner-plan.json")["selected_candidates"]
                    ),
                    "service_changes": 0,
                    "model_calls": 0,
                }
            )
        )
        raise SystemExit(0)
    require(AUTH_SHA is not None, "exact root authorization SHA required")
    owner_lock = (P / "owner.lock").open("a")
    fcntl.flock(owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state["plan_sha256"] = PLAN_SHA
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
        signal.signal(sig, stop_signal)
    error = None
    try:
        run()
    except BaseException as exc:
        error = type(exc).__name__
        state.update(status="failed", error_type=error, failed_at=stamp())
        publish()
    finally:
        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if (
                active_cid is None
                and active_proc is not None
                and active_job is not None
            ):
                # A signal can precede the first CID poll. The fresh owner-labeled name is
                # inspected only to recover the exact owned container, never another service.
                name = active_job["run_argv"][
                    active_job["run_argv"].index("--name") + 1
                ]
                q = subprocess.run(
                    ["docker", "inspect", name], capture_output=True, text=True
                )
                if q.returncode == 0:
                    owned = json.loads(q.stdout)[0]
                    if owned["Image"] == plan["image"] and owned["Config"].get(
                        "Labels", {}
                    ).get("kairyu.observation.owner") == plan.get(
                        "owner_label", "current-ad-functional-six-finite-v1"
                    ):
                        active_cid = owned["Id"]
            if active_cid:
                subprocess.run(
                    ["docker", "stop", "--time", "30", active_cid], capture_output=True
                )
            if active_proc:
                try:
                    active_proc.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    active_proc.terminate()
                    active_proc.wait(timeout=15)
        except BaseException as exc:
            state["collector_cleanup_error_type"] = type(exc).__name__
            error = error or type(exc).__name__
        finally:
            if stopped:
                state["status"] = "restoring_same_gemma0"
                state["restore_started_at"] = stamp()
                try:
                    publish()
                except BaseException as exc:
                    state["restore_status_write_error_type"] = type(exc).__name__
                    error = error or type(exc).__name__
                try:
                    cid = plan["serving_baseline"]["gemma-0"]["id"]
                    q = subprocess.run(["docker", "start", cid], capture_output=True)
                    require(q.returncode == 0, "same CID restart failed")
                    deadline = time.monotonic() + 600
                    while time.monotonic() < deadline:
                        snap = snapshot("gemma-0")
                        if snap["running"] and snap["health"] == "healthy":
                            break
                        time.sleep(2)
                    require(
                        canonical(snap)
                        == canonical(plan["serving_baseline"]["gemma-0"]),
                        "same CID restore not healthy/exact",
                    )
                    validate(True)
                    state["same_container_restored"] = True
                    state["restore_finished_at"] = stamp()
                except BaseException as exc:
                    state["restore_error_type"] = type(exc).__name__
                    error = error or type(exc).__name__
        state["status"] = (
            "completed"
            if not error
            and state.get("finite_completed")
            and state.get("same_container_restored")
            else "failed"
        )
        state["finished_at"] = stamp()
        state["finished_epoch"] = time.time()
        publish()
        if owned_execution:
            with (P / "completion.json").open("x") as final:
                json.dump(state, final, indent=2)
                final.write("\n")
        else:
            print(
                json.dumps(
                    {
                        "status": "preflight_failed_without_ownership",
                        "error_type": error,
                        "existing_receipts_unchanged": True,
                    }
                )
            )
    raise SystemExit(0 if state["status"] == "completed" else 1)
