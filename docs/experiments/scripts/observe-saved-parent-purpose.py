#!/usr/bin/env python3
"""Preview or execute pinned saved-parent authored observations; provisional observation does not adopt or edit."""

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
P = None
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
    "model_calls_scope": "Plan-pinned authored train cases on explicit saved observed parent; four teacher-forced derivative collectors only",
    "benchmark_calls": 0,
    "checkpoint_writes": 0,
}
plan = None
stopped = False
active = {}
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
        sha(P / "plan.json") == PLAN_SHA
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
        sha(plan["full_ci_results_path"]) == plan["full_ci_results_sha256"]
        and read(plan["full_ci_results_path"])["source_changed_during_checks"] == [],
        "CI receipt changed",
    )
    require(
        plan["stop_service"] == "gemma-2"
        and plan["protected_services"] == ["gemma-0", "gemma-1", "gemma-3"]
        and plan["physical_gpus"] == [4, 5],
        "fixed O2 observation lifecycle scope",
    )
    require(
        plan["status"] == "planned_not_executed_requires_root_authorization"
        and plan["source_freeze_required"] is False
        and len(plan["source_sha256"]) == 6,
        "source pins not bound",
    )
    require(
        [(j["collector_kind"], j["objective"]) for j in plan["jobs"]]
        == [
            ("native", "correct-logprob"),
            ("native", "margin"),
            ("matrix", "correct-logprob"),
            ("matrix", "margin"),
        ],
        "four-stage contract",
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
    observed = plan["observed_parent"]
    current = read(plan["parent"]["path"])
    require(
        observed["mode"] in {"accepted", "provisional"},
        "explicit observed-parent mode required",
    )
    require(
        data["checkpoint"]["path"] == observed["checkpoint_path"]
        and data["checkpoint"]["sha256"] == observed["tree_sha256"],
        "observed checkpoint differs from attestation",
    )
    require(
        data["source_freeze"] == observed["freeze"], "observed freeze binding differs"
    )
    require(
        observed["service"] in plan["protected_services"],
        "observed serving parent must be protected",
    )
    serving_source = plan["serving_baseline"][observed["service"]]
    require(
        serving_source["checkpoint_path"] == observed["checkpoint_path"]
        and serving_source["tree"] == observed["tree_sha256"],
        "observed service/source differs",
    )
    if observed["mode"] == "accepted":
        require(
            current["checkpoint_path"] == observed["checkpoint_path"]
            and current["control_tree_sha256"] == observed["tree_sha256"],
            "accepted observation is not current incumbent",
        )
    groups = plan["cases"]["groups"]
    require(
        plan["cases"]["purposes"] == len(groups)
        and plan["cases"]["folds"] == sum(len(counts) for counts in groups.values())
        and plan["cases"]["count"] == sum(sum(counts) for counts in groups.values()),
        "case count/group metadata differs",
    )
    require(
        set(plan["serving_baseline"]) == {"gemma-0", "gemma-1", "gemma-2", "gemma-3"}
        and len(plan["embedding_baseline"]) == 2,
        "complete serving/embedding binding required",
    )
    for job in plan["jobs"]:
        for argv in [job["preview_argv"], job["run_argv"]]:
            for flag, value in {
                "--checkpoint": observed["checkpoint_path"],
                "--cases": plan["cases"]["path"],
                "--checkpoint-attestation": att["path"],
                "--attestation-sha256": att["sha256"],
                "--output": job["output"],
            }.items():
                require(
                    argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value,
                    "collector observed-parent/data argument differs",
                )
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
            "observed-parent file stamp changed",
        )
    if serving:
        for svc, want in plan["serving_baseline"].items():
            require(same_service(svc, want), "serving snapshot changed " + svc)
    else:
        for svc in plan["protected_services"]:
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
    global plan, stopped, owned_execution
    os.umask(0o077)
    require(os.getuid() == 1003, "ordinary owner required")
    plan = read(P / "plan.json")
    validate()
    state["observed_parent"] = plan["observed_parent"]
    state["incumbent"] = plan["parent"]
    state["observation_is_adoption"] = False
    state["new_edit_eligibility"] = (
        "Use only if this exact observed path/tree is the accepted parent when creating a future edit; rebind current incumbent then."
    )
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
    c = guard.inspect(plan["serving_baseline"]["gemma-2"]["id"])
    ips = [
        n["IPAddress"]
        for n in c["NetworkSettings"]["Networks"].values()
        if n.get("IPAddress")
    ]
    require(len(ips) == 1, "gemma2 network mapping")
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
            str(P / "gemma2-before-stop-counters.json"),
            "--capture",
        ],
        capture_output=True,
    )
    require(q.returncode == 0, "idle metadata snapshot failed")
    metrics = read(P / "gemma2-before-stop-counters.json")
    for metric in ["vllm:num_requests_running", "vllm:num_requests_waiting"]:
        vals = [float(v["value"]) for v in metrics["series"] if v["metric"] == metric]
        require(vals and all(v == 0 for v in vals), "gemma2 not idle")
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
    outroot = Path(plan["jobs"][0]["output"]).parent
    outroot.mkdir()
    for j in plan["jobs"]:
        Path(j["temp_dir"]).mkdir()
    state["status"] = "stopping_exact_idle_gemma2"
    state["stop_requested_at"] = stamp()
    publish()
    stopped = True
    q = subprocess.run(["docker", "stop", "--time", "90", c["Id"]], capture_output=True)
    require(q.returncode == 0, "gemma2 stop failed")
    require(not guard.inspect(c["Id"])["State"]["Running"], "gemma2 still running")
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
    state["gpus_after_stop"] = []
    for index in [4, 5]:
        row = next(x for x in rows if x[0] == str(index))
        state["gpus_after_stop"].append(
            {
                "index": index,
                "uuid": row[1],
                "total_MiB": int(row[2]),
                "free_MiB": int(row[3]),
            }
        )
        require(int(row[3]) >= 70 * 1024, "collector existing70GiB free requirement")
    state["status"] = "collecting_parallel_families"
    publish()
    collection_deadline = time.monotonic() + 1800
    queues = {
        kind: [j for j in plan["jobs"] if j["collector_kind"] == kind]
        for kind in ["native", "matrix"]
    }

    def launch(kind):
        validate(False)
        require(time.monotonic() < collection_deadline, "collection window exhausted")
        job = queues[kind].pop(0)
        rec = {
            "label": job["label"],
            "collector_kind": kind,
            "physical_gpu": job["physical_gpu"],
            "status": "running",
            "started_at": stamp(),
            "started_epoch": time.time(),
            "output": job["output"],
            "argv": job["run_argv"],
        }
        state["jobs"].append(rec)
        publish()
        log = Path(job["log"]).open("xb")
        entry = {
            "job": job,
            "record": rec,
            "log": log,
            "proc": None,
            "cid": None,
            "deadline": min(
                time.monotonic() + job["timeout_seconds"], collection_deadline
            ),
        }
        active[kind] = entry
        entry["proc"] = subprocess.Popen(
            job["run_argv"], stdout=log, stderr=subprocess.STDOUT
        )

    for kind in ["native", "matrix"]:
        launch(kind)
    while active:
        for kind in list(active):
            entry = active[kind]
            job = entry["job"]
            rec = entry["record"]
            proc = entry["proc"]
            require(proc is not None, "collector process absent")
            if proc.poll() is None:
                require(
                    time.monotonic() < entry["deadline"],
                    "bounded collector elapsed " + job["label"],
                )
                if entry["cid"] is None:
                    name = job["run_argv"][job["run_argv"].index("--name") + 1]
                    x = subprocess.run(
                        ["docker", "inspect", name], capture_output=True, text=True
                    )
                    if x.returncode == 0:
                        found = json.loads(x.stdout)[0]
                        require(
                            found["Image"] == plan["image"]
                            and found["Config"]
                            .get("Labels", {})
                            .get("kairyu.observation.owner")
                            == "accepted-parent-purpose-v1",
                            "collector identity mismatch",
                        )
                        entry["cid"] = found["Id"]
                        rec["container_id"] = entry["cid"]
                        publish()
                continue
            entry["log"].close()
            rc = proc.returncode
            rec.update(
                returncode=rc,
                finished_at=stamp(),
                finished_epoch=time.time(),
                log_sha256=sha(job["log"]),
            )
            require(rc == 0, "collector nonzero exit " + job["label"])
            out = Path(job["output"])
            summary = read(out / "summary.json")
            frame = read(out / "frame-evidence.json")
            require(
                summary["status"] == "completed"
                and len(summary["folds"]) == plan["cases"]["folds"]
                and summary["source_values_unchanged"] is True,
                "collector completion incomplete",
            )
            require(
                summary["objective_kind"] == job["objective"]
                and summary["answer_batching"] == "serial",
                "collector objective/batching differs",
            )
            require(
                sum(row["examples"] for row in summary["folds"])
                == plan["cases"]["count"],
                "collected case count differs",
            )
            observed_groups = {
                (row["purpose"], row["fold"]): row["examples"]
                for row in summary["folds"]
            }
            expected_groups = {
                (purpose, fold): count
                for purpose, counts in plan["cases"]["groups"].items()
                for fold, count in enumerate(counts)
            }
            require(
                observed_groups == expected_groups,
                "collected purpose/fold case groups differ",
            )
            if kind == "native":
                require(
                    summary["coverage"]["native_tensors"] == 421
                    and summary["coverage"]["native_coordinates"] == 1331516,
                    "native inventory differs",
                )
            else:
                require(
                    summary["coverage"]["sampled_matrices"] == 410
                    and summary["coverage"]["sampled_coordinates"] == 209920
                    and summary["coverage"]["sampling_mode"] == "mlp-triplets",
                    "matrix inventory differs",
                )
            require(
                frame == read(plan["frame"]["historical_evidence_path"]),
                "observed-parent tokenizer frame differs",
            )
            rec.update(
                status="completed",
                summary_sha256=sha(out / "summary.json"),
                provenance_sha256=sha(out / "provenance.json"),
                frame_sha256=sha(out / "frame-evidence.json"),
                collector_seconds=summary["seconds"],
                runtime=summary["runtime"],
                folds=len(summary["folds"]),
                source_values_unchanged=True,
            )
            publish()
            del active[kind]
            validate(False)
            if queues[kind]:
                launch(kind)
        if active:
            time.sleep(2)
    state["collections_completed"] = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--authorization-sha")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    P = args.plan.resolve().parent
    PLAN_SHA = args.plan_sha
    AUTH_SHA = args.authorization_sha
    require(args.plan.resolve() == P / "plan.json", "bound file must be plan.json")
    require(sha(P / "plan.json") == PLAN_SHA, "plan SHA mismatch")
    if not args.apply:
        print(
            json.dumps(
                {
                    "status": "preview_no_operations",
                    "plan_sha256": PLAN_SHA,
                    "runtime_sha256": sha(__file__),
                    "jobs": 4,
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
            for entry in list(active.values()):
                try:
                    job = entry["job"]
                    proc = entry["proc"]
                    cid = entry["cid"]
                    if cid is None and proc is not None:
                        name = job["run_argv"][job["run_argv"].index("--name") + 1]
                        q = subprocess.run(
                            ["docker", "inspect", name], capture_output=True, text=True
                        )
                        if q.returncode == 0:
                            owned = json.loads(q.stdout)[0]
                            if (
                                owned["Image"] == plan["image"]
                                and owned["Config"]
                                .get("Labels", {})
                                .get("kairyu.observation.owner")
                                == "accepted-parent-purpose-v1"
                            ):
                                cid = owned["Id"]
                    if cid:
                        subprocess.run(
                            ["docker", "stop", "--time", "30", cid], capture_output=True
                        )
                    if proc:
                        try:
                            proc.wait(timeout=45)
                        except subprocess.TimeoutExpired:
                            proc.terminate()
                            proc.wait(timeout=15)
                except BaseException as exc:
                    state.setdefault("collector_cleanup_errors", []).append(
                        {
                            "label": entry["job"]["label"],
                            "error_type": type(exc).__name__,
                        }
                    )
                    error = error or type(exc).__name__
                finally:
                    try:
                        entry["log"].close()
                    except BaseException:
                        pass
        finally:
            if stopped:
                state["status"] = "restoring_same_gemma2"
                state["restore_started_at"] = stamp()
                try:
                    publish()
                except BaseException as exc:
                    state["restore_status_write_error_type"] = type(exc).__name__
                    error = error or type(exc).__name__
                try:
                    cid = plan["serving_baseline"]["gemma-2"]["id"]
                    q = subprocess.run(["docker", "start", cid], capture_output=True)
                    require(q.returncode == 0, "same CID restart failed")
                    deadline = time.monotonic() + 600
                    while time.monotonic() < deadline:
                        snap = snapshot("gemma-2")
                        if snap["running"] and snap["health"] == "healthy":
                            break
                        time.sleep(2)
                    require(
                        canonical(snap)
                        == canonical(plan["serving_baseline"]["gemma-2"]),
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
            and state.get("collections_completed")
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
