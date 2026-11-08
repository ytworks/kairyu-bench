#!/usr/bin/env python3
"""Fresh ordinary TP2 AD plus one or two saved AN/AR finalists on the fixed all14 cohort; no deployment or automatic adoption."""

import argparse
import fcntl
import importlib.util
import json
import os
import subprocess
import time
import datetime
from pathlib import Path
from types import SimpleNamespace

R = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
A = R / "analysis/code90-resume-20261106"
D = R / "runtime/code90-resume-20261106"
P = None
PLAN = None
PLAN_SHA = None
spec = importlib.util.spec_from_file_location(
    "fixed_four", D / "base-alignment-fourway-m-v1/dev90-four.py"
)
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
require, sha, read, save = F.require, F.sha, F.read, F.save
PY = R / "infra/weight-tools-ci/bin/python"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def canonical(s):
    return {
        **s,
        "mounts": sorted(
            s.get("mounts", []), key=lambda x: json.dumps(x, sort_keys=True)
        ),
    }


def metadata(ready):
    require(os.getuid() == 1003 and sha(PLAN) == PLAN_SHA, "UID/plan differs")
    plan = read(PLAN)
    require(plan.get("binding_complete") is True, "final selection/snapshots not bound")
    require(
        plan["selection"]["count"] == 778
        and sum(plan["selection"]["source_counts"].values()) == 778,
        "all14 cohort differs",
    )
    reference = read(plan["full14_protocol_reference"]["path"])
    require(
        sha(plan["full14_protocol_reference"]["path"])
        == plan["full14_protocol_reference"]["sha256"],
        "full14 reference changed",
    )
    require(
        plan["selection"]["source_counts"] == reference["selection_source_counts"]
        and plan["env"] == reference["env"],
        "full14 source counts/environment differs",
    )
    for job in plan["jobs"]:
        require(
            job["command"]
            == [
                str(Path(plan["source_path"]) / "kairyu-bench"),
                "wsp",
                "run",
                job["endpoint"],
                "--results-dir",
                str(R / "results"),
                "--tier",
                "10",
                "--split",
                "dev",
                "--label",
                job["tree"],
                "--run-id",
                job["run_id"],
                "--chat-concurrency",
                "32",
                "--logprob-concurrency",
                "8",
                "--agent-concurrency",
                "1",
                "--logprob-endpoint",
                job["endpoint"],
                "--observe-config",
                job["observer"]["path"],
            ],
            "fixed all14 command differs",
        )
    proof = {}
    for key, row in plan["pins"].items():
        require(sha(row["path"]) == row["sha256"], "bound proof differs: " + key)
        proof[key] = read(row["path"])
    require(proof["incumbent"]["short_name"] == "AD", "current incumbent is not AD")
    for rel, digest in plan["archived_source_files_sha256"].items():
        require(
            sha(Path(plan["source_path"]) / rel) == digest, "archive source differs"
        )
    require(len(plan["archived_source_files_sha256"]) == 76, "archive pins missing")
    for row in plan["operational_helpers"].values():
        require(sha(row["path"]) == row["sha256"], "helper differs")
    require(
        sha(D / "base-alignment-fourway-m-v1/dev90-four.py")
        == "1e5e2248d060d058aab0eb223966dc2421bdb7936601bb9490cc74d0121efad6",
        "base runtime differs",
    )
    labels = plan["candidate_roles"]
    require(
        1 <= len(labels) <= 2
        and len(set(labels)) == len(labels)
        and set(labels) <= {"AN", "AR"},
        "one or two fixed finalists required",
    )
    require(
        set(plan["selected_slots"]) == set(labels)
        and set(plan["selected_slots"].values()) <= {"gemma-0", "gemma-3"}
        and len(set(plan["selected_slots"].values())) == len(labels),
        "distinct selected slot mapping required",
    )
    require(
        len(plan["jobs"]) == len(labels) + 1
        and {j["role"] for j in plan["jobs"]} == {"ad"} | {x.lower() for x in labels},
        "exact selected arms required",
    )
    require(
        all(
            j["service"]
            == (
                {"ad": "gemma-1"}
                | {k.lower(): v for k, v in plan["selected_slots"].items()}
            )[j["role"]]
            for j in plan["jobs"]
        ),
        "role/service differs",
    )
    for label in labels:
        role = label.lower()
        job = next(j for j in plan["jobs"] if j["role"] == role)
        require(
            proof[label + "_pipeline"]["status"] == "completed"
            and proof[label + "_pipeline"]["freeze_sha256"]
            == plan["pins"][label + "_freeze"]["sha256"],
            "sparse create/verify/deploy incomplete",
        )
        require(
            proof[label + "_deployment"]["status"] == "completed"
            and canonical(proof[label + "_deployment"]["after"])
            == canonical(job["snapshot"]),
            "deployment differs",
        )
    finite = proof["finite"]
    require(
        finite["status"] == "completed"
        and finite["source_stable"] is True
        and finite["base_restoration"]["storage_bytes_exact"] is True,
        "finite restoration proof missing",
    )
    for identifier in [
        j["finite_candidate_id"] for j in plan["jobs"] if j["role"] != "ad"
    ]:
        require(
            finite["selected_byte_proofs"][identifier]["storage_bytes_exact"] is True,
            "selected finite byte proof missing",
        )
    require(
        proof["used_ci"]["all_passed"] is True
        and proof["used_ci"]["source_changed_during_checks"] == [],
        "used runtime CI differs",
    )
    guard, _ = F.U.helper()
    identities = {}
    for role, key in [("ad", "AD_freeze")] + [
        (label.lower(), label + "_freeze") for label in labels
    ]:
        job = next(j for j in plan["jobs"] if j["role"] == role)
        identities[role] = guard.identity(
            SimpleNamespace(
                candidate=Path(job["candidate"]), freeze=Path(plan["pins"][key]["path"])
            )
        )
        require(
            identities[role]["tree_sha256"] == job["tree"], "candidate tree differs"
        )
    for label in labels:
        nf = proof[label + "_freeze"]
        require(
            nf["status"] == "verified_sparse_editor_full_file_hashes"
            and nf["source_checkpoint_path"] == identities["ad"]["path"]
            and nf["source_tree_sha256"] == identities["ad"]["tree_sha256"]
            and nf["source_inventory_stat"] == identities["ad"]["stat"],
            "candidate must derive from exact accepted AD",
        )
        require(
            nf["selected_byte_parity"]["finite_selected_byte_targets_exact"] is True
            and nf["selected_byte_parity"][
                "saved_manifest_verified_against_actual_files"
            ]
            is True,
            "saved versus measured sparse bytes differ",
        )
        require(
            nf["manifest"]["sha256"] == plan["pins"][label + "_manifest"]["sha256"],
            "manifest mismatch",
        )
        require(
            nf["spec"]["sha256"] == plan["pins"][label + "_spec"]["sha256"]
            and nf["parent_state"]["sha256"] == plan["pins"]["incumbent"]["sha256"],
            "sparse spec/parent mismatch",
        )
    for filename in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "processor_config.json",
    ):
        require(
            len(
                {
                    next(r["sha256"] for r in x["files"] if r["path"] == filename)
                    for x in identities.values()
                }
            )
            == 1,
            "protocol metadata differs",
        )
    observer_policies = []
    for job in plan["jobs"]:
        op = job["observer"]
        require(sha(op["path"]) == op["sha256"], "observer config changed")
        config = read(op["path"])
        outdir = Path(config["output_dir"])
        require(
            config["schema_version"] == 1
            and Path(op["path"]).parent == outdir
            and outdir.is_dir(),
            "observer directory differs",
        )
        require(
            config["limits"] == {"storage_bytes": 268435456}, "observer quota differs"
        )
        require(
            len(config["capture_id"]) == 32
            and all(c in "0123456789abcdef" for c in config["capture_id"]),
            "observer capture ID differs",
        )
        require(
            job["command"].count("--observe-config") == 1
            and job["command"][job["command"].index("--observe-config") + 1]
            == op["path"],
            "observer command differs",
        )
        observer_policies.append(
            {k: v for k, v in config.items() if k not in ("capture_id", "output_dir")}
        )
    require(
        all(p == observer_policies[0] for p in observer_policies),
        "observer policy differs",
    )
    for job in plan["jobs"]:
        c = guard.inspect("gemma-4-31b-8gpu-" + job["service"] + "-1")
        snap = guard.snapshot(c)
        require(
            canonical(snap) == canonical(job["snapshot"])
            and snap["running"]
            and snap["health"] == "healthy"
            and snap["checkpoint_readonly"],
            "live serving differs",
        )
        ips = [
            n["IPAddress"]
            for n in c["NetworkSettings"]["Networks"].values()
            if n.get("IPAddress")
        ]
        require(
            len(ips) == 1 and job["endpoint"] == "http://" + ips[0] + ":8000/v1",
            "direct endpoint differs",
        )
        cmd = c["Config"]["Cmd"]
        require(
            cmd[cmd.index("--tensor-parallel-size") + 1] == "2"
            and json.loads(cmd[cmd.index("--speculative-config") + 1])[
                "num_speculative_tokens"
            ]
            == 4,
            "TP/MTP differs",
        )
    for field in (
        "image_id",
        "command_sha256",
        "environment_sha256",
        "user",
        "working_dir",
        "healthcheck_sha256",
        "ipc_mode",
        "networks",
    ):
        require(
            len(
                {json.dumps(j["snapshot"][field], sort_keys=True) for j in plan["jobs"]}
            )
            == 1,
            "pair serving config differs",
        )
    if ready:
        decision = read(plan["root_decision_path"])
        require(
            decision["status"] == "proceed_to_official_full14_after_saved_dev90"
            and decision["issued_by"] == "root"
            and decision["plan_sha256"] == PLAN_SHA
            and decision["runtime_sha256"] == sha(__file__)
            and decision["finite_summary_sha256"] == plan["pins"]["finite"]["sha256"]
            and decision["parent_incumbent_sha256"]
            == plan["pins"]["incumbent"]["sha256"],
            "root decision missing/different",
        )
        require(
            decision["selected_candidates"] == labels,
            "selected full14 finalists differ",
        )
        require(
            decision["ordinary_dev90_closure_sha256"]
            == plan["pins"]["ordinary_dev90_closure"]["sha256"],
            "saved dev90 decision closure differs",
        )
        for job in plan["ordinary_dev90_runs"]:
            run = R / "results" / job["run_id"]
            require(
                read(run / "run.json")["status"] == "completed"
                and (run / "report.json").is_file(),
                "saved ordinary dev90 not completed",
            )
    return plan, guard, identities


def counters(plan, job):
    target = P / (job["role"] + "-full14-before.json")
    tool = plan["operational_helpers"]["counters"]["path"]
    p = subprocess.run(
        [
            str(PY),
            tool,
            "snapshot",
            "--endpoint",
            job["endpoint"].removesuffix("/v1") + "/metrics",
            "--model",
            "gemma-4-31b",
            "--container-id",
            job["snapshot"]["id"],
            "--attestation-sha256",
            PLAN_SHA,
            "--output",
            str(target),
            "--capture",
        ],
        capture_output=True,
        timeout=20,
    )
    require(p.returncode == 0, "counter capture failed")
    data = read(target)
    for metric in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        values = [v["value"] for v in data["series"] if v["metric"] == metric]
        require(values and all(float(v) == 0 for v in values), "service not idle")
    return {"path": str(target), "sha256": sha(target)}


def require_results_mount(container):
    requested = R / "results"
    covering = []
    for mount in container["Mounts"]:
        destination = Path(mount["Destination"])
        if requested.is_relative_to(destination):
            covering.append(
                (len(destination.parts), mount, requested.relative_to(destination))
            )
    require(bool(covering), "results directory is not mounted")
    _, mount, relative = max(covering, key=lambda row: row[0])
    require(
        mount["Type"] == "bind"
        and mount["RW"] is True
        and Path(mount["Source"]) / relative == requested,
        "results path must map to the same writable host directory",
    )


def capture(plan, guard):
    pending = {j["run_id"]: j for j in plan["jobs"]}
    rows = {}
    # Build may take a minute. This watcher has no launch path and never reruns jobs.
    for _ in range(90):
        ids = guard.command(
            ["docker", "ps", "--no-trunc", "--format", "{{.ID}}"]
        ).split()
        for cid in ids:
            p = subprocess.run(
                ["docker", "inspect", cid], capture_output=True, text=True
            )
            if p.returncode:
                continue
            c = json.loads(p.stdout)[0]
            cmd = c["Config"].get("Cmd") or []
            matches = [r for r in pending if r in cmd]
            if not matches:
                continue
            require(len(matches) == 1, "ambiguous runner")
            job = pending.pop(matches[0])
            require(cmd == job["command"][1:], "actual command mismatch")
            im = json.loads(guard.command(["docker", "image", "inspect", c["Image"]]))[
                0
            ]
            env = c["Config"].get("Env") or []
            keys = [v.split("=", 1)[0] for v in env]
            safe = [
                v
                for v in env
                if not any(
                    k in v.split("=", 1)[0].upper()
                    for k in (
                        "TOKEN",
                        "SECRET",
                        "KEY",
                        "PASS",
                        "AUTH",
                        "CREDENTIAL",
                        "COOKIE",
                    )
                )
            ]
            normalized = [
                (
                    "__ENDPOINT__"
                    if v == job["endpoint"]
                    else "__TREE__"
                    if v == job["tree"]
                    else "__RUN_ID__"
                    if v == job["run_id"]
                    else "__OBSERVE_CONFIG__"
                    if v == job["observer"]["path"]
                    else v
                )
                for v in cmd
            ]
            row = {
                "run_id": job["run_id"],
                "container_id": c["Id"],
                "image_id": c["Image"],
                "rootfs": im["RootFS"],
                "full_image_config_sha256": guard.digest(im["Config"]),
                "full_host_config_sha256": guard.digest(c["HostConfig"]),
                "command": cmd,
                "normalized_command": normalized,
                "noncredential_environment_sha256": guard.digest(sorted(safe)),
                "duplicate_environment_keys": len(keys) != len(set(keys)),
                "user": c["Config"].get("User"),
                "working_dir": c["Config"].get("WorkingDir"),
                "runtime": c["HostConfig"].get("Runtime"),
                "network_mode": c["HostConfig"].get("NetworkMode"),
                "mounts": [
                    {k: v.get(k) for k in ("Type", "Source", "Destination", "RW")}
                    for v in c["Mounts"]
                ],
                "captured_at": now(),
                "credentials_persisted": False,
            }
            require(not row["duplicate_environment_keys"], "duplicate runner env keys")
            rows[job["role"]] = row
            save(
                P / "full14-actual-runner-oci.json",
                {
                    "status": "captured" if not pending else "partial_capture",
                    "rows": rows,
                    "pending_run_ids": list(pending),
                },
            )
            require_results_mount(c)
            # Fixed installed package files only; never run/result data or credentials.
            audit_code = "import hashlib,importlib.util,importlib.metadata,json,sys;from pathlib import Path;root=Path(importlib.util.find_spec('kairyu_bench').origin).parent;names=json.loads(sys.argv[1]);print(json.dumps({'source_files':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names},'python':sys.version,'packages':dict(sorted((x.metadata['Name'],x.version) for x in importlib.metadata.distributions() if x.metadata['Name']))}))"
            try:
                audit_run = subprocess.run(
                    [
                        "docker",
                        "exec",
                        cid,
                        "python",
                        "-B",
                        "-c",
                        audit_code,
                        json.dumps(sorted(plan["installed_source_files_sha256"])),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                require(audit_run.returncode == 0, "runner source metadata read failed")
                audit = json.loads(audit_run.stdout)
                require(
                    audit["source_files"] == plan["installed_source_files_sha256"],
                    "installed runner source/resources differ",
                )
                row["installed_source_audit"] = {"status": "matched_archive", **audit}
            except BaseException as error:
                row["installed_source_audit"] = {
                    "status": "unavailable_or_mismatch",
                    "error_type": type(error).__name__,
                }
                save(
                    P / "full14-actual-runner-oci.json",
                    {
                        "status": "source_attestation_failed_no_rerun",
                        "rows": rows,
                        "pending_run_ids": list(pending),
                    },
                )
                raise
            save(
                P / "full14-actual-runner-oci.json",
                {
                    "status": "captured" if not pending else "partial_capture",
                    "rows": rows,
                    "pending_run_ids": list(pending),
                },
            )
        if not pending:
            return rows
        time.sleep(1)
    return rows


def main():
    global PLAN_SHA, PLAN, P
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    PLAN = args.plan
    P = PLAN.resolve().parent
    PLAN_SHA = args.plan_sha256
    require(sha(PLAN) == PLAN_SHA, "plan changed")
    draft = read(PLAN)
    if not args.apply and not draft.get("binding_complete"):
        print(
            json.dumps(
                {
                    "status": "prepared_unbound_no_launch",
                    "plan_sha256": PLAN_SHA,
                    "candidate_roles": draft["candidate_roles"],
                    "selection_count": 778,
                    "Docker_calls": 0,
                    "HTTP_calls": 0,
                    "model_calls": 0,
                    "launches": 0,
                    "requires_saved_dev90_root_selection": True,
                }
            )
        )
        return
    plan, guard, ids = metadata(args.apply)
    if not args.apply:
        print(
            json.dumps(
                {
                    "status": "metadata_preview",
                    "plan_sha256": PLAN_SHA,
                    "run_ids": [j["run_id"] for j in plan["jobs"]],
                    "model_calls": 0,
                    "launches": 0,
                }
            )
        )
        return
    with (P / "full14-owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = P / "full14-three-launch.json"
        require(
            not os.path.lexists(path) and bool(os.environ.get("HF_TOKEN")),
            "existing launch or missing private auth",
        )
        for job in plan["jobs"]:
            F.U.require_absent(job["run_id"])
            require(not os.path.lexists(job["log"]), "log exists")
        before = {j["role"]: counters(plan, j) for j in plan["jobs"]}
        metadata(True)
        record = {
            "schema_version": 1,
            "status": "reserved",
            "plan_sha256": PLAN_SHA,
            "source_head": plan["source_head"],
            "source_path": plan["source_path"],
            "incumbent_sha256": plan["pins"]["incumbent"]["sha256"],
            "root_decision_sha256": sha(plan["root_decision_path"]),
            "launcher_sha256": sha(__file__),
            "jobs": plan["jobs"],
            "before_counters": before,
            "started_at": now(),
            "started_epoch": time.time(),
            "scope": "One fresh AD control plus selected saved finalists; identical numeric observer policy, no duplicate baseline or automatic follow-up.",
            "credentials_persisted": False,
        }
        logs = []
        try:
            for job in plan["jobs"]:
                logs.append(open(job["log"], "xb"))
            save(path, record, exclusive=True)
            for job, log in zip(record["jobs"], logs):
                F.U.require_absent(job["run_id"])
                child = subprocess.Popen(
                    job["command"],
                    cwd=R / "results",
                    env={**os.environ, **plan["env"]},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                job.update(pid=child.pid, started_at=now())
                save(path, record)
            record["status"] = "running"
            save(path, record)
            print(
                json.dumps(
                    {
                        "status": "running",
                        "jobs": [
                            {k: j[k] for k in ("role", "pid", "run_id")}
                            for j in record["jobs"]
                        ],
                    }
                ),
                flush=True,
            )
        finally:
            for log in logs:
                log.close()
        rows = capture(plan, guard)
        record["runner_capture_status"] = (
            "captured_all_selected"
            if len(rows) == len(plan["jobs"])
            else "incomplete_no_rerun"
        )
        save(path, record)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(
            json.dumps(
                {"status": "refused_or_failed", "error_type": type(error).__name__}
            ),
            flush=True,
        )
        raise SystemExit(1)
