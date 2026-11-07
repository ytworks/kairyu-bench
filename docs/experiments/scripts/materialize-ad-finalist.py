#!/usr/bin/env python3
"""Selected sparse finalist from a pinned saved parent: one create, one verify, optional one-service deploy."""

import argparse
import datetime
import hashlib
import struct
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

R = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
A = R / "analysis/code90-resume-20261106"
W = R / "worktrees/kairyu-bench-weight-tools"
CPU_PY = R / "infra/weight-tools-ci/bin/python"
PLAN = None
PLAN_SHA = None
FINITE = None
FINITE_OWNER_SHA = None
FINITE_PLAN_SHA = None
S_ID = None


def read(p):
    return json.loads(Path(p).read_bytes())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def require(value, message):
    if not value:
        raise ValueError(message)


def save(path, data, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open("x") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    else:
        fd, tmp = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(data, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


def load_helper(plan):
    helper = Path(
        next(p for p in plan["source_pins"] if p.endswith("/redeploy-replica.py"))
    )
    sys.path.insert(0, str(W / "src"))
    spec = importlib.util.spec_from_file_location("s_native_redeploy", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(snapshot):
    result = dict(snapshot)
    result["mounts"] = sorted(
        result.get("mounts", []), key=lambda x: json.dumps(x, sort_keys=True)
    )
    return result


def parent_check(plan, job, guard):
    parent = plan["parent"]
    require(
        sha(parent["state"]["path"]) == parent["state"]["sha256"],
        "incumbent state changed",
    )
    current = read(parent["state"]["path"])
    require(
        current["short_name"] == parent["short_name"]
        and current["checkpoint_path"] == parent["checkpoint_path"]
        and current["control_tree_sha256"] == parent["tree_sha256"],
        "declared parent mismatch",
    )
    require(
        sha(parent["freeze"]["path"]) == parent["freeze"]["sha256"],
        "parent freeze changed",
    )
    require(
        guard.inventory(Path(parent["checkpoint_path"]))
        == parent["source_inventory_stat"],
        "parent inventory changed",
    )
    require(sha(job["spec"]["path"]) == job["spec"]["sha256"], "finalist spec changed")
    spec = read(job["spec"]["path"])
    require(
        spec["source_checkpoint"] == parent["checkpoint_path"]
        and spec["base_tree_sha256"] == parent["tree_sha256"],
        "declared spec parent differs",
    )
    require(sha(PLAN) == PLAN_SHA, "materialization plan changed")
    for path, digest in plan["source_pins"].items():
        require(sha(path) == digest, "used source changed: " + path)


def finite_check(plan, job, auth, guard):
    require(
        sha(FINITE / "owner-plan.json") == FINITE_OWNER_SHA
        and sha(FINITE / "plan.json") == FINITE_PLAN_SHA,
        "finite plans changed",
    )
    owner = read(FINITE / "owner-plan.json")
    require(
        owner["parent"]["sha256"] == plan["parent"]["state"]["sha256"],
        "finite parent differs",
    )
    completion_pin = auth["finite_completion"]
    summary_pin = auth["finite_summary"]
    require(
        Path(completion_pin["path"]) == FINITE / "completion.json"
        and sha(completion_pin["path"]) == completion_pin["sha256"],
        "finite completion pin",
    )
    require(
        Path(summary_pin["path"]) == FINITE / "replay/summary.json"
        and sha(summary_pin["path"]) == summary_pin["sha256"],
        "finite summary pin",
    )
    completed = read(completion_pin["path"])
    summary = read(summary_pin["path"])
    require(
        completed["status"] == "completed"
        and completed["finite_completed"] is True
        and completed["same_container_restored"] is True
        and completed["plan_sha256"] == FINITE_OWNER_SHA,
        "finite owner not closed/restored",
    )
    require(
        summary["status"] == "completed"
        and summary["source_stable"] is True
        and summary["error_type"] is None,
        "finite source not stable",
    )
    restored = summary["base_restoration"]
    require(
        restored["matched"] is True and restored["storage_bytes_exact"] is True,
        "finite base restoration proof",
    )
    require(
        summary["selected_byte_proofs"][S_ID]["storage_bytes_exact"] is True,
        "S selected-byte proof",
    )
    match = [c for c in read(FINITE / "plan.json")["candidates"] if c["id"] == S_ID]
    require(
        len(match) == 1
        and match[0]["spec"] == job["spec"]["path"]
        and match[0]["sha256"] == job["spec"]["sha256"],
        "finite S differs",
    )
    for row in plan["screen_evidence"].values():
        require(sha(row["path"]) == row["sha256"], "screen evidence changed")
    screen = read(plan["screen_evidence"]["completion"]["path"])
    require(
        screen["status"] == "completed"
        and screen["screen_completed"] is True
        and screen["final_source_serving_checks"] is True,
        "screen not closed",
    )
    require(
        set(screen["restored"]) == {"gemma-0", "gemma-1", "gemma-2", "gemma-3"}
        and all(
            x.get("same_cid_exact_healthy") is True for x in screen["restored"].values()
        ),
        "screen originals not restored",
    )
    o2 = guard.snapshot(guard.inspect(owner["serving_baseline"]["gemma-2"]["id"]))
    require(
        plan["post_screen_serving_baseline"]["gemma-2"]["id"]
        == owner["serving_baseline"]["gemma-2"]["id"],
        "O2 CID changed",
    )
    require(
        canonical(o2) == canonical(plan["post_screen_serving_baseline"]["gemma-2"])
        and o2["running"]
        and o2["health"] == "healthy",
        "O2 not exact post-screen restored healthy CID",
    )
    return owner


def protected(guard, owner, target):
    result = {"replicas": {}, "embeddings": {}}
    for service in [
        s for s in ["gemma-3", "gemma-0", "gemma-2", "gemma-1"] if s != target
    ]:
        result["replicas"][service] = canonical(
            guard.snapshot(guard.inspect("gemma-4-31b-8gpu-" + service + "-1"))
        )
    for row in owner["embedding_baseline"]:
        c = guard.inspect(row["id"])
        result["embeddings"][row["id"]] = {
            "id": c["Id"],
            "image_id": c["Image"],
            "running": c["State"]["Running"],
            "started_at": c["State"]["StartedAt"],
            "config_sha256": guard.digest(c["Config"]),
            "host_config_sha256": guard.digest(c["HostConfig"]),
            "mounts_sha256": guard.digest(
                sorted(
                    c["Mounts"],
                    key=lambda x: (
                        x.get("Destination", ""),
                        x.get("Source", ""),
                        x.get("Type", ""),
                    ),
                )
            ),
        }
    return result


def selected_manifest_parity(manifest, plan, summary):
    pin = plan["selected_byte_expectations"]
    require(sha(pin["path"]) == pin["sha256"], "selected expectations changed")
    expected = read(pin["path"])
    grouped = {}
    for patch in manifest["edits"]:
        require(patch["dtype"] == "BF16", "BF16 patch required")
        before = bytes.fromhex(patch["before_hex"])
        after = bytes.fromhex(patch["after_hex"])
        require(len(before) == len(after) == patch["size"], "saved patch size differs")
        for index in patch["changed_indices"]:
            offset = 0 if len(patch["shape"]) == 2 and patch["size"] == 2 else index * 2
            require(offset + 2 <= len(before), "patch index outside stored bytes")
            grouped.setdefault(patch["tensor"], []).append(
                (index, before[offset : offset + 2], after[offset : offset + 2])
            )
    actual = []
    for name, parts in sorted(grouped.items()):
        parts.sort(key=lambda p: p[0])
        require(len({p[0] for p in parts}) == len(parts), "duplicate saved coordinates")

        def h(data):
            return hashlib.sha256(data).hexdigest()

        actual.append(
            {
                "tensor": name,
                "selected_count": len(parts),
                "indices_sha256": h(b"".join(struct.pack("<q", p[0]) for p in parts)),
                "original_bf16_sha256": h(b"".join(p[1] for p in parts)),
                "candidate_bf16_sha256": h(b"".join(p[2] for p in parts)),
            }
        )
    require(
        actual == expected["edited_subset_tensors"],
        "saved manifest differs from measured candidate selected BF16 targets",
    )
    proof = summary["selected_byte_proofs"][S_ID]
    normalized = [
        {
            **{k: v for k, v in row.items() if k != "actual_bf16_sha256"},
            "candidate_bf16_sha256": row["actual_bf16_sha256"],
        }
        for row in proof["tensors"]
    ]
    require(
        normalized == expected["finite_union_tensors"] and proof["storage_bytes_exact"],
        "finite actual union targets differ",
    )
    return {
        "selected_tensors": len(actual),
        "selected_elements": sum(x["selected_count"] for x in actual),
        "finite_union_tensors": len(normalized),
        "finite_union_selected_elements": sum(x["selected_count"] for x in normalized),
        "saved_manifest_verified_against_actual_files": True,
        "finite_selected_byte_targets_exact": True,
        "extra_payload_scan": False,
        "expectations": pin,
    }


def main():
    global PLAN, PLAN_SHA, FINITE, FINITE_OWNER_SHA, FINITE_PLAN_SHA, S_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--save-only", action="store_true")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    PLAN = args.plan
    PLAN_SHA = args.plan_sha256
    os.umask(0o077)
    sys.dont_write_bytecode = True
    require(os.getuid() == 1003, "ordinary UID1003 required")
    require(
        Path(sys.executable).absolute() == CPU_PY,
        "invoke using existing managed CPU_PY",
    )
    require(sha(PLAN) == PLAN_SHA, "plan mismatch")
    plan = read(PLAN)
    require(len(plan["jobs"]) == 1, "one finalist per owner")
    job = plan["jobs"][0]
    require(
        job["label"] in {"AN", "AO", "AP", "AQ", "AR", "AS", "AV", "BA"},
        "unknown finalist",
    )
    guard = load_helper(plan)
    FINITE = Path(plan["finite_owner"]["path"]).parent
    FINITE_OWNER_SHA = plan["finite_owner"]["sha256"]
    FINITE_PLAN_SHA = plan["finite_plan"]["sha256"]
    S_ID = plan["candidate_id"]
    require(
        plan["target_slot"] in (0, 3)
        and job["service"] == "gemma-" + str(plan["target_slot"]),
        "finalist service mismatch",
    )
    parent_check(plan, job, guard)
    out = Path(job["metadata_dir"])
    candidate = Path(job["candidate"])
    manifest = Path(job["manifest"])
    require(
        not any(os.path.lexists(p) for p in [out, candidate, manifest]),
        "fresh outputs required",
    )
    require(
        shutil.disk_usage(candidate.parent).free
        >= plan["storage"]["required_free_before_each_create"],
        "copy plus reserve unavailable",
    )
    if not args.apply:
        print(
            json.dumps(
                {
                    "status": "metadata_preview",
                    "plan_sha256": PLAN_SHA,
                    "driver_sha256": sha(__file__),
                    "label": job["label"],
                    "create_verify_invocations": 0,
                    "checkpoint_writes": 0,
                    "service_changes": 0,
                }
            )
        )
        return
    require(
        args.authorization
        and args.authorization_sha256
        and sha(args.authorization) == args.authorization_sha256,
        "exact root authorization required",
    )
    auth = read(args.authorization)
    require(
        auth["status"] == "authorized"
        and auth["issued_by"] == "root"
        and auth["plan_sha256"] == PLAN_SHA
        and auth["driver_sha256"] == sha(__file__),
        "authorization identity",
    )
    require(
        auth["selected_label"] == job["label"]
        and auth["spec_sha256"] == job["spec"]["sha256"]
        and auth["parent_state_sha256"] == plan["parent"]["state"]["sha256"],
        "authorization candidate/source",
    )
    require(
        auth["scope"] == "create_verify_freeze_deploy_" + job["label"] + "_only",
        "authorization scope",
    )
    require(
        sha(auth["selection"]["path"]) == auth["selection"]["sha256"],
        "root selection changed",
    )
    owner = finite_check(plan, job, auth, guard)
    before = protected(guard, owner, job["service"])
    target_before = canonical(
        guard.snapshot(guard.inspect("gemma-4-31b-8gpu-" + job["service"] + "-1"))
    )
    require(
        target_before["checkpoint_path"] == plan["target_previous_checkpoint"],
        "finalist target previous checkpoint changed",
    )
    require(
        before["replicas"][plan.get("parent_service", "gemma-1")]["tree"]
        == plan["parent"]["tree_sha256"]
        and before["replicas"][plan.get("parent_service", "gemma-1")]["health"]
        == "healthy",
        "declared parent service not healthy",
    )
    # Reservation happens once after all authorization/closure checks. No resume or retry path.
    out.mkdir()
    record = {
        "schema_version": 1,
        "status": "starting",
        "pid": os.getpid(),
        "started_at": utc(),
        "started_epoch": time.time(),
        "scope": auth["scope"],
        "plan_sha256": PLAN_SHA,
        "driver_sha256": sha(__file__),
        "authorization": {
            "path": str(args.authorization),
            "sha256": args.authorization_sha256,
        },
        "host_uid": os.getuid(),
        "cpu_cores": plan["cpu_cores"],
        "jobs": [],
        "protected_before": before,
        "target_before": target_before,
        "benchmark_calls": 0,
        "authored_smoke_calls": 0,
    }

    def publish():
        save(out / "execution-status.json", record)

    env = {**os.environ, **plan["environment"]}

    def stable(check_target=True):
        parent_check(plan, job, guard)
        require(
            protected(guard, owner, job["service"]) == before,
            "protected services changed",
        )
        require(
            sha(args.authorization) == args.authorization_sha256,
            "root authorization changed",
        )
        if check_target:
            require(
                canonical(
                    guard.snapshot(
                        guard.inspect("gemma-4-31b-8gpu-" + job["service"] + "-1")
                    )
                )
                == target_before,
                "finalist target changed before deployment",
            )

    def run_phase(label, argv):
        require(str(CPU_PY) in argv, "CPU_PY missing")
        r = {
            "label": label,
            "status": "running",
            "argv": argv,
            "started_at": utc(),
            "started_epoch": time.time(),
        }
        record["status"] = label
        record["jobs"].append(r)
        publish()
        output = out / (label + "-output.json")
        error = out / (label + ".log")
        try:
            with output.open("xb") as stdout, error.open("xb") as stderr:
                cp = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr)
                try:
                    cp.wait()
                except BaseException:
                    if cp.poll() is None:
                        cp.send_signal(signal.SIGTERM)
                        try:
                            cp.wait(timeout=1300)
                        except subprocess.TimeoutExpired:
                            cp.kill()
                            cp.wait()
                    raise
            r.update(
                returncode=cp.returncode,
                finished_at=utc(),
                finished_epoch=time.time(),
                seconds=time.time() - r["started_epoch"],
                stdout_sha256=sha(output),
                stderr_sha256=sha(error),
            )
            require(cp.returncode == 0, label + " failed; no automatic retry")
            data = read(output)
            r["status"] = "completed"
            save(out / (label + "-completion.json"), r, True)
            publish()
            return data
        except BaseException as error:
            r.update(
                status="failed",
                error_type=type(error).__name__,
                finished_at=utc(),
                finished_epoch=time.time(),
            )
            if not (out / (label + "-completion.json")).exists():
                save(out / (label + "-completion.json"), r, True)
            raise

    publish()
    try:
        stable()
        created = run_phase("create", job["create_apply_argv"])
        require(created["applied"] is True, "create not applied")
        candidate_stat = guard.inventory(candidate)
        require(
            set(candidate_stat) == set(plan["parent"]["source_inventory_stat"]),
            "candidate inventory",
        )
        for rel, s in candidate_stat.items():
            require(
                s["uid"] == 1003
                and s["nlink"] == 1
                and not os.path.samefile(
                    candidate / rel, Path(plan["parent"]["checkpoint_path"]) / rel
                ),
                "candidate not private",
            )
        save(out / "post-create-inventory.json", candidate_stat, True)
        manifest_sha = sha(manifest)
        stable()
        verified = run_phase("verify", job["verify_argv"])
        require(
            verified["verified"] is True
            and verified["legacy_full_hashes_missing"] is False,
            "full native verify proof missing",
        )
        stable()
        require(
            guard.inventory(candidate) == candidate_stat
            and sha(manifest) == manifest_sha,
            "candidate/manifest changed during verify",
        )
        m = read(manifest)
        parity = selected_manifest_parity(m, plan, read(auth["finite_summary"]["path"]))
        save(out / "selected-byte-parity.json", parity, True)
        require(
            m["source"] == plan["parent"]["checkpoint_path"]
            and m["candidate"] == str(candidate)
            and m["spec_sha256"] == job["spec"]["sha256"],
            "manifest identity",
        )
        expected = {r["path"]: r for r in plan["parent"]["all_checkpoint_files"]}
        mf = {r["path"]: r for r in m["files"]}
        vf = {r["path"]: r for r in verified["files"]}
        require(
            len(mf) == len(m["files"]) == len(vf) == len(verified["files"]) == 9
            and set(expected) == set(mf) == set(vf) == set(candidate_stat),
            "exact9 full hashes required",
        )
        rows = []
        for rel in sorted(expected):
            require(
                mf[rel]["method"] == "private-copy"
                and mf[rel]["size"]
                == expected[rel]["size"]
                == candidate_stat[rel]["size"],
                "private copy layout",
            )
            require(
                vf[rel]["source_sha256"]
                == mf[rel]["source_sha256"]
                == expected[rel]["sha256"]
                and vf[rel]["candidate_sha256"] == mf[rel]["candidate_sha256"],
                "full file hash proof mismatch",
            )
            rows.append(
                {
                    "path": rel,
                    "size": mf[rel]["size"],
                    "sha256": vf[rel]["candidate_sha256"],
                }
            )
        freeze = {
            "schema_version": 1,
            "status": "verified_sparse_editor_full_file_hashes",
            "verified_at": utc(),
            "checkpoint_path": str(candidate),
            "control_tree_sha256": guard.digest(rows),
            "all_checkpoint_files": rows,
            "candidate_inventory_stat": candidate_stat,
            "source_checkpoint_path": plan["parent"]["checkpoint_path"],
            "source_tree_sha256": plan["parent"]["tree_sha256"],
            "source_inventory_stat": plan["parent"]["source_inventory_stat"],
            "parent_state": plan["parent"]["state"],
            "spec": job["spec"],
            "manifest": {"path": str(manifest), "sha256": manifest_sha},
            "plan": {"path": str(PLAN), "sha256": PLAN_SHA},
            "implementation_files": plan["source_pins"],
            "verify_receipt": {
                "path": str(out / "verify-completion.json"),
                "sha256": sha(out / "verify-completion.json"),
            },
            "verify_output": {
                "path": str(out / "verify-output.json"),
                "sha256": sha(out / "verify-output.json"),
            },
            "finite_summary": auth["finite_summary"],
            "finite_completion": auth["finite_completion"],
            "selected_byte_parity": parity,
            "actual_saved_edit_metrics": {
                k: m[k] for k in ["changed_elements", "changed_tensors", "delta_l2"]
            },
            "verification": {
                "verified": True,
                "scope": verified["scope"],
                "proof_kind": "existing sparse editor verify fresh full source/candidate SHA and original-byte restoration",
                "source_hashes_match_parent_freeze": True,
                "all_regular_files_verified": True,
                "freeze_added_payload_scan": False,
                "all_410_matrices_cpu_regenerated": False,
            },
        }
        save(out / "freeze.json", freeze, True)
        stable()
        require(
            guard.inventory(candidate) == candidate_stat,
            "candidate changed before deploy",
        )
        if not args.save_only:
            run_phase("deployment", job["redeploy_apply_argv"])
            receipt = read(out / "deployment/receipt.json")
            require(
                receipt["status"] == "completed"
                and receipt["service"] == job["service"]
                and receipt["candidate_identity"]["tree_sha256"]
                == freeze["control_tree_sha256"],
                "deployment incomplete",
            )
        stable(args.save_only)
        require(
            guard.inventory(candidate) == candidate_stat,
            "candidate changed after save/deployment",
        )
        record.update(
            status="completed",
            deployment_performed=not args.save_only,
            freeze_sha256=sha(out / "freeze.json"),
            tree=freeze["control_tree_sha256"],
            deployment_receipt_sha256=sha(out / "deployment/receipt.json")
            if not args.save_only
            else None,
            protected_after=protected(guard, owner, job["service"]),
            actual_saved_edit_metrics=freeze["actual_saved_edit_metrics"],
        )
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        record.update(finished_at=utc(), finished_epoch=time.time())
        publish()
        save(out / "completion.json", record, True)
    print(
        json.dumps(
            {"status": record["status"], "output": str(out), "tree": record["tree"]}
        )
    )


if __name__ == "__main__":

    def interrupt(signum, frame):
        raise InterruptedError("owner interrupted")

    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
        signal.signal(sig, interrupt)
    main()
