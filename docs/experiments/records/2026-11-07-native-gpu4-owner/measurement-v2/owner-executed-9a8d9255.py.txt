#!/usr/bin/env python3
"""Preview, or explicitly own two native collections and restore exact gemma-2."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
TARGET = "gemma-4-31b-8gpu-gemma-2-1"
CID = "fd4ec2eb9d7deba2ce3e790add6b50c0ee79a6390495320d8697873b32957bcf"
IMAGE = "sha256:af9e746b913088cbf6c4464a31d6904a6b7e1d3ba62f0895362ddf13595f7abe"
TREE = "5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf"
OBJECTIVES = ("correct-logprob", "margin")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, data):
    with Path(path).open("x") as out:
        json.dump(data, out, indent=2, allow_nan=False)
        out.write("\n")


def pinned(pin):
    require(sha(pin["path"]) == pin["sha256"], "pinned file changed")
    return read(pin["path"])


def native_args(plan, objective, run=False):
    args = [str(Path(plan["source_root"]) / "scripts/experiments/gemma4-purpose/native.py"),
            "--checkpoint", plan["checkpoint"], "--cases", plan["cases"]["path"],
            "--output", str(Path(plan["output_root"]) / "work" / objective),
            "--checkpoint-attestation", plan["attestation"]["path"],
            "--attestation-sha256", plan["attestation"]["sha256"],
            "--mode", "chat-public-answer", "--objective", objective,
            "--answer-batching", "serial", "--max-cases-per-microbatch", "1",
            "--saved-activation-storage", "cpu-fp32", "--max-frame-tokens", "1024",
            "--text-attention-backend", "sdpa", "--cuda-memory-cap-gib", "88",
            "--execution-order", "longest_frame_first", "--device", "cuda:0",
            "--threads", "4", "--seed", "0", "--gpu-exclusive-approved"]
    return args + (["--run"] if run else [])


def command(plan, objective, plan_sha):
    require(re.fullmatch(r"[0-9a-f]{64}", plan_sha), "plan SHA label required")
    work = Path(plan["output_root"]) / "work"
    args = ["docker", "run", "--rm", "--pull=never", "--network", "none",
            "--user", "1003:1003", "--gpus", "device=4", "--cpuset-cpus", "32-39",
            "--memory", "128g", "--memory-swap", "128g", "--pids-limit", "512",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only",
            "--shm-size", "1g", "--entrypoint", "/usr/bin/python3",
            "--name", plan["container_prefix"] + "-" + objective,
            "--cidfile", str(work / (objective + ".cid")),
            "--label", "kairyu.native.owner-plan-sha256=" + plan_sha,
            "--label", "kairyu.native.owner-root=" + plan["output_root"],
            "--mount", f"type=bind,src={ROOT},dst={ROOT},readonly",
            "--mount", f"type=bind,src={work},dst={work}",
            "--mount", f"type=bind,src={work / 'tmp'},dst=/tmp", "--workdir", str(work)]
    env = dict(PYTHONPATH=str(Path(plan["source_root"]) / "src"), PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HOME=str(work / "cache/hf"),
               TORCH_HOME=str(work / "cache/torch"), TRITON_CACHE_DIR=str(work / "cache/triton"),
               TORCHINDUCTOR_CACHE_DIR=str(work / "cache/inductor"), CUDA_CACHE_PATH=str(work / "cache/cuda"),
               XDG_CACHE_HOME=str(work / "cache"), TMPDIR="/tmp", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    for key, value in env.items():
        args.extend(["--env", key + "=" + value])
    return args + [IMAGE, *native_args(plan, objective, run=True)]


def validate(plan, for_run=False):
    require(plan["schema_version"] == 1 and plan["maximum_collection_seconds"] == 900, "fixed collection scope")
    require(plan["target"]["id"] == CID and plan["target"]["image"] == IMAGE, "exact target/image required")
    protected_ids = {v["id"] for v in plan["protected"].values()}
    require(len(plan["protected"]) == len(protected_ids) == 26 and CID not in protected_ids, "other26 unique required")
    require(re.fullmatch(r"[a-z0-9][a-z0-9_.-]{1,90}", plan["container_prefix"]), "safe owned-container prefix required")
    require(plan["checkpoint"] == str(ROOT / "checkpoints/code90-current-ad-helper-av-v1"), "AV checkpoint required")
    for key in ("source_root", "output_root"):
        path = Path(plan[key])
        require(path.is_absolute() and path.resolve().is_relative_to(ROOT), "NVMe task path required")
    require(Path(plan["output_root"]).resolve().is_relative_to(ROOT / "analysis"), "analysis output required")
    require(str(Path(__file__).resolve()) in plan["source_pins"], "owner source must be pinned")
    work = Path(plan["output_root"]).resolve() / "work"
    inputs = [plan[k]["path"] for k in ("cases", "frame", "attestation", "native_behavior_result")]
    inputs += list(plan["source_pins"]) + [v["path"] for v in plan["cpu_previews"].values()]
    require(all(not Path(p).resolve().is_relative_to(work) for p in inputs), "inputs must stay outside writable work")
    source = Path(plan["source_root"])
    recipe = source / "scripts/experiments/gemma4-purpose"
    required = {str((recipe / name).resolve()) for name in ("native.py", "recipe_common.py", "objective_variants.py")}
    required |= {str(p.resolve()) for p in (source / "src").rglob("*.py")}
    required.add(str(Path(__file__).with_name("fresh-f01-finite.py").resolve()))
    require(len(required) > 4 and required <= set(plan["source_pins"]), "complete executable source pins required")
    for path, expected in plan["source_pins"].items():
        require(sha(path) == expected, "source changed: " + path)
    att = pinned(plan["attestation"])["checkpoint"]
    require(att["path"] == plan["checkpoint"] and att["sha256"] == TREE and att["hash_files"] is True
            and att["identity_level"] == "content_sha256" and len(att["files"]) == 9, "full AV attestation required")
    for row in att["files"]:
        st = (Path(plan["checkpoint"]) / row["path"]).stat()
        actual = dict(device=st.st_dev, inode=st.st_ino, mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)
        require(st.st_size == row["size"] and actual == row["stat"], "checkpoint stat changed")
    cases = pinned(plan["cases"])
    require(cases.get("split", "train") == "train" and cases.get("gradient_allowed") is not False,
            "transfer/evaluation envelope prohibited")
    groups = cases.get("cases", cases)
    require(list(groups) == list("ABCDEF"), "exact six train groups required")
    folds = [(group, i, rows) for group, fs in groups.items() for i, rows in enumerate(fs)]
    require(len(folds) == 12 and all(len(rows) == 1 for _, _, rows in folds), "12 singleton train folds required")
    expected = [(g, i, "behavior-" + g + str(i + 1)) for g in "ABCDEF" for i in range(2)]
    require([(g, i, rows[0]["id"]) for g, i, rows in folds] == expected
            and all(rows[0].get("split") == "train" and rows[0].get("gradient_allowed") is not False
                    for _, _, rows in folds), "exact train12 IDs/row split required; transfer/duplicates prohibited")
    ids = [row[2] for row in expected]
    frames = pinned(plan["frame"])
    observed = {(g, i, row["id"]) for g, fs in frames["frames"].items() for i, rows in enumerate(fs) for row in rows}
    require(observed == {(g, i, rows[0]["id"]) for g, i, rows in folds}
            and sum(len(rows) for fs in frames["frames"].values() for rows in fs) == 12, "exact train frame join required")
    require(all(row["prefix_tokens"] + max(row["correct_tokens"], row["foil_tokens"]) <= 1024
                for fs in frames["frames"].values() for rows in fs for row in rows), "complete frame exceeds1024")
    for objective in OBJECTIVES:
        preview = pinned(plan["cpu_previews"][objective])
        require(preview["returncode"] == 0 and preview["model_calls"] == 0
                and preview["collector_arguments"] == native_args(plan, objective), "CPU metadata preview mismatch")
    pinned(plan["native_behavior_result"])
    if for_run:
        require(plan.get("approved_for_execution") is True, "root-reviewed approval required")
    return ids


def helper(plan):
    path = Path(__file__).with_name("fresh-f01-finite.py")
    require(plan["source_pins"].get(str(path.resolve())) == sha(path), "helper must be pinned")
    spec = importlib.util.spec_from_file_location("native_owner_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def services(plan, guard, serving=True):
    for name, expected in plan["protected"].items():
        require(guard.snapshot(name) == expected, "protected container changed: " + name)
    if serving:
        require(guard.snapshot(TARGET) == plan["target"], "target changed before stop")


def idle(guard):
    result = guard.target_metrics(CID)
    require(all(v == 0 for values in result["queues"].values() for v in values), "target not idle")
    return result


def cleanup_collectors(plan, guard, status, plan_sha, phase):
    """Fresh cidfile/daemon checks on every pass; no cached absence proof."""
    safe = True
    status.setdefault("collector_cleanup", [])
    for objective in OBJECTIVES:
        name = plan["container_prefix"] + "-" + objective
        cidfile = Path(plan["output_root"]) / "work" / (objective + ".cid")
        cid = cidfile.read_text().strip() if cidfile.exists() else None
        record = dict(objective=objective, cid=cid, phase=phase)
        status["collector_cleanup"].append(record)
        def lookup():
            try:
                owned = guard.inspect(cid or name)
            except subprocess.CalledProcessError:
                # A failed inspect alone cannot prove absence (daemon might be down).
                all_ids = guard.docker("ps", "-aq", "--no-trunc").split()
                names = guard.docker("ps", "-aq", "--no-trunc", "--filter", "name=^/" + name + "$").split()
                require(not names and (cid is None or cid not in all_ids), "collector absence not proven")
                return None
            labels = owned["Config"].get("Labels", {}) or {}
            require((cid is None or owned["Id"] == cid) and owned["Image"] == IMAGE
                    and owned["Name"] == "/" + name
                    and labels.get("kairyu.native.owner-plan-sha256") == plan_sha
                    and labels.get("kairyu.native.owner-root") == plan["output_root"], "collector ownership differs")
            return owned
        try:
            owned = lookup()
            if owned is not None and owned["State"]["Running"]:
                try:
                    guard.docker("stop", "--time", "30", owned["Id"])
                except subprocess.CalledProcessError:
                    record["stop_failed"] = True
                owned = lookup()  # Recheck ownership immediately before any kill.
                if owned is not None and owned["State"]["Running"]:
                    guard.docker("kill", owned["Id"])
                    record["owned_kill"] = True
            owned = lookup()
            require(owned is None or not owned["State"]["Running"], "owned collector remains active")
            record["proven_inactive"] = True
        except BaseException as error:
            safe = False
            record["error_type"] = type(error).__name__
    return safe


def finish(plan, guard, process, stopped, status, plan_sha):
    """Client exit must precede a fresh final collector inactivity proof."""
    initial_safe = cleanup_collectors(plan, guard, status, plan_sha, "before_client_wait")
    client_finished = False
    try:
        if process is not None and process.poll() is None:
            process.wait(timeout=45)
        client_finished = process is None or process.poll() is not None
        require(client_finished, "collector client still running")
    except BaseException as error:
        status["cleanup_error_type"] = type(error).__name__
    final_safe = cleanup_collectors(plan, guard, status, plan_sha, "after_client_wait")
    safe = initial_safe and client_finished and final_safe
    status["collector_client_finished"] = client_finished
    if stopped and safe:
        try:
            guard.docker("start", CID)
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                actual = guard.snapshot(TARGET)
                require(all(actual[k] == v for k, v in plan["target"].items() if k not in
                            ("started_at", "running", "health")), "restored identity differs")
                if actual["running"] and actual["health"] == "healthy":
                    status["after_queues"] = idle(guard)
                    status["restored"] = True
                    break
                time.sleep(2)
            require(status["restored"], "restoration health deadline")
        except BaseException as error:
            status["restoration_error_type"] = type(error).__name__
    elif stopped:
        status["restoration_blocked"] = "collector inactivity/ownership not proven; original remains stopped"
    try:
        services(plan, guard, serving=False)
        status["protected_exact"] = True
    except BaseException as error:
        status["protected_error_type"] = type(error).__name__


def run(plan, expected_sha, plan_path):
    require(expected_sha == sha(plan_path), "explicit exact plan SHA required")
    require(plan.get("approved_for_execution") is True, "root-reviewed approval required")
    validate(plan, for_run=True)
    require(os.getuid() == 1003, "ordinary owner UID required")
    guard = helper(plan)
    out = Path(plan["output_root"])
    require(not (out / "work").exists() and not (out / "launch.json").exists(), "attempt already started")
    status = dict(pid=os.getpid(), started_epoch=time.time(), jobs=[], restored=False, protected_exact=False)
    with ExitStack() as stack:
        for path in (ROOT / "candidate-evaluation.lock", ROOT / "gpu4-native.lock"):
            lock = stack.enter_context(path.open("a+"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status["lock_proof"] = guard.lock_owner_proof(ROOT / "candidate-evaluation.lock", os.getpid(), 1003)
        validate(plan, for_run=True)
        services(plan, guard)
        status["before_queues"] = idle(guard)
        require(set(guard.docker("ps", "-aq", "--no-trunc").split()) ==
                {CID, *(v["id"] for v in plan["protected"].values())}, "container inventory changed")
        save(out / "launch.json", status)
        for path in (out / "work", out / "work/tmp", out / "work/cache"):
            path.mkdir()
        stopped, process = False, None
        def interrupted(signum, frame):
            raise InterruptedError("owner interrupted")
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, interrupted)
        try:
            stopped = True  # Stop may succeed even if its client is interrupted.
            guard.docker("stop", "--time", "90", CID)
            require(not guard.inspect(CID)["State"]["Running"], "target still running")
            free = int(subprocess.check_output(["nvidia-smi", "-i", "4", "--query-gpu=memory.free",
                                                "--format=csv,noheader,nounits"], text=True).strip())
            require(free >= 92 * 1024, "need88GiB allocator plus4GiB reserve")
            status["gpu4_free_mib"] = free
            deadline = time.monotonic() + 900
            for objective in OBJECTIVES:
                validate(plan, for_run=True)
                services(plan, guard, serving=False)
                require(time.monotonic() < deadline, "collection deadline")
                entry = dict(objective=objective, started_epoch=time.time(), command=command(plan, objective, expected_sha))
                status["jobs"].append(entry)
                save(out / (objective + "-start.json"), entry)
                with (out / (objective + ".log")).open("x") as log:
                    process = subprocess.Popen(entry["command"], stdout=log, stderr=subprocess.STDOUT)
                    entry["returncode"] = process.wait(timeout=max(.01, deadline - time.monotonic()))
                entry["finished_epoch"] = time.time()
                require(entry["returncode"] == 0, "collector failed")
                result = out / "work" / objective
                summary = read(result / "summary.json")
                require(summary["status"] == "completed" and summary["objective_kind"] == objective
                        and summary["source_values_unchanged"] is True and len(summary["folds"]) == 12
                        and sum(v["examples"] for v in summary["folds"]) == 12, "partial collector result")
                require(summary["coverage"]["native_tensors"] == 421
                        and summary["coverage"]["native_coordinates"] == 1331516, "native inventory differs")
                require(read(result / "frame-evidence.json") == pinned(plan["frame"]), "measured frame mismatch")
                entry["summary_sha256"] = sha(result / "summary.json")
                save(out / (objective + "-complete.json"), entry)
            validate(plan, for_run=True)
            status["collections_completed"] = True
        except BaseException as error:
            status["error_type"] = type(error).__name__
        finally:
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                signal.signal(sig, signal.SIG_IGN)
            finish(plan, guard, process, stopped, status, expected_sha)
            status["finished_epoch"] = time.time()
            status["elapsed_seconds"] = status["finished_epoch"] - status["started_epoch"]
            status["status"] = "completed" if (status.get("collections_completed") and status["restored"]
                and status["protected_exact"] and not any(k.endswith("error_type") for k in status)) else "failed"
            save(out / "completion.json", status)
        require(status["status"] == "completed", "preserved incomplete attempt; inspect completion")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--approved-plan-sha256")
    args = parser.parse_args()
    plan = read(args.plan)
    if args.run:
        try:
            result = run(plan, args.approved_plan_sha256, args.plan)
        except BaseException as error:
            out = Path(plan["output_root"])
            if out.is_dir() and out.resolve().is_relative_to(ROOT / "analysis"):
                save(out / f"failed-launch-{time.time_ns()}.json", dict(error_type=type(error).__name__,
                     plan_sha256=sha(args.plan), model_calls_claim="see launch/completion; preflight may have rejected"))
            raise
        print(json.dumps({k: result[k] for k in ("status", "elapsed_seconds", "restored", "protected_exact")}))
    else:
        ids = validate(plan)
        print(json.dumps(dict(status="preview_no_execution", plan_sha256=sha(args.plan), case_ids=ids,
                              commands=[command(plan, o, sha(args.plan)) for o in OBJECTIVES], model_calls=0, service_changes=0)))


if __name__ == "__main__":
    main()
