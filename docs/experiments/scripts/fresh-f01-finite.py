#!/usr/bin/env python3
"""Prepare, CPU-check, or explicitly run one subleased GPU6 AV/F01/AV replay."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import urllib.request


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    with Path(path).open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def docker(*args):
    return subprocess.run(["docker", *args], check=True, capture_output=True, text=True, timeout=120).stdout


def inspect(name):
    return json.loads(docker("inspect", name))[0]


def snapshot(name):
    d = inspect(name)
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return dict(id=d["Id"], image=d["Image"], running=d["State"]["Running"],
                health=d["State"].get("Health", {}).get("Status"),
                started_at=d["State"]["StartedAt"], config_sha256=digest(d["Config"]),
                host_config_sha256=digest(d["HostConfig"]),
                cpuset=d["HostConfig"]["CpusetCpus"],
                devices=d["HostConfig"]["DeviceRequests"])


def target_metrics(cid):
    d = inspect(cid)
    ips = [v["IPAddress"] for v in d["NetworkSettings"]["Networks"].values()]
    if len(ips) != 1 or not ips[0]:
        raise ValueError("expected one direct target endpoint")
    with urllib.request.urlopen("http://" + ips[0] + ":8000/metrics", timeout=10) as response:
        raw = response.read()
    values = {}
    for line in raw.decode().splitlines():
        for key in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
            if line.startswith(key + "{") or line.startswith(key + " "):
                values.setdefault(key, []).append(float(line.rsplit(" ", 1)[-1]))
    if set(values) != {"vllm:num_requests_running", "vllm:num_requests_waiting"}:
        raise ValueError("missing target queue metrics")
    return dict(metrics_sha256=hashlib.sha256(raw).hexdigest(), queues=values)


def lock_owner_proof(lock_path, pid, uid):
    process = Path("/proc") / str(pid)
    if not process.exists() or process.stat().st_uid != uid:
        raise ValueError("subleasing primary owner is no longer alive with expected UID")
    os.kill(pid, 0)
    lock = Path(lock_path)
    stat = lock.stat()
    identity = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}"
    matching = []
    for line in Path("/proc/locks").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[5] == identity:
            matching.append(parts)
    if not any(p[1] == "FLOCK" and p[3] == "WRITE" and p[4] == str(pid) for p in matching):
        raise ValueError("primary owner does not hold the real global flock")
    return dict(pid=pid, uid=uid, lock_identity=identity,
                lock_rows=matching, observed_epoch=time.time())


def primary_owner_proof(sublease_path):
    lease = read(sublease_path)["primary_owner"]
    return lock_owner_proof(lease["global_lock"], lease["pid"], lease["uid"])


def active_owner_proof(plan):
    if plan["lock_mode"] == "exclusive_global":
        return lock_owner_proof(plan["global_lock"], os.getpid(), 1003)
    return primary_owner_proof(plan["sublease"]["path"])


def checkpoint_stats(checkpoint, files):
    return {entry["path"]: dict(size=(st := (checkpoint / entry["path"]).stat()).st_size,
                               mtime_ns=st.st_mtime_ns, inode=st.st_ino, device=st.st_dev)
            for entry in files}


def materialized_binding(parent, checkpoint, spec):
    manifest_path = parent / "materialization/edit-manifest.json"
    verify_path = parent / "materialization/verify.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    verification = read(verify_path)
    candidate = parent.parents[1] / "checkpoints" / read(spec)["id"]
    if (manifest["source"] != str(checkpoint) or manifest["candidate"] != str(candidate)
            or manifest["spec_sha256"] != sha(spec) or manifest["spec"] != read(spec)):
        raise ValueError("saved checkpoint does not bind the same AV and fixed F01 spec")
    if not verification["verified"] or verification["legacy_full_hashes_missing"]:
        raise ValueError("complete editor verification required")
    for key in ("changed_elements", "changed_tensors", "delta_l2"):
        if manifest[key] != verification[key]:
            raise ValueError("editor verify differs from manifest: " + key)
    expected = [{k: entry[k] for k in ("path", "source_sha256", "candidate_sha256")}
                for entry in manifest["files"]]
    if expected != verification["files"] or len(expected) != 9:
        raise ValueError("all nine saved file hashes required")
    return dict(checkpoint=str(candidate), spec_sha256=sha(spec),
                manifest=dict(path=str(manifest_path), sha256=hashlib.sha256(manifest_bytes).hexdigest()),
                verify=dict(path=str(verify_path), sha256=sha(verify_path)),
                proof="Editor reconstructs every declared BF16 edit, verifies all nine candidate files and original-byte restoration; finite replay uses the identical spec on the identical AV base.",
                verification=verification, candidate_file_stats=checkpoint_stats(candidate, expected))


def prepare(root, project, repo, candidate_id, unsaved_spec, exclusive_global):
    root.mkdir(exist_ok=False)
    shutil.copyfile(__file__, root / "owner-source.py")
    parent = root.parent
    proposal = read(parent / "prepared/plan.json")
    sublease_path = project / "analysis/fresh-purpose-20261107-av-live10-v1/finite-gpu6-sublease.json"
    sublease = None if exclusive_global else read(sublease_path)
    prior = project / "analysis/code90-resume-20261106"
    collection = prior / "science-gpu-runs/provisional-av-long-purpose-v3/matrix-correct-logprob"
    cases = Path(proposal["cases"]["path"])
    all_cases = read(cases)
    if sum(len(fold) for folds in all_cases.values() for fold in folds) != 120:
        raise ValueError("all 120 fixed authored cases required")
    checkpoint = Path(proposal["checkpoint"]["path"])
    spec = parent / "proposal" / (candidate_id + ".json")
    finite = dict(
        schema_version=1, architecture="gemma4_sparse_bf16",
        cases=proposal["cases"], frame_evidence=dict(path=str(collection / "frame-evidence.json"),
                                                  sha256=sha(collection / "frame-evidence.json")),
        checkpoint_attestation=proposal["checkpoint"]["attestation"],
        checkpoint_index_sha256=sha(checkpoint / "model.safetensors.index.json"),
        base_tree_sha256=proposal["checkpoint"]["tree_sha256"],
        candidates=[dict(id=candidate_id, spec=str(spec), sha256=sha(spec))],
        observation_policy=read(collection / "provenance.json")["observation_policy"],
    )
    save(root / "finite-plan.json", finite)
    shutil.copytree(repo / "src", root / "source", ignore=shutil.ignore_patterns("__pycache__"))
    if exclusive_global:
        # Parent delegated only this exact existing target; no expired sublease is used.
        target_name = "gemma-4-31b-8gpu-gemma-3-1"
        target_cid = "64f27c717165a4e4ea37abec691b5871f87fdfc8ddcd8dd2ecddfd832e5df550"
        protected_names = [f"gemma-4-31b-8gpu-gemma-{i}-1" for i in (0, 1, 2)]
        protected_names += ["gemma-4-31b-8gpu-embed-0-1", "gemma-4-31b-8gpu-embed-1-1"]
    else:
        target_name = sublease["sublease"]["target_service"]
        target_cid = sublease["sublease"]["target_snapshot"]["container_id"]
        protected_names = list(sublease["protected_services"])
    target = snapshot(target_name)
    if target["id"] != target_cid:
        raise ValueError("explicit exact target CID differs")
    protected = {name: snapshot(name) for name in protected_names}
    for name, current in protected.items():
        if not exclusive_global and current["id"] != sublease["protected_services"][name]["container_id"]:
            raise ValueError("subleased protected CID differs")
    materialized = None if unsaved_spec else materialized_binding(parent, checkpoint, spec)
    pins = {str(p): sha(p) for p in (root / "finite-plan.json", cases, spec,
                                    Path(finite["frame_evidence"]["path"]),
                                    Path(finite["checkpoint_attestation"]["path"]))}
    if not exclusive_global:
        pins[str(sublease_path)] = sha(sublease_path)
    if materialized is not None:
        pins.update({entry["path"]: entry["sha256"] for entry in (materialized["manifest"], materialized["verify"])})
    else:
        postcheck = parent / "independent-postcheck.json"
        proof = read(postcheck)
        if not proof["current_source_words_exact"] or proof["inputs_sha256"][spec.name] != sha(spec):
            raise ValueError("exact source/spec CPU postcheck required for unsaved RAM candidate")
        pins[str(postcheck)] = sha(postcheck)
    pins.update({str(p): sha(p) for p in (root / "source").rglob("*.py")})
    plan = dict(schema_version=1, status="prepared_no_gpu_execution", checkpoint=str(checkpoint),
                candidate_id=candidate_id, saved_checkpoint_verified=materialized is not None,
                candidate_scope="saved checkpoint verified" if materialized else "unsaved RAM spec only; runtime selected BF16 byte proof required, no saved checkpoint exists",
                container_name="gemma-f01-finite-gpu6-20261107-v1" if candidate_id == "fresh-code-joint-f01" else candidate_id + "-finite-gpu6-20261107",
                lock_mode="exclusive_global" if exclusive_global else "subleased_gpu6",
                global_lock=str(project / "candidate-evaluation.lock"),
                materialized_candidate=materialized,
                finite_plan=str(root / "finite-plan.json"), sublease=None if exclusive_global else dict(path=str(sublease_path), sha256=sha(sublease_path)),
                target_name=target_name, target=target, protected=protected, image=target["image"],
                gpu=6, cpuset="48-55", memory="128g", maximum_seconds=600,
                uid_gid="1003:1003", network="none", cases=120,
                target_queue_metrics=target_metrics(target["id"]), pins=pins,
                primary_owner_at_preparation=None if exclusive_global else primary_owner_proof(sublease_path),
                owner_source_sha256=sha(__file__), output=str(root / "work/replay"),
                gpu_lock=str(project / "gpu6-finite.lock"),
                timing_note="Global-exclusive candidate evaluation; no overlapping benchmark permitted while global flock is held." if exclusive_global else "May overlap gemma0 WSP on disjoint GPUs; shared host resources mean no uncontended latency claim.")
    save(root / "owner-plan.json", plan)
    print(json.dumps(dict(status="prepared", plan_sha256=sha(root / "owner-plan.json"), model_calls=0)))


def validate(root, plan, serving=True):
    if os.getuid() != 1003 or sha(__file__) != plan["owner_source_sha256"]:
        raise ValueError("owner UID/source mismatch")
    for path, expected in plan["pins"].items():
        if sha(path) != expected:
            raise ValueError("pinned input changed: " + path)
    materialized = plan["materialized_candidate"]
    if materialized is not None and checkpoint_stats(Path(materialized["checkpoint"]), materialized["verification"]["files"]) != materialized["candidate_file_stats"]:
        raise ValueError("saved F01 checkpoint stat changed after binding")
    for name, expected in plan["protected"].items():
        if snapshot(name) != expected:
            raise ValueError("protected service changed: " + name)
    if serving and snapshot(plan["target_name"]) != plan["target"]:
        raise ValueError("target service changed")
    if plan["gpu"] != 6 or plan["cpuset"] != "48-55" or plan["maximum_seconds"] != 600:
        raise ValueError("fixed hardware/time scope differs")
    if not root.is_relative_to(Path(plan["checkpoint"]).parents[1] / "analysis"):
        raise ValueError("output must stay in task analysis root")


def command(root, project, plan, preview=False):
    work = root / "work"
    for path in (work, work / "cache", work / "tmp"):
        path.mkdir(exist_ok=True)
    argv = ["docker", "run", "--rm", "--network", "none", "--user", "1003:1003",
            "--memory", "128g", "--cpuset-cpus", "48-55", "--pids-limit", "512",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only",
            "--shm-size", "1g", "--entrypoint", "/usr/bin/python3",
            "--mount", f"type=bind,src={project},dst={project},readonly",
            "--mount", f"type=bind,src={work},dst={work}",
            "--mount", f"type=bind,src={work / 'tmp'},dst=/tmp",
            "--workdir", str(work)]
    for key, value in dict(PYTHONPATH=str(root / "source"), PYTHONDONTWRITEBYTECODE="1",
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           HF_HOME=str(work / "cache/hf"), TORCH_HOME=str(work / "cache/torch"),
                           TRITON_CACHE_DIR=str(work / "cache/triton"),
                           TORCHINDUCTOR_CACHE_DIR=str(work / "cache/inductor"),
                           CUDA_CACHE_PATH=str(work / "cache/cuda"),
                           XDG_CACHE_HOME=str(work / "cache"), TMPDIR="/tmp",
                           OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4").items():
        argv.extend(["--env", key + "=" + value])
    if preview:
        code = "\n".join([
            "import json,torch,transformers",
            "from pathlib import Path",
            "from transformers import AutoTokenizer",
            "from kairyu_bench.wsp.weights.finite_replay import load_plan,prepare_frames",
            "from kairyu_bench.wsp.weights.checkpoint import Checkpoint",
            f"cp=Checkpoint({plan['checkpoint']!r})",
            f"plan,specs,hashes=load_plan(Path({plan['finite_plan']!r}),cp)",
            "tok=AutoTokenizer.from_pretrained(cp.path,local_files_only=True)",
            "cases=json.loads(Path(plan['cases']['path']).read_text())",
            "prepared,frame=prepare_frames(tok,cases,max_frame_tokens=2048)",
            "assert frame==json.loads(Path(plan['frame_evidence']['path']).read_text())",
            "print(json.dumps(dict(status='exact_frames_and_spec_pass',cases=sum(len(f) for fs in cases.values() for f in fs),torch=torch.__version__,transformers=transformers.__version__,cuda_devices=torch.cuda.device_count(),model_calls=0)))",
        ])
        return argv + [plan["image"], "-c", code]
    argv += ["--gpus", "device=6", "--name", plan["container_name"],
             "--cidfile", str(work / "container.cid"), plan["image"],
             "-m", "kairyu_bench.wsp.weights.finite_replay",
             "--checkpoint", plan["checkpoint"], "--plan", plan["finite_plan"],
             "--output", plan["output"], "--device", "cuda:0", "--maximum-seconds", "600",
             "--answer-batching", "serial", "--threads", "4", "--seed", "0",
             "--max-frame-tokens", "2048", "--text-attention-backend", "sdpa",
             "--cuda-memory-cap-gib", "88", "--execution-order", "longest_frame_first",
             "--gpu-exclusive-approved"]
    return argv


def run(root, project, plan, approved_sha):
    if approved_sha != sha(root / "owner-plan.json"):
        raise ValueError("explicit matching plan SHA required")
    if not (root / "cpu-preview.json").exists() or read(root / "cpu-preview.json")["returncode"] != 0:
        raise ValueError("successful exact-frame CPU preview required")
    if (root / "completion.json").exists() or (root / "work/container.cid").exists():
        raise ValueError("finite attempt already exists")
    with ExitStack() as stack:
        lock_paths = ([plan["global_lock"]] if plan["lock_mode"] == "exclusive_global" else []) + [plan["gpu_lock"]]
        for lock_path in lock_paths:
            lock = stack.enter_context(open(lock_path, "a+"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate(root, plan)
        owner_proof = active_owner_proof(plan)
        before = target_metrics(plan["target"]["id"])
        if any(value != 0 for values in before["queues"].values() for value in values):
            raise ValueError("target is serving active or waiting requests")
        status = dict(status="starting", pid=os.getpid(), plan_sha256=approved_sha,
                      started_epoch=time.time(), before=before, primary_owner_before_run=owner_proof,
                      stopped=False, restored=False)
        save(root / "launch.json", status)
        stopped = False
        process = None
        def interrupted(signum, frame):
            raise InterruptedError(f"finite owner interrupted {signum}")
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, interrupted)
        try:
            if snapshot(plan["target_name"]) != plan["target"]:
                raise ValueError("exact target CID changed before stop")
            save(root / "primary-owner-before-stop.json", active_owner_proof(plan))
            stopped = True  # A stop command may succeed before its client is interrupted.
            docker("stop", "--time", "60", plan["target"]["id"])
            status["stopped"] = True
            gpu = subprocess.run(["nvidia-smi", "-i", "6", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
            if int(gpu.stdout.strip()) < 88 * 1024:
                raise ValueError("GPU6 lacks the declared 88 GiB free allocator capacity")
            validate(root, plan, serving=False)
            argv = command(root, project, plan)
            save(root / "gpu-command.json", dict(argv=argv, gpu6_free_mib=int(gpu.stdout.strip())))
            with (root / "gpu-replay.log").open("x") as log:
                process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
                status["returncode"] = process.wait(timeout=780)
            status["status"] = "finished" if status["returncode"] == 0 else "failed"
        except BaseException as error:
            status.update(status="failed", error_type=type(error).__name__, error=str(error))
        finally:
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                signal.signal(sig, signal.SIG_IGN)
            try:
                cidfile = root / "work/container.cid"
                if cidfile.exists():
                    cid = cidfile.read_text().strip()
                    try:
                        owned = inspect(cid)
                    except subprocess.CalledProcessError:
                        owned = None  # --rm already removed the owned container.
                    if owned is not None:
                        if owned["Id"] != cid or owned["Name"] != "/" + plan["container_name"]:
                            raise ValueError("owned replay container identity differs")
                        if owned["State"]["Running"]:
                            docker("stop", "--time", "90", cid)
                if process is not None and process.poll() is None:
                    process.wait(timeout=120)
                if stopped:
                    docker("start", plan["target"]["id"])
                    deadline = time.monotonic() + 480
                    while time.monotonic() < deadline:
                        restored = snapshot(plan["target_name"])
                        keys = ("id", "image", "config_sha256", "host_config_sha256", "cpuset", "devices")
                        if any(restored[k] != plan["target"][k] for k in keys):
                            raise ValueError("restored target identity differs")
                        if restored["running"] and restored["health"] == "healthy":
                            target_metrics(restored["id"])
                            status["restored"] = True
                            break
                        time.sleep(3)
                    if not status["restored"]:
                        status["restoration_error"] = "target health deadline"
                status["protected_after"] = {name: snapshot(name) for name in plan["protected"]}
                status["protected_exact"] = status["protected_after"] == plan["protected"]
                status["protected_changes"] = {
                    name: {k: dict(before=expected[k], after=status["protected_after"][name][k])
                           for k in expected if expected[k] != status["protected_after"][name][k]}
                    for name, expected in plan["protected"].items()
                    if expected != status["protected_after"][name]
                }
                if status["protected_changes"]:
                    status["protected_change_note"] = "Observed other-actor change; owner commands only target exact CID and its own replay container. Target restoration still attempted."

            except BaseException as error:
                status["restoration_error"] = dict(type=type(error).__name__, message=str(error))
            finally:
                status["elapsed_seconds"] = time.time() - status["started_epoch"]
                save(root / "completion.json", status)
        if status["status"] != "finished" or not status["restored"] or not status.get("protected_exact", False):
            raise RuntimeError("finite attempt incomplete; inspect preserved completion receipt")
        print(json.dumps({k: status[k] for k in ("status", "returncode", "restored", "protected_exact", "elapsed_seconds")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preview", "run"), default="preview", nargs="?")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--approved-plan-sha256")
    parser.add_argument("--revision", type=int, default=1)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--candidate-id", default="fresh-code-joint-f01")
    parser.add_argument("--unsaved-spec", action="store_true")
    parser.add_argument("--exclusive-global", action="store_true")
    args = parser.parse_args()
    project, repo = args.project.resolve(), args.repository.resolve()
    experiment = (args.experiment_root or project / "analysis/fresh-purpose-20261107-f01").resolve()
    if not experiment.is_relative_to(project / "analysis"):
        raise ValueError("experiment root must remain in task analysis root")
    root = experiment / f"finite-gpu6-v{args.revision}"
    if args.action == "prepare":
        prepare(root, project, repo, args.candidate_id, args.unsaved_spec, args.exclusive_global)
        return
    plan = read(root / "owner-plan.json")
    if args.action == "preview":
        validate(root, plan)
        argv = command(root, project, plan, preview=True)
        start = time.monotonic()
        result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        save(root / "cpu-preview.json", dict(argv=argv, returncode=result.returncode,
                                            stdout=result.stdout, stderr=result.stderr,
                                            seconds=time.monotonic() - start, model_calls=0))
        print(json.dumps(dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr[-3000:])))
        raise SystemExit(result.returncode)
    run(root, project, plan, args.approved_plan_sha256)


if __name__ == "__main__":
    main()
