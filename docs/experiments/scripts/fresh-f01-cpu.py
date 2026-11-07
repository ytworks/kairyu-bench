#!/usr/bin/env python3
"""Run the fixed F01 CPU recipe with exact historical input bytes in a namespace."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preview", "solve"))
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--candidate-id", default="fresh-code-joint-f01")
    parser.add_argument("--target-purposes", nargs="+", default=["algorithm", "implementation"])
    args = parser.parse_args()
    root = (args.experiment_root or args.project.resolve() / "analysis/fresh-purpose-20261107-f01").resolve()
    if not root.is_relative_to(args.project.resolve() / "analysis"):
        raise ValueError("experiment root must remain in task analysis root")
    root.mkdir(parents=True, exist_ok=True)
    repo = args.repository.resolve()
    source = root / "solver-snapshot"
    receipt = root / (args.action + f"-namespace-receipt-r{args.attempt}.json")
    if receipt.exists():
        raise ValueError("receipt already exists; no silent retry")
    legacy_path = repo / "src/kairyu_bench/wsp/weights/finite_replay.py"
    legacy_snapshot = root / "historical-inputs/finite_replay.py"
    legacy_sha = "e28f3ebeab3eea32a9685311fd3e937f2b1d9d827ec7cbced1ece57cbca29090"
    if args.action == "prepare" and not source.exists():
        source.mkdir()
        shutil.copytree(repo / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__"))
        recipes = Path("scripts/experiments/gemma4-purpose")
        shutil.copytree(repo / recipes, source / recipes, ignore=shutil.ignore_patterns("__pycache__"))
        legacy_snapshot.parent.mkdir()
        old = subprocess.run(
            ["git", "show", "fba68c3:src/kairyu_bench/wsp/weights/finite_replay.py"],
            cwd=repo, check=True, stdout=subprocess.PIPE,
        ).stdout
        if hashlib.sha256(old).hexdigest() != legacy_sha:
            raise ValueError("historical source SHA differs from collector provenance")
        legacy_snapshot.write_bytes(old)
    if sha(legacy_snapshot) != legacy_sha:
        raise ValueError("historical snapshot changed")
    python = args.project.resolve() / "infra/weight-tools-ci/bin/python"
    recipes = source / "scripts/experiments/gemma4-purpose"
    past = args.project.resolve() / "analysis/code90-resume-20261106"
    if args.action == "prepare":
        command = [str(python), str(recipes / "fresh_joint_f01.py"),
                   "--collection-root", str(past / "science-gpu-runs/provisional-av-long-purpose-v3"),
                   "--cases", str(past / "long-helper-chain-curriculum-v1/train-cases.json"),
                   "--checkpoint", str(args.project.resolve() / "checkpoints/code90-current-ad-helper-av-v1"),
                   "--attestation", str(past / "provisional-av-long-purpose-observation-v1/checkpoint-attestation.json"),
                   "--output", str(root / "prepared"), "--candidate-id", args.candidate_id,
                   "--target-purposes", *args.target_purposes]
    else:
        command = [str(python), str(recipes / "propose_native_objectives.py"),
                   "--plan", str(root / "prepared/plan.json"), "--output", str(root / "proposal")]
        if args.action == "solve":
            command.extend(["--run", "--write-spec", "--maximum-seconds", "120"])
    namespace = ["bwrap", "--unshare-net", "--ro-bind", "/", "/",
                 "--bind", str(root), str(root),
                 "--ro-bind", str(source), str(source),
                 "--ro-bind", str(legacy_snapshot), str(legacy_snapshot),
                 "--ro-bind", str(legacy_snapshot), str(legacy_path),
                 "--dev", "/dev",
                 "--chdir", str(source)]
    environment = {**os.environ, "PYTHONPATH": str(source / "src"),
                   "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "4",
                   "OMP_NUM_THREADS": "4", "TMPDIR": str(root)}
    started = time.monotonic()
    probe = subprocess.run(
        namespace + ["/bin/sh", "-c", "id; readlink /proc/self/ns/net; cat /proc/net/route; sha256sum \"$1\" \"$2\"", "probe",
                     str(legacy_path), str(source / "src/kairyu_bench/wsp/weights/finite_replay.py")],
        env=environment, check=True, capture_output=True, text=True,
    )
    result = subprocess.run(namespace + command, env=environment, capture_output=True, text=True)
    data = dict(
        action=args.action, attempt=args.attempt, launcher_sha256=sha(Path(__file__)), returncode=result.returncode, seconds=time.monotonic() - started,
        namespace_argv=namespace, command=command, namespace_probe=probe.stdout,
        host_uid=os.getuid(), host_network_namespace=os.readlink("/proc/self/ns/net"),
        historical_source_sha256=legacy_sha, host_current_source_sha256=sha(legacy_path),
        solver_snapshot_finite_replay_sha256=sha(source / "src/kairyu_bench/wsp/weights/finite_replay.py"),
        snapshot_source_hashes={str(p.relative_to(source)): sha(p) for p in source.rglob("*.py")},
        stdout=result.stdout, stderr=result.stderr,
        model_calls=0, checkpoint_writes=0, wsp_calls=0,
    )
    with receipt.open("x") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: data[k] for k in ("action", "returncode", "seconds")}))
    if result.returncode:
        print(result.stderr[-4000:])
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
