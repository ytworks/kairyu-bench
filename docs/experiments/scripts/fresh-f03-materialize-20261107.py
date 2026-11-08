#!/usr/bin/env python3
"""Authorized F03 only: one CPU preview/create/full verify under the global lock."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

R = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
B = R / "worktrees/kairyu-bench-weight-tools"
P = R / "analysis/fresh-purpose-20261107-f03-native-v1"
M = P / "materialization"
SOURCE = R / "checkpoints/code90-current-ad-helper-av-v1"
TARGET = R / "checkpoints/fresh-native-contract-f03"
SPEC = P / "proposal/fresh-native-contract-f03.json"
SPEC_SHA = "e10018748de2795a8af03b98abb6e18362393a152c74f46b7a7892fe4975e878"
ATTEST = R / "analysis/fresh-purpose-20261107-av-dev30-dp4-v1/http-trace/capture.json"
ATTEST_SHA = "8e4b57eacaeca88260281c4d39a0f347b55ed0a7f1cabcac3ff0507d1e6d9e74"
TREE = "5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf"
PYTHON = R / "infra/weight-tools-ci/bin/python"
RESERVE = 50 * 2**30


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open("x") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())


def inventory(root, names):
    rows = {}
    for name in names:
        s = (root / name).stat(follow_symlinks=False)
        require(stat.S_ISREG(s.st_mode), "regular file required")
        rows[name] = dict(dev=s.st_dev, inode=s.st_ino, size=s.st_size, mtime_ns=s.st_mtime_ns,
                          ctime_ns=s.st_ctime_ns, mode=s.st_mode, uid=s.st_uid, gid=s.st_gid, nlink=s.st_nlink)
    return rows


def totals(result):
    require(result["changed_elements"] == 60473 and result["changed_tensors"] == 112
            and math.isclose(result["delta_l2"], .0043893681725189855, rel_tol=1e-12, abs_tol=1e-15),
            "fixed F03 actual diff differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="explicit authorization for this exact fixed F03 checkpoint")
    args = parser.parse_args()
    require(args.apply, "requires explicit --apply; this fixed recipe previews before writing")
    require(not M.exists() and not TARGET.exists(), "fresh materialization and candidate paths required")
    M.mkdir()
    started = time.time()
    phase = "preflight"
    acquired = False
    with (R / "candidate-evaluation.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
            save(M / "lock-acquired.json", {"pid": os.getpid(), "path": str(R / "candidate-evaluation.lock"),
                "epoch": time.time(), "exclusive_nonblocking": True, "held_until_complete_or_failure": True})
            require(sha(SPEC) == SPEC_SHA and sha(ATTEST) == ATTEST_SHA, "spec or AV attestation SHA differs")
            spec = read(SPEC)
            require(spec["source_checkpoint"] == str(SOURCE) and spec["base_tree_sha256"] == TREE,
                    "fixed F03 spec parent differs")
            relative_code = ["src/kairyu_bench/__init__.py", "src/kairyu_bench/wsp/__init__.py",
                "src/kairyu_bench/wsp/weights/__init__.py", "src/kairyu_bench/wsp/weights/checkpoint.py",
                "src/kairyu_bench/wsp/weights/editor.py"]
            snapshot = M / "source"
            pins = {}
            for name in relative_code:
                original = B / name
                target = snapshot / name
                target.parent.mkdir(parents=True, exist_ok=True)
                raw = original.read_bytes()
                target.write_bytes(raw)
                target.chmod(0o444)
                digest = hashlib.sha256(raw).hexdigest()
                require(sha(original) == digest, "source changed while snapshotting")
                pins[str(target)] = digest
            recipe_copy = snapshot / Path(__file__).name
            recipe_copy.write_bytes(Path(__file__).read_bytes())
            recipe_copy.chmod(0o444)
            pins[str(recipe_copy)] = sha(recipe_copy)
            sys.path.insert(0, str(snapshot / "src"))
            from kairyu_bench.wsp.weights.checkpoint import Checkpoint
            source = Checkpoint(SOURCE)
            source_before = source.identity(hash_files=False)
            old = read(ATTEST)["checkpoint"]
            require(old["sha256"] == TREE and old["hash_files"] is True and len(old["files"]) == 9
                    and len(source_before["files"]) == 9, "complete AV attestation required")
            require(all(all(a[k] == b[k] for k in ["path", "size", "stat"])
                        for a, b in zip(old["files"], source_before["files"], strict=True)), "AV attestation stat changed")
            names = [f["path"] for f in source_before["files"]]
            source_full_stat = inventory(SOURCE, names)
            plan = dict(source=str(SOURCE), candidate=str(TARGET), manifest=str(M / "edit-manifest.json"),
                spec={"path":str(SPEC),"sha256":SPEC_SHA},
                source_checkpoint_attestation={"path":str(ATTEST),"sha256":ATTEST_SHA,"tree_sha256":TREE},
                source_checkpoint_current_stat=source_before, source_full_stat=source_full_stat,
                reserve_bytes=RESERVE, free_bytes_before=shutil.disk_usage(R).free, source_sha256=pins,
                cpu_python=str(PYTHON), authorized_action="one preview, one apply, one fresh full editor verify",
                gpu_calls=0, model_calls=0, service_operations=0)
            save(M / "plan.json", plan)
            env = dict(os.environ, PYTHONPATH=str(snapshot / "src"), PYTHONDONTWRITEBYTECODE="1",
                       TMPDIR=str(M / "tmp"), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
            (M / "tmp").mkdir()
            command = [str(PYTHON), "-m", "kairyu_bench.wsp.weights.editor"]

            def stable():
                require(sha(SPEC) == SPEC_SHA and sha(ATTEST) == ATTEST_SHA, "input identity changed")
                require(all(sha(Path(p)) == h for p, h in pins.items()), "immutable CPU snapshot changed")
                require(source.identity(hash_files=False) == source_before
                        and inventory(SOURCE, names) == source_full_stat, "AV source stat changed")

            def stage(name, argv):
                stable()
                begin = time.time()
                with (M / (name + ".stdout")).open("xb") as stdout, (M / (name + ".stderr")).open("xb") as stderr:
                    result = subprocess.run(argv, cwd=snapshot, env=env, stdout=stdout, stderr=stderr,
                                            pass_fds=(lock.fileno(),), check=False)
                finished = time.time()
                save(M / (name + "-receipt.json"), {"argv":argv,"started_epoch":begin,"finished_epoch":finished,
                    "seconds":finished-begin,"returncode":result.returncode,"source_sha256":pins})
                require(result.returncode == 0, name + " failed; no automatic retry")
                value = read(M / (name + ".stdout"))
                save(M / (name + ".json"), value)
                totals(value)
                stable()
                return value

            arguments = ["create", str(SOURCE), str(TARGET), "--spec", str(SPEC),
                         "--manifest", str(M / "edit-manifest.json"), "--reserve-bytes", str(RESERVE)]
            phase = "preview"
            preview = stage("preview", command + arguments)
            require(preview["applied"] is False and preview["hardlink_bytes"] == 0, "expected private-copy preview")
            phase = "create"
            created = stage("create", command + arguments + ["--apply"])
            require(created["applied"] is True and shutil.disk_usage(R).free >= RESERVE, "create/reserve condition failed")
            before = Checkpoint(TARGET).identity(hash_files=False)
            before_full = inventory(TARGET, names)
            save(M / "candidate-before-verify-stat.json", {"checkpoint":before,"full_stat":before_full,"epoch":time.time()})
            phase = "verify"
            verified = stage("verify", command + ["verify", str(M / "edit-manifest.json")])
            require(verified["verified"] is True and verified["legacy_full_hashes_missing"] is False
                    and len(verified["files"]) == 9, "fresh all-nine-file verification required")
            after = Checkpoint(TARGET).identity(hash_files=False)
            after_full = inventory(TARGET, names)
            require(before == after and before_full == after_full, "candidate stat changed across fresh verify")
            save(M / "candidate-after-verify-stat.json", {"checkpoint":after,"full_stat":after_full,"epoch":time.time()})
            by_name = {f["path"]:f for f in verified["files"]}
            require(set(by_name) == set(names), "verified file inventory differs")
            identity = copy.deepcopy(after)
            for row in identity["files"]:
                row["sha256"] = by_name[row["path"]]["candidate_sha256"]
            canonical = [{k:f[k] for k in ["path","size","sha256"]} for f in identity["files"]]
            identity["sha256"] = hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(",",":")).encode()).hexdigest()
            identity.update(hash_files=True, identity_level="content_sha256")
            source_canonical = [{"path":f["path"],"size":f["size"],
                                 "sha256":by_name[f["path"]]["source_sha256"]} for f in after["files"]]
            source_tree = hashlib.sha256(json.dumps(source_canonical,sort_keys=True,separators=(",",":")).encode()).hexdigest()
            require(source_tree == TREE, "fresh verification source tree differs from AV")
            derived = {"checkpoint":identity,"derivation":"Fresh editor verify candidate SHA bound to identical full Checkpoint.identity(hash_files=False) before/after verify; canonical Checkpoint tree serialization; no duplicate candidate payload hash.",
                "verify":{"path":str(M/"verify.json"),"sha256":sha(M/"verify.json")},
                "manifest":{"path":str(M/"edit-manifest.json"),"sha256":sha(M/"edit-manifest.json")},
                "candidate_full_stat":after_full,"source_full_stat":source_full_stat,"source_tree_sha256":source_tree,
                "candidate_stat_unchanged_across_verify":True,"extra_payload_hash_pass":False,
                "spec_sha256":SPEC_SHA,"source_sha256":pins,"created_epoch":time.time()}
            save(M / "fullcandidateidentity.json", derived)
            stable()
            phase = "complete"
            save(M / "completion.json", {"status":"completed","seconds":time.time()-started,
                "candidate_tree_sha256":identity["sha256"],"identity_sha256":sha(M/"fullcandidateidentity.json"),
                "spec_sha256":SPEC_SHA,"verified":True,"fresh_all9_file_hashes":True,
                "changed_elements":verified["changed_elements"],"changed_tensors":verified["changed_tensors"],
                "delta_l2":verified["delta_l2"],"free_bytes_after":shutil.disk_usage(R).free,
                "model_calls":0,"gpu_calls":0,"service_operations":0,"apply_calls":1,"fresh_verify_calls":1})
            print(json.dumps(read(M / "completion.json")), flush=True)
        except BaseException as error:
            save(M / "failure.json", {"phase":phase,"type":type(error).__name__,"message":str(error),
                                      "seconds":time.time()-started,"automatic_retry":False})
            raise
        finally:
            if acquired:
                fcntl.flock(lock, fcntl.LOCK_UN)
            save(M / "lock-released.json", {"epoch":time.time(),"pid":os.getpid(),"released":acquired,"phase":phase})


if __name__ == "__main__":
    main()
