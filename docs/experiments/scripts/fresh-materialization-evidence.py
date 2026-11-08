#!/usr/bin/env python3
"""Bind completed editor verification to post-verify full stats, without payload rehash."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import time


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root, names):
    result = {}
    for name in names:
        value = (root/name).stat(follow_symlinks=False)
        assert stat.S_ISREG(value.st_mode)
        result[name] = dict(dev=value.st_dev,inode=value.st_ino,size=value.st_size,
                            mtime_ns=value.st_mtime_ns,ctime_ns=value.st_ctime_ns,mode=value.st_mode,
                            uid=value.st_uid,gid=value.st_gid,nlink=value.st_nlink)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization",required=True,type=Path)
    parser.add_argument("--repository",required=True,type=Path)
    args = parser.parse_args()
    root,repo = args.materialization,args.repository
    plan = read(root/"plan.json")
    verification = read(root/"verify.json")
    create_receipt,verify_receipt = read(root/"create-receipt.json"),read(root/"verify-receipt.json")
    assert create_receipt["returncode"]==verify_receipt["returncode"]==0
    assert verification["verified"] and not verification["legacy_full_hashes_missing"]
    names=sorted(entry["path"] for entry in verification["files"])
    assert len(names)==9
    started=time.time()
    source=inventory(Path(plan["source"]),names)
    candidate=inventory(Path(plan["candidate"]),names)
    captured=time.time()
    for entry in plan["source_checkpoint_current_stat"]["files"]:
        actual=source[entry["path"]]
        expected=entry["stat"]
        assert actual["size"]==entry["size"]
        assert all(actual[a]==expected[b] for a,b in (("dev","device"),("inode","inode"),("mtime_ns","mtime_ns"),("ctime_ns","ctime_ns")))
    files=[]
    for entry in verification["files"]:
        assert source[entry["path"]]["size"]==candidate[entry["path"]]["size"]
        files.append(dict(**entry,size=candidate[entry["path"]]["size"]))
    def tree(prefix):
        rows=[dict(path=f["path"],size=f["size"],sha256=f[prefix+"_sha256"]) for f in sorted(files,key=lambda x:x["path"])]
        return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    source_tree,candidate_tree=tree("source"),tree("candidate")
    assert source_tree==plan["source_checkpoint_attestation"]["tree_sha256"]
    code={}
    for name,expected in plan["code"]["source_sha256"].items():
        actual=sha(repo/name)
        assert actual==expected
        code[name]=actual
    manifest=Path(plan["manifest"])
    receipt_time=(root/"verify-receipt.json").stat().st_mtime_ns/1e9
    output=dict(status="full_editor_verification_bound_to_postverify_stat",
        source=plan["source"],candidate=plan["candidate"],files=files,
        source_tree_sha256=source_tree,candidate_tree_sha256=candidate_tree,
        source_inventory_stat=source,candidate_inventory_stat=candidate,
        stat_capture_started_epoch=started,stat_capture_epoch=captured,
        verify_finished_epoch=receipt_time,
        verify_finished_epoch_note="Upper bound from receipt write immediately after verify subprocess returned; not a separately sampled process-exit timestamp.",
        verify_completion_window_epoch=[(root/"verify.stdout").stat().st_mtime_ns/1e9,receipt_time],
        manifest=dict(path=str(manifest),sha256=sha(manifest)),
        verify=dict(path=str(root/"verify.json"),sha256=sha(root/"verify.json")),
        spec=plan["spec"],scope=verification["scope"],legacy_full_hashes_missing=False,
        changed_elements=verification["changed_elements"],changed_tensors=verification["changed_tensors"],
        delta_l2=verification["delta_l2"],code_sha256=code,
        git_head_at_preview=plan["code"]["git_head"],
        git_head_at_stat_capture=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),
        hash_evidence="All nine file SHA values come from completed full editor verification; canonical tree reconstructed using Checkpoint.identity ordering/serialization. This evidence builder did not rehash checkpoint payload.",
        fresh_payload_rehash_by_evidence_builder=False,source_stat_unchanged_since_preview=True,
        seconds=dict(create=create_receipt["seconds"],verify=verify_receipt["seconds"]),
        script_sha256=sha(Path(__file__)))
    target=root/"full-verification-freeze-input.json"
    with target.open("x") as f:
        json.dump(output,f,indent=2,allow_nan=False)
        f.write(chr(10))
    print(json.dumps(dict(path=str(target),sha256=sha(target),candidate_tree_sha256=candidate_tree,
                         manifest_sha256=output["manifest"]["sha256"],verify_sha256=output["verify"]["sha256"],
                         stat_capture_epoch=captured,verify_finished_epoch=receipt_time)))


if __name__=="__main__":
    main()
