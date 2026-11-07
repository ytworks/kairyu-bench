#!/usr/bin/env python3
"""Fixed non-destructive scalar archive; full streamed verification, never unlink."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import time

R = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
B = R / "worktrees/kairyu-bench-weight-tools"
D = R / "analysis/fresh-purpose-20261107-scalar-l41-minus4-release-plan-v1"
A = R / "analysis/fresh-purpose-20261107-scalar-l41-minus4-lossless-v1"
P = D / "plan.json"
C = R / "checkpoints/scalar-l41-minus4"
M = R / "manifests/scalar-l41-minus4.json"
S = R / "specs/scalar-l41-minus4.json"
PY = R / "infra/weight-tools-ci/bin/python"
FIELDS = ("device", "inode", "size", "mtime_ns", "ctime_ns", "mode", "uid", "gid", "nlink")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def stamp(path):
    s = Path(path).lstat()
    return dict(zip(FIELDS, (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns,
                s.st_ctime_ns, s.st_mode, s.st_uid, s.st_gid, s.st_nlink), strict=True))


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    started = time.time()
    assert sha(P) == "a05e488a2bdd6f88b31dc51f8a654676e41a9857d756e74d332f1102141873ac"
    plan = read(P)
    assert all(sha(p) == h for p, h in plan["source_pins"].items())
    manifest = read(M)
    base = Path(manifest["source"])
    audit = read(plan["audit"]["path"])
    for root, states in ((base, audit["source_stats"]), (C, audit["candidate_stats"])):
        assert all(stamp(root / name) == state for name, state in states.items())
    assert not A.exists()
    A.mkdir()
    shutil.copyfile(__file__, A / "preparation-source.py.txt")
    shutil.copytree(B / "src", A / "source", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    save(D / "archive-start.json", dict(pid=os.getpid(), started_epoch=started,
         plan_sha256=sha(P), source_sha256=sha(__file__), model_calls=0, payload_mutations=0))
    cmd = [str(PY), "-B", "-m", "kairyu_bench.wsp.weights.editor", "verify", str(M)]
    env = dict(os.environ, PYTHONPATH=str(A / "source"), PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    save(D / "verify-command.json", dict(argv=cmd, environment={k: env[k] for k in
        ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}))
    t = time.time()
    with (D / "fresh-verify.json").open("x") as out, (D / "fresh-verify.stderr").open("x") as err:
        result = subprocess.run(cmd, env=env, stdout=out, stderr=err, timeout=900)
    save(D / "verify-execution.json", dict(returncode=result.returncode, seconds=time.time() - t))
    assert result.returncode == 0
    verified = read(D / "fresh-verify.json")
    assert verified["verified"] and verified["legacy_full_hashes_missing"] and len(verified["files"]) == 12
    by_name = {r["path"]: r for r in verified["files"]}
    assert set(by_name) == set(audit["candidate_stats"])
    for root, states in ((base, audit["source_stats"]), (C, audit["candidate_stats"])):
        assert all(stamp(root / name) == state for name, state in states.items())
    for name, path in (("spec.json.gz", S), ("original-manifest.json.gz", M)):
        raw = path.read_bytes()
        (A / name).write_bytes(gzip.compress(raw, mtime=0))
        assert gzip.decompress((A / name).read_bytes()) == raw
    metadata = A / "metadata"
    for name, row in by_name.items():
        if name.endswith(".safetensors"):
            continue
        p = metadata / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C / name, p)
        assert sha(p) == row["candidate_sha256"]
    shutil.copyfile(D / "fresh-verify.json", A / "fresh-verify.json")
    shutil.copyfile(P, A / "approved-scope-plan.json")
    shutil.copyfile(plan["audit"]["path"], A / "prearchive-readonly-audit.json")
    archive = dict(schema_version=1, candidate_path=str(C), mandatory_base_path=str(base),
        original_spec_sha256=sha(S), original_manifest_sha256=sha(M),
        original_manifest_legacy_hashes_missing=True, fresh_full_verify_sha256=sha(A / "fresh-verify.json"),
        streamed_restoration_equals_original_base=True, changed_elements=1, changed_bytes=2,
        patches=manifest["edits"], files=verified["files"], metadata_count=10,
        source_stats=audit["source_stats"], candidate_stats=audit["candidate_stats"],
        release_scope=plan["target_payloads"], physical_full_reconstruction=False,
        historical_tree_identity_not_rehashed=False,
        model_calls=0, gpu_calls=0, service_changes=0, checkpoint_mutations=0)
    save(A / "archive.json", archive)
    for label, field in (("BASE", "source_sha256"), ("CANDIDATE", "candidate_sha256")):
        (A / (label + "-SHA256SUMS")).write_text("".join(r[field] + "  " + r["path"] + "\n" for r in verified["files"]))
    text = ("# Exact dependent scalar archive\n\n"
      "Original source is mandatory; AV and faststore are not dependencies. Preserve all original records. "
      "Fresh existing editor.verify checked all 12 files, candidate bytes, and the candidate stream with "
      "the declared two-byte edit restored against original source SHA. The historical manifest lacked hashes; "
      "that fact remains true and the new full verification is recorded separately. No physical 62GB restore was made.\n\n"
      "To reconstruct, first sha256sum -c SHA256SUMS in this archive and BASE-SHA256SUMS in the original base. "
      "Decompress spec.json.gz to a fresh metadata path. Use the archived source with the existing editor create "
      "<original-base> <new-checkpoint> --spec <spec> --manifest <new-manifest> --apply. "
      "Run editor verify on that new manifest, then compare all 12 reconstructed files against CANDIDATE-SHA256SUMS.\n")
    (A / "README.md").write_text(text)
    save(A / "archive-validation.json", dict(status="passed", all12_fresh_hashes=True,
         all10_metadata_copied_and_hash_matched=True, spec_manifest_gzip_roundtrip=True,
         all_source_candidate_stats_unchanged=True, physical_restore=False, released_payloads=0))
    for root, states in ((base, audit["source_stats"]), (C, audit["candidate_stats"])):
        assert all(stamp(root / name) == state for name, state in states.items())
    files = sorted(p for p in A.rglob("*") if p.is_file())
    checksums = "".join(sha(p) + "  " + str(p.relative_to(A)) + "\n" for p in files)
    (A / "SHA256SUMS").write_text(checksums)
    for line in checksums.splitlines():
        digest, name = line.split("  ", 1)
        assert sha(A / name) == digest
    for p in A.rglob("*"):
        if p.is_file():
            with p.open("rb") as stream:
                os.fsync(stream.fileno())
    for directory in [A, *[p for p in A.rglob("*") if p.is_dir()]]:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    save(D / "archive-completion.json", dict(status="completed", elapsed_seconds=time.time() - started,
         fresh_verify_sha256=sha(A / "fresh-verify.json"), archive_sha256=sha(A / "archive.json"),
         checksum_list_sha256=sha(A / "SHA256SUMS"), archive_files=len(files) + 1,
         archive_bytes=sum(p.stat().st_size for p in A.rglob("*") if p.is_file()),
         payloads_released=0, payloads_unchanged=True, source_unchanged=True))
    print(json.dumps(read(D / "archive-completion.json")), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        save(D / ("archive-failed-" + str(time.time_ns()) + ".json"),
             dict(error_type=type(error).__name__, payloads_released=0, model_calls=0))
        raise
