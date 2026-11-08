"""Metadata fingerprints and read-only validation of development followup evidence.

These functions do not query serving, skip recipe verification, or run a benchmark.
A current serving/full-tree snapshot must be produced by the caller under its lock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

DENSE_KIND = "dense_lora_recipe_v1"
CREATOR_PROOF = "Exact FP32 recipe regeneration and BF16 bytes, full source/target tree hashes, independent touched-shard inodes, every nonrecipe byte unchanged."


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("receipt must be a JSON object")
    return value


def _state(path: Path) -> dict[str, int]:
    if path.is_symlink():
        raise ValueError("receipt inputs cannot be symlinks")
    s = path.stat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError("receipt inputs must be regular files")
    return {"device": s.st_dev, "inode": s.st_ino, "size": s.st_size,
            "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns}


def _sha(path: Path) -> str:
    before = _state(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    if _state(path) != before:
        raise ValueError("input changed during fingerprint")
    return digest.hexdigest()


def fingerprints(files: dict[str, Path]) -> dict[str, Any]:
    """Hash explicitly supplied manifest/spec/code files; never read environments."""
    if not files or any(not isinstance(k, str) or not k for k in files):
        raise ValueError("nonempty labelled file mapping required")
    rows: list[dict[str, Any]] = []
    for label, path in sorted(files.items()):
        supplied = path.expanduser()
        if supplied.is_symlink():
            raise ValueError("receipt inputs cannot be symlinks")
        path = supplied.resolve()
        before = _state(path)
        digest = _sha(path)
        if _state(path) != before:
            raise ValueError("input changed during fingerprint")
        rows.append({"label": label, "path": str(path), "sha256": digest, "stat": before})
    for row in rows:
        if _state(Path(row["path"])) != row["stat"]:
            raise ValueError("input set changed while fingerprinting")
    return {"schema_version": 1, "kind": "explicit_file_fingerprints_v1", "files": rows,
            "scope": "Only the explicitly listed files; content fingerprints are not signed attestations"}


def freeze(files: dict[str, Path], output: Path, *, apply: bool = False) -> dict[str, Any]:
    if os.path.lexists(output) or not output.parent.is_dir():
        raise ValueError("output must be new with an existing parent")
    result = fingerprints(files)
    if not apply:
        return {**result, "applied": False}
    temporary = None
    try:
        fd, name = tempfile.mkstemp(prefix=".fingerprints-", dir=output.parent)
        temporary = Path(name)
        with os.fdopen(fd, "w") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    return {**result, "applied": True}


def verify_receipt(path: Path) -> dict[str, Any]:
    receipt_stamp = _state(path)
    record = _read(path)
    if record.get("schema_version") != 1 or record.get("kind") != "explicit_file_fingerprints_v1":
        raise ValueError("unsupported fingerprint receipt")
    rows = record["files"]
    files = {row["label"]: Path(row["path"]) for row in rows}
    if len(files) != len(rows):
        raise ValueError("duplicate fingerprint label")
    current = fingerprints(files)
    for expected, actual in zip(sorted(rows, key=lambda r: r["label"]), current["files"], strict=True):
        if any(expected[k] != actual[k] for k in ("label", "path", "sha256")):
            raise ValueError("frozen file fingerprint differs")
    if _state(path) != receipt_stamp:
        raise ValueError("receipt changed during validation")
    return {"verified": True, "files": len(rows), "scope": record["scope"],
            "receipt_sha256": _sha(path)}


def validate_followup(ticket: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Pure comparison of the legacy ticket and explicit current evidence metadata."""
    if ticket.get("schema_version") != 1 or current.get("split") != "dev":
        raise ValueError("only development followup tickets are supported")
    if not ticket.get("candidate") or current.get("candidate") != ticket["candidate"]:
        raise ValueError("candidate differs")
    expected = ticket["fingerprints"]
    required = {"manifest_sha256", "spec_sha256", "candidate_tree_sha256", "source_tree_sha256",
                "source_states", "code_sha256", "donor_sha256", "donor_config_sha256",
                "implementation_sha256", "recipe_sha256", "proof_input_states"}
    if not isinstance(expected, dict) or not required <= expected.keys() or not expected.get("code_sha256") or current.get("fingerprints") != expected:
        raise ValueError("source/manifest/spec/donor/implementation fingerprints differ")
    if (current.get("full_candidate_hash_recomputed") is not True or
            current.get("fresh_tree_sha256") != expected["candidate_tree_sha256"]):
        raise ValueError("fresh full candidate tree evidence required")
    manifest = current["manifest"]
    if (manifest.get("schema_version") != 2 or manifest.get("kind") != DENSE_KIND or
            manifest.get("candidate_tree_sha256") != expected["candidate_tree_sha256"] or
            manifest.get("source_tree_sha256") != expected["source_tree_sha256"] or
            current.get("checkpoint_path") != manifest.get("candidate")):
        raise ValueError("current checkpoint/creator manifest differs")
    if (manifest.get("verification") != CREATOR_PROOF or
            _json_hash(manifest["source_files"]) != manifest["source_tree_sha256"] or
            _json_hash(manifest["candidate_files"]) != manifest["candidate_tree_sha256"] or
            _json_hash(manifest["recipe"]) != expected["recipe_sha256"] or
            manifest["recipe"]["implementation_sha256"] != expected["implementation_sha256"] or
            _json_hash(manifest["recipe"]["implementation_files"]) != expected["implementation_sha256"] or
            manifest["spec_sha256"] != expected["spec_sha256"]):
        raise ValueError("original complete creator proof is missing or inconsistent")
    if any(expected["code_sha256"].get(name) != digest
           for name, digest in manifest["recipe"]["implementation_files"].items()):
        raise ValueError("code fingerprints do not cover the complete implementation bundle")
    replicas = current["replicas"]
    names = [r["name"] for r in replicas]
    original = ticket["replica_ids"]
    if not original or len(set(names)) != len(names):
        raise ValueError("duplicate or empty replica set")
    if sorted([{k: r[k] for k in ("name", "id", "image")} for r in replicas], key=lambda r: r["name"]) != sorted(original, key=lambda r: r["name"]):
        raise ValueError("serving replica identities differ")
    if any(r.get("running") is not True or r.get("healthy") is not True or
           r.get("checkpoint_path") != manifest["candidate"] or
           r.get("tree_sha256") != expected["candidate_tree_sha256"] for r in replicas):
        raise ValueError("all serving replicas must be healthy and use the same checkpoint")
    if "creator" not in ticket["creator"].get("proof", []):
        raise ValueError("original completed creator proof is required")
    for receipt in (ticket["creator"], ticket["completed_receipt"]):
        if not receipt.get("run_id") or not set(receipt.get("proof", [])) & {"creator", "full_recipe_verified"}:
            raise ValueError("completed creator/full verification receipt is required")
    return {"metadata_valid": True, "candidate": ticket["candidate"], "replicas": len(replicas),
            "automatic_skip_authorized": False,
            "scope": "Conditional metadata comparison only; caller must acquire fresh evidence under the evaluation lock",
            "required_outer_checks": ["current serving and mounted checkpoint identity",
                                      "fresh full candidate hash", "completed official dev scoring",
                                      "no active mutation between evidence collection and followup",
                                      "full manual/final/HF recipe verification remains required"]}


def verify_followup(ticket_path: Path, current_path: Path) -> dict[str, Any]:
    """Check current file evidence; no Docker, model, evaluation or queue operations."""
    initial = {_p: _state(_p) for _p in (ticket_path, current_path)}
    ticket, current = _read(ticket_path), _read(current_path)
    manifest_path = Path(current["manifest_path"])
    manifest_stamp = _state(manifest_path)
    manifest = _read(manifest_path)
    if _sha(manifest_path) != ticket["fingerprints"]["manifest_sha256"]:
        raise ValueError("manifest fingerprint differs")
    current = {**current, "manifest": manifest}
    result = validate_followup(ticket, current)
    tracked: dict[Path, dict[str, int]] = {**initial, manifest_path: manifest_stamp}
    proof = ticket["fingerprints"]
    spec_path = Path(manifest["spec_path"])
    required_paths = [manifest_path, spec_path]
    if _sha(spec_path) != proof["spec_sha256"] or _read(spec_path) != manifest["spec"]:
        raise ValueError("specification bytes differ")
    recipe = manifest["recipe"]
    for key in ("donor", "donor_config"):
        path = Path(recipe[key + "_path"])
        required_paths.append(path)
        if recipe[key + "_sha256"] != proof[key + "_sha256"] or _sha(path) != proof[key + "_sha256"]:
            raise ValueError("donor/config bytes differ")
    known_paths = [Path(value) for value in proof["proof_input_states"]]
    for label, digest in proof["code_sha256"].items():
        matches = [path for path in known_paths if path.name == label]
        if len(matches) != 1 or _sha(matches[0]) != digest:
            raise ValueError("implementation code bytes differ or are ambiguous")
        required_paths.append(matches[0])
    if any(str(path) not in proof["proof_input_states"] for path in required_paths):
        raise ValueError("required proof input is missing its continuity state")
    for receipt in (ticket["creator"], ticket["completed_receipt"]):
        for key in ("status", "log"):
            path = Path(receipt[key + "_path"])
            tracked[path] = _state(path)
            if _sha(path) != receipt[key + "_sha256"]:
                raise ValueError("completed evidence file changed")
        status = _read(Path(receipt["status_path"]))
        if (status.get("status") != "completed" or status.get("split") != "dev" or
                status.get("candidate") != ticket["candidate"] or status.get("run_id") != receipt["run_id"] or
                status["checkpoint"]["path"] != manifest["candidate"] or
                status["checkpoint"]["tree_sha256"] != manifest["candidate_tree_sha256"] or
                "postcheck_seconds" not in status.get("timings", {})):
            raise ValueError("completed dev/postcheck evidence differs")
        proven = set()
        with Path(receipt["log_path"]).open() as stream:
            for line in stream:
                if not line.startswith("{"):
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(row, dict) or row.get("kind") != DENSE_KIND:
                    continue
                if (row.get("candidate_tree_sha256") != manifest["candidate_tree_sha256"] or
                        row.get("changed_elements") != manifest["changed_elements"] or
                        row.get("delta_l2") != manifest["delta_l2"]):
                    continue
                if row.get("verified") is True:
                    proven.add("full_recipe_verified")
                if all(row.get(k) == manifest.get(k) for k in ("candidate", "source", "elapsed_seconds")):
                    proven.add("creator")
        if not set(receipt["proof"]) <= proven:
            raise ValueError("claimed creator/full-verification log proof is absent")
    for path_string, expected in proof["proof_input_states"].items():
        path = Path(path_string)
        if _state(path) != expected:
            raise ValueError("proof input state changed")
        tracked[path] = expected
    source = Path(manifest["source"])
    actual_source = {}
    for path in source.rglob("*"):
        rel = path.relative_to(source)
        if ".cache" in rel.parts or rel.name == ".kairyu-model-attestation.json":
            continue
        if path.is_symlink():
            raise ValueError("source inventory contains a symlink")
        if path.is_dir():
            continue
        actual_source[str(rel)] = _state(path)
    if actual_source != proof["source_states"]:
        raise ValueError("complete source inventory/state changed")
    for relative_name, value in actual_source.items():
        tracked[source / relative_name] = value
    if any(_state(path) != expected for path, expected in tracked.items()):
        raise ValueError("evidence changed during validation")
    return {**result, "evidence_files_verified": True, "ticket_sha256": _sha(ticket_path),
            "current_snapshot_sha256": _sha(current_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freezer = sub.add_parser("freeze")
    freezer.add_argument("--file", action="append", required=True, help="LABEL=PATH; explicit metadata/code inputs")
    freezer.add_argument("--output", type=Path, required=True)
    freezer.add_argument("--apply", action="store_true")
    verifier = sub.add_parser("verify")
    verifier.add_argument("receipt", type=Path)
    followup = sub.add_parser("followup")
    followup.add_argument("ticket", type=Path)
    followup.add_argument("current", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        pairs = [value.split("=", 1) for value in args.file]
        if any(len(pair) != 2 for pair in pairs) or len({pair[0] for pair in pairs}) != len(pairs):
            raise ValueError("unique LABEL=PATH arguments required")
        result = freeze({label: Path(path) for label, path in pairs}, args.output, apply=args.apply)
    elif args.command == "verify":
        result = verify_receipt(args.receipt)
    else:
        result = verify_followup(args.ticket, args.current)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
