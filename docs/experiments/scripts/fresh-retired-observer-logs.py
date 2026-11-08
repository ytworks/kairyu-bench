#!/usr/bin/env python3
"""Preserve only owned stopped observer logs after exact healthy restoration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def redact(data, secrets):
    count = 0
    for value in sorted(secrets, key=len, reverse=True):
        encoded = value.encode()
        count += data.count(encoded)
        data = data.replace(encoded, b"[REDACTED_KNOWN_SECRET]")
    for pattern in [rb"hf_[A-Za-z0-9]{8,}", rb"sk[-_][A-Za-z0-9_-]{8,}",
                    rb"(?i)Bearer[ \t]+[^\s,\"']+",
                    rb"(?i)\b(?:api_key|password|secret|token)[ \t]*[:=][ \t]*[^,;\s]+"]:
        data, hits = re.subn(pattern, b"[REDACTED_CREDENTIAL_FIELD]", data)
        count += hits
    assert all(value.encode() not in data for value in secrets)
    return data, count


def inspect(cid):
    return json.loads(subprocess.check_output(["docker", "inspect", cid]))[0]


def private_write(path, payload):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        safe, count = redact(b"metrics=1 secret-value-123 Bearer abcDEF987 hf_abcdefgh1234", {"secret-value-123"})
        assert count == 3 and b"secret-value-123" not in safe and b"abcDEF987" not in safe and b"hf_abcdefgh1234" not in safe
        assert redact(b"metrics=1 requests=20", set()) == (b"metrics=1 requests=20", 0)
        print("redaction checks passed; no Docker or filesystem operation")
        return
    assert args.artifacts is not None and os.getuid() == 1003
    root = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
    artifacts = args.artifacts.resolve()
    assert artifacts.is_relative_to(root)
    plan = json.loads((artifacts / "plan.json").read_text())
    backup = Path(plan["capture"]["backup"]) / plan["original_container"]["name"]
    state = json.loads((backup / "state.json").read_text())
    assert state["status"] == "restored"
    assert state["original_id"] == plan["original_container"]["id"]
    original = inspect(state["original_id"])
    observer = inspect(state["observer_id"])
    assert original["State"]["Health"]["Status"] == "healthy"
    assert original["Name"].lstrip("/") == plan["original_container"]["name"]
    assert not observer["State"]["Running"] and "observer-retired-" in observer["Name"]
    labels = observer["Config"].get("Labels", {})
    assert state["operation_id"] in labels.values()
    assert labels["kairyu.bench.capture_id"] == state["capture_id"]
    closed = json.loads((artifacts / "worker-closed.json").read_text())
    assert len(closed) == 2
    assert all(x["writer_queue"]["depth"] == 0 and x["writer_queue"]["pending_bytes"] == 0 for x in closed.values())
    secrets = set()
    for item in [original, observer]:
        for entry in item["Config"].get("Env", []):
            key, sep, value = entry.partition("=")
            if sep and value and re.search(r"(?:TOKEN|SECRET|PASSWORD|API_KEY|ACCESS_KEY)(?:_ID)?$", key, re.I) and value.lower() not in {"none", "null", "empty"}:
                secrets.add(value)
        command = [*(item["Config"].get("Entrypoint") or []), *(item["Config"].get("Cmd") or [])]
        for index, argument in enumerate(command):
            if re.fullmatch(r"--(?:api-key|hf-token|password|secret)", argument) and index + 1 < len(command):
                secrets.add(command[index + 1])
    result = subprocess.run(["docker", "logs", "--timestamps", state["observer_id"]], capture_output=True)
    assert result.returncode == 0, "docker logs failed; no log payload saved"
    directory = artifacts / "retired-observer-logs"
    directory.mkdir(mode=0o700)
    records = {}
    for name, raw in [("stdout.log", result.stdout), ("stderr.log", result.stderr)]:
        safe, count = redact(raw, secrets)
        private_write(directory / name, safe)
        records[name] = {"input_bytes": len(raw), "saved_bytes": len(safe), "redactions": count, "sha256": hashlib.sha256(safe).hexdigest()}
    receipt = {"recorded_epoch": time.time(), "original_cid": original["Id"], "original_healthy": True,
               "observer_cid": observer["Id"], "observer_name": observer["Name"], "observer_state": observer["State"]["Status"],
               "compose_labels": {k: v for k, v in labels.items() if k.startswith("com.docker.compose.") and k in {"com.docker.compose.project", "com.docker.compose.service", "com.docker.compose.container-number", "com.docker.compose.oneoff"}},
               "closed_worker_count": sum(bool(x["closed"]) for x in closed.values()),
               "worker_count": len(closed), "writer_close_acknowledged": all(x["closed"] for x in closed.values()),
               "worker_status_unmodified": True, "files": records, "credential_values_or_env_dump_saved": False,
               "method": "docker logs --timestamps; no --details; known environment/command secret values and credential-shaped fields redacted in memory before write",
               "output_permissions": "directory0700/files0600", "service_or_container_mutation": False,
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    private_write(directory / "receipt.json", (json.dumps(receipt, indent=2) + "\n").encode())
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
