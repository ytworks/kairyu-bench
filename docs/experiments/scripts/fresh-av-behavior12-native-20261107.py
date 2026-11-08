"""One pinned native train cohort; preview only unless its exact plan is approved."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.request

IDS = [f"behavior-{letter}{i}" for letter in "ABCDEF" for i in (1, 2)]
TREE = "5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf"
IMAGE = "sha256:af9e746b913088cbf6c4464a31d6904a6b7e1d3ba62f0895362ddf13595f7abe"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def validate_train(document, expected_ids=None):
    groups = document.get("cases", document)
    if expected_ids is None:
        expected_ids = IDS
        if list(groups) != list("ABCDEF") or any(len(folds) != 2 for folds in groups.values()):
            raise ValueError("legacy six purposes and two folds required")
    if (not isinstance(expected_ids, list) or not 1 <= len(expected_ids) <= 256
            or len(set(expected_ids)) != len(expected_ids)
            or any(not isinstance(i, str) or not i for i in expected_ids)
            or not isinstance(groups, dict) or not groups
            or any(not folds or any(len(fold) != 1 for fold in folds) for folds in groups.values())):
        raise ValueError("unique explicit IDs and singleton train folds required")
    rows = [r for folds in groups.values() for fold in folds for r in fold]
    if [r.get("id") for r in rows] != expected_ids or any(r.get("split") != "train" for r in rows):
        raise ValueError("exact ordered train IDs and train split required")
    if any(not isinstance(r.get(k), str) or not r[k] for r in rows for k in ("prompt", "correct", "foil")):
        raise ValueError("complete original authored row fields required")
    return rows


def command(plan, run=False):
    result = [
        plan["python"], plan["generator"], "--cases", plan["cases"], "--split", "train",
        "--endpoint", plan["gateway"]["endpoint"], "--model", "gemma-4-31b",
        "--checkpoint-sha256", plan["serving"]["tree_sha256"], "--output", plan["generation_output"],
        "--concurrency", "16", "--max-tokens", "32768", "--timeout", "1200",
        "--seed", "0", "--thinking-control", "server-default",
    ]
    return result + (["--run"] if run else [])


def check_pins(plan):
    for pin in plan["pins"]:
        if sha(pin["path"]) != pin["sha256"]:
            raise ValueError("pinned source/input changed")
    if sha(__file__) != plan["owner_sha256"]:
        raise ValueError("owner source changed")
    for old in plan["checkpoint_files"]:
        s = (Path(plan["serving"]["checkpoint"]) / old["path"]).stat()
        current = {"device": s.st_dev, "inode": s.st_ino, "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns}
        if current != old["stat"] or s.st_size != old["size"]:
            raise ValueError("parent checkpoint stat continuity lost")


def check_plan(plan):
    required = {str(Path(plan[k]).resolve()) for k in ("python", "generator", "cases", "frame_evidence")}
    if not required <= {str(Path(p["path"]).resolve()) for p in plan["pins"]}:
        raise ValueError("all executable and input sources must be pinned")
    check_pins(plan)
    explicit = plan.get("schema_version", 1) == 2
    validate_train(json.loads(Path(plan["cases"]).read_text()), plan["ids"] if explicit else None)
    tree = plan["serving"]["tree_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", tree) or (not explicit and (plan["ids"] != IDS or tree != TREE)):
        raise ValueError("pinned cohort/checkpoint identity required")
    if list(plan["serving"]["service_cids"]) != [f"gemma-{i}" for i in range(4)]:
        raise ValueError("all four pinned replicas required")
    if plan["runtime_image"] != IMAGE or len(plan["protected_containers"]) != 23:
        raise ValueError("fixed image and protected service set required")
    evidence = json.loads(Path(plan["frame_evidence"]).read_text())
    lengths = evidence["lengths"]
    expected_ids = plan["ids"] if explicit else IDS + [f"behavior-{a}3" for a in "ABCDEF"]
    if (evidence["expected_count"] != len(expected_ids) or evidence["over_limit_ids"]
            or [r["id"] for r in lengths] != expected_ids
            or evidence["input_sha256"]["train-cases.json"] != sha(plan["cases"])
            or any(max(r["correct_complete"], r["foil_complete"]) > 1024 for r in lengths)
            or evidence["frame_protocol"] != "gemma4-public-answer-v2-forced-empty-thought"):
        raise ValueError("complete matching frames must pass 1024 before generation")
    if plan["command"] != command(plan) or plan["thinking_control"] != "server-default":
        raise ValueError("generation settings changed")
    if any(Path(plan[k]).exists() for k in ("generation_output", "owner_output")):
        raise FileExistsError("a fresh output is required; never resume/retry")


def require_authorization(plan, actual_sha, approved_sha):
    if plan["approved_for_execution"] is not True or not plan.get("authorization") or approved_sha != actual_sha:
        raise ValueError("explicit approval of the exact plan SHA required")


def inspect(cid):
    return json.loads(subprocess.check_output(["docker", "inspect", cid], text=True))[0]


def get_json(url):
    with urllib.request.urlopen(url, timeout=15) as response:
        return json.load(response)


def preflight(plan):
    records = []
    for expected in plan["protected_containers"]:
        d = inspect(expected["cid"])
        actual = {"cid": d["Id"], "name": d["Name"], "image": d["Image"],
                  "running": d["State"]["Running"], "status": d["State"]["Status"],
                  "health": d["State"].get("Health", {}).get("Status")}
        if actual != expected:
            raise ValueError("protected container changed")
    for expected in plan["services"]:
        d = inspect(expected["cid"])
        labels = d["Config"]["Labels"]
        members = subprocess.check_output([
            "docker", "ps", "-aq", "--no-trunc", "--filter",
            "label=com.docker.compose.project=gemma-4-31b-8gpu",
            "--filter", "label=com.docker.compose.service=" + expected["service"],
        ], text=True).split()
        mounts = [{"source": m["Source"], "destination": m["Destination"], "rw": m["RW"]}
                  for m in d["Mounts"] if m["Destination"] == "/checkpoint"]
        cmd_sha = hashlib.sha256(json.dumps(d["Config"]["Cmd"], separators=(",", ":")).encode()).hexdigest()
        if not (members == [expected["cid"]] and d["Id"] == expected["cid"] and d["Image"] == IMAGE
                and d["State"]["Running"] and d["State"]["Health"]["Status"] == "healthy"
                and labels["com.docker.compose.service"] == expected["service"]
                and mounts == [expected["readonly_model_mount"]] and cmd_sha == expected["cmd_sha256"]):
            raise ValueError("ordinary homogeneous parent serving identity changed")
        ip = d["NetworkSettings"]["Networks"][plan["network"]]["IPAddress"]
        with urllib.request.urlopen(f"http://{ip}:8000/metrics", timeout=15) as response:
            metrics = response.read().decode()
        counts = [float(x) for x in re.findall(
            r"^vllm:num_requests_(?:running|waiting)(?:\{[^}]*\})?\s+([0-9.eE+-]+)", metrics, re.M)]
        if len(counts) < 2 or any(x != 0 for x in counts):
            raise ValueError("serving requests not idle")
        ids = [x["id"] for x in get_json(f"http://{ip}:8000/v1/models")["data"]]
        if "gemma-4-31b" not in ids:
            raise ValueError("direct model identity mismatch")
        records.append({"cid": d["Id"], "ip": ip, "idle_counts": counts, "model_ids": ids})
    gateway = plan["gateway"]
    d = inspect(gateway["cid"])
    ip = d["NetworkSettings"]["Networks"][plan["network"]]["IPAddress"]
    if (d["Image"] != gateway["image_id"] or not d["State"]["Running"]
            or d["State"]["Health"]["Status"] != "healthy"
            or gateway["endpoint"] != f"http://{ip}:8000/v1"):
        raise ValueError("gateway identity changed")
    ids = [x["id"] for x in get_json(gateway["endpoint"] + "/models")["data"]]
    if "gemma-4-31b" not in ids:
        raise ValueError("gateway model identity mismatch")
    return {"epoch": time.time(), "services": records, "gateway_model_ids": ids, "protected": 23}


def summary(plan):
    path = Path(plan["generation_output"]) / "train-responses.json"
    rows = json.loads(path.read_text())["responses"] if path.exists() else []
    ids = [r["id"] for r in rows]
    expected = plan["ids"]
    if len(ids) != len(set(ids)) or not set(ids) <= set(expected):
        raise ValueError("response IDs duplicate or outside fixed train12")
    return {"expected": len(expected), "received": len(rows), "missing_ids": [i for i in expected if i not in ids],
            "errors": sum("error_type" in r for r in rows),
            "empty": sum(r.get("public_answer") == "" for r in rows),
            "length_finished": sum(r.get("finish_reason") == "length" for r in rows),
            "responses_sha256": sha(path) if path.exists() else None,
            "rows": [{k: v for k, v in r.items() if k != "public_answer"} for r in rows]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--approved-plan-sha256")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    check_plan(plan)
    digest = sha(args.plan)
    if not args.run:
        print(json.dumps({"status": "preview", "plan_sha256": digest, "cases": len(plan["ids"]),
                          "approved": plan["approved_for_execution"], "model_calls": 0, "network_calls": 0}))
        return
    require_authorization(plan, digest, args.approved_plan_sha256)
    output = Path(plan["owner_output"])
    output.mkdir()
    stopped = []
    def terminate(signum, _frame):
        stopped.append({"signal": signum, "epoch": time.time()})
    old_handlers = {sig: signal.signal(sig, terminate) for sig in (signal.SIGINT, signal.SIGTERM)}
    process, started = None, time.time()
    try:
        with Path(plan["global_lock"]).open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            save(output / "start.json", {"pid": os.getpid(), "epoch": started, "plan_sha256": digest})
            try:
                check_pins(plan)
                save(output / "preflight.json", preflight(plan))
                if stopped:
                    raise RuntimeError("stop requested before generation")
                with (output / "generator.log").open("x") as log:
                    process = subprocess.Popen(command(plan, True), cwd=plan["cwd"], stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    save(output / "child.json", {"pid": process.pid, "command": command(plan, True)})
                    returncode = process.wait()
                result = summary(plan)
                save(output / "generation-complete.json", {"returncode": returncode, **result})
                if returncode != 0 or result["missing_ids"]:
                    raise RuntimeError("generator process failed or response IDs are incomplete")
            finally:
                if process is not None:
                    # Repeated handled stop signals cannot release the lock before child/HTTP closure.
                    process.wait()
                    attempt = 0
                    while True:
                        try:
                            closure = preflight(plan)
                            save(output / "serving-closure.json", closure)
                            break
                        except Exception as error:
                            save(output / f"closure-pending-{attempt}.json",
                                 {"epoch": time.time(), "error_type": type(error).__name__, "lock_retained": True})
                            attempt += 1
                            time.sleep(5)
                    save(output / "observed-responses.json", summary(plan))
                check_pins(plan)
                save(output / "owner-closure.json", {"epoch": time.time(), "elapsed_seconds": time.time()-started,
                     "stop_requests": stopped, "child_finished": process is None or process.poll() is not None,
                     "model_requests_issued": process is not None, "retries": 0, "services_changed": False})
        save(output / "normal-lock-release.json", {"epoch": time.time(), "released": True})
    except BaseException as error:
        save(output / "failure.json", {"epoch": time.time(), "error_type": type(error).__name__,
                                      "no_automatic_retry": True, "stop_requests": stopped})
        raise
    finally:
        for sig, previous in old_handlers.items():
            signal.signal(sig, previous)


if __name__ == "__main__":
    main()
