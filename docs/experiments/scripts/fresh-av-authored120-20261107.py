"""One fixed four-batch authored generation; no scoring or service mutation."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.request


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def check_pins(plan):
    for pin in plan["pins"]:
        assert sha(pin["path"]) == pin["sha256"], "pinned input changed"
    for item in plan["checkpoint_files"]:
        st = (Path(plan["serving"]["checkpoint"]) / item["path"]).stat()
        assert st.st_size == item["size"]
        assert {"device": st.st_dev, "inode": st.st_ino, "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns} == item["stat"]


def preflight(plan):
    services = []
    for service, cid in plan["serving"]["service_cids"].items():
        data = json.loads(subprocess.check_output(["docker", "inspect", cid], text=True))[0]
        assert data["Id"] == cid and data["State"]["Running"] and data["State"]["Health"]["Status"] == "healthy"
        assert data["Config"]["Labels"].get("com.docker.compose.service") == service
        assert any(x["Source"] == plan["serving"]["checkpoint"] and not x["RW"] for x in data["Mounts"])
        ip = data["NetworkSettings"]["Networks"][plan["network"]]["IPAddress"]
        with urllib.request.urlopen(f"http://{ip}:8000/metrics", timeout=15) as response:
            metrics = response.read().decode()
        counts = [float(x) for x in re.findall(r"^vllm:num_requests_(?:running|waiting)(?:\{[^}]*\})?\s+([0-9.eE+-]+)", metrics, re.M)]
        assert len(counts) >= 2 and all(x == 0 for x in counts)
        with urllib.request.urlopen(f"http://{ip}:8000/v1/models", timeout=15) as response:
            ids = [x["id"] for x in json.load(response)["data"]]
        assert plan["serving"]["model"] in ids
        services.append({"service": service, "cid": cid, "ip": ip, "healthy": True, "idle_counts": counts, "model_ids": ids})
    gateway = plan["gateway"]
    data = json.loads(subprocess.check_output(["docker", "inspect", gateway["cid"]], text=True))[0]
    assert data["Id"] == gateway["cid"] and data["State"]["Running"] and data["State"]["Health"]["Status"] == "healthy"
    ip = data["NetworkSettings"]["Networks"][plan["network"]]["IPAddress"]
    assert gateway["endpoint"] == f"http://{ip}:8000/v1"
    with urllib.request.urlopen(gateway["endpoint"] + "/models", timeout=15) as response:
        ids = [x["id"] for x in json.load(response)["data"]]
    assert plan["serving"]["model"] in ids
    return {"epoch": time.time(), "services": services, "gateway_cid": gateway["cid"], "gateway_endpoint": gateway["endpoint"], "gateway_model_ids": ids, "generation_calls": 0}


def batch_summary(batch):
    output = Path(batch["output"])
    path = output / "train-responses.json"
    receipt = json.loads((output / "receipt.json").read_text()) if (output / "receipt.json").exists() else None
    rows = json.loads(path.read_text())["responses"] if path.exists() else []
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)) and set(ids) <= set(batch["ids"])
    return {"batch": batch["name"], "expected": len(batch["ids"]), "received": len(rows), "missing_ids": [x for x in batch["ids"] if x not in ids], "errors": sum("error_type" in x for x in rows), "length_finished": sum(x.get("finish_reason") == "length" for x in rows), "receipt": receipt, "responses_sha256": sha(path) if path.exists() else None, "rows": [{k: v for k, v in x.items() if k != "public_answer"} for x in rows], "answer_execution": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    check_pins(plan)
    assert sha(__file__) == plan["owner_sha256"]
    assert [b["name"] for b in plan["batches"]] == ["basic", "advanced-v2", "library-train-v1", "unsupported72"]
    assert [len(b["ids"]) for b in plan["batches"]] == [16, 16, 16, 72]
    spec = importlib.util.spec_from_file_location("authored_generation_input", plan["generator"])
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    thinking_control = plan.get("thinking_control", "request")
    assert thinking_control in {"request", "server-default"}
    ids = []
    for batch in plan["batches"]:
        rows = generator.public_cases(json.loads(Path(batch["cases"]).read_text()), "train")
        assert [x["id"] for x in rows] == batch["ids"]
        ids.extend(batch["ids"])
        expected = [plan["python"], plan["generator"], "--cases", batch["cases"], "--split", "train", "--endpoint", plan["gateway"]["endpoint"], "--model", plan["serving"]["model"], "--checkpoint-sha256", plan["serving"]["tree_sha256"], "--output", batch["output"], "--concurrency", "16", "--max-tokens", "32768", "--timeout", "1200", "--seed", "0"]
        if thinking_control == "server-default":
            expected += ["--thinking-control", "server-default"]
        assert batch["command"] == expected and not Path(batch["output"]).exists()
    assert len(ids) == len(set(ids)) == 120 and set(ids) == set(plan["original_ids"])
    if not args.run:
        print(json.dumps({"status": "preview", "plan_sha256": sha(args.plan), "approved": plan["approved_for_execution"], "cases": 120, "batch_sizes": [16, 16, 16, 72], "generation_calls": 0, "network_calls": 0, "service_changes": 0}))
        return 0
    assert plan["approved_for_execution"] is True and plan["authorization"]
    output = Path(plan["owner_output"])
    output.mkdir()
    started, summaries = time.time(), []
    def terminate(_signum, _frame):
        raise KeyboardInterrupt("termination requested")

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        with Path(plan["global_lock"]).open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            save(output / "owner-start.json", {"pid": os.getpid(), "epoch": started, "plan_sha256": sha(args.plan), "owner_sha256": sha(__file__), "lock_acquired": True})
            try:
                for index, batch in enumerate(plan["batches"]):
                    check_pins(plan)
                    save(output / f"preflight-{index}.json", preflight(plan))
                    began = time.time()
                    with (output / f"batch-{index}.log").open("x") as log:
                        process = subprocess.Popen(batch["command"] + ["--run"], cwd=plan["cwd"], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                        save(output / f"batch-{index}-start.json", {"pid": process.pid, "epoch": began, "command": batch["command"] + ["--run"]})
                        try:
                            code = process.wait()
                        finally:
                            if process.poll() is None:
                                process.wait()  # Keep the lock while an issued batch finishes.
                    summary = {**batch_summary(batch), "returncode": code, "elapsed_seconds": time.time() - began}
                    summaries.append(summary)
                    save(output / f"batch-{index}-complete.json", summary)
                    assert code == 0 and summary["receipt"]["status"] == "completed" and not summary["missing_ids"]
                check_pins(plan)
                save(output / "final-preflight.json", preflight(plan))
                save(output / "owner-complete.json", {"epoch": time.time(), "elapsed_seconds": time.time() - started, "batches": summaries, "cases_received": sum(x["received"] for x in summaries), "errors": sum(x["errors"] for x in summaries), "services_changed": False, "answer_execution": False, "repeated_cases": 0})
            finally:
                save(output / "all-batches-observed.json", {"epoch": time.time(), "batches": [batch_summary(b) for b in plan["batches"]], "missing_or_error_is_not_a_score": True})
    except BaseException as error:
        save(output / "owner-failure.json", {"epoch": time.time(), "elapsed_seconds": time.time() - started, "error_type": type(error).__name__, "no_automatic_retry": True, "services_changed": False})
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
