"""Small canonical WSP wrapper owner; no model deployment or cleanup."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    output = args.plan.parent
    command = plan["command"]
    assert command[1:3] == ["wsp", "run"]
    assert not any(x in command for x in ["--only", "--exclude-item", "--limit", "--incomplete-manifest"])
    assert plan["selection"]["items"] == 2338 and plan["selection"]["probe_count_without_official_area"] == 96
    assert len(plan["selection"]["named_area_counts"]) == 14
    for flag, expected in [("--tier", "30"), ("--split", "dev"), ("--chat-concurrency", "256"), ("--logprob-concurrency", "32"), ("--agent-concurrency", "4")]:
        assert command[command.index(flag) + 1] == expected
    assert command.count("--logprob-endpoint") == 4
    config_path = Path(plan["http_trace"]["config"]["path"])
    assert sha(config_path) == plan["http_trace"]["config"]["sha256"]
    assert sha(Path(command[0])) == plan["source_identity"]["wrapper"]["sha256"]
    result_dir = Path(plan["cwd"]) / plan["run_id"]
    assert not result_dir.exists()
    assert not (config_path.parent / "requests-client.jsonl").exists()
    if not args.run:
        print(json.dumps({"preview": True, "plan_sha256": sha(args.plan), "approved": plan["approved_for_execution"], "items": 2338, "model_calls": 0}))
        return 0
    assert plan["approved_for_execution"] is True and plan["handoff"]
    serving = plan["expected_serving"]
    assert len(serving["service_cids"]) == 4 and len(serving["closures"]) == 4
    started = time.time()
    with Path(plan["global_lock"]).open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(output / "owner-start.json", {"pid": os.getpid(), "epoch": started, "plan_sha256": sha(args.plan), "owner_sha256": sha(Path(__file__)), "lock_acquired": True})
        try:
            for pin in serving["closures"]:
                assert sha(Path(pin["path"])) == pin["sha256"]
            config = json.loads(config_path.read_text())
            for item in config["checkpoint"]["files"]:
                st = (Path(serving["checkpoint"]) / item["path"]).stat()
                assert st.st_size == item["size"]
                assert {"device": st.st_dev, "inode": st.st_ino, "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns} == item["stat"]
            env = {**os.environ, **plan["environment_allowlist"]}
            for name in ["KAIRYU_BENCH_DEEPSWE_TASKS", "KAIRYU_BENCH_TERMINAL_BENCH_TASKS", "KAIRYU_BENCH_SWEBENCH_PRO_INSTANCES"]:
                assert not env.get(name), "task selector override is forbidden"
            image = env["KAIRYU_BENCH_RUNNER_IMAGE_ID"]
            assert subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True).strip() == image
            services = []
            for service, cid in sorted(serving["service_cids"].items()):
                data = json.loads(subprocess.check_output(["docker", "inspect", cid], text=True))[0]
                assert data["Id"] == cid and data["State"]["Running"] and data["State"]["Health"]["Status"] == "healthy"
                assert data["Config"]["Labels"].get("com.docker.compose.service") == service
                assert any(x["Source"] == serving["checkpoint"] and not x["RW"] for x in data["Mounts"])
                ip = data["NetworkSettings"]["Networks"][env["KAIRYU_BENCH_DOCKER_NETWORK"]]["IPAddress"]
                with urllib.request.urlopen(f"http://{ip}:8000/metrics", timeout=15) as response:
                    metrics = response.read().decode()
                counts = [float(x) for x in re.findall(r"^vllm:num_requests_(?:running|waiting)(?:\{[^}]*\})?\s+([0-9.eE+-]+)", metrics, re.M)]
                assert len(counts) >= 2 and all(x == 0 for x in counts)
                with urllib.request.urlopen(f"http://{ip}:8000/v1/models", timeout=15) as response:
                    model_ids = [x["id"] for x in json.load(response)["data"]]
                assert serving["model"] in model_ids
                services.append({"service": service, "cid": cid, "ip": ip, "healthy": True, "idle_counts": counts, "model_ids": model_ids})
            save(output / "launch-preflight.json", {"epoch": time.time(), "services": services, "checkpoint_stats_equal": True, "model_generation_calls": 0})
            with (output / "wsp.log").open("x") as log:
                process = subprocess.Popen(command, cwd=plan["cwd"], env=env, stdout=log, stderr=subprocess.STDOUT)
                save(output / "wrapper-start.json", {"pid": process.pid, "epoch": time.time(), "command": command, "environment_allowlist": plan["environment_allowlist"]})
                returncode = process.wait()
            save(output / "owner-complete.json", {"epoch": time.time(), "elapsed_seconds": time.time() - started, "returncode": returncode, "services_changed_by_owner": False})
            return returncode
        except BaseException as error:
            save(output / "owner-failure.json", {"epoch": time.time(), "elapsed_seconds": time.time() - started, "error_type": type(error).__name__, "message": str(error), "no_automatic_rerun": True, "services_changed_by_owner": False})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
