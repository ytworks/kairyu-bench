"""Validate public MBPP training references in the existing isolated CPU Docker boundary."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time
import uuid

import science_sandbox as sandbox

CHILD = r"""
import contextlib,json,os,sys,time
row=json.loads(sys.stdin.read())
report={"task_id":row["task_id"],"scopes":{}}
for scope in ("test_list","challenge_test_list"):
    tests=row.get(scope,[])
    started=time.monotonic()
    target={"planned":len(tests),"tests":[],"setup_error":None}
    namespace={"__name__":"__main__"}
    with open(os.devnull,"w") as sink,contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):
        try:
            exec(compile(row["code"],"<public-reference>","exec"),namespace)
            exec(compile(row.get("test_setup_code",""),"<public-test-setup>","exec"),namespace)
        except BaseException as error:
            target["setup_error"]=type(error).__name__
        for index,test in enumerate(tests):
            begin=time.monotonic()
            if target["setup_error"] is not None:
                result={"index":index,"status":"unexecuted","exception_type":target["setup_error"]}
            else:
                try:
                    exec(compile(test,"<public-test>","exec"),namespace)
                    result={"index":index,"status":"passed","exception_type":None}
                except BaseException as error:
                    result={"index":index,"status":"failed","exception_type":type(error).__name__}
            result["seconds"]=time.monotonic()-begin
            target["tests"].append(result)
    target["seconds"]=time.monotonic()-started
    report["scopes"][scope]=target
print(json.dumps(report,allow_nan=False))
"""

WORKER = """
import json,os,selectors,subprocess,sys,threading,time
""" + inspect.getsource(sandbox.bounded_command).replace("max_output: int = MAX_OUTPUT", "max_output: int = 1048576") + """
payload=json.loads(sys.stdin.read())
for row in payload["rows"]:
    started=time.monotonic()
    result={"task_id":row["task_id"],"split":row["_split"],"status":"unknown","planned_tests":len(row["test_list"]),"planned_challenge_tests":len(row.get("challenge_test_list",[]))}
    try:
        code,output=bounded_command([sys.executable,"-I","-c",payload["child"]],payload=json.dumps(row).encode(),seconds=10,max_output=1048576)
        if code:
            result.update(error="child_exit",returncode=code)
        else:
            scored=json.loads(output)
            assert scored["task_id"]==row["task_id"]
            result.update(status="evaluated",scopes=scored["scopes"],error=None)
    except BaseException as error:
        result.update(error=type(error).__name__)
    result["seconds"]=time.monotonic()-started
    print(json.dumps(result,allow_nan=False),flush=True)
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows, pins = [], {}
    for split in ("train", "holdout"):
        path = args.inputs / (split + ".jsonl")
        pins[str(path)] = sha(path)
        data = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(data) == {"train": 204, "holdout": 51}[split]
        rows += [dict(row, _split=split) for row in data]
    assert len({row["task_id"] for row in rows}) == 255
    image = "sha256:c4ab50223f03fca17bf871e4d05f77001e76525a1c574b6c6ee7024db06cfdc2"
    docker = sandbox.Docker(
        Path("/var/run/docker.sock"), "3e518d2b-6972-43f8-b325-f7f33697a2d2",
        image, seconds=300, memory_mib=512, cpus=1,
    )
    docker.check_image()

    def inventory():
        code, output = docker.cli(["ps", "-aq", "--no-trunc"], max_output=65536)
        assert code == 0
        ids = output.decode().splitlines()
        code, output = docker.cli(["inspect", *ids], max_output=4 * 1024 * 1024)
        assert code == 0
        return {x["Id"]: {
            "image": x["Image"],
            "state": {key: x["State"].get(key) for key in ("Status", "Running", "Pid", "StartedAt", "FinishedAt")},
            "mounts": sorted([{"source": m["Source"], "target": m["Destination"], "readonly": not m["RW"]} for m in x["Mounts"]], key=lambda m: m["target"]),
        } for x in json.loads(output)}

    before = inventory()
    assert len(before) == 27
    nonce = uuid.uuid4().hex
    name = "kairyu-mbpp-reference-" + nonce
    argv = docker.arguments(name, nonce)
    assert argv[-1] == sandbox.WORKER
    argv[-1] = WORKER
    plan = {
        "source_sha256": sha(__file__), "sandbox_source_sha256": sha(sandbox.__file__),
        "input_sha256": pins, "rows": 255, "split_counts": {"train": 204, "holdout": 51},
        "container_name": name, "ownership_nonce": nonce, "argv": argv,
        "worker_sha256": hashlib.sha256(WORKER.encode()).hexdigest(),
        "child_sha256": hashlib.sha256(CHILD.encode()).hexdigest(),
        "container_timeout_seconds": 300, "case_timeout_seconds": 10,
        "scope": "Original public TRAIN reference and original tests; challenge separately reported; no generated code or official benchmark gold.",
        "model_calls": 0, "host_data_code_exec": False,
    }
    save(args.output / "plan.json", plan)
    save(args.output / "protected-before.json", before)
    error, output, returncode = None, b"", None
    try:
        returncode, output = docker.cli(
            argv, payload=json.dumps({"rows": rows, "child": CHILD}).encode(),
            seconds=300, max_output=4 * 1024 * 1024,
        )
    except BaseException as exc:
        error = type(exc).__name__
    finally:
        docker.cleanup(name, nonce)
    (args.output / "raw-worker.jsonl").write_bytes(output)
    observed = {}
    for line in output.splitlines():
        row = json.loads(line)
        assert row["task_id"] not in observed
        observed[row["task_id"]] = row
    ledger = []
    for row in rows:
        result = observed.get(row["task_id"], {
            "task_id": row["task_id"], "split": row["_split"], "status": "unknown",
            "error": error or "missing_worker_result", "seconds": None,
            "planned_tests": len(row["test_list"]),
            "planned_challenge_tests": len(row.get("challenge_test_list", [])),
        })
        original = result.get("scopes", {}).get("test_list", {})
        tests = original.get("tests", [])
        passed = result["status"] == "evaluated" and original.get("setup_error") is None and len(tests) == len(row["test_list"]) and len(tests) > 0 and all(x["status"] == "passed" for x in tests)
        result["reference_original_tests_passed"] = passed
        result["quality_decision"] = (
            "eligible_training_reference" if passed and row["_split"] == "train" else
            "holdout_reference_original_tests_confirmed" if passed else
            "exclude_unverified_training_reference" if row["_split"] == "train" else
            "holdout_reference_not_confirmed"
        )
        ledger.append(result)
    save(args.output / "all255-quality.json", ledger)
    after = inventory()
    save(args.output / "protected-after.json", after)
    summary = {
        "status": "completed" if len(observed) == 255 and error is None and returncode == 0 else "partial",
        "rows": len(ledger), "observed": len(observed), "elapsed_seconds": time.monotonic() - started,
        "container_returncode": returncode, "container_error": error,
        "input_unchanged": all(sha(path) == value for path, value in pins.items()),
        "protected27_unchanged": before == after, "owned_cleanup_completed": True,
        "model_calls": 0, "original_test_count": sum(len(x["test_list"]) for x in rows),
        "challenge_test_count": sum(len(x.get("challenge_test_list", [])) for x in rows),
        "splits": {},
    }
    for split in ("train", "holdout"):
        subset = [x for x in ledger if x["split"] == split]
        summary["splits"][split] = {
            "rows": len(subset), "passed": sum(x["reference_original_tests_passed"] for x in subset),
            "unconfirmed_ids": [x["task_id"] for x in subset if not x["reference_original_tests_passed"]],
            "unknown": sum(x["status"] == "unknown" for x in subset),
            "eligible_ids": [x["task_id"] for x in subset if x["reference_original_tests_passed"]],
        }
    save(args.output / "summary.json", summary)
    print(json.dumps(summary))
    assert summary["input_unchanged"] and summary["protected27_unchanged"]


if __name__ == "__main__":
    main()
