#!/usr/bin/env python3
"""Run pinned matrix calibrations and BF16 proposals once, with two bounded CPU workers."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

import propose_native_objectives as adapter

STOP = threading.Event()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def terminate(child):
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)


def command(argv, log, seconds, cpu_set, environment):
    started = time.monotonic()
    with log.open("x") as stream:
        child = subprocess.Popen(["taskset", "-c", cpu_set, *argv], stdout=stream,
                                 stderr=subprocess.STDOUT, env=environment,
                                 start_new_session=True)
        try:
            while child.poll() is None:
                if STOP.is_set():
                    raise InterruptedError("batch interrupted")
                if time.monotonic() - started >= seconds:
                    raise TimeoutError("fixed process deadline; no retry")
                time.sleep(.1)
            return dict(returncode=child.returncode, seconds=time.monotonic() - started)
        finally:
            terminate(child)


def prepare(args):
    pins = {}
    roster = adapter.document(args.roster, pins, args.roster_sha256)
    adapter.capture(Path(__file__), pins)
    policy = adapter.document(Path(roster["policy"]["path"]), pins, roster["policy"]["sha256"])
    receipt = adapter.document(Path(roster["collection_receipt"]["path"]), pins,
                               roster["collection_receipt"]["sha256"])
    expected_calibration = dict(calibration_seconds=120, candidate_seconds=180,
                               fractional_buffer=1/128, floor=1e-6,
                               dual_tolerance=2e-10, maximum_iterations=20000)
    calibration = dict(policy["analytic_calibration"])
    declared_metric = calibration.pop("metric", None)
    raw_cap = policy["realized_raw_cap"]
    if (roster["schema_version"] != 1 or policy["protocol"] != "functional_targets_v1"
            or policy["parent_state"] != roster["parent"]
            or calibration != expected_calibration
            or declared_metric not in (None, "euclidean", "relative")
            or not policy["native_frozen"] or type(raw_cap) not in (int, float)
            or not math.isfinite(raw_cap) or raw_cap <= 0
            or not receipt["all_stat_stable"]):
        raise ValueError("batch policy or observation contract differs")
    cpu_sets = roster["cpu_sets"]
    if (len(cpu_sets) != 2 or len(set(cpu_sets)) != 2
            or any(not re.fullmatch(r"\d+-\d+", value) for value in cpu_sets)):
        raise ValueError("two explicit CPU ranges required")
    ranges = [set(range(int(v.split("-")[0]), int(v.split("-")[1]) + 1)) for v in cpu_sets]
    if any(not v for v in ranges) or ranges[0] & ranges[1]:
        raise ValueError("CPU ranges overlap or are empty")
    for filename, digest in roster["source_sha256"].items():
        adapter.capture(Path(filename), pins, digest)
    here = Path(__file__).resolve().parent
    derive, postcheck = here / "derive_matrix_radius.py", here / "summarize_dependency_set_proposals.py"
    for tool in (derive, postcheck, Path(__file__).resolve()):
        if roster["source_sha256"].get(str(tool)) != sha(tool):
            raise ValueError("batch tool source pin missing")
    observed_hashes, observed_stats = {}, {}
    for collection in receipt["outputs"].values():
        for basename, digest in collection["files"].items():
            observed_hashes[str(Path(collection["path"]) / basename)] = digest
        observed_stats.update(collection["stat"])
    fields = ("device", "inode", "size", "mtime_ns", "ctime_ns")
    stats = {name: dict(zip(fields, value, strict=True)) for name, value in observed_stats.items()}
    for name, expected in observed_stats.items():
        if tuple(expected) != adapter.stamp(Path(name)):
            raise ValueError("completed observation stat changed")
    rows, checkpoints = [], []
    labels = [r["label"] for r in roster["rows"]]
    if (not 1 <= len(labels) <= 6 or len(labels) != len(set(labels))
            or any(not re.fullmatch(r"[A-Z]{1,3}", label) for label in labels)):
        raise ValueError("one through six unique safe labels required")
    for row in roster["rows"]:
        path = Path(row["plan"]).resolve()
        checkpoint, plan, collections, _, found = adapter.prepare(path)
        if found[str(path)][0] != row["plan_sha256"]:
            raise ValueError("reference plan hash differs")
        pins.update(found)
        if (plan["parent_state"] != roster["parent"] or plan["eligible_blocks"] != ["matrix"]
                or len(plan["candidates"]) != 1 or plan["candidates"][0]["id"] != row["id"]
                or plan["prospective_method"] != roster["policy"]):
            raise ValueError("reference plan identity differs")
        if (declared_metric is not None
                and plan["candidates"][0]["cost_metric"]["type"] != declared_metric):
            raise ValueError("declared calibration metric differs from candidate")
        if any(observed_hashes.get(name) != digest
               for name, digest in plan["completed_collection_sha256"].items()):
            raise ValueError("reference observation hashes differ")
        observation_root = Path(roster["observation_root"]).resolve()
        if any(value["path"].resolve().parent != observation_root for value in collections.values()):
            raise ValueError("observation root differs")
        adapter.check_output(args.output.resolve(), checkpoint, path, collections, pins)
        rows.append((row, plan))
        checkpoints.append(checkpoint)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("fresh output required")
    for checkpoint in checkpoints:
        adapter.stable(checkpoint, pins)
    return roster, policy, pins, stats, rows, checkpoints, derive, postcheck


def execute_method(row, reference, policy, stats, roster_path, pins, root, derive, cpu_set, env):
    folder = root / row["label"]
    folder.mkdir()
    result = dict(id=row["id"], label=row["label"], status="not_started", cpu_set=cpu_set)
    try:
        if STOP.is_set():
            raise InterruptedError("batch interrupted before method")
        plan = copy.deepcopy(reference)
        plan["reference_plan"] = dict(path=row["plan"], sha256=row["plan_sha256"])
        plan["completed_collection_stat"] = stats
        plan["analytic_calibration"] = dict(policy["analytic_calibration"],
                                            metric=plan["candidates"][0]["cost_metric"]["type"])
        plan["fixed_method_inputs"].update({name: value[0] for name, value in pins.items()
                                            if name == str(roster_path.resolve())
                                            or name == str(Path(__file__).resolve())})
        calibration_plan = folder / "calibration-plan.json"
        write(calibration_plan, plan)
        cal = folder / "calibration"
        result["calibration"] = command(
            [sys.executable, "-B", str(derive), "--plan", str(calibration_plan),
             "--plan-sha256", sha(calibration_plan), "--output", str(cal), "--run"],
            folder / "calibration.log", 120, cpu_set, env)
        if result["calibration"]["returncode"] != 0:
            result["status"] = "calibration_failed"
            return result
        derived = cal / "candidate-plan.json"
        completion = json.loads((cal / "completion.json").read_text())
        if completion["status"] != "completed":
            raise ValueError("calibration completion missing")
        candidate = folder / "candidate"
        result["candidate"] = command(
            [sys.executable, "-B", str(derive), "--plan", str(derived),
             "--plan-sha256", sha(derived), "--phase", "candidate",
             "--output", str(candidate), "--run"],
            folder / "candidate.log", 180, cpu_set, env)
        if result["candidate"]["returncode"] != 0:
            result["status"] = "candidate_failed"
            return result
        if json.loads((candidate / "completion.json").read_text())["status"] != "completed":
            raise ValueError("candidate completion missing")
        result.update(status="completed", candidate_row=dict(
            label=row["label"], plan=str(derived), plan_sha256=sha(derived),
            output=str(candidate / "proposals"), ids=[row["id"]]))
        return result
    except (Exception, KeyboardInterrupt) as exc:
        result.update(status="failed", exception_type=type(exc).__name__)
        return result
    finally:
        write(folder / "stage-result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", required=True, type=Path)
    parser.add_argument("--roster-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    started, owned = time.monotonic(), False
    old_handlers = {}
    try:
        roster, policy, pins, stats, rows, checkpoints, derive, postcheck = prepare(args)
        if not args.run:
            print(json.dumps(dict(status="metadata_preview", methods=len(rows), workers=2,
                                  calibration_seconds=120, candidate_seconds=180,
                                  gradient_reads=0, solves=0, model_calls=0)))
            return
        args.output.mkdir(parents=True)
        owned = True
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, lambda *_: STOP.set())
        write(args.output / "started.json", dict(pid=os.getpid(), epoch=time.time(),
              roster_sha256=args.roster_sha256, source_sha256={p: v[0] for p, v in pins.items()}))
        env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
        repo = Path(__file__).resolve().parents[3]
        env["PYTHONPATH"] = str(repo / "src")
        def worker(index):
            return [execute_method(row, plan, policy, stats, args.roster, pins,
                                   args.output, derive, roster["cpu_sets"][index], env)
                    for row, plan in rows[index::2]]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, i) for i in range(2)]
            results = [value for future in futures for value in future.result()]
        results.sort(key=lambda r: [x["label"] for x in roster["rows"]].index(r["label"]))
        completed = [r["candidate_row"] for r in results if r["status"] == "completed"]
        candidate_roster = args.output / "candidate-roster.json"
        write(candidate_roster, dict(schema_version=1, parent=roster["parent"], rows=completed,
              failures=[r for r in results if r["status"] != "completed"]))
        numeric = args.output / "numeric-postcheck.json"
        if completed and not STOP.is_set():
            checked = command([sys.executable, "-B", str(postcheck),
                               "--roster", str(candidate_roster),
                               "--observation-root", roster["observation_root"],
                               "--output", str(numeric)],
                              args.output / "postcheck.log", 180,
                              ",".join(roster["cpu_sets"]), env)
            if checked["returncode"] != 0:
                raise ValueError("numeric postcheck failed")
        for checkpoint in checkpoints:
            adapter.stable(checkpoint, pins)
        result = dict(status="completed" if len(completed) == len(rows) else "completed_with_method_failures",
                      seconds=time.monotonic()-started, results=results, inputs_stable=True,
                      no_retries=True, model_calls=0, checkpoint_writes=0,
                      candidate_roster_sha256=sha(candidate_roster),
                      numeric_postcheck_sha256=sha(numeric) if numeric.exists() else None)
        if STOP.is_set():
            result["status"] = "interrupted"
        write(args.output / "completion.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "results"}))
    except BaseException as exc:
        STOP.set()
        if owned and not (args.output / "failure.json").exists():
            write(args.output / "failure.json", dict(status="failed", exception_type=type(exc).__name__,
                  seconds=time.monotonic()-started, no_automatic_retry=True))
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
