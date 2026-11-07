#!/usr/bin/env python3
"""Run a pinned alternate derived-radius runtime without changing historical source paths."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--derive", required=True, type=Path)
    parser.add_argument("--derive-sha256", required=True)
    parser.add_argument("--roster", required=True, type=Path)
    parser.add_argument("--roster-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    script_dir = args.repo.resolve() / "scripts/experiments/gemma4-purpose"
    sys.path[:0] = [str(script_dir), str(args.repo.resolve() / "src")]
    import run_derived_matrix_batch as driver
    started, owned, handlers = time.monotonic(), False, {}
    try:
        roster, policy, pins, stats, rows, checkpoints, original, postcheck = driver.prepare(args)
        runtime = roster["compatibility_runtime"]
        derive = args.derive.resolve()
        if runtime != {"path": str(derive), "sha256": args.derive_sha256,
                       "original_path": str(original), "original_sha256": driver.sha(original)}:
            raise ValueError("alternate runtime/source correspondence differs")
        driver.adapter.capture(derive, pins, args.derive_sha256)
        driver.adapter.capture(Path(__file__), pins, roster["orchestration_sha256"])
        for checkpoint in checkpoints:
            driver.adapter.check_output(args.output.resolve(), checkpoint, args.roster, {}, pins)
            driver.adapter.stable(checkpoint, pins)
        if not args.run:
            print(json.dumps(dict(status="metadata_preview", methods=len(rows), workers=2,
                                  gradient_reads=0, solves=0, model_calls=0)))
            return
        args.output.mkdir(parents=True)
        owned = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, lambda *_: driver.STOP.set())
        driver.write(args.output / "started.json", dict(pid=os.getpid(), epoch=time.time(),
                     roster_sha256=args.roster_sha256, source_sha256={p: v[0] for p, v in pins.items()}))
        env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONPATH=os.pathsep.join([str(script_dir), str(args.repo.resolve() / "src")]))
        def worker(index):
            return [driver.execute_method(row, plan, policy, stats, args.roster, pins,
                    args.output, derive, roster["cpu_sets"][index], env)
                    for row, plan in rows[index::2]]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, index) for index in range(2)]
            results = [row for future in futures for row in future.result()]
        results.sort(key=lambda row: [r["label"] for r in roster["rows"]].index(row["label"]))
        completed = [r["candidate_row"] for r in results if r["status"] == "completed"]
        candidate_roster = args.output / "candidate-roster.json"
        driver.write(candidate_roster, dict(schema_version=1, parent=roster["parent"], rows=completed,
                     failures=[r for r in results if r["status"] != "completed"]))
        numeric = args.output / "numeric-postcheck.json"
        if completed and not driver.STOP.is_set():
            checked = driver.command([sys.executable, "-B", str(postcheck), "--roster",
                    str(candidate_roster), "--observation-root", roster["observation_root"],
                    "--output", str(numeric)], args.output / "postcheck.log", 180,
                    ",".join(roster["cpu_sets"]), env)
            if checked["returncode"] != 0:
                raise ValueError("numeric postcheck failed")
        for checkpoint in checkpoints:
            driver.adapter.stable(checkpoint, pins)
        result = dict(status="completed" if len(completed) == len(rows) else "completed_with_method_failures",
                      seconds=time.monotonic()-started, results=results, inputs_stable=True,
                      no_retries=True, model_calls=0, checkpoint_writes=0,
                      candidate_roster_sha256=driver.sha(candidate_roster),
                      numeric_postcheck_sha256=driver.sha(numeric) if numeric.exists() else None)
        if driver.STOP.is_set():
            result["status"] = "interrupted"
        driver.write(args.output / "completion.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "results"}))
    except BaseException as exc:
        driver.STOP.set()
        if owned and not (args.output / "failure.json").exists():
            driver.write(args.output / "failure.json", dict(status="failed",
                         exception_type=type(exc).__name__, seconds=time.monotonic()-started,
                         no_automatic_retry=True))
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
