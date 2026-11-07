#!/usr/bin/env python3
"""Score all16 fixed helper stages from an unchanged72-response envelope."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import helper_chain_cases as short
import long_helper_chain_cases as long
import science_generate as generator
import science_sandbox as sandbox

CONTRACT_ISSUES = {
    "chain-whitened-fit-1",
    "chain-quadrature-mixture-1",
    "chain-boundary-response-1",
}


def targets():
    result = {}
    for module in (short, long):
        for chain in module.catalog():
            if chain.split == "train":
                for stage in (0, 1):
                    row = chain.row(stage)
                    if row["id"] in result:
                        raise ValueError("duplicate source stage")
                    result[row["id"]] = (module, chain, stage)
    if len(result) != 16:
        raise ValueError("requires all16 fixed train stages")
    return result


def read_inputs(cases_path, responses_path, cases_sha256, responses_sha256):
    case_bytes, case_receipt = sandbox.capture(cases_path, 8 * 1024 * 1024)
    response_bytes, response_receipt = sandbox.capture(responses_path, 16 * 1024 * 1024)
    if (case_receipt["sha256"] != cases_sha256
            or response_receipt["sha256"] != responses_sha256):
        raise ValueError("pinned input SHA mismatch")
    data, responses = json.loads(case_bytes), json.loads(response_bytes)
    public = generator.public_cases(data, "train")
    if len(public) != 72:
        raise ValueError("requires the complete72 generation input")
    rows = {row["id"]: row for folds in data["cases"].values()
            for fold in folds for row in fold}
    selected = targets()
    for identifier, (_, chain, stage) in selected.items():
        if rows.get(identifier) != chain.row(stage):
            raise ValueError("missing or changed fixed helper stage")
    if (responses.get("schema_version") != 1 or responses.get("split") != "train"
            or responses.get("cases_sha256") != cases_sha256):
        raise ValueError("case/response envelope identity mismatch")
    answers, generation = sandbox.read_public_answers(responses, set(rows))
    return selected, answers, generation, [case_receipt, response_receipt], rows


def score_answer(target, row, run, seed, trials):
    module, chain, stage = target
    result = {key: row[key] for key in (
        "request_id", "finish_reason", "usage", "seconds", "error_type", "http_status"
    ) if key in row}
    result.update(status="missing_response", passed=None)
    answer = row.get("public_answer")
    if row.get("error") or row.get("error_type"):
        result["status"] = "generation_error"
        return result
    if answer is None:
        return result
    result["public_answer_sha256"] = sandbox.sha(answer.encode())
    try:
        code = sandbox.extract(answer)
    except (ValueError, SyntaxError):
        result.update(status="format_or_syntax_error", passed=False)
        return result
    result["public_code_sha256"] = sandbox.sha(code.encode())
    # This helper is already in the fixed public prompt, never the generated
    # stage1 answer. long.score adds its public support exactly once.
    if stage:
        helper = short.as_helper(chain.helper.code())
        result["supplied_helper_sha256"] = sandbox.sha(helper.encode())
        code = helper + code
    result["public_support_sha256"] = (
        sandbox.sha(chain.support.encode()) if module is long else None
    )
    began = time.monotonic()
    try:
        result.update(module.score(chain, stage, code, run, seed, trials))
    except sandbox.SafetyError:
        raise
    except TimeoutError:
        result.update(status="execution_timeout", passed=None, error_category="wall_timeout")
    except sandbox.SandboxError as error:
        result.update(status="sandbox_or_protocol_error", passed=None,
                      error_category=error.category, return_code=error.return_code)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        result.update(status="sandbox_or_protocol_error", passed=None,
                      error_category=type(error).__name__)
    result["execution_seconds"] = time.monotonic() - began
    return result


def counts(rows, expected):
    passed = sum(row["passed"] is True for row in rows)
    failed = sum(row["passed"] is False for row in rows)
    return {"expected": expected, "passed": passed, "failed": failed,
            "unknown": expected - passed - failed}


def main(argv=None, docker_factory=sandbox.Docker):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "responses"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docker-socket", type=Path, required=True)
    parser.add_argument("--daemon-id", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--memory-mib", type=int, default=512)
    parser.add_argument("--cpus", type=float, default=1)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if (not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image)
            or not args.docker_socket.is_absolute() or not args.daemon_id
            or any(not re.fullmatch(r"[0-9a-f]{64}", x)
                   for x in (args.cases_sha256, args.responses_sha256))):
        raise ValueError("explicit pinned input/image/socket/daemon required")
    if (not 4 <= args.trials <= 100 or args.trials % 2
            or not 1 <= args.seconds <= 120 or not 128 <= args.memory_mib <= 4096
            or not 0 < args.cpus <= 4):
        raise ValueError("resource bounds outside allowed range")
    if os.path.lexists(args.output):
        raise FileExistsError(args.output)
    selected, answers, generation, inputs, rows = read_inputs(
        args.cases, args.responses, args.cases_sha256, args.responses_sha256
    )
    for source in (Path(__file__), Path(short.__file__), Path(long.__file__),
                   Path(sandbox.__file__), Path(generator.__file__)):
        _, receipt = sandbox.capture(source, 1024 * 1024)
        inputs.append(receipt)
    output = args.output.resolve()
    if any(output == Path(item["path"]).resolve()
           or Path(item["path"]).resolve().is_relative_to(output) for item in inputs):
        raise ValueError("output overlaps pinned inputs")
    report = {
        "schema_version": 1, "status": "preview", "inputs": inputs,
        "generation": generation, "input_cases": 72, "target_cases": 16, "not_targeted_cases": 56,
        "target_ids": list(selected), "not_targeted_ids": sorted(set(rows) - set(selected)),
        "response_ids": sorted(answers), "missing_response_ids": sorted(set(rows) - set(answers)),
        "scope": "Additional known calibration diagnostic; separate from the frozen48 and WSP.",
        "contract_issue_ids": sorted(CONTRACT_ISSUES), "results": [],
        "seed": args.seed, "trials": args.trials,
        "sandbox": {"socket": str(args.docker_socket), "daemon_id": args.daemon_id,
                    "image": args.image, "seconds": args.seconds,
                    "memory_mib": args.memory_mib, "cpus": args.cpus,
                    "network": "none", "user": "1003:1003", "host_mounts": [], "gpus": False},
        "runtime": {"python": sys.version, **{
            name: importlib.metadata.version(name) for name in ("numpy",)
        }},
        "limits": [
            "All16 fixed stages; other56 are not targeted, never counted as failures.",
            "Three short-helper dtype counterexamples are not covered by the float64 sampler.",
            "Numeric fixtures do not prove helper reuse, non-redefinition or all prompt contracts.",
            "Only public prompt helper/support is prepended; expected values stay on the host.",
            "No model request, generated-helper substitution, or host execution of generated code.",
        ],
    }
    output.mkdir(parents=True)
    report_path = output / "report.json"
    sandbox.save(report_path, report)
    if not args.run:
        return report
    docker = docker_factory(args.docker_socket, args.daemon_id, args.image,
                            seconds=args.seconds, memory_mib=args.memory_mib, cpus=args.cpus)
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(_signum, _frame):
        raise KeyboardInterrupt("termination requested")

    signal.signal(signal.SIGTERM, terminate)
    try:
        docker.check_image()
        report["status"] = "running"
        for identifier, target in selected.items():
            sandbox.continuity(inputs)
            item = score_answer(target, answers.get(identifier, {}), docker.run,
                                args.seed, args.trials)
            item.update(id=identifier, contract_issue=identifier in CONTRACT_ISSUES)
            report["results"].append(item)
            sandbox.save(report_path, report)
        sandbox.continuity(inputs)
        report["status"] = "completed"
    except BaseException:
        report["status"] = "interrupted_or_error"
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        report["counts"] = counts(report["results"], 16)
        report["strata"] = {
            "known_dtype_contract_issue": counts(
                [x for x in report["results"] if x["contract_issue"]], 3),
            "other": counts([x for x in report["results"] if not x["contract_issue"]], 13),
        }
        sandbox.save(report_path, report)
    return report


if __name__ == "__main__":
    main()
