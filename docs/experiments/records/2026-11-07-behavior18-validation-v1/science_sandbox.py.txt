#!/usr/bin/env python3
"""Execute public science answers only inside a bounded, explicitly selected Docker.

This is an experiment recipe, not the WSP scorer. The host never executes an
answer. Only sampled inputs (no expected values or reference code) enter the
container; numeric comparison takes place on the host.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import threading
import time
import uuid

import numpy as np

import science_cases


class SafetyError(Exception):
    """Fail closed instead of continuing after an ownership/cleanup failure."""


class SandboxError(RuntimeError):
    def __init__(self, category: str, return_code: int | None = None):
        super().__init__(category)
        self.category = category
        self.return_code = return_code


MAX_OUTPUT = 1024 * 1024
LABEL = "kairyu.science-sandbox"
WORKER = r"""
import contextlib, inspect, json, sys
import numpy as np
payload = json.load(sys.stdin)
def decode(v):
    if isinstance(v, dict):
        return np.asarray(v["data"], dtype=v["dtype"]).reshape(v["shape"])
    return v
namespace = {}
rows = []
with open("/dev/null", "w") as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
    try:
        exec(compile(payload["code"], "<public-answer>", "exec"), namespace)
        solve = namespace["solve"]
        if list(inspect.signature(solve).parameters) != payload["parameters"]:
            raise TypeError("interface")
    except BaseException as error:
        result = {"setup_error": type(error).__name__}
    else:
        for packed in payload["trials"]:
            args = [decode(v) for v in packed]
            copies = [v.copy() if isinstance(v, np.ndarray) else v for v in args]
            try:
                with np.errstate(all="raise", under="ignore" if payload.get("allow_underflow") is True else "raise"):
                    out = solve(*args)
                unchanged = all(not isinstance(a, np.ndarray) or np.array_equal(a, b)
                                for a, b in zip(copies, args))
                if not isinstance(out, np.ndarray):
                    row = {"error": "output_type"}
                elif out.size > 65536:
                    row = {"error": "output_size"}
                elif out.dtype != np.float64:
                    row = {"error": "output_dtype"}
                elif not np.isfinite(out).all():
                    row = {"error": "nonfinite"}
                else:
                    row = {"shape": list(out.shape), "dtype": str(out.dtype),
                           "data": out.tolist(), "unchanged": bool(unchanged)}
            except BaseException as error:
                row = {"error": type(error).__name__}
            rows.append(row)
        result = {"trials": rows}
sys.stdout.write(json.dumps(result, allow_nan=False))
"""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stamp(path: Path) -> dict:
    value = path.stat()
    return {
        key: getattr(value, "st_" + key)
        for key in ("dev", "ino", "size", "mtime_ns", "ctime_ns")
    }


def capture(path: Path, limit: int) -> tuple[bytes, dict]:
    before = stamp(path)
    if before["size"] > limit or not path.is_file():
        raise ValueError("input too large or not regular file")
    data = path.read_bytes()
    if stamp(path) != before or len(data) != before["size"]:
        raise ValueError("input changed during read")
    return data, {"path": str(path.resolve()), "sha256": sha(data), "stat": before}


def continuity(inputs: list[dict]) -> None:
    for row in inputs:
        path = Path(row["path"])
        data, now = capture(path, 16 * 1024 * 1024)
        if now["stat"] != row["stat"] or sha(data) != row["sha256"]:
            raise ValueError("input or source changed")


def extract(answer: str) -> str:
    text = answer.strip()
    if text.startswith("~~~"):
        raise ValueError("unsupported code fence")
    if text.startswith(chr(96) * 3):
        match = re.fullmatch(r"\x60{3}(?:python|py)?\s*\n(.*?)\n\x60{3}", text, re.S)
        if not match:
            raise ValueError("answer is not one complete Python code block")
        text = match.group(1)
    if chr(96) * 3 in text:
        raise ValueError("mixed prose or multiple code blocks")
    ast.parse(text)
    return text + "\n"


def pack(value):
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def prepare(
    case, code: str, seed: int, trials: int, sample_fn=science_cases.sample
) -> tuple[dict, list]:
    seed += int.from_bytes(hashlib.sha256(case.name.encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    packed, expected = [], []
    for trial in range(trials):
        args, reference = sample_fn(case.name, rng, trial)
        packed.append([pack(value) for value in args])
        expected.append(reference)
    return {
        "code": code,
        "parameters": case.parameters.split(", "),
        "trials": packed,
        **(
            {"allow_underflow": True}
            if getattr(case, "allow_underflow", False) is True
            else {}
        ),
    }, expected


def compare(result: dict, expected: list) -> dict:
    def exception_fields(value, phase: str) -> dict:
        # Worker class names are untrusted. Export only fixed categories, never
        # custom names, messages, tracebacks, or arbitrary protocol values.
        allowed = {
            "NameError", "UnboundLocalError", "TypeError", "ValueError",
            "IndexError", "KeyError", "AttributeError", "ZeroDivisionError",
            "FloatingPointError", "OverflowError", "MemoryError", "RecursionError",
            "ImportError", "ModuleNotFoundError", "SyntaxError", "IndentationError",
            "LinAlgError", "RuntimeError", "NotImplementedError",
        }
        category = value if type(value) is str and value in allowed else "other"
        return {"phase": phase, "exception_category": category}

    if result.get("setup_error"):
        return {
            "status": "evaluated",
            "passed": False,
            "tests_passed": 0,
            "tests_total": len(expected),
            "failures": [{
                "kind": "runtime_or_interface",
                **exception_fields(result["setup_error"], "setup"),
            }],
        }
    rows = result.get("trials")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("sandbox response trial count mismatch")
    failures = []
    for index, (row, reference) in enumerate(zip(rows, expected)):
        kind = None
        diagnostic = {}
        if not isinstance(row, dict):
            raise ValueError("invalid sandbox trial")
        if "error" in row:
            known = {"output_type", "output_size", "output_dtype", "nonfinite"}
            if type(row["error"]) is str and row["error"] in known:
                kind = row["error"]
            else:
                kind = "runtime"
                diagnostic = exception_fields(row["error"], "execution")
        else:
            if row.get("dtype") != "float64":
                kind = "output_dtype"
            elif row.get("shape") != list(reference.shape):
                kind = "output_shape"
            elif row.get("unchanged") is not True:
                kind = "input_mutation"
            else:
                try:
                    value = np.asarray(row["data"], dtype=np.float64)
                    if value.shape != reference.shape:
                        kind = "output_shape"
                    elif not np.isfinite(value).all():
                        kind = "nonfinite"
                    elif not np.allclose(value, reference, rtol=2e-6, atol=2e-7):
                        kind = "numeric"
                except (KeyError, TypeError, ValueError, OverflowError):
                    raise ValueError("invalid sandbox numeric output") from None
        if kind:
            failures.append({"trial": index, "kind": kind, **diagnostic})
    return {
        "status": "evaluated",
        "passed": not failures,
        "tests_passed": len(expected) - len(failures),
        "tests_total": len(expected),
        "failures": failures,
    }


def bounded_command(
    argv: list[str],
    *,
    payload: bytes = b"",
    seconds: float = 20,
    max_output: int = MAX_OUTPUT,
) -> tuple[int, bytes]:
    """Bound both pipes; never retain/print container stderr or arbitrary errors."""
    process = subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    selector = selectors.DefaultSelector()
    output = bytearray()
    total = 0
    writer = None
    try:

        def write():
            try:
                process.stdin.write(payload)
                process.stdin.close()
            except (OSError, ValueError):
                pass

        writer = threading.Thread(target=write, daemon=True)
        writer.start()
        for pipe in (process.stdout, process.stderr):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ)
        deadline = time.monotonic() + seconds
        while selector.get_map():
            if time.monotonic() >= deadline:
                raise TimeoutError("sandbox command exceeded wall limit")
            for key, _ in selector.select(
                min(0.1, max(0, deadline - time.monotonic()))
            ):
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                total += len(chunk)
                if total > max_output:
                    raise ValueError("sandbox command output limit")
                if key.fileobj is process.stdout:
                    output.extend(chunk)
        remaining = max(0.01, deadline - time.monotonic())
        return process.wait(timeout=remaining), bytes(output)
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
        process.wait()
        if writer:
            writer.join(timeout=1)
        for pipe in (process.stdin, process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()


class Docker:
    def __init__(
        self,
        socket: Path,
        daemon_id: str,
        image: str,
        *,
        seconds: float,
        memory_mib: int,
        cpus: float,
        command=bounded_command,
    ):
        self.socket, self.daemon_id, self.image = socket, daemon_id, image
        self.seconds, self.memory_mib, self.cpus = seconds, memory_mib, cpus
        self.command = command
        self.socket_stat = stamp(socket)
        if not stat.S_ISSOCK(socket.stat().st_mode):
            raise ValueError("Docker socket must be an existing Unix socket")

    def cli(self, args: list[str], **kwargs):
        return self.command(
            ["docker", "--host", "unix://" + str(self.socket), *args], **kwargs
        )

    def guard(self):
        if stamp(self.socket) != self.socket_stat:
            raise SafetyError("Docker socket changed")
        code, output = self.cli(["info", "--format", "{{.ID}}"], max_output=65536)
        if code or output.decode().strip() != self.daemon_id:
            raise SafetyError("unexpected Docker daemon")

    def check_image(self):
        self.guard()
        code, output = self.cli(
            ["image", "inspect", "--format", "{{.Id}}", self.image], max_output=65536
        )
        if code or output.decode().strip() != self.image:
            raise ValueError("pinned image not available locally")

    def arguments(self, name: str, nonce: str) -> list[str]:
        return [
            "run",
            "--rm",
            "-i",
            "--name",
            name,
            "--label",
            LABEL + "=" + nonce,
            "--pull",
            "never",
            "--runtime",
            "runc",
            "--network",
            "none",
            "--read-only",
            "--user",
            "1003:1003",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--cpus",
            str(self.cpus),
            "--memory",
            str(self.memory_mib) + "m",
            "--memory-swap",
            str(self.memory_mib) + "m",
            "--pids-limit",
            "64",
            "--ulimit",
            "nofile=64:64",
            "--log-driver",
            "none",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=16m,mode=1777",
            "--workdir",
            "/tmp",
            "--env",
            "OPENBLAS_NUM_THREADS=1",
            "--env",
            "OMP_NUM_THREADS=1",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "CUDA_VISIBLE_DEVICES=",
            "--env",
            "NVIDIA_VISIBLE_DEVICES=void",
            "--entrypoint",
            "python3",
            self.image,
            "-I",
            "-c",
            WORKER,
        ]

    def cleanup(self, name: str, nonce: str):
        self.guard()
        code, listing = self.cli(
            [
                "container",
                "ls",
                "--all",
                "--filter",
                "name=^/" + name + "$",
                "--format",
                "{{.ID}}",
            ],
            max_output=65536,
        )
        if code:
            raise SafetyError("own container presence could not be checked")
        if not listing.strip():
            return
        template = '{{json .Id}} {{json (index .Config.Labels "' + LABEL + '")}}'
        code, output = self.cli(
            ["container", "inspect", "--format", template, name], max_output=65536
        )
        if code:
            raise SafetyError("own container identity could not be checked")
        match = re.fullmatch(r'"([0-9a-f]{64})" "([0-9a-f]{32})"\s*', output.decode())
        if not match or match.group(2) != nonce:
            raise SafetyError("refuse cleanup: container ownership differs")
        self.guard()
        code, _ = self.cli(
            ["container", "rm", "--force", match.group(1)], max_output=65536
        )
        if code:
            raise SafetyError("own container cleanup failed")

    def run(self, payload: dict) -> dict:
        nonce = uuid.uuid4().hex
        name = "kairyu-science-" + nonce
        self.guard()
        try:
            code, output = self.cli(
                self.arguments(name, nonce),
                payload=json.dumps(payload, allow_nan=False).encode(),
                seconds=self.seconds,
            )
            if code:
                raise SandboxError("container_exit", code)
            try:
                result = json.loads(output)
            except (ValueError, UnicodeDecodeError):
                raise SandboxError("invalid_json_protocol", code) from None
            if not isinstance(result, dict):
                raise SandboxError("nonobject_protocol", code)
            return result
        finally:
            self.cleanup(name, nonce)


def read_inputs(
    cases_path: Path,
    responses_path: Path,
    case_module=science_cases,
    catalog_revision: int = 1,
):
    case_bytes, case_receipt = capture(cases_path, 8 * 1024 * 1024)
    response_bytes, response_receipt = capture(responses_path, 16 * 1024 * 1024)
    data, responses = json.loads(case_bytes), json.loads(response_bytes)
    if responses.get("schema_version") != 1 or responses.get("split") not in {
        "train",
        "transfer",
    }:
        raise ValueError("response schema/split unsupported")
    if responses.get("cases_sha256") != case_receipt["sha256"]:
        raise ValueError("case/response identity mismatch")
    split = responses["split"]
    if split == "transfer":
        if data.get("split") != split or data.get("gradient_allowed") is not False:
            raise ValueError("invalid transfer envelope")
        rows = data["cases"]
    else:
        mapping = data.get("cases", data)
        rows = [row for folds in mapping.values() for fold in folds for row in fold]
    cases = (
        case_module.catalog()
        if case_module is science_cases
        else case_module.catalog(revision=catalog_revision)
    )
    catalog = {"science-" + c.name: c for c in cases if c.split == split}
    if len(rows) != len(catalog) or {row["id"] for row in rows} != set(catalog):
        raise ValueError("requires the complete authored split")
    for row in rows:
        if row != catalog[row["id"]].public():
            raise ValueError("authored case content mismatch")
    answers, generation = read_public_answers(responses, set(catalog))
    return catalog, answers, generation, [case_receipt, response_receipt]


def read_public_answers(responses: dict, allowed_ids: set[str]):
    """Validate public-only response rows without changing their envelope identity."""
    generation = responses.get("generation", {})
    if generation.get("reasoning_saved") is not False:
        raise ValueError("public-answer-only provenance required")
    allowed_generation = {
        "model_id",
        "checkpoint_sha256",
        "frame",
        "enable_thinking",
        "thinking_control",
        "reasoning_saved",
        "sampling",
        "temperature",
        "top_p",
        "max_tokens",
        "seed",
    }
    if set(generation) - allowed_generation:
        raise ValueError("unsupported generation provenance fields")
    if "thinking_control" in generation:
        control = generation["thinking_control"]
        if control not in ("request", "server-default") or (
            control == "request" and generation.get("enable_thinking") is not True
        ) or (
            control == "server-default" and generation.get("enable_thinking") is not None
        ):
            raise ValueError("inconsistent thinking-control provenance")
    answers = {}
    for row in responses.get("responses", []):
        if set(row) - {
            "id",
            "public_answer",
            "request_id",
            "finish_reason",
            "usage",
            "seconds",
            "error",
            "error_type",
            "http_status",
        }:
            raise ValueError(
                "unsupported response fields; no reasoning content accepted"
            )
        if row.get("id") not in allowed_ids or row["id"] in answers:
            raise ValueError("unknown or duplicate response ID")
        answer = row.get("public_answer")
        if "http_status" in row and (
            type(row["http_status"]) is not int
            or not 400 <= row["http_status"] <= 599
            or row.get("error_type") != "HTTPError"
            or answer is not None
        ):
            raise ValueError("invalid HTTP failure status")
        if answer is not None and (
            not isinstance(answer, str) or len(answer.encode()) > 262144
        ):
            raise ValueError("invalid or oversized public answer")
        usage = row.get("usage", {})
        if not isinstance(usage, dict) or any(
            key
            not in {
                "input_tokens",
                "output_tokens",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            }
            or (
                value is not None
                and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                )
            )
            for key, value in usage.items()
        ):
            raise ValueError("usage must contain numeric token counts only")
        answers[row["id"]] = row
    return answers, generation


def save(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def main(argv=None, docker_factory=Docker):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument(
        "--catalog",
        choices=(
            "basic",
            "advanced",
            "advanced-v2",
            "library-train-v1",
            "library-transfer-v1",
        ),
        default="basic",
    )
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--docker-socket", required=True, type=Path)
    parser.add_argument("--daemon-id", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--memory-mib", type=int, default=512)
    parser.add_argument("--cpus", type=float, default=1)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image):
        raise ValueError("image must be an existing immutable sha256 image ID")
    if not args.docker_socket.is_absolute() or not args.daemon_id:
        raise ValueError("explicit absolute Docker socket and daemon ID required")
    if not (
        4 <= args.trials <= 100
        and 1 <= args.seconds <= 120
        and 128 <= args.memory_mib <= 4096
        and 0 < args.cpus <= 4
    ):
        raise ValueError("resource bounds outside allowed range")
    if os.path.lexists(args.output):
        raise FileExistsError(args.output)
    case_module = science_cases
    if args.catalog in ("advanced", "advanced-v2"):
        import advanced_science_cases

        case_module = advanced_science_cases
    elif args.catalog == "library-train-v1":
        import library_train_cases

        case_module = library_train_cases
    elif args.catalog == "library-transfer-v1":
        import library_transfer_cases

        case_module = library_transfer_cases
    catalog, answers, generation, inputs = read_inputs(
        args.cases,
        args.responses,
        case_module=case_module,
        catalog_revision=2 if args.catalog == "advanced-v2" else 1,
    )
    sources = [Path(__file__), Path(science_cases.__file__)]
    if case_module is not science_cases:
        sources.append(Path(case_module.__file__))
    for source in sources:
        _, receipt = capture(source, 1024 * 1024)
        inputs.append(receipt)
    report = {
        "schema_version": 1,
        "status": "preview",
        "split": next(iter(catalog.values())).split,
        "catalog": args.catalog,
        "generation": generation,
        "inputs": inputs,
        "seed": args.seed,
        "trials": args.trials,
        "expected_cases": len(catalog),
        "responses_received": len(answers),
        "sandbox": {
            "image": args.image,
            "daemon_id": args.daemon_id,
            "socket": str(args.docker_socket),
            "seconds": args.seconds,
            "memory_mib": args.memory_mib,
            "cpus": args.cpus,
            "network": "none",
            "user": "1003:1003",
            "read_only": True,
            "host_mounts": [],
            "gpus": False,
            "output_limit_bytes": MAX_OUTPUT,
        },
        "results": [],
        "scope": "Authored calibration execution, not WSP score.",
        "limits": [
            "Only plain Python or one whole-answer Python fence is accepted; this differs from WSP extraction.",
            "Finite randomized tests do not prove correctness for all inputs.",
            "Docker is process isolation, not a formal hostile-code security boundary.",
            "Input mutation is checked inside the container; hostile protocol forgery is not ruled out.",
            "Transfer results must not be used as gradient input; repeated selection makes them known.",
        ],
    }
    args.output.mkdir(parents=True)
    output = args.output / "report.json"
    save(output, report)
    if not args.run:
        return report
    docker = docker_factory(
        args.docker_socket,
        args.daemon_id,
        args.image,
        seconds=args.seconds,
        memory_mib=args.memory_mib,
        cpus=args.cpus,
    )
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminate(_signum, _frame):
        raise KeyboardInterrupt("termination requested")

    signal.signal(signal.SIGTERM, terminate)
    try:
        docker.check_image()
        report["status"] = "running"
        for case_id, case in catalog.items():
            continuity(inputs)
            row = answers.get(case_id, {})
            item = {
                key: row[key]
                for key in (
                    "request_id", "finish_reason", "usage", "seconds",
                    "error_type", "http_status",
                )
                if key in row
            }
            item.update({"id": case_id, "status": "missing_response", "passed": None})
            if row.get("error") or row.get("error_type"):
                item["status"] = "generation_error"
            answer = row.get("public_answer")
            if (
                answer is not None
                and not row.get("error")
                and not row.get("error_type")
            ):
                item["public_answer_sha256"] = sha(answer.encode())
                try:
                    code = extract(answer)
                except (ValueError, SyntaxError):
                    item.update(status="format_or_syntax_error", passed=False)
                else:
                    item["code_sha256"] = sha(code.encode())
                    payload, expected = prepare(
                        case, code, args.seed, args.trials, sample_fn=case_module.sample
                    )
                    began = time.monotonic()
                    try:
                        item.update(compare(docker.run(payload), expected))
                    except TimeoutError:
                        item.update(
                            status="execution_timeout",
                            passed=None,
                            error_category="wall_timeout",
                        )
                    except SandboxError as error:
                        item.update(
                            status="sandbox_or_protocol_error",
                            passed=None,
                            error_category=error.category,
                            return_code=error.return_code,
                        )
                    except (
                        ValueError,
                        RuntimeError,
                        OSError,
                        subprocess.SubprocessError,
                    ) as error:
                        item.update(
                            status="sandbox_or_protocol_error",
                            passed=None,
                            error_category=type(error).__name__,
                        )
                    item["execution_seconds"] = time.monotonic() - began
            report["results"].append(item)
            save(output, report)
        continuity(inputs)
        report["status"] = "completed"
    except BaseException:
        report["status"] = "interrupted_or_error"
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        report["passed"] = sum(row["passed"] is True for row in report["results"])
        report["failed"] = sum(row["passed"] is False for row in report["results"])
        report["unknown"] = len(catalog) - report["passed"] - report["failed"]
        save(output, report)
    return report


if __name__ == "__main__":
    result = main()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in ("status", "expected_cases", "passed", "failed", "unknown")
            }
        )
    )
