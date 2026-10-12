"""Terminal-Bench 4.0 preflight (`prepare`) and artifact finalization (`finalize`)."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from kairyu_bench.artifacts import read_object, redact_artifacts, write_json
from kairyu_bench.terminal_bench4 import (
    BENCHMARK,
    LABEL,
    build_job_config,
    load_lock,
    model_name,
    select_tasks,
)
from kairyu_bench.terminal_bench4_accounting import summarize

SECRET_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "KAIRYU_API_KEY", "HF_TOKEN")
UNSUPPORTED_EXIT = 10
HOST_ALIAS = "host.docker.internal"


class Unsupported(RuntimeError):
    """The Docker daemon cannot run the selected official tasks unchanged."""


def docker_info() -> dict[str, Any]:
    output = subprocess.run(
        ["docker", "info", "--format", "{{json .}}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
    info = json.loads(output)
    if not isinstance(info, dict):
        raise ValueError("docker info did not return an object")
    return info


def egress_control_supported() -> bool:
    from harbor.environments.docker.docker import DockerEnvironment

    return bool(DockerEnvironment._egress_control_kernel_support())


def gateway_address(endpoint: str, resolve: Callable[[str], str] = socket.gethostbyname) -> str:
    """Nested task containers cannot use Docker's host-gateway inside a DinD daemon."""
    if urlsplit(endpoint).hostname != HOST_ALIAS:
        return "host-gateway"
    address = resolve(HOST_ALIAS)
    ipaddress.IPv4Address(address)
    return address


def write_overlay(raw_dir: Path, address: str) -> Path:
    path = raw_dir / "host-gateway.json"
    write_json(path, {"services": {"main": {"extra_hosts": [f"{HOST_ALIAS}:{address}"]}}})
    return path


def check_daemon(
    info: Mapping[str, Any], lock: Mapping[str, Any], tasks: list[str], egress: Callable[[], bool]
) -> dict[str, Any]:
    required = max(lock["tasks"][task]["cpus"] for task in tasks)
    if int(info.get("NCPU") or 0) < required:
        raise Unsupported(
            f"{LABEL} needs Docker to expose at least {required} CPUs; found {info.get('NCPU')}"
        )
    offline = sorted(set(tasks) & set(lock["offline_verifier_tasks"]))
    supported = egress() if offline else None
    if offline and not supported:
        raise Unsupported(
            f"{LABEL} offline verifiers ({', '.join(offline)}) need nftables egress control"
        )
    return {
        "server_version": info.get("ServerVersion"),
        "ncpu": info.get("NCPU"),
        "mem_total": info.get("MemTotal"),
        "runtimes": sorted(info.get("Runtimes") or {}),
        "required_cpus": required,
        "gpu_available": "nvidia" in (info.get("Runtimes") or {}),
        "egress_control_supported": supported,
    }


def prepare(
    context: Mapping[str, Any],
    raw_dir: Path,
    info: Mapping[str, Any],
    egress: Callable[[], bool],
    harbor_version: str,
    resolve: Callable[[str], str] = socket.gethostbyname,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    lock = load_lock()
    if harbor_version != lock["harbor_version"]:
        raise ValueError(f"{LABEL} needs Harbor {lock['harbor_version']}, found {harbor_version}")
    if context["agent"] == "codex" and "/" in context["model_id"]:
        # Pinned Harbor's Codex agent sends only the text after the last '/'.
        raise Unsupported(
            f"Harbor's Codex agent cannot preserve model IDs containing '/': {context['model_id']}"
        )
    tasks = select_tasks(lock, context.get("limit"))
    daemon = check_daemon(info, lock, tasks, egress)
    overlay = write_overlay(raw_dir, gateway_address(context["endpoint"], resolve))
    # Keep downloaded task definitions on the benchmark cache, not in the runner.
    task_cache = (cache_dir or Path(os.environ.get("KAIRYU_BENCH_CACHE_DIR", raw_dir))) / "harbor-tasks"
    config = build_job_config(context, raw_dir, overlay, task_cache)
    conditions = context["conditions"]
    write_json(raw_dir / "job-config.json", config)
    write_json(
        raw_dir / "selected.json",
        {
            "problem_ids": tasks,
            "attempts": conditions["attempts_per_task"],
            "agent": context["agent"],
            "model_id": context["model_id"],
            "model_name": model_name(conditions, context["model_id"]),
            "agent_kwargs": conditions["agent_kwargs"],
            "job_name": config["job_name"],
            "dataset": context["dataset"],
        },
    )
    (raw_dir / "instance-ids.txt").write_text("".join(task + "\n" for task in tasks))
    runtime = {
        "harbor_version": harbor_version,
        "harbor_revision": context["source"]["revision"],
        "dataset_ref": lock["dataset_ref"],
        "conditions": conditions,
        **daemon,
    }
    write_json(raw_dir / "runtime.json", runtime)
    return runtime


def finalize(raw_dir: Path, env: Mapping[str, str] = os.environ) -> None:
    redact_artifacts(raw_dir, SECRET_KEYS, env)
    if (raw_dir / "selected.json").is_file():
        write_json(raw_dir / "summary.json", summarize(raw_dir))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m kairyu_bench.terminal_bench4_cli")
    parser.add_argument("command", choices=("prepare", "finalize"))
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("--harbor-version", default="")
    args = parser.parse_args(argv)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.command == "finalize":
            finalize(args.raw_dir)
            return 0
        context = read_object(Path(os.environ["KAIRYU_BENCH_CONTEXT"]), LABEL)
        if context.get("benchmark") != BENCHMARK:
            raise ValueError(f"{LABEL} prepare received context for {context.get('benchmark')}")
        runtime = prepare(
            context, args.raw_dir, docker_info(), egress_control_supported, args.harbor_version
        )
    except Unsupported as error:
        (args.raw_dir / "unsupported.txt").write_text(str(error) + "\n", encoding="utf-8")
        print(f"{LABEL} unsupported: {error}", file=sys.stderr)
        return UNSUPPORTED_EXIT
    except (Exception, KeyboardInterrupt) as error:
        write_json(args.raw_dir / "runtime-error.json", {"type": type(error).__name__, "stage": args.command})
        print(f"{LABEL} {args.command} failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(
        f"{LABEL} prepared: tasks={len(read_object(args.raw_dir / 'selected.json', LABEL)['problem_ids'])} "
        f"attempts={runtime['conditions']['attempts_per_task']} docker_cpus={runtime['ncpu']} "
        f"gpu_available={runtime['gpu_available']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
