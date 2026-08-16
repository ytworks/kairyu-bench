from __future__ import annotations

import argparse
import os
import re
import secrets
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from kairyu_bench.benchmarks import BENCHMARK_NAMES, HARBOR_AGENT_NAMES
from kairyu_bench.manifest import ManifestError, load_manifest, select_benchmarks
from kairyu_bench.runner import RunConfig, run_benchmarks
from kairyu_bench.reporting import compare_runs, comparison_markdown
from kairyu_bench.target import Endpoint, PreflightError, TargetClient


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) or ".." in value:
        raise argparse.ArgumentTypeError(
            "must contain only letters, digits, dot, underscore, or dash"
        )
    return value


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{secrets.token_hex(4)}"


def _default_app_root() -> Path:
    configured = os.environ.get("KAIRYU_BENCH_APP_ROOT")
    if configured:
        return Path(configured)
    repository = Path(__file__).resolve().parents[2]
    return repository if (repository / "adapters").is_dir() else Path("/app")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kairyu-bench")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="list supported benchmark adapters")

    run = commands.add_parser("run", help="run benchmarks against an API")
    run.add_argument("endpoint", help="Kairyu-compatible API base URL")
    run.add_argument(
        "--only",
        help="comma-separated benchmark names (default: all)",
    )
    run.add_argument(
        "--limit", type=_positive_integer, help="maximum problems per benchmark"
    )
    run.add_argument(
        "--harbor-agent",
        choices=HARBOR_AGENT_NAMES,
        default="terminus-2",
        help="agent for Harbor benchmarks (default: terminus-2)",
    )
    run.add_argument(
        "--results-dir",
        type=Path,
        default=Path(os.environ.get("KAIRYU_BENCH_RESULTS_DIR", "/work/results")),
        help=argparse.SUPPRESS,
    )
    run.add_argument("--run-id", type=_run_id, default=None, help=argparse.SUPPRESS)
    run.add_argument(
        "--adapter-timeout",
        type=_positive_integer,
        default=None,
        help=argparse.SUPPRESS,
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="validate arguments without contacting the endpoint",
    )

    compare = commands.add_parser("compare", help="compare two result runs")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        print("\n".join(BENCHMARK_NAMES))
        return 0
    if args.command == "run":
        try:
            endpoint = Endpoint.parse(args.endpoint)
            selected = select_benchmarks(args.only)
        except (ValueError, ManifestError) as error:
            parser.error(str(error))
        if args.dry_run:
            print(f"endpoint: {endpoint.base_url}")
            print(f"benchmarks: {len(selected)}")
            print("selected: " + ",".join(selected))
            print(f"limit: {args.limit if args.limit is not None else 'all'}")
            print(f"harbor agent: {args.harbor_agent}")
            return 0

        client = TargetClient(
            endpoint,
            api_key=os.environ.get("KAIRYU_API_KEY"),
        )
        config = RunConfig(
            endpoint=endpoint,
            selected=tuple(selected),
            limit=args.limit,
            results_root=args.results_dir,
            run_id=args.run_id or _new_run_id(),
            app_root=_default_app_root(),
            harbor_agent=args.harbor_agent,
            adapter_timeout=args.adapter_timeout,
        )
        try:
            outcome = run_benchmarks(config, client, load_manifest())
        except (PreflightError, FileExistsError, OSError) as error:
            print(f"kairyu-bench: {error}", file=sys.stderr)
            return 2
        print(f"run: {outcome.run_dir.name}")
        print(f"model: {outcome.model_id}")
        print(f"results: {outcome.run_dir}")
        return outcome.exit_code
    if args.command == "compare":
        try:
            comparison = compare_runs(Path(args.baseline), Path(args.candidate))
        except (OSError, ValueError) as error:
            print(f"kairyu-bench: {error}", file=sys.stderr)
            return 2
        print(comparison_markdown(comparison), end="")
        return 0
    raise AssertionError(f"unknown command: {args.command}")
