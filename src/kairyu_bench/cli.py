from __future__ import annotations

import argparse
from collections.abc import Sequence

from kairyu_bench.benchmarks import BENCHMARK_NAMES


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
    run.add_argument("--limit", type=int, help="maximum problems per benchmark")
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
    args = build_parser().parse_args(argv)
    if args.command == "list":
        print("\n".join(BENCHMARK_NAMES))
        return 0
    if args.command == "run" and args.dry_run:
        print(f"endpoint: {args.endpoint}")
        return 0
    if args.command == "run":
        raise SystemExit("benchmark execution is not implemented yet")
    raise SystemExit("comparison is not implemented yet")
