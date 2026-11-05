"""Host CPU commands for configuring, merging and viewing activation captures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from kairyu_bench.artifacts import read_object, validate_output_path, write_json
from kairyu_bench.wsp.observe.core import compare_reports, initialize_capture, merge_capture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kairyu-bench wsp observe")
    commands = parser.add_subparsers(dest="operation", required=True)
    init = commands.add_parser("init", help="fingerprint a checkpoint and create capture configuration")
    init.add_argument("checkpoint", type=Path)
    init.add_argument("--output-dir", type=Path, required=True)
    init.add_argument("--model", required=True)
    init.add_argument("--tp-size", type=int, default=2)
    init.add_argument("--replicas", type=int, default=4)
    init.add_argument("--mtp-tokens", type=int, default=4)
    init.add_argument("--mode", choices=("live", "replay", "teacher_forced"), default="live")
    merge = commands.add_parser("merge", help="join worker records and benchmark outcomes")
    merge.add_argument("capture", type=Path)
    merge.add_argument("--results", type=Path)
    merge.add_argument("--token-bin-size", type=int, default=64)
    merge.add_argument("--task", help="limit the detailed report to this item ID")
    merge.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare", help="combine observations for a paired display")
    compare.add_argument("base", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    view = commands.add_parser("view", help="write a self-contained interactive HTML report")
    view.add_argument("report", type=Path)
    view.add_argument("--weights", type=Path, help="optional weights inspect/diff JSON")
    view.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.operation == "init":
            result = initialize_capture(
                args.checkpoint, args.output_dir, model=args.model, tp_size=args.tp_size,
                replicas=args.replicas, mtp_tokens=args.mtp_tokens, mode=args.mode,
            )
            print(Path(result["output_dir"]) / "capture.json")
            return 0
        if args.operation == "merge":
            validate_output_path(args.output, [args.capture, *([args.results] if args.results else [])])
            result = merge_capture(
                args.capture, results=args.results, token_bin_size=args.token_bin_size,
                task_filter=args.task,
            )
        elif args.operation == "compare":
            validate_output_path(args.output, [args.base, args.candidate])
            result = compare_reports(read_object(args.base, "base report"), read_object(args.candidate, "candidate report"))
        else:
            from kairyu_bench.wsp.observe.report import write_report

            validate_output_path(args.output, [args.report, *([args.weights] if args.weights else [])])
            result = read_object(args.report, "observation report")
            if args.weights:
                result["weights"] = read_object(args.weights, "weight observations")
            write_report(result, args.output)
            print(args.output)
            return 0
        write_json(args.output, result)
        print(args.output)
        return 0
    except (ValueError, OSError) as error:
        parser.exit(2, f"kairyu-bench wsp observe: {error}\n")
